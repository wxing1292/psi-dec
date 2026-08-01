//! Model-neutral DSpark Markov sampling and confidence composition.
//!
//! ```text
//! input_token_ids[i]
//!   i = 0: sampled_token / anchor_token
//!   i > 0: spec_tokens[i - 1]
//!          |
//!          v
//! quantized Markov W1 embedding
//!          |
//!          +---------------------------------------------+
//!          |                                             |
//!          v                                             v
//! quantized Markov W2                         [hidden[i], Markov W1 embedding]
//!          |                                             |
//!          v                                             v
//! base_logits[i] + correction                 confidence projection + sigmoid
//!          |                                             |
//!          v                                             v
//! tile-local Top-K                              spec_confidences[i]
//!          |
//!          v
//! Top-K merge and sampling
//!          |
//!          +--> step_outputs[i].token_ids       = spec_tokens[i]
//!          +--> step_outputs[i].token_probs     = spec_probs[i]
//!          +--> sparse draft distribution
//! ```
//!
//! `DSparkMarkovTopKMapKernel` computes both branches. The confidence branch
//! reuses the current step's W1 embedding. It does not add a replay command.

use inference_backend_metal::components::DSparkConfidenceBuffers;
use inference_backend_metal::components::DSparkConfidenceConfig as DSparkBackendConfidenceConfig;
use inference_backend_metal::components::DSparkMarkovTopKMapBuffers;
use inference_backend_metal::components::DSparkMarkovTopKMapConfig;
use inference_backend_metal::components::DSparkMarkovTopKMapKernel;
use inference_backend_metal::components::DSparkMarkovTopKMapShape;
use inference_backend_metal::components::TopKMergeKernels;
use inference_backend_metal::components::TopKSampleAndWriteDistributionBuffers;
use inference_backend_metal::components::TopKSampleShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::replay_op::ReplayOp;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingRuntimeParams;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DSparkMarkovReplayShape {
    pub num_requests: u32,
    pub sampling: TopKSamplingShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DSparkProposal {
    pub token_ids: Vec<Vec<u32>>,
    pub token_probs: Vec<Vec<f32>>,
    pub confidences: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkMarkovSamplingConfig {
    pub block_size: usize,
    pub vocab_size: u32,
    pub rank: u32,
    pub w1_group_size: u32,
    pub w1_bits: u32,
    pub w2_group_size: u32,
    pub w2_bits: u32,
    pub io_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
    pub confidence: Option<DSparkMarkovConfidenceConfig>,
    pub sampling: TopKSamplingBounds,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkMarkovConfidenceConfig {
    pub hidden_dim: u32,
    pub with_markov: bool,
}

#[derive(Clone, Copy)]
pub struct DSparkMarkovWeights<'a> {
    pub w1_weight: &'a Buffer,
    pub w1_scales: &'a Buffer,
    pub w1_biases: &'a Buffer,
    pub w2_weight: &'a Buffer,
    pub w2_scales: &'a Buffer,
    pub w2_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct DSparkConfidenceWeights<'a> {
    pub weight: &'a Buffer,
    pub bias: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct DSparkConfidenceInput<'a> {
    pub hidden: &'a Buffer,
    pub weights: DSparkConfidenceWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct DSparkMarkovInput<'a> {
    pub shape: DSparkMarkovReplayShape,
    pub base_logits: &'a Buffer,
    pub distribution_store: &'a SpecProbsStore,
    pub weights: DSparkMarkovWeights<'a>,
    pub confidence: Option<DSparkConfidenceInput<'a>>,
}

pub struct DSparkMarkovSampling {
    block_size: usize,
    max_requests: usize,
    confidence_output: Option<Buffer>,
    top_k_map: DSparkMarkovTopKMapKernel,
    sample_reduce: TopKMergeKernels,
    anchor_token_ids: Buffer,
    tile_token_ids: Buffer,
    tile_logits: Buffer,
    step_params: Vec<TopKSamplingRuntimeParams>,
    step_outputs: Vec<TopKSamplingOutputBuffers>,
    step_distribution_indices: Vec<Buffer>,
}

impl DSparkMarkovSampling {
    pub fn new(device: &Device, config: DSparkMarkovSamplingConfig) -> Self {
        config.validate();
        let max_requests = config.sampling.max_sampling_inputs as usize;
        let confidence_output = config.confidence.map(|_| {
            Buffer::new_zeroed_elements(
                device,
                config
                    .block_size
                    .checked_mul(max_requests)
                    .expect("DSpark confidence output capacity must fit usize"),
                Dtype::Float32,
            )
        });
        let top_k_map = DSparkMarkovTopKMapKernel::new(device, map_config(config));
        let candidate_count = top_k_map.candidate_count(component_shape(config.sampling.max_shape()));
        let mut step_params = Vec::with_capacity(config.block_size);
        let mut step_outputs = Vec::with_capacity(config.block_size);
        let mut step_distribution_indices = Vec::with_capacity(config.block_size);
        for _ in 0..config.block_size {
            step_params.push(TopKSamplingRuntimeParams::new(device, config.sampling));
            step_outputs.push(TopKSamplingOutputBuffers::new(device, config.sampling));
            step_distribution_indices.push(Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32));
        }
        Self {
            block_size: config.block_size,
            max_requests,
            confidence_output,
            top_k_map,
            sample_reduce: TopKMergeKernels::new(device),
            anchor_token_ids: Buffer::new_zeroed_elements(device, max_requests, Dtype::Int32),
            tile_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            tile_logits: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32),
            step_params,
            step_outputs,
            step_distribution_indices,
        }
    }

    pub fn prepare(
        &self,
        req_slots: &[u32],
        anchor_token_ids: &[u32],
        anchor_positions: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> DSparkMarkovReplayShape {
        assert!(!req_slots.is_empty(), "DSpark Markov sampling requires requests");
        assert_eq!(req_slots.len(), anchor_token_ids.len());
        assert_eq!(req_slots.len(), anchor_positions.len());
        assert_eq!(req_slots.len(), sampler_configs.len());
        assert!(req_slots.len() <= self.max_requests);
        self.anchor_token_ids.write_typed(
            0,
            &anchor_token_ids
                .iter()
                .map(|&token_id| {
                    i32::try_from(token_id).expect("DSpark anchor token ID must fit the model i32 token domain")
                })
                .collect::<Vec<_>>(),
        );

        let mut sampling = None;
        for step_index in 0..self.block_size {
            self.step_distribution_indices[step_index].write_typed(
                0,
                &req_slots
                    .iter()
                    .map(|&req_slot| distribution_store.draft_distribution_index(req_slot, step_index))
                    .collect::<Vec<_>>(),
            );
            let sample_positions = anchor_positions
                .iter()
                .map(|&anchor_position| {
                    anchor_position
                        .checked_add(
                            u32::try_from(step_index)
                                .expect("DSpark Markov step index must fit u32")
                                .checked_add(1)
                                .expect("DSpark Markov step offset must fit u32"),
                        )
                        .expect("DSpark proposal sample position must fit u32")
                })
                .collect::<Vec<_>>();
            self.step_params[step_index].set_configs(sampler_configs, &sample_positions, SamplingDomain::Draft);
            let active = self.step_params[step_index].active_shape(sampler_configs);
            let step_shape = active.with_num_total_sampling_inputs(replay_bucket_capacity(
                active.num_active_sampling_inputs,
                self.max_requests
                    .try_into()
                    .expect("DSpark maximum requests must fit u32"),
            ));
            match sampling {
                Some(expected) => assert_eq!(step_shape, expected),
                None => sampling = Some(step_shape),
            }
        }
        DSparkMarkovReplayShape {
            num_requests: req_slots.len().try_into().expect("DSpark request count must fit u32"),
            sampling: sampling.expect("DSpark Markov requires steps"),
        }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: DSparkMarkovInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = input.shape;
        assert!(shape.num_requests > 0 && shape.num_requests as usize <= self.max_requests);
        assert_eq!(shape.sampling.num_active_sampling_inputs, shape.num_requests);
        let sampling = component_shape(shape.sampling);
        for step_index in 0..self.block_size {
            let input_token_ids = if step_index == 0 {
                &self.anchor_token_ids
            } else {
                &self.step_outputs[step_index - 1].token_ids
            };
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.top_k_map.invoke_replay(
                    DSparkMarkovTopKMapShape {
                        sampling,
                        base_logits_row_offset: u32::try_from(step_index)
                            .expect("DSpark Markov step index must fit u32")
                            .checked_mul(shape.num_requests)
                            .expect("DSpark Markov base-logit row offset must fit u32"),
                    },
                    DSparkMarkovTopKMapBuffers {
                        input_token_ids,
                        base_logits: input.base_logits,
                        w1_weight: input.weights.w1_weight,
                        w1_scales: input.weights.w1_scales,
                        w1_biases: input.weights.w1_biases,
                        w2_weight: input.weights.w2_weight,
                        w2_scales: input.weights.w2_scales,
                        w2_biases: input.weights.w2_biases,
                        tile_token_ids: &self.tile_token_ids,
                        tile_logits: &self.tile_logits,
                        confidence: match (self.confidence_output.as_ref(), input.confidence) {
                            (Some(output), Some(confidence)) => {
                                Some(DSparkConfidenceBuffers {
                                    hidden: confidence.hidden,
                                    weight: confidence.weights.weight,
                                    bias: confidence.weights.bias,
                                    output,
                                })
                            },
                            (None, None) => None,
                            _ => panic!("DSpark confidence config and input must match"),
                        },
                    },
                ),
            ));
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.sample_reduce
                    .invoke_sample_and_write_distribution_with_vocab_tile_size(
                        sampling,
                        TopKSampleAndWriteDistributionBuffers {
                            tile_token_ids: &self.tile_token_ids,
                            tile_logits: &self.tile_logits,
                            sampled_token_ids: &self.step_outputs[step_index].token_ids,
                            sampled_token_probs: &self.step_outputs[step_index].token_probs,
                            distribution_token_ids: input.distribution_store.draft_token_ids(),
                            distribution_probs: input.distribution_store.draft_probs(),
                            runtime_params: self.step_params[step_index].buffer(),
                            output_distribution_indices: &self.step_distribution_indices[step_index],
                            max_k: input
                                .distribution_store
                                .max_k()
                                .try_into()
                                .expect("DSpark distribution width must fit u32"),
                            num_output_distributions: input.distribution_store.num_draft_distributions(),
                        },
                        self.top_k_map.vocab_tile_size(),
                    ),
            ));
        }
    }

    pub fn add_replay_arguments(&self, shape: DSparkMarkovReplayShape, arguments: &mut ReplayArguments) {
        for params in &self.step_params {
            params.consume(shape.sampling);
        }
        self.top_k_map.add_replay_arguments(
            component_shape(shape.sampling),
            shape.sampling.num_active_sampling_inputs,
            arguments,
        );
        self.sample_reduce.add_replay_arguments(
            component_shape(shape.sampling),
            shape.sampling.num_active_sampling_inputs,
            arguments,
        );
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> DSparkProposal {
        assert!(!req_slots.is_empty() && req_slots.len() <= self.max_requests);
        let step_token_ids = self
            .step_outputs
            .iter()
            .map(|output| output.token_ids.read_typed::<i32>(0, req_slots.len()))
            .collect::<Vec<_>>();
        let step_token_probs = self
            .step_outputs
            .iter()
            .map(|output| output.token_probs.read_typed::<f32>(0, req_slots.len()))
            .collect::<Vec<_>>();
        let mut token_ids = vec![Vec::with_capacity(self.block_size); req_slots.len()];
        let mut token_probs = vec![Vec::with_capacity(self.block_size); req_slots.len()];
        let step_confidences = self
            .confidence_output
            .as_ref()
            .map(|output| output.read_typed::<f32>(0, self.block_size * req_slots.len()));
        let mut confidences = vec![Vec::with_capacity(self.block_size); req_slots.len()];
        for step_index in 0..self.block_size {
            for (request_index, &req_slot) in req_slots.iter().enumerate() {
                let token_id: u32 = step_token_ids[step_index][request_index]
                    .try_into()
                    .expect("DSpark sampler returned a negative token ID");
                distribution_store.set_expected_draft_token(req_slot, step_index, token_id);
                token_ids[request_index].push(token_id);
                token_probs[request_index].push(step_token_probs[step_index][request_index]);
                confidences[request_index].push(
                    step_confidences
                        .as_ref()
                        .map_or(1.0, |values| values[step_index * req_slots.len() + request_index]),
                );
            }
        }
        DSparkProposal {
            token_ids,
            token_probs,
            confidences,
        }
    }
}

impl DSparkMarkovSamplingConfig {
    pub fn validate(self) {
        assert!(self.block_size > 0, "DSpark Markov sampling requires steps");
        map_config(self).validate();
        self.sampling.validate();
        assert_eq!(
            self.vocab_size, self.sampling.vocab_size,
            "DSpark Markov map and sampling vocabularies must match"
        );
    }
}

fn map_config(config: DSparkMarkovSamplingConfig) -> DSparkMarkovTopKMapConfig {
    DSparkMarkovTopKMapConfig {
        vocab_size: config.vocab_size,
        rank: config.rank,
        w1_group_size: config.w1_group_size,
        w1_bits: config.w1_bits,
        w2_group_size: config.w2_group_size,
        w2_bits: config.w2_bits,
        io_dtype: config.io_dtype,
        scale_bias_dtype: config.scale_bias_dtype,
        confidence: config.confidence.map(|confidence| {
            DSparkBackendConfidenceConfig {
                hidden_dim: confidence.hidden_dim,
                with_markov: confidence.with_markov,
            }
        }),
    }
}

fn replay_bucket_capacity(num_active: u32, max_capacity: u32) -> u32 {
    assert!(num_active > 0 && num_active <= max_capacity);
    num_active
        .checked_next_power_of_two()
        .map_or(max_capacity, |bucket| bucket.min(max_capacity))
}

fn component_shape(shape: TopKSamplingShape) -> TopKSampleShape {
    TopKSampleShape {
        num_total_sampling_inputs: shape.num_total_sampling_inputs,
        vocab_size: shape.vocab_size,
        top_k: shape.top_k,
    }
}

#[cfg(test)]
#[path = "dspark_markov_test.rs"]
mod tests;
