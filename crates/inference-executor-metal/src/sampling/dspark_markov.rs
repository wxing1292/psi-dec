use inference_backend_metal::components::DSparkMarkovTopKMapBuffers;
use inference_backend_metal::components::DSparkMarkovTopKMapConfig;
use inference_backend_metal::components::DSparkMarkovTopKMapKernel;
use inference_backend_metal::components::DSparkMarkovTopKMapShape;
use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::components::TopKMergeKernels;
use inference_backend_metal::components::TopKSampleAndWriteDistributionBuffers;
use inference_backend_metal::components::TopKSampleShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMarkovWeightBindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkPlan;
use crate::model::qwen::v3_x::weight::quant_weight;
use crate::model::qwen::v3_x::weight::typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingRuntimeParams;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DSparkMarkovShape {
    pub num_requests: u32,
    pub sampling: TopKSamplingShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DSparkProposal {
    pub token_ids: Vec<Vec<u32>>,
    pub token_probs: Vec<Vec<f32>>,
}

struct DSparkMarkovWeights {
    w1_weight: Buffer,
    w1_scales: Buffer,
    w1_biases: Buffer,
    w2_weight: Buffer,
    w2_scales: Buffer,
    w2_biases: Buffer,
}

pub struct DSparkMarkovSampling {
    block_size: usize,
    max_requests: usize,
    weights: DSparkMarkovWeights,
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
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkPlan,
        bindings: &Qwen3xDSparkMarkovWeightBindings,
        max_requests: usize,
        sampler_bounds: TopKSamplingBounds,
    ) -> Result<Self, ModelExecutorError> {
        assert!(plan.block_size > 0, "DSpark Markov sampling requires steps");
        assert!(max_requests > 0, "DSpark Markov sampling requires requests");
        sampler_bounds.validate();
        let rank = to_u32("DSpark Markov rank", plan.markov_w1.embedding_dim);
        let vocab_size = to_u32("DSpark Markov vocabulary", plan.markov_w1.num_embeddings);
        assert_eq!(
            vocab_size as usize, plan.markov_w2.output_dim,
            "DSpark Markov W1 and W2 vocabularies must match"
        );
        assert_eq!(
            rank as usize, plan.markov_w2.input_dim,
            "DSpark Markov W1 and W2 ranks must match"
        );
        assert!(max_requests <= sampler_bounds.max_sampling_inputs as usize);
        let markov_sampler_bounds = TopKSamplingBounds {
            max_sampling_inputs: max_requests
                .try_into()
                .expect("DSpark maximum requests must fit sampling bounds"),
            ..sampler_bounds
        };
        markov_sampler_bounds.validate();
        let w1_config = QuantizedEmbeddingConfig {
            vocab_size,
            hidden_dim: rank,
            group_size: plan.markov_w1.group_size,
            bits: plan.markov_w1.bits,
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        w1_config.validate();
        let weights = DSparkMarkovWeights::load(device, store, plan, bindings, w1_config)?;
        let top_k_map_config = DSparkMarkovTopKMapConfig {
            vocab_size,
            rank,
            w1_group_size: plan.markov_w1.group_size,
            w1_bits: plan.markov_w1.bits,
            w2_group_size: plan.markov_w2.group_size,
            w2_bits: plan.markov_w2.bits,
        };
        let top_k_map = DSparkMarkovTopKMapKernel::new(device, top_k_map_config);
        let candidate_count = top_k_map.candidate_count(component_shape(markov_sampler_bounds.max_shape()));
        let mut step_params = Vec::with_capacity(plan.block_size);
        let mut step_outputs = Vec::with_capacity(plan.block_size);
        let mut step_distribution_indices = Vec::with_capacity(plan.block_size);
        for _ in 0..plan.block_size {
            step_params.push(TopKSamplingRuntimeParams::new(device, markov_sampler_bounds));
            step_outputs.push(TopKSamplingOutputBuffers::new(device, markov_sampler_bounds));
            step_distribution_indices.push(Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32));
        }
        Ok(Self {
            block_size: plan.block_size,
            max_requests,
            weights,
            top_k_map,
            sample_reduce: TopKMergeKernels::new(device),
            anchor_token_ids: Buffer::new_zeroed_elements(device, max_requests, Dtype::Int32),
            tile_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            tile_logits: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32),
            step_params,
            step_outputs,
            step_distribution_indices,
        })
    }

    pub fn prepare(
        &self,
        req_slots: &[u32],
        anchor_token_ids: &[u32],
        anchor_positions: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> DSparkMarkovShape {
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
        DSparkMarkovShape {
            num_requests: req_slots.len().try_into().expect("DSpark request count must fit u32"),
            sampling: sampling.expect("DSpark Markov requires steps"),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: DSparkMarkovShape,
        base_logits: &'a Buffer,
        distribution_store: &'a SpecProbsStore,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(shape.num_requests > 0 && shape.num_requests as usize <= self.max_requests);
        assert_eq!(shape.sampling.num_active_sampling_inputs, shape.num_requests);
        let sampling = component_shape(shape.sampling);
        for step_index in 0..self.block_size {
            let previous_token_ids = if step_index == 0 {
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
                        previous_token_ids,
                        base_logits,
                        w1_weight: &self.weights.w1_weight,
                        w1_scales: &self.weights.w1_scales,
                        w1_biases: &self.weights.w1_biases,
                        w2_weight: &self.weights.w2_weight,
                        w2_scales: &self.weights.w2_scales,
                        w2_biases: &self.weights.w2_biases,
                        tile_token_ids: &self.tile_token_ids,
                        tile_logits: &self.tile_logits,
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
                            distribution_token_ids: distribution_store.draft_token_ids(),
                            distribution_probs: distribution_store.draft_probs(),
                            runtime_params: self.step_params[step_index].buffer(),
                            output_distribution_indices: &self.step_distribution_indices[step_index],
                            max_k: distribution_store
                                .max_k()
                                .try_into()
                                .expect("DSpark distribution width must fit u32"),
                            num_output_distributions: distribution_store.num_draft_distributions(),
                        },
                        self.top_k_map.vocab_tile_size(),
                    ),
            ));
        }
    }

    pub fn add_replay_arguments(&self, shape: DSparkMarkovShape, arguments: &mut ReplayArguments) {
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
        for step_index in 0..self.block_size {
            for (request_index, &req_slot) in req_slots.iter().enumerate() {
                let token_id: u32 = step_token_ids[step_index][request_index]
                    .try_into()
                    .expect("DSpark sampler returned a negative token ID");
                distribution_store.set_expected_draft_token(req_slot, step_index, token_id);
                token_ids[request_index].push(token_id);
                token_probs[request_index].push(step_token_probs[step_index][request_index]);
            }
        }
        DSparkProposal { token_ids, token_probs }
    }
}

impl DSparkMarkovWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkPlan,
        bindings: &Qwen3xDSparkMarkovWeightBindings,
        w1_config: QuantizedEmbeddingConfig,
    ) -> Result<Self, ModelExecutorError> {
        let w1_weight = quant_weight(store, &bindings.w1.weight)?;
        let w1_scales = typed_tensor(store, &bindings.w1.scales, safetensors::Dtype::BF16)?.into_data();
        let w1_biases = typed_tensor(store, &bindings.w1.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("DSpark Markov W1 weight", w1_weight.len(), w1_config.weight_bytes())?;
        let w1_affine_bytes = w1_config
            .num_affine_params()
            .checked_mul(Dtype::Bfloat16.item_size())
            .expect("DSpark Markov W1 affine byte length must fit usize");
        validate_len("DSpark Markov W1 scales", w1_scales.len(), w1_affine_bytes)?;
        validate_len("DSpark Markov W1 biases", w1_biases.len(), w1_affine_bytes)?;

        let w2_config = w2_config(plan);
        let w2_weight = quant_weight(store, &bindings.w2.weight)?;
        let w2_scales = typed_tensor(store, &bindings.w2.scales, safetensors::Dtype::BF16)?.into_data();
        let w2_biases = typed_tensor(store, &bindings.w2.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("DSpark Markov W2 weight", w2_weight.len(), w2_config.weight_bytes())?;
        validate_len(
            "DSpark Markov W2 scales",
            w2_scales.len(),
            w2_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "DSpark Markov W2 biases",
            w2_biases.len(),
            w2_config.scale_or_bias_bytes(),
        )?;
        Ok(Self {
            w1_weight: Buffer::from_slice(device, &w1_weight),
            w1_scales: Buffer::from_slice(device, &w1_scales),
            w1_biases: Buffer::from_slice(device, &w1_biases),
            w2_weight: Buffer::from_slice(device, &w2_weight),
            w2_scales: Buffer::from_slice(device, &w2_scales),
            w2_biases: Buffer::from_slice(device, &w2_biases),
        })
    }
}

fn w2_config(plan: &Qwen3xDSparkPlan) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig::same_dtype(
        plan.markov_w2
            .output_dim
            .try_into()
            .expect("DSpark vocabulary must fit i32"),
        plan.markov_w2
            .input_dim
            .try_into()
            .expect("DSpark Markov rank must fit i32"),
        plan.markov_w2
            .group_size
            .try_into()
            .expect("DSpark Markov group size must fit i32"),
        plan.markov_w2.bits.try_into().expect("DSpark Markov bits must fit i32"),
        Dtype::Bfloat16,
    )
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

fn to_u32(name: &str, value: usize) -> u32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit u32"))
}

#[cfg(test)]
#[path = "dspark_markov_test.rs"]
mod tests;
