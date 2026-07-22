use inference_backend_metal::components::QuantizedEmbeddingBuffers;
use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::components::QuantizedEmbeddingKernel;
use inference_backend_metal::components::QuantizedEmbeddingShape;
use inference_backend_metal::components::RowwiseAddBuffers;
use inference_backend_metal::components::RowwiseAddConfig;
use inference_backend_metal::components::RowwiseAddKernel;
use inference_backend_metal::components::RowwiseAddShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkMarkovWeightBindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::dspark::weights::Qwen35DSparkMarkovWeights;
use crate::model::qwen::v3_5::plan::Qwen35DSparkPlan;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingSparseDistributionOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkMarkovShape {
    pub num_requests: u32,
    pub sampling: TopKSamplingShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35DSparkProposal {
    pub token_ids: Vec<Vec<u32>>,
    pub token_probs: Vec<Vec<f32>>,
}

pub struct Qwen35DSparkMarkov {
    block_size: usize,
    max_requests: usize,
    rank: u32,
    vocab_size: u32,
    w1_config: QuantizedEmbeddingConfig,
    w2_group_size: u32,
    w2_bits: u32,
    weights: Qwen35DSparkMarkovWeights,
    w1: QuantizedEmbeddingKernel,
    w2_qmv: AffineQuantizedMatmulKernel,
    w2_qmm: AffineQuantizedMatmulKernel,
    add_bias: RowwiseAddKernel,
    anchor_token_ids: Buffer,
    latent: Buffer,
    bias_logits: Buffer,
    corrected_logits: Buffer,
    step_samplers: Vec<TopKSampling>,
    step_outputs: Vec<TopKSamplingOutputBuffers>,
    step_distribution_indices: Vec<Buffer>,
}

impl Qwen35DSparkMarkov {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkPlan,
        weight_bindings: &Qwen35DSparkMarkovWeightBindings,
        max_requests: usize,
        sampler_bounds: TopKSamplingBounds,
    ) -> Result<Self, ModelExecutorError> {
        assert!(plan.block_size > 0);
        assert!(max_requests > 0);
        sampler_bounds.validate();
        let rank = to_u32("DSpark Markov rank", plan.markov_w1.embedding_dim);
        let vocab_size = to_u32("DSpark Markov vocabulary", plan.markov_w1.num_embeddings);
        assert_eq!(vocab_size as usize, plan.markov_w2.output_dim);
        assert_eq!(rank as usize, plan.markov_w2.input_dim);
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
            affine_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        w1_config.validate();
        let weights = Qwen35DSparkMarkovWeights::load(device, store, plan, weight_bindings)?;
        let qmm_rows = qmv_batch_limit(rank, vocab_size);
        let mut step_samplers = Vec::with_capacity(plan.block_size);
        let mut step_outputs = Vec::with_capacity(plan.block_size);
        let mut step_distribution_indices = Vec::with_capacity(plan.block_size);
        for _ in 0..plan.block_size {
            step_samplers.push(TopKSampling::new(device, markov_sampler_bounds));
            step_outputs.push(TopKSamplingOutputBuffers::new(device, markov_sampler_bounds));
            step_distribution_indices.push(Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32));
        }
        let latent_elements = max_requests
            .checked_mul(rank as usize)
            .expect("DSpark Markov latent capacity must fit usize");
        let logit_elements = max_requests
            .checked_mul(vocab_size as usize)
            .expect("DSpark Markov logit capacity must fit usize");
        Ok(Self {
            block_size: plan.block_size,
            max_requests,
            rank,
            vocab_size,
            w1_config,
            w2_group_size: plan.markov_w2.group_size,
            w2_bits: plan.markov_w2.bits,
            weights,
            w1: QuantizedEmbeddingKernel::new(device, w1_config),
            w2_qmv: AffineQuantizedMatmulKernel::new(device, w2_shape(plan, 1)),
            w2_qmm: AffineQuantizedMatmulKernel::new(device, w2_shape(plan, qmm_rows)),
            add_bias: RowwiseAddKernel::new(
                device,
                RowwiseAddConfig {
                    row_width: vocab_size,
                    dtype: Dtype::Bfloat16,
                },
            ),
            anchor_token_ids: Buffer::new_zeroed_elements(device, max_requests, Dtype::Int32),
            latent: Buffer::new_zeroed_elements(device, latent_elements, Dtype::Bfloat16),
            bias_logits: Buffer::new_zeroed_elements(device, logit_elements, Dtype::Bfloat16),
            corrected_logits: Buffer::new_zeroed_elements(device, logit_elements, Dtype::Bfloat16),
            step_samplers,
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
    ) -> Qwen35DSparkMarkovShape {
        assert!(!req_slots.is_empty());
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
            self.step_samplers[step_index].set_configs(sampler_configs, &sample_positions, SamplingDomain::Draft);
            let active = self.step_samplers[step_index].active_shape(sampler_configs);
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
        Qwen35DSparkMarkovShape {
            num_requests: req_slots.len().try_into().expect("DSpark request count must fit u32"),
            sampling: sampling.expect("DSpark Markov requires steps"),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: Qwen35DSparkMarkovShape,
        base_logits: &'a Buffer,
        distribution_store: &'a SpecProbsStore,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(shape.num_requests > 0 && shape.num_requests as usize <= self.max_requests);
        assert_eq!(shape.sampling.num_active_sampling_inputs, shape.num_requests);
        let w2 = if shape.num_requests >= qmv_batch_limit(self.rank, self.vocab_size) {
            &self.w2_qmm
        } else {
            &self.w2_qmv
        };
        for step_index in 0..self.block_size {
            let previous_token_ids = if step_index == 0 {
                &self.anchor_token_ids
            } else {
                &self.step_outputs[step_index - 1].token_ids
            };
            recorder.record_with_barrier_before(ReplayOp::opaque(self.w1.invoke(
                QuantizedEmbeddingShape {
                    num_tokens: shape.num_requests,
                },
                QuantizedEmbeddingBuffers {
                    token_ids: previous_token_ids,
                    weight: &self.weights.w1_weight,
                    scales: &self.weights.w1_scales,
                    biases: &self.weights.w1_biases,
                    output: &self.latent,
                },
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(w2.invoke_with_shape(
                self.w2_shape(shape.num_requests),
                &self.bias_logits,
                0,
                &self.latent,
                0,
                &self.weights.w2_weight,
                0,
                &self.weights.w2_scales,
                0,
                &self.weights.w2_biases,
                0,
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.add_bias.invoke(
                    RowwiseAddShape {
                        num_rows: shape.num_requests,
                        lhs_row_offset: u32::try_from(step_index)
                            .expect("DSpark Markov step index must fit u32")
                            .checked_mul(shape.num_requests)
                            .expect("DSpark Markov base-logit row offset must fit u32"),
                    },
                    RowwiseAddBuffers {
                        lhs: base_logits,
                        rhs: &self.bias_logits,
                        output: &self.corrected_logits,
                    },
                ),
            ));
            self.step_samplers[step_index].record_bf16_with_sparse_distribution(
                recorder,
                shape.sampling,
                TopKSamplingInputs {
                    logits: &self.corrected_logits,
                    logits_offset_bytes: 0,
                },
                self.step_outputs[step_index].as_output(),
                TopKSamplingSparseDistributionOutput {
                    token_ids: distribution_store.draft_token_ids(),
                    probs: distribution_store.draft_probs(),
                    output_distribution_indices: &self.step_distribution_indices[step_index],
                    max_k: distribution_store
                        .max_k()
                        .try_into()
                        .expect("DSpark distribution width must fit u32"),
                    num_output_distributions: distribution_store.num_draft_distributions(),
                },
            );
        }
    }

    pub fn add_replay_arguments(&self, shape: Qwen35DSparkMarkovShape, arguments: &mut ReplayArguments) {
        for sampler in &self.step_samplers {
            sampler.add_replay_arguments(shape.sampling, arguments);
        }
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> Qwen35DSparkProposal {
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
        Qwen35DSparkProposal { token_ids, token_probs }
    }

    fn w2_shape(&self, num_requests: u32) -> AffineQuantizedMatmulShape {
        AffineQuantizedMatmulShape::same_dtype(
            num_requests.try_into().expect("DSpark Markov rows must fit i32"),
            self.vocab_size.try_into().expect("DSpark vocabulary must fit i32"),
            self.rank.try_into().expect("DSpark Markov rank must fit i32"),
            self.w2_group_size
                .try_into()
                .expect("DSpark Markov group size must fit i32"),
            self.w2_bits.try_into().expect("DSpark Markov bits must fit i32"),
            Dtype::Bfloat16,
        )
    }
}

fn w2_shape(plan: &Qwen35DSparkPlan, num_requests: u32) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape::same_dtype(
        num_requests.try_into().expect("DSpark Markov rows must fit i32"),
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

fn qmv_batch_limit(input_dim: u32, output_dim: u32) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

fn to_u32(name: &str, value: usize) -> u32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit u32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_sampling_positions_start_after_the_anchor() {
        let anchors = [11, 25];
        let positions = (0..3)
            .map(|step| anchors.map(|anchor| anchor + step + 1))
            .collect::<Vec<_>>();
        assert_eq!(positions, [[12, 26], [13, 27], [14, 28]]);
    }

    #[test]
    fn sampling_bucket_caps_at_non_power_of_two_request_capacity() {
        assert_eq!(replay_bucket_capacity(3, 6), 4);
        assert_eq!(replay_bucket_capacity(5, 6), 6);
    }
}
