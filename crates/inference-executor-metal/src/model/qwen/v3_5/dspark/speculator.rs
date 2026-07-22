use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::components::RowGatherBuffers;
use inference_backend_metal::components::RowGatherKernel;
use inference_backend_metal::components::RowGatherShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkWeightBindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::TopKSamplingBounds;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::dspark::Qwen35DSparkBlockRequest;
use crate::model::qwen::v3_5::dspark::context::Qwen35DSparkContextScratch;
use crate::model::qwen::v3_5::dspark::layer::Qwen35DSparkLayer;
use crate::model::qwen::v3_5::dspark::layer::Qwen35DSparkLayerCapacities;
use crate::model::qwen::v3_5::dspark::layer::Qwen35DSparkLayerContextInput;
use crate::model::qwen::v3_5::dspark::layer::Qwen35DSparkLayerInput;
use crate::model::qwen::v3_5::dspark::layer::Qwen35DSparkLayerScratch;
use crate::model::qwen::v3_5::dspark::markov::Qwen35DSparkMarkov;
use crate::model::qwen::v3_5::dspark::markov::Qwen35DSparkMarkovShape;
use crate::model::qwen::v3_5::dspark::markov::Qwen35DSparkProposal;
use crate::model::qwen::v3_5::dspark::target::Qwen35DSparkTargetProjector;
use crate::model::qwen::v3_5::dspark::weights::Qwen35DSparkFinalWeights;
use crate::model::qwen::v3_5::plan::Qwen35DSparkPlan;
use crate::sampling::spec_probs::SpecProbsStore;

#[derive(Clone, Copy)]
pub struct Qwen35DSparkTargetContextInput<'a> {
    pub num_tokens: u32,
    pub gqa_batch_metadata: &'a GQAMetadataBuffers,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35DSparkBlockInput<'a> {
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkBlockShape {
    pub attention: inference_executor_core::attn::GQAReplayShape,
    pub num_mask_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkProposalShape {
    pub block: Qwen35DSparkBlockShape,
    pub markov: Qwen35DSparkMarkovShape,
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen35DSparkSpeculatorConfig {
    pub max_tokens: usize,
    pub max_requests: usize,
    pub sampler_bounds: TopKSamplingBounds,
    pub target_page_table_layout: GQAPageTableLayout,
}

pub struct Qwen35DSparkSpeculator {
    block_size: usize,
    mask_token_id: usize,
    hidden_dim: u32,
    final_norm_eps: f32,
    target_projector: Qwen35DSparkTargetProjector,
    layers: Vec<Qwen35DSparkLayer>,
    context_scratch: Qwen35DSparkContextScratch,
    block_request: Qwen35DSparkBlockRequest,
    block_scratch: Qwen35DSparkLayerScratch,
    block_token_ids: Buffer,
    block_hidden: Buffer,
    block_final_hidden: Buffer,
    mask_row_indices: Buffer,
    mask_hidden: Buffer,
    final_norm: RMSNormKernel,
    final_weights: Qwen35DSparkFinalWeights,
    mask_gather: RowGatherKernel,
    markov: Qwen35DSparkMarkov,
    max_block_tokens: usize,
    kv_token_tile_size: u32,
    num_target_gqa_layers: u32,
    main_page_table_layout: GQAPageTableLayout,
}

impl Qwen35DSparkSpeculator {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: Qwen35DSparkPlan,
        weight_bindings: Qwen35DSparkWeightBindings,
        config: Qwen35DSparkSpeculatorConfig,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35DSparkSpeculatorConfig {
            max_tokens,
            max_requests,
            sampler_bounds,
            target_page_table_layout,
        } = config;
        target_page_table_layout.validate();
        assert!(!plan.layers.is_empty(), "DSpark speculator requires draft layers");
        assert_eq!(
            weight_bindings.layers.len(),
            plan.layers.len(),
            "DSpark checkpoint bindings must match the planned draft layer count"
        );
        for (expected_index, layer) in plan.layers.iter().enumerate() {
            assert_eq!(
                layer.dspark_layer_index, expected_index,
                "DSpark speculator layers must use compact ordered indices"
            );
        }
        let num_context_layers: u32 = plan
            .layers
            .len()
            .try_into()
            .expect("DSpark context layer count must fit u32");
        let main_page_table_layout = extend_main_page_table_layout(target_page_table_layout, num_context_layers);
        let local_block_size = plan
            .block_size
            .checked_add(1)
            .expect("DSpark anchor/MASK block size must fit usize");
        let kv_token_tile_size = plan.layers[0].attention_metal.context_parallel_kv_token_tile_size;
        let hidden_dim: u32 = plan.layers[0]
            .attention_core
            .hidden_dim
            .try_into()
            .expect("DSpark hidden dimension must fit u32");
        let block_size = plan.block_size;
        let mask_token_id = plan.mask_token_id;
        let final_norm_eps = plan.norm_eps;
        let (max_block_tokens, max_sdpa_map_task_templates) = block_capacities(max_requests, local_block_size);
        for layer in &plan.layers[1..] {
            assert_layer_scratch_compatible(&plan.layers[0], layer);
        }

        let target_projector =
            Qwen35DSparkTargetProjector::load(device, store, &plan, &weight_bindings.target_projection, max_tokens)?;
        store.unload_all();
        let final_weights =
            Qwen35DSparkFinalWeights::load(device, store, &weight_bindings.final_norm_weight, plan.fc.output_dim)?;
        store.unload_all();
        let markov = Qwen35DSparkMarkov::load(
            device,
            store,
            &plan,
            &weight_bindings.markov,
            max_requests,
            sampler_bounds,
        )?;
        store.unload_all();
        let context_scratch = Qwen35DSparkContextScratch::new(device, &plan.layers[0], max_tokens);
        let block_request =
            Qwen35DSparkBlockRequest::new(device, max_requests, local_block_size, max_sdpa_map_task_templates);
        let block_scratch =
            Qwen35DSparkLayerScratch::new(device, &plan.layers[0], max_block_tokens, max_sdpa_map_task_templates);
        let max_hidden_elements = max_block_tokens
            .checked_mul(hidden_dim as usize)
            .expect("DSpark block hidden capacity must fit usize");
        let max_mask_rows = max_requests
            .checked_mul(plan.block_size)
            .expect("DSpark MASK-row capacity must fit usize");
        let max_mask_hidden_elements = max_mask_rows
            .checked_mul(hidden_dim as usize)
            .expect("DSpark MASK hidden capacity must fit usize");
        let mut layers = Vec::with_capacity(plan.layers.len());
        for (layer_plan, layer_bindings) in plan.layers.iter().zip(&weight_bindings.layers) {
            layers.push(Qwen35DSparkLayer::load(
                device,
                store,
                layer_plan,
                layer_bindings,
                Qwen35DSparkLayerCapacities {
                    max_target_tokens: max_tokens,
                    max_block_tokens,
                    max_sdpa_map_task_templates,
                    local_block_size: local_block_size
                        .try_into()
                        .expect("DSpark local block size must fit u32"),
                },
            )?);
            store.unload_all();
        }
        Ok(Self {
            block_size,
            mask_token_id,
            hidden_dim,
            final_norm_eps,
            target_projector,
            layers,
            context_scratch,
            block_request,
            block_scratch,
            block_token_ids: Buffer::new_zeroed_elements(device, max_block_tokens, Dtype::Int32),
            block_hidden: Buffer::new_zeroed_elements(device, max_hidden_elements, Dtype::Bfloat16),
            block_final_hidden: Buffer::new_zeroed_elements(device, max_hidden_elements, Dtype::Bfloat16),
            mask_row_indices: Buffer::new_zeroed_elements(device, max_mask_rows, Dtype::Uint32),
            mask_hidden: Buffer::new_zeroed_elements(device, max_mask_hidden_elements, Dtype::Bfloat16),
            final_norm: RMSNormKernel::new(device),
            final_weights,
            mask_gather: RowGatherKernel::new(device),
            markov,
            max_block_tokens,
            kv_token_tile_size,
            num_target_gqa_layers: target_page_table_layout.num_gqa_layers,
            main_page_table_layout,
        })
    }

    pub fn num_speculative_tokens(&self) -> usize {
        self.block_size
    }

    pub fn max_block_tokens(&self) -> usize {
        self.max_block_tokens
    }

    pub fn main_page_table_layout(&self) -> GQAPageTableLayout {
        self.main_page_table_layout
    }

    pub fn duplicate_residual_output_for_model_layer(
        &self,
        model_layer_index: usize,
    ) -> Option<inference_backend_metal::components::DuplicateResidualOutput<'_>> {
        self.target_projector
            .duplicate_residual_output_for_model_layer(model_layer_index)
    }

    pub fn record_target_context<'a, R>(
        &'a self,
        recorder: &mut R,
        input: Qwen35DSparkTargetContextInput<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert_eq!(
            input.gqa_batch_metadata.replay_shape().num_tokens,
            input.num_tokens,
            "DSpark context append token count must match the target GQA request"
        );
        let projected_target = self.target_projector.record(recorder, input.num_tokens);
        for layer in &self.layers {
            layer.record_target_context(
                recorder,
                Qwen35DSparkLayerContextInput {
                    num_tokens: input.num_tokens,
                    num_target_gqa_layers: self.num_target_gqa_layers,
                    projected_target,
                    req_slots: input.gqa_batch_metadata.req_slots(),
                    flat_token_indices: input.gqa_batch_metadata.flat_token_indices(),
                    pages: input.pages,
                    page_ids: input.page_ids,
                    page_table_layout: self.main_page_table_layout,
                    scratch: &self.context_scratch,
                },
            );
        }
        projected_target
    }

    pub fn update_block_request(
        &self,
        req_slots: &[u32],
        anchor_positions: &[u32],
        anchor_token_ids: &[u32],
    ) -> Qwen35DSparkBlockShape {
        assert_eq!(req_slots.len(), anchor_token_ids.len());
        assert_eq!(anchor_positions.len(), anchor_token_ids.len());
        let local_block_size = self.block_request.local_block_size() as usize;
        let (block_token_ids, mask_row_indices) =
            build_block_tokens(anchor_token_ids, self.mask_token_id, self.block_size);
        assert_eq!(block_token_ids.len(), req_slots.len() * local_block_size);
        self.block_token_ids.write_typed(0, &block_token_ids);
        self.mask_row_indices.write_typed(0, &mask_row_indices);
        let attention = self
            .block_request
            .update(req_slots, anchor_positions, self.kv_token_tile_size);
        Qwen35DSparkBlockShape {
            attention,
            num_mask_rows: mask_row_indices
                .len()
                .try_into()
                .expect("DSpark MASK row count must fit u32"),
        }
    }

    pub fn prepare_proposal(
        &self,
        req_slots: &[u32],
        anchor_positions: &[u32],
        anchor_token_ids: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> Qwen35DSparkProposalShape {
        Qwen35DSparkProposalShape {
            block: self.update_block_request(req_slots, anchor_positions, anchor_token_ids),
            markov: self.markov.prepare(
                req_slots,
                anchor_token_ids,
                anchor_positions,
                sampler_configs,
                distribution_store,
            ),
        }
    }

    pub fn block_token_ids(&self) -> &Buffer {
        &self.block_token_ids
    }

    pub fn block_hidden(&self) -> &Buffer {
        &self.block_hidden
    }

    pub fn record_block<'a, R>(&'a self, recorder: &mut R, input: Qwen35DSparkBlockInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(!self.layers.is_empty());
        let local_block_size = self.block_request.local_block_size();
        let mut hidden_state = &self.block_hidden;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let next_hidden_state = self.block_scratch.residual_stream(layer_index);
            hidden_state = layer.record(
                recorder,
                Qwen35DSparkLayerInput {
                    block_request: &self.block_request,
                    local_block_size,
                    num_target_gqa_layers: self.num_target_gqa_layers,
                    page_table_layout: self.main_page_table_layout,
                    pages: input.pages,
                    page_ids: input.page_ids,
                    hidden_state,
                    next_hidden_state,
                    scratch: &self.block_scratch,
                },
            );
        }
        let replay_shape = self.block_request.replay_shape();
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.final_norm.invoke(
            RMSNormShape::bf16(replay_shape.num_tokens, self.hidden_dim),
            RMSNormBuffers {
                input: hidden_state,
                weight: &self.final_weights.norm_weight,
                output: &self.block_final_hidden,
            },
            self.final_norm_eps,
        )));
        let num_requests = replay_shape.num_tokens / local_block_size;
        let num_mask_rows = num_requests
            .checked_mul(self.block_size.try_into().expect("DSpark block size must fit u32"))
            .expect("DSpark MASK row count must fit u32");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.mask_gather.invoke(
            RowGatherShape::bf16(num_mask_rows, self.hidden_dim),
            RowGatherBuffers {
                input: &self.block_final_hidden,
                row_indices: &self.mask_row_indices,
                output: &self.mask_hidden,
            },
        )));
        &self.mask_hidden
    }

    pub fn record_markov<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: Qwen35DSparkMarkovShape,
        base_logits: &'a Buffer,
        distribution_store: &'a SpecProbsStore,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.markov.record(recorder, shape, base_logits, distribution_store);
    }

    pub fn add_markov_replay_arguments(&self, shape: Qwen35DSparkMarkovShape, arguments: &mut ReplayArguments) {
        self.markov.add_replay_arguments(shape, arguments);
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> Qwen35DSparkProposal {
        self.markov.read_proposal(req_slots, distribution_store)
    }
}

fn assert_layer_scratch_compatible(
    first: &crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan,
    other: &crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan,
) {
    assert_eq!(first.attention_core.hidden_dim, other.attention_core.hidden_dim);
    assert_eq!(first.attention_core.head_dim, other.attention_core.head_dim);
    assert_eq!(first.attention_core.num_q_heads, other.attention_core.num_q_heads);
    assert_eq!(first.attention_core.num_kv_heads, other.attention_core.num_kv_heads);
    assert_eq!(first.mlp_core.hidden_dim, other.mlp_core.hidden_dim);
    assert_eq!(first.mlp_core.intermediate_dim, other.mlp_core.intermediate_dim);
    assert_eq!(first.attention_metal.dtype, other.attention_metal.dtype);
    assert_eq!(first.mlp_metal.dtype, other.mlp_metal.dtype);
    assert_eq!(
        first.attention_metal.context_parallel_kv_token_tile_size,
        other.attention_metal.context_parallel_kv_token_tile_size
    );
}

fn extend_main_page_table_layout(
    target_page_table_layout: GQAPageTableLayout,
    num_context_layers: u32,
) -> GQAPageTableLayout {
    target_page_table_layout.validate();
    assert!(num_context_layers > 0, "DSpark requires context layers");
    let layout = GQAPageTableLayout {
        num_gqa_layers: target_page_table_layout
            .num_gqa_layers
            .checked_add(num_context_layers)
            .expect("DSpark lane-0 GQA layer count must fit u32"),
        ..target_page_table_layout
    };
    layout.validate();
    layout
}

fn block_capacities(max_requests: usize, local_block_size: usize) -> (usize, usize) {
    assert!(max_requests > 0);
    assert!(local_block_size > 1);
    let max_block_tokens = max_requests
        .checked_mul(local_block_size)
        .expect("DSpark maximum block token count must fit usize");
    // Match the bounded map-TaskTemplate contract used by the target GQA request.
    // Every Q token needs at least one persistent partial and one block-local
    // partial; one TaskTemplate may cover multiple KV-token tiles at long history.
    let max_sdpa_map_task_templates = max_block_tokens
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .expect("DSpark SDPA map TaskTemplate capacity must fit usize");
    (max_block_tokens, max_sdpa_map_task_templates)
}

fn build_block_tokens(anchor_token_ids: &[u32], mask_token_id: usize, block_size: usize) -> (Vec<i32>, Vec<u32>) {
    assert!(!anchor_token_ids.is_empty());
    assert!(block_size > 0);
    let local_block_size = block_size
        .checked_add(1)
        .expect("DSpark local block size must fit usize");
    let mut block_token_ids = Vec::<i32>::with_capacity(anchor_token_ids.len() * local_block_size);
    let mut mask_row_indices = Vec::<u32>::with_capacity(anchor_token_ids.len() * block_size);
    let mask_token_id: i32 = mask_token_id
        .try_into()
        .expect("DSpark MASK token ID must fit the model i32 token domain");
    for &anchor_token_id in anchor_token_ids {
        block_token_ids.push(
            anchor_token_id
                .try_into()
                .expect("DSpark anchor token ID must fit the model i32 token domain"),
        );
        block_token_ids.extend(std::iter::repeat_n(mask_token_id, block_size));
    }
    // Base logits are step-major so each Markov step consumes one contiguous
    // [num_requests, vocab] slice while its predecessor tokens are sampled
    // left-to-right.
    for local_index in 1..local_block_size {
        mask_row_indices.extend((0..anchor_token_ids.len()).map(|request_index| {
            request_index
                .checked_mul(local_block_size)
                .and_then(|block_start| block_start.checked_add(local_index))
                .and_then(|index| u32::try_from(index).ok())
                .expect("DSpark MASK row index must fit u32")
        }));
    }
    (block_token_ids, mask_row_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_zero_layout_appends_context_layers_without_new_cache_lane() {
        let target = GQAPageTableLayout {
            num_req_slots: 4,
            num_gqa_layers: 16,
            num_blocks: 128,
            num_page_ids_per_block: 4,
        };
        let combined = extend_main_page_table_layout(target, 5);
        assert_eq!(combined.num_req_slots, target.num_req_slots);
        assert_eq!(combined.num_gqa_layers, 21);
        assert_eq!(combined.num_blocks, target.num_blocks);
        assert_eq!(combined.num_page_ids_per_block, target.num_page_ids_per_block);
    }

    #[test]
    fn proposal_scratch_is_bounded_by_request_blocks() {
        let (max_block_tokens, max_sdpa_map_task_templates) = block_capacities(4, 9);
        assert_eq!(max_block_tokens, 36);
        assert_eq!(max_sdpa_map_task_templates, 128);
    }

    #[test]
    fn block_tokens_keep_anchor_rows_but_gather_only_mask_outputs() {
        let (token_ids, mask_rows) = build_block_tokens(&[17, 23], 151_671, 2);
        assert_eq!(token_ids, [17, 151_671, 151_671, 23, 151_671, 151_671]);
        assert_eq!(mask_rows, [1, 4, 2, 5]);
    }
}
