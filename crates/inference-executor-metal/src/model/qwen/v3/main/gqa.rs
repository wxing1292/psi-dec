use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::ungated_backend::UngatedGQA;
use crate::attn::gqa::ungated_backend::UngatedGQAInput;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::layer::Qwen3xUngatedGQAWeightBuffers;

pub struct Qwen3MainGQA {
    model_layer_index: usize,
    weights: Qwen3xUngatedGQAWeightBuffers,
    backend: Rc<UngatedGQA>,
    scratch: Rc<UngatedGQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
}

pub struct Qwen3MainGQAState {
    backend: Rc<UngatedGQA>,
    scratch: Rc<UngatedGQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
    metadata: GQAMetadataBuffers,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl Qwen3MainGQA {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
        bindings: Qwen3xGQAWeightBindings,
        state: &Qwen3MainGQAState,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            model_layer_index: core.model_layer_index,
            weights: Qwen3xUngatedGQAWeightBuffers::load(device, store, &bindings, core, metal)?,
            backend: Rc::clone(&state.backend),
            scratch: Rc::clone(&state.scratch),
            request_page_table: Rc::clone(&state.request_page_table),
        })
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        pages: &'a Buffer,
        metadata: &'a GQAMetadataBuffers,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <UngatedGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            UngatedGQAInput {
                page_table_layout: self.request_page_table.layout(),
                gqa_layer_index: self
                    .model_layer_index
                    .try_into()
                    .expect("qwen3 Main GQA layer index must fit u32"),
                batch_metadata: metadata,
                hidden_state: input,
                next_hidden_state: output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table.page_ids_buffer(),
                },
                weights: self.weights.as_borrowed(),
                scratch: self.scratch.bindings(),
            },
        );
    }
}

impl Qwen3MainGQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        core: UngatedGQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
        cache_lane: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3 Main GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3 Main cache page IDs must fit u32"
        );
        page_table_layout.validate();
        Self {
            backend: Rc::new(UngatedGQA::new(device, core.clone(), metal)),
            scratch: Rc::new(UngatedGQAScratch::new(device, &core, metal, max_tokens)),
            request_page_table: Rc::new(GQARequestPageTable::new(device, page_table_layout)),
            metadata: GQAMetadataBuffers::new(device, max_tokens),
            num_cache_pages,
            cache_lane,
        }
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.backend.num_tokens_per_page() as usize
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        &self.metadata
    }

    pub fn prepare_pages(&self, core_batch: &BatchDeviceRequest) {
        self.request_page_table
            .prepare(core_batch, self.cache_lane, self.num_cache_pages);
    }

    pub fn prepare_page_span(
        &self,
        core_batch: &BatchDeviceRequest,
        num_runtime_page_ids_per_block: usize,
        page_id_offset: usize,
    ) {
        self.request_page_table.prepare_span(
            core_batch,
            self.cache_lane,
            self.num_cache_pages,
            num_runtime_page_ids_per_block,
            page_id_offset,
        );
    }

    pub fn prepare_metadata(&self, req_slots: &[u32], token_indices: &[u32], cu_tokens: &[u32]) -> GQAReplayShape {
        self.backend
            .prepare(&self.metadata, req_slots, token_indices, cu_tokens)
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table.reset_req_slots(req_slots);
    }
}
