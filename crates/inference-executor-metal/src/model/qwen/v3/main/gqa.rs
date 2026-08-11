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
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

pub struct Qwen3MainGQA {
    model_layer_index: usize,
    weights: Option<Qwen3xUngatedGQAWeightBuffers>,
    backend: Option<Rc<UngatedGQA>>,
    scratch: Option<Rc<UngatedGQAScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
}

pub struct Qwen3MainGQAState {
    backend: Rc<UngatedGQA>,
    scratch: Rc<UngatedGQAScratch>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    page_table_layout: GQAPageTableLayout,
    metadata: GQAMetadataBuffers,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl Qwen3MainGQA {
    pub fn new(core: &UngatedGQACore, state: &Qwen3MainGQAState) -> Self {
        Self {
            model_layer_index: core.model_layer_index,
            weights: None,
            backend: Some(Rc::clone(&state.backend)),
            scratch: Some(Rc::clone(&state.scratch)),
            request_page_table: Some(Rc::clone(state.request_page_table())),
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
        bindings: Qwen3xGQAWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "Qwen3 Main GQA weights are already loaded");
        self.weights = Some(Qwen3xUngatedGQAWeightBuffers::load(
            device, store, &bindings, core, metal,
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "Qwen3 Main GQA weights are not loaded");
        self.weights.take();
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.backend.is_some() && self.scratch.is_some() && self.request_page_table.is_some(),
            "qwen3 Main GQA state is not loaded"
        );
        self.request_page_table.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn load_state(&mut self, state: &Qwen3MainGQAState) {
        assert!(
            self.backend.is_none() && self.scratch.is_none() && self.request_page_table.is_none(),
            "qwen3 Main GQA state is already loaded"
        );
        self.backend = Some(Rc::clone(&state.backend));
        self.scratch = Some(Rc::clone(&state.scratch));
        self.request_page_table = Some(Rc::clone(state.request_page_table()));
    }

    fn weights(&self) -> &Qwen3xUngatedGQAWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3 Main GQA weights must be loaded before execution")
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
            self.backend(),
            recorder,
            UngatedGQAInput {
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: self
                    .model_layer_index
                    .try_into()
                    .expect("qwen3 Main GQA layer index must fit u32"),
                batch_metadata: metadata,
                hidden_state: input,
                next_hidden_state: output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.scratch().bindings(),
            },
        );
    }

    fn backend(&self) -> &UngatedGQA {
        self.backend
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }

    fn scratch(&self) -> &UngatedGQAScratch {
        self.scratch
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }
}

impl Qwen3MainGQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        let backend = Rc::new(UngatedGQA::new(device, core, metal));
        let scratch = Rc::new(backend.new_scratch(max_tokens));
        Self {
            backend,
            scratch,
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            page_table_layout,
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

    fn request_page_table(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("qwen3 Main GQA request page-table state must be loaded")
    }

    pub fn prepare_pages(&self, core_batch: &BatchDeviceRequest) {
        self.request_page_table()
            .prepare(core_batch, self.cache_lane, self.num_cache_pages);
    }

    pub fn prepare_page_span(
        &self,
        core_batch: &BatchDeviceRequest,
        num_runtime_page_ids_per_block: usize,
        page_id_offset: usize,
    ) {
        self.request_page_table().prepare_span(
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
        self.request_page_table().reset_req_slots(req_slots);
    }

    pub fn write_full_state(&self, writer: &mut StateSnapshotWriter, resource: u32) -> Result<(), ModelExecutorError> {
        self.request_page_table().write_full_state(writer, resource)
    }

    pub fn unload_state(&mut self) {
        self.request_page_table
            .take()
            .expect("qwen3 Main GQA request page-table state must be loaded");
    }

    pub fn load_state(&mut self, device: &Device) {
        assert!(
            self.request_page_table.is_none(),
            "qwen3 Main GQA request page-table state is already loaded"
        );
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
    }

    pub fn read_full_state(&self, reader: &mut StateSnapshotReader, resource: u32) -> Result<(), ModelExecutorError> {
        self.request_page_table().read_full_state(reader, resource)
    }
}
