use std::rc::Rc;

use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::backend::add_gqa_private_replay_arguments;
use crate::attn::gqa::backend::add_gqa_replay_arguments;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

pub struct Qwen3xGQAState {
    backend: Rc<GQA>,
    scratch: Rc<GQAScratch>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    page_table_layout: GQAPageTableLayout,
    metadata: GQAMetadataBuffers,
    replay_bucket_policy: GQAReplayBucketPolicy,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl Qwen3xGQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        core: GQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
        cache_lane: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3.x GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3.x cache page IDs must fit u32"
        );
        page_table_layout.validate();
        let backend = Rc::new(GQA::new(device, core, metal));
        let scratch = Rc::new(backend.new_scratch(max_tokens));
        let max_tokens_u32 = max_tokens.try_into().expect("qwen3.x GQA token capacity must fit u32");
        let replay_bucket_policy = backend.replay_bucket_policy(max_tokens_u32);
        Self {
            backend,
            scratch,
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            page_table_layout,
            metadata: GQAMetadataBuffers::new(device, max_tokens),
            replay_bucket_policy,
            num_cache_pages,
            cache_lane,
        }
    }

    pub fn backend(&self) -> &Rc<GQA> {
        &self.backend
    }

    pub fn scratch(&self) -> &Rc<GQAScratch> {
        &self.scratch
    }

    pub fn request_page_table(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("Qwen3.x GQA request page-table state must be loaded")
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        &self.metadata
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

    pub fn prepare_metadata_bucketed(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
    ) -> GQAReplayShape {
        self.backend.prepare_bucketed(
            &self.metadata,
            req_slots,
            token_indices,
            cu_tokens,
            &self.replay_bucket_policy,
        )
    }

    pub fn prepare_metadata_bucketed_with_token_capacity(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        total_tokens: u32,
    ) -> GQAReplayShape {
        self.backend.prepare_bucketed_with_token_capacity(
            &self.metadata,
            req_slots,
            token_indices,
            cu_tokens,
            &self.replay_bucket_policy,
            total_tokens,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        self.backend.replay_token_topology_boundaries()
    }

    pub fn replay_topology(&self) -> GQAReplayTopology {
        self.backend.replay_topology(&self.metadata)
    }

    pub fn add_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gqa_replay_arguments(self.metadata.replay_shape(), self.replay_topology(), arguments);
    }

    pub fn add_private_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gqa_private_replay_arguments(self.metadata.replay_shape(), self.replay_topology(), arguments);
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
            .expect("Qwen3.x GQA request page-table state must be loaded");
    }

    pub fn load_state(&mut self, device: &Device) {
        assert!(
            self.request_page_table.is_none(),
            "Qwen3.x GQA request page-table state is already loaded"
        );
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
    }

    pub fn read_full_state(&self, reader: &mut StateSnapshotReader, resource: u32) -> Result<(), ModelExecutorError> {
        self.request_page_table().read_full_state(reader, resource)
    }
}
