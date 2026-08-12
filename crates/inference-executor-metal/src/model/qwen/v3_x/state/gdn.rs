use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gdn::backend::GDNReplayTopology;
use crate::attn::gdn::backend::add_gdn_private_replay_arguments;
use crate::attn::gdn::backend::add_gdn_replay_arguments;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::batch_metadata::GDNReplayBucketPolicy;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::attn::gdn::state_table::GDNRequestStateResources;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::attn::gdn::state_table::GDNStateCapacity;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;
use crate::trace;

mod file_io;

pub struct Qwen3xGDNState {
    backend: Option<Rc<GDN>>,
    scratch: Option<Rc<GDNScratch>>,
    metadata: Option<GDNMetadataBuffers>,
    representative_core: GDNCore,
    metal: GDNMetalConfig,
    num_req_slots: usize,
    max_tokens: usize,
    replay_bucket_policy: GDNReplayBucketPolicy,
    request_state_table: GDNRequestStateTable,
    state_restore: Replay<GDNStateRestore>,
    pending_publish: Option<MetalReplaySubmission>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GDNStateRestoreKey {
    num_state_io_requests: usize,
}

#[derive(Clone, Copy)]
pub struct GDNStateRestoreInput<'a> {
    request_state_table: &'a GDNRequestStateTable,
    pages: &'a Buffer,
}

pub struct GDNStateRestore;

impl ReplayComponent for GDNStateRestore {
    type Key = GDNStateRestoreKey;
    type Input<'a> = GDNStateRestoreInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let num_state_io_requests = input.request_state_table.restores().len();
        assert!(num_state_io_requests > 0, "GDN restore replay requires restore jobs");
        GDNStateRestoreKey { num_state_io_requests }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        input.request_state_table.record_restore(recorder, input.pages);
    }
}

impl Qwen3xGDNState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        cores: &[GDNCore],
        metal: GDNMetalConfig,
        num_req_slots: usize,
        state_capacity: GDNStateCapacity,
        max_tokens: usize,
        num_tokens_per_block: usize,
        num_cache_pages: usize,
        page_bytes: usize,
    ) -> Self {
        let representative = cores
            .first()
            .expect("qwen3.x GDN state requires at least one GDN layer");
        let request_state_table = GDNRequestStateTable::new(
            device,
            cores,
            num_req_slots,
            state_capacity,
            num_tokens_per_block,
            num_cache_pages,
            page_bytes,
        );
        let backend = Rc::new(GDN::new(device, representative.clone(), metal));
        let max_requests = num_req_slots
            .try_into()
            .expect("qwen3.x GDN request capacity must fit u32");
        let max_tokens_u32 = max_tokens.try_into().expect("qwen3.x GDN token capacity must fit u32");
        let replay_bucket_policy = backend.replay_bucket_policy(max_requests, max_tokens_u32);
        Self {
            backend: Some(backend),
            scratch: Some(Rc::new(GDNScratch::new(device, representative, max_tokens))),
            metadata: Some(GDNMetadataBuffers::new(device, num_req_slots, max_tokens)),
            representative_core: representative.clone(),
            metal,
            num_req_slots,
            max_tokens,
            replay_bucket_policy,
            request_state_table,
            state_restore: Replay::new("qwen3.x GDN state restore", GDNStateRestore),
            pending_publish: None,
        }
    }

    pub fn backend(&self) -> &Rc<GDN> {
        self.backend.as_ref().expect("Qwen3.x GDN backend state must be loaded")
    }

    pub fn scratch(&self) -> &Rc<GDNScratch> {
        self.scratch.as_ref().expect("Qwen3.x GDN scratch state must be loaded")
    }

    pub fn request_state_resources(&self) -> &Rc<GDNRequestStateResources> {
        self.request_state_table.resources()
    }

    pub fn metadata(&self) -> &GDNMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("Qwen3.x GDN metadata state must be loaded")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_states(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        state_txns: &[GDNStateTxn],
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNPreparedRequestState {
        self.request_state_table.prepare(
            req_slots,
            block_indices,
            token_indices,
            cu_tokens,
            state_txns,
            state_page_ids_by_req,
        )
    }

    pub fn prepare_metadata(&self, cu_tokens: &[u32], prepared: &GDNPreparedRequestState) -> GDNReplayShape {
        self.backend().prepare(self.metadata(), cu_tokens, prepared)
    }

    pub fn prepare_metadata_bucketed(&self, cu_tokens: &[u32], prepared: &GDNPreparedRequestState) -> GDNReplayShape {
        self.backend()
            .prepare_bucketed(self.metadata(), cu_tokens, prepared, &self.replay_bucket_policy)
    }

    pub fn prepare_metadata_bucketed_with_token_capacity(
        &self,
        cu_tokens: &[u32],
        prepared: &GDNPreparedRequestState,
        total_tokens: u32,
    ) -> GDNReplayShape {
        self.backend().prepare_bucketed_with_token_capacity(
            self.metadata(),
            cu_tokens,
            prepared,
            &self.replay_bucket_policy,
            total_tokens,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        self.backend().replay_token_topology_boundaries()
    }

    pub fn replay_topology(&self) -> GDNReplayTopology {
        self.backend().replay_topology(self.metadata(), true)
    }

    pub fn add_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gdn_replay_arguments(self.metadata().replay_shape(), arguments);
    }

    pub fn add_private_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gdn_private_replay_arguments(self.metadata().replay_shape(), arguments);
    }

    pub fn restore(&mut self, runtime: &MetalReplayRuntime<'_>, pages: &Buffer) {
        if !self.request_state_table.prepare_restore() {
            trace::gdn_state(|| "event=gdn_restore skipped=true".to_string());
            return;
        }
        let input = GDNStateRestoreInput {
            request_state_table: &self.request_state_table,
            pages,
        };
        let (key, cache_hit) = self.state_restore.record(runtime, &input);
        trace::gdn_state(|| format!("event=gdn_restore key={key:?} cache_hit={cache_hit}"));
        runtime.submit_replay(self.state_restore.replay(&key)).wait();
        self.request_state_table.finish_restore();
    }

    pub fn commit(&mut self, runtime: &MetalReplayRuntime<'_>, pages: &Buffer, state_versions: &[u32]) {
        assert!(
            self.pending_publish.is_none(),
            "GDN cache publish cannot overlap a previous publish"
        );
        self.request_state_table.commit(state_versions);
        let mut recorder = runtime.create_recorder();
        if self.request_state_table.record_publish(&mut recorder, pages) {
            self.pending_publish = Some(runtime.submit_replay(&recorder.build()));
        }
    }

    pub fn finish_publish(&mut self) {
        if let Some(submission) = self.pending_publish.take() {
            submission.wait();
        }
        self.request_state_table.finish_publish();
    }

    pub fn clear_replay_cache(&mut self) {
        assert!(
            self.pending_publish.is_none(),
            "GDN replay cache cannot be cleared while a state publish is pending"
        );
        self.state_restore.clear();
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.backend.is_some() && self.scratch.is_some() && self.metadata.is_some(),
            "Qwen3.x GDN state resources are not loaded"
        );
        self.request_state_table.release_resources();
        self.metadata.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.backend.is_none() && self.scratch.is_none() && self.metadata.is_none(),
            "Qwen3.x GDN state resources are already loaded"
        );
        let backend = Rc::new(GDN::new(device, self.representative_core.clone(), self.metal));
        self.scratch = Some(Rc::new(GDNScratch::new(
            device,
            &self.representative_core,
            self.max_tokens,
        )));
        self.metadata = Some(GDNMetadataBuffers::new(device, self.num_req_slots, self.max_tokens));
        self.backend = Some(backend);
        self.request_state_table.allocate_resources(device);
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_state_table.reset_req_slots(req_slots);
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        self.request_state_table.num_pages_per_state_slot()
    }
}
