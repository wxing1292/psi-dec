use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;
use crate::trace;

pub struct Qwen35GDNState {
    backend: Rc<GDN>,
    scratch: Rc<GDNScratch>,
    request_state_table: Rc<GDNRequestStateTable>,
    metadata: GDNMetadataBuffers,
    state_restore: Replay<Rc<GDNRequestStateTable>>,
    pending_publish: Option<MetalReplaySubmission>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GDNStateRestoreKey {
    num_state_io_requests: usize,
}

#[derive(Clone, Copy)]
pub struct GDNStateRestoreInput<'a> {
    pages: &'a Buffer,
}

impl ReplayComponent for Rc<GDNRequestStateTable> {
    type Key = GDNStateRestoreKey;
    type Input<'a> = GDNStateRestoreInput<'a>;

    fn replay_key(&self, _input: &Self::Input<'_>) -> Self::Key {
        let num_state_io_requests = self.restores().len();
        assert!(num_state_io_requests > 0, "GDN restore replay requires restore jobs");
        GDNStateRestoreKey { num_state_io_requests }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        GDNRequestStateTable::record_restore(self, recorder, input.pages);
    }
}

impl Qwen35GDNState {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        cores: &[GDNCore],
        metal: GDNMetalConfig,
        num_req_slots: usize,
        max_spec_tokens: usize,
        max_tokens: usize,
        max_tokens_per_request: usize,
        num_tokens_per_block: usize,
        page_bytes: usize,
    ) -> Self {
        let representative = cores
            .first()
            .expect("qwen3.5 GDN state requires at least one GDN layer");
        let request_state_table = Rc::new(GDNRequestStateTable::new(
            device,
            cores,
            num_req_slots,
            max_spec_tokens,
            max_tokens_per_request,
            num_tokens_per_block,
            page_bytes,
        ));
        Self {
            backend: Rc::new(GDN::new(device, representative.clone(), metal)),
            scratch: Rc::new(GDNScratch::new(device, representative, metal, max_tokens)),
            request_state_table: Rc::clone(&request_state_table),
            metadata: GDNMetadataBuffers::new(device, num_req_slots, max_tokens),
            state_restore: Replay::new("qwen3.5 GDN state restore", Rc::clone(&request_state_table)),
            pending_publish: None,
        }
    }

    pub fn backend(&self) -> &Rc<GDN> {
        &self.backend
    }

    pub fn scratch(&self) -> &Rc<GDNScratch> {
        &self.scratch
    }

    pub fn request_state_table(&self) -> &Rc<GDNRequestStateTable> {
        &self.request_state_table
    }

    pub fn metadata(&self) -> &GDNMetadataBuffers {
        &self.metadata
    }

    pub fn prepare_states(&self, microbatch: &Qwen35Microbatch) -> GDNPreparedRequestState {
        self.request_state_table.prepare(
            microbatch.req_slots(),
            microbatch.block_indices(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            microbatch.gdn_state_txns(),
            microbatch.gdn_state_page_ids_by_req(),
        )
    }

    pub fn prepare_metadata(
        &self,
        microbatch: &Qwen35Microbatch,
        prepared: &GDNPreparedRequestState,
    ) -> GDNReplayShape {
        self.backend.prepare(&self.metadata, microbatch.cu_tokens(), prepared)
    }

    pub fn restore(&mut self, runtime: &MetalReplayRuntime<'_>, pages: &Buffer) {
        if !self.request_state_table.prepare_restore() {
            trace::qwen35_state(|| "event=gdn_restore skipped=true".to_string());
            return;
        }
        let input = GDNStateRestoreInput { pages };
        let (key, cache_hit) = self.state_restore.record(runtime, &input);
        trace::qwen35_state(|| format!("event=gdn_restore key={key:?} cache_hit={cache_hit}"));
        runtime.submit_replay(self.state_restore.replay(&key)).wait();
    }

    pub fn commit(&mut self, runtime: &MetalReplayRuntime<'_>, pages: &Buffer, verified_state_versions: &[u32]) {
        assert!(
            self.pending_publish.is_none(),
            "GDN cache publish cannot overlap a previous publish"
        );
        self.request_state_table.commit(verified_state_versions);
        let mut recorder = runtime.create_recorder();
        if self.request_state_table.record_publish(&mut recorder, pages) {
            self.pending_publish = Some(runtime.submit_replay(&recorder.build()));
        }
    }

    pub fn finish_publish(&mut self) {
        let Some(submission) = self.pending_publish.take() else {
            return;
        };
        submission.wait();
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_state_table.reset_req_slots(req_slots);
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        self.request_state_table.num_pages_per_state_slot()
    }
}
