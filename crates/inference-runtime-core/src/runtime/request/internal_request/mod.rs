use std::collections::VecDeque;
use std::sync::Arc;

use async_channel::Sender;
use async_channel::TrySendError;

use crate::config::SamplingConfig;
use crate::runtime::RawRequestID;
use crate::runtime::RawRequestSlot;
use crate::runtime::RequestInputPositions;
use crate::runtime::RequestSlot;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::CompletionReason;
use crate::runtime::request::RequestStatus;
use crate::runtime::request::TokenProbs;
use crate::runtime::scheduler::ComputePhase;
use crate::runtime::tasks::ResourceProcessor;

mod req_resp;

mod stop_sequence;
pub use stop_sequence::StopSequenceMatch;

pub struct InternalRequest<const N: usize, const P: usize, const L: usize, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    req_id: RawRequestID,
    req_slot: RequestSlot,
    req_status: AtomicRequestStatus,
    decoder_blocks: TrieDecoderBlocks<N, P, L, DBC>,
    input_positions: Option<RequestInputPositions>,
    resource_processor: Arc<ResourceProcessor>,
    in_flight_computes: VecDeque<ComputePhase>,
    num_in_flight_blocking_async_tasks: usize,
    num_in_flight_nonblocking_async_tasks: usize,
    token_prob_tx: Sender<TokenProbs>,

    sampling_config: SamplingConfig,
    context_window: usize,
}

impl<const N: usize, const P: usize, const L: usize, DBC> InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        req_id: RawRequestID,
        req_slot: RequestSlot,
        req_status: AtomicRequestStatus,
        decoder_blocks: TrieDecoderBlocks<N, P, L, DBC>,
        input_positions: Option<RequestInputPositions>,
        resource_processor: Arc<ResourceProcessor>,
        token_prob_tx: Sender<TokenProbs>,
        sampling_config: SamplingConfig,
        context_window: usize,
    ) -> Self {
        Self {
            req_id,
            req_slot,
            req_status,
            decoder_blocks,
            input_positions,
            resource_processor,
            in_flight_computes: VecDeque::new(),
            num_in_flight_blocking_async_tasks: 0,
            num_in_flight_nonblocking_async_tasks: 0,
            token_prob_tx,
            sampling_config,
            context_window,
        }
    }

    pub fn req_id(&self) -> RawRequestID {
        self.req_id
    }

    pub fn req_slot(&self) -> RawRequestSlot {
        self.req_slot.raw() as RawRequestSlot
    }

    pub fn status(&self) -> RequestStatus {
        self.req_status.load()
    }

    delegate::delegate! {
        to self.req_status {
            pub fn store_running(&self) -> bool;
            pub fn store_swapped(&self) -> bool;
            pub fn store_timed_out(&self) -> bool;
            pub fn store_aborted(&self) -> bool;
            pub fn store_completed(&self, completion: CompletionReason) -> bool;
        }
    }

    pub fn send_token_probs(&self, token_probs: TokenProbs) {
        match self.token_prob_tx.try_send(token_probs) {
            Ok(()) => {},
            Err(TrySendError::Full(_)) => {
                panic!("request token output channel is full; the client is not consuming committed output")
            },
            Err(TrySendError::Closed(_)) => {
                assert!(
                    self.status().is_terminal(),
                    "active request token output channel closed unexpectedly"
                );
            },
        }
    }

    pub fn sampling_config(&self) -> &SamplingConfig {
        &self.sampling_config
    }
}

impl<const N: usize, const P: usize, const L: usize, DBC> Drop for InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    fn drop(&mut self) {
        if self.req_status.store_aborted() {
            tracing::debug!(
                target: "inference-runtime-core::request",
                phase = "request.aborted",
                request_id = self.req_id,
                "request aborted"
            );
        }
        self.token_prob_tx.close();
    }
}
