use std::collections::VecDeque;
use std::sync::Arc;

use async_channel::Sender;
use async_channel::TrySendError;

use crate::config::SamplingConfig;
use crate::runtime::RawRequestID;
use crate::runtime::RawRequestSlot;
use crate::runtime::RequestSlot;
use crate::runtime::RequestTokenPositions;
use crate::runtime::Token;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::CompletionReason;
use crate::runtime::request::RequestEvent;
use crate::runtime::request::RequestStatus;
use crate::runtime::request::TokenProbs;
use crate::runtime::resource::processor::ResourceProcessors;
use crate::runtime::scheduler::ComputePhase;

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
    token_positions: Option<RequestTokenPositions>,
    resource_processors: Arc<ResourceProcessors>,
    in_flight_computes: VecDeque<ComputePhase>,
    num_in_flight_blocking_async_tasks: usize,
    num_in_flight_nonblocking_async_tasks: usize,
    event_tx: Sender<RequestEvent>,

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
        token_positions: Option<RequestTokenPositions>,
        resource_processors: Arc<ResourceProcessors>,
        event_tx: Sender<RequestEvent>,
        sampling_config: SamplingConfig,
        context_window: usize,
    ) -> Self {
        Self {
            req_id,
            req_slot,
            req_status,
            decoder_blocks,
            token_positions,
            resource_processors,
            in_flight_computes: VecDeque::new(),
            num_in_flight_blocking_async_tasks: 0,
            num_in_flight_nonblocking_async_tasks: 0,
            event_tx,
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

        to self.decoder_blocks {
            pub fn num_history_tokens(&self) -> usize;
            pub fn num_prompt_tokens(&self) -> usize;
            pub fn num_sampled_tokens(&self) -> usize;
            pub fn num_total_tokens(&self) -> usize;
        }
    }

    pub fn send_token_probs(&self, token_probs: TokenProbs) {
        self.send_event(RequestEvent::TokenProbs(token_probs));
    }

    pub fn send_turn_completed(&self, reason: CompletionReason) {
        self.send_event(RequestEvent::TurnCompleted(reason));
    }

    fn send_event(&self, event: RequestEvent) {
        match self.event_tx.try_send(event) {
            Ok(()) => {},
            Err(TrySendError::Full(_)) => {
                unreachable!("unbounded request event channel cannot be full")
            },
            Err(TrySendError::Closed(_)) => {
                assert!(
                    self.status().is_terminal(),
                    "active request event channel closed unexpectedly"
                );
            },
        }
    }

    pub fn start_turn(&mut self, prompt_tokens: Vec<Token>, sampling_config: SamplingConfig) {
        assert!(self.status().is_running(), "only a running request can start a turn");
        assert!(
            self.in_flight_computes.is_empty(),
            "a new turn cannot have in-flight computes"
        );
        assert_eq!(
            self.num_in_flight_blocking_async_tasks, 0,
            "a new turn cannot have blocking async tasks"
        );
        assert_eq!(
            self.num_in_flight_nonblocking_async_tasks, 0,
            "a new turn cannot have nonblocking async tasks"
        );
        assert!(
            !prompt_tokens.is_empty(),
            "a new turn must include at least one prompt token"
        );
        let num_session_tokens = self.decoder_blocks.num_total_tokens() + prompt_tokens.len();
        assert!(
            num_session_tokens < self.context_window,
            "a new turn must fit the request context window"
        );
        self.decoder_blocks.start_turn(prompt_tokens);
        self.sampling_config = sampling_config;
    }

    pub fn finish_turn(&mut self) {
        self.decoder_blocks.finish_turn();
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
        self.event_tx.close();
    }
}
