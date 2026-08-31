use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_channel::Receiver as AsyncReceiver;
use async_channel::Sender as AsyncSender;
use async_channel::unbounded as async_unbounded;
use crossbeam_channel::Sender as SyncSender;
use crossbeam_channel::TrySendError as SyncTrySendError;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::AtomicRequestStatus;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RequestEvent;
use inference_runtime_core::runtime::RequestStatus;
use inference_runtime_core::runtime::RequestTokenPositions;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::decoder::trie_cache::TrieDecoderBlocks;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;

use super::RuntimeBlockCache;
use super::RuntimeQueuedRequest;
use super::RuntimeRequest;

pub struct DecodeSessions<const N: usize, const L: usize, const P: usize> {
    model_runtime_config: RuntimeConfig,
    block_cache: Arc<RuntimeBlockCache<P, L>>,
    resource_processors: Arc<ResourceProcessors>,
    new_req_tx: SyncSender<RuntimeQueuedRequest<N, P, L>>,
    ready_req_tx: SyncSender<RuntimeRequest<N, P, L>>,
    pending: Arc<Mutex<HashMap<RawRequestID, RuntimeRequest<N, P, L>>>>,
    cancel_tx: AsyncSender<RawRequestID>,
}

impl<const N: usize, const L: usize, const P: usize> DecodeSessions<N, L, P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_runtime_config: RuntimeConfig,
        block_cache: Arc<RuntimeBlockCache<P, L>>,
        resource_processors: Arc<ResourceProcessors>,
        new_req_tx: SyncSender<RuntimeQueuedRequest<N, P, L>>,
        ready_req_tx: SyncSender<RuntimeRequest<N, P, L>>,
        completed_turn_rx: AsyncReceiver<(RuntimeRequest<N, P, L>, CompletionReason)>,
        shutdown: Shutdown,
        async_task_handle: &tokio::runtime::Handle,
    ) -> Self {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (cancel_tx, cancel_rx) = async_unbounded();
        let event_loop_pending = pending.clone();
        let event_loop_shutdown = shutdown.clone();
        let event_loop = async_task_handle.spawn(async move {
            let _shutdown_guard = ShutdownGuard::new(event_loop_shutdown.clone());
            run_event_loop(completed_turn_rx, cancel_rx, event_loop_pending, event_loop_shutdown).await
        });
        drop(event_loop);
        Self {
            model_runtime_config,
            block_cache,
            resource_processors,
            new_req_tx,
            ready_req_tx,
            pending,
            cancel_tx,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_session(
        &self,
        request_id: RawRequestID,
        history_tokens: Vec<Token>,
        prompt_tokens: Vec<Token>,
        sampled_tokens: Vec<Token>,
        token_positions: Option<RequestTokenPositions>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling_config: SamplingConfig,
    ) -> Result<ExternalRequest> {
        let (queued_request, external_request) = self.initialize_session(
            request_id,
            history_tokens,
            prompt_tokens,
            sampled_tokens,
            token_positions,
            resource_entries,
            sampling_config,
        )?;
        self.submit_new(queued_request)?;
        Ok(external_request)
    }

    pub fn continue_session(
        &self,
        external_request: &ExternalRequest,
        prompt_tokens: Vec<Token>,
        sampling_config: SamplingConfig,
    ) -> Result<()> {
        let request_id = external_request.req_id();
        match external_request.status() {
            RequestStatus::Running => {},
            status if status.is_terminal() => {
                return Err(status_error(status));
            },
            status => panic!("decode session cannot continue from request status {status:?}"),
        }

        let mut pending = self.pending.lock().unwrap();
        let request = pending
            .get_mut(&request_id)
            .ok_or_else(|| Error::unavailable("decode session is not resident; retry with the complete input"))?;

        // TODO(session-residency): Restore a partially evicted or SSD-offloaded request here before starting the next
        // turn. A fully evicted session must remain a cache miss and return Unavailable to the caller.
        let num_session_tokens = request.num_total_tokens() + prompt_tokens.len();
        if num_session_tokens >= self.model_runtime_config.context_window {
            return Err(Error::invalid_argument(format!(
                "decode session token count={num_session_tokens} must be less than context window={}",
                self.model_runtime_config.context_window
            )));
        }
        request.start_turn(prompt_tokens, sampling_config);
        let request = pending
            .remove(&request_id)
            .expect("validated pending session must remain present until ownership transfer");
        drop(pending);

        match self.ready_req_tx.try_send(request) {
            Ok(()) => Ok(()),
            Err(SyncTrySendError::Full(_)) => {
                unreachable!("ready-request queue capacity must cover all resident sessions")
            },
            Err(SyncTrySendError::Disconnected(_)) => Err(Error::unavailable("runtime is stopped")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize_session(
        &self,
        request_id: RawRequestID,
        history_tokens: Vec<Token>,
        prompt_tokens: Vec<Token>,
        sampled_tokens: Vec<Token>,
        token_positions: Option<RequestTokenPositions>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling_config: SamplingConfig,
    ) -> Result<(RuntimeQueuedRequest<N, P, L>, ExternalRequest)> {
        let (resources, resource_placements): (Vec<_>, Vec<_>) = resource_entries.into_iter().unzip();
        let num_initial_tokens = history_tokens.len() + prompt_tokens.len() + sampled_tokens.len();
        assert!(
            token_positions
                .as_ref()
                .is_none_or(|positions| positions.initial().len() == num_initial_tokens),
            "explicit request token positions must match the initial token count"
        );
        let min_initial_tokens = usize::max(1, L - 1);
        if num_initial_tokens < min_initial_tokens {
            return Err(Error::invalid_argument(format!(
                "decode request minimum initial token count is {min_initial_tokens} for {L} cache lanes, got {}",
                num_initial_tokens
            )));
        }
        if num_initial_tokens >= self.model_runtime_config.context_window {
            return Err(Error::invalid_argument(format!(
                "decode request initial token count={num_initial_tokens} must be less than context window={}",
                self.model_runtime_config.context_window
            )));
        }
        if let Some(token_positions) = &token_positions {
            let max_continuation_index = self.model_runtime_config.context_window - 1 - num_initial_tokens;
            assert!(
                token_positions
                    .continuation_start()
                    .iter()
                    .all(|&position| max_continuation_index <= u32::MAX as usize - position as usize),
                "explicit request continuation token positions must fit u32 through the context window"
            );
        }
        if sampled_tokens.len() >= sampling_config.max_sampled_tokens {
            return Err(Error::invalid_argument(format!(
                "decode request initial sampled token count={} must be less than max_sampled_tokens={}",
                sampled_tokens.len(),
                sampling_config.max_sampled_tokens
            )));
        }
        inference_runtime_core::runtime::validate_resources(&resources, &resource_placements, num_initial_tokens)?;
        let req_status = AtomicRequestStatus::new();
        let decoder_blocks = TrieDecoderBlocks::new(
            self.block_cache.clone(),
            resources,
            resource_placements,
            history_tokens,
            prompt_tokens,
            sampled_tokens,
        );
        let (event_tx, event_rx) = async_unbounded::<RequestEvent>();
        let queued_request = RuntimeQueuedRequest::new(
            request_id,
            req_status.clone(),
            decoder_blocks,
            token_positions,
            self.resource_processors.clone(),
            event_tx,
            sampling_config,
            self.model_runtime_config.context_window,
        );
        let external_request = ExternalRequest::new(request_id, req_status, event_rx, self.cancel_tx.clone());
        Ok((queued_request, external_request))
    }

    pub fn submit_new(&self, queued_request: RuntimeQueuedRequest<N, P, L>) -> Result<()> {
        let request_id = queued_request.req_id();
        match self.new_req_tx.try_send(queued_request) {
            Ok(()) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.queued",
                    request_id,
                    "decode request queued"
                );
                Ok(())
            },
            Err(SyncTrySendError::Full(_)) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.queue_full",
                    request_id,
                    "request queue is full"
                );
                Err(Error::resource_exhausted("decode queue is full"))
            },
            Err(SyncTrySendError::Disconnected(_)) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.runtime_stopped",
                    request_id,
                    "runtime is stopped"
                );
                Err(Error::unavailable("runtime is stopped"))
            },
        }
    }
}

async fn run_event_loop<const N: usize, const L: usize, const P: usize>(
    completed_turn_rx: AsyncReceiver<(RuntimeRequest<N, P, L>, CompletionReason)>,
    cancel_rx: AsyncReceiver<RawRequestID>,
    pending: Arc<Mutex<HashMap<RawRequestID, RuntimeRequest<N, P, L>>>>,
    shutdown: Shutdown,
) {
    let shutdown_rx = shutdown.async_rx().clone();
    while !shutdown.is_shutdown() {
        tokio::select! {
            completed = completed_turn_rx.recv() => {
                let Ok((request, reason)) = completed else {
                    shutdown.shutdown();
                    break;
                };
                if request.status().is_terminal() {
                    continue;
                }
                let request_id = request.req_id();
                let mut pending = pending.lock().unwrap();
                assert!(
                    pending.insert(request_id, request).is_none(),
                    "a decode session cannot own two pending requests",
                );
                pending
                    .get_mut(&request_id)
                    .expect("inserted decode session must remain pending")
                    .send_turn_completed(reason);
            },
            cancelled = cancel_rx.recv() => {
                let Ok(request_id) = cancelled else {
                    shutdown.shutdown();
                    break;
                };
                pending.lock().unwrap().remove(&request_id);
            },
            stopped = shutdown_rx.recv() => {
                let _ = stopped;
                break;
            },
        }
    }
    pending.lock().unwrap().clear();
}

fn status_error(status: RequestStatus) -> Error {
    match status {
        RequestStatus::Cancelled => Error::cancelled("decode session was cancelled"),
        RequestStatus::TimedOut => Error::deadline_exceeded("decode session timed out"),
        RequestStatus::Aborted => Error::aborted("decode session was aborted"),
        RequestStatus::Completed(_) => Error::invalid_argument("decode session has terminal completion"),
        RequestStatus::Initialized | RequestStatus::Running | RequestStatus::Swapped => {
            panic!("active request status cannot map to a terminal session error")
        },
    }
}
