use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_channel::bounded as async_bounded;
use async_channel::unbounded as async_unbounded;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TrySendError;
use crossbeam_channel::bounded as sync_bounded;
use inference_executor_core::model::ReplayableModel;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::memory::U32IDAllocator;
use inference_runtime_core::runtime::AtomicRequestStatus;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::InternalRequest;
use inference_runtime_core::runtime::QueuedRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::RequestInputPositions;
use inference_runtime_core::runtime::RequestSlotAllocator;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::decoder::TPKVBlockAllocator;
use inference_runtime_core::runtime::decoder::TPStateBlockAllocator;
use inference_runtime_core::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use inference_runtime_core::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use inference_runtime_core::runtime::decoder::trie_cache::TrieDecoderBlocks;
use inference_runtime_core::runtime::scheduler::EventLoop;
use inference_runtime_core::runtime::scheduler::FIFOBatcher;
use inference_runtime_core::runtime::scheduler::InstrumentedScheduler;
use inference_runtime_core::runtime::scheduler::ScheduleQueue;
use inference_runtime_core::runtime::scheduler::SimpleScheduler;
use inference_runtime_core::runtime::tasks::AsyncTaskPool;
use inference_runtime_core::runtime::tasks::AsyncTaskResp;
use inference_runtime_core::runtime::tasks::ResourceProcessor;
use inference_runtime_core::runtime::validate_resources;

use crate::api::Inference;
use crate::consts::NUM_TRIE_PARTITION;
use crate::executor::ReplayableModelEventLoop;
use crate::rpc;
use crate::rpc::HTTPService;

type RuntimeBlockCache<const P: usize, const L: usize> =
    MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>;
type RuntimeQueuedRequest<const N: usize, const P: usize, const L: usize> =
    QueuedRequest<N, P, L, RuntimeBlockCache<P, L>>;
type RuntimeRequest<const N: usize, const P: usize, const L: usize> = InternalRequest<N, P, L, RuntimeBlockCache<P, L>>;

pub struct InferenceRuntime<const N: usize, const L: usize, const P: usize> {
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,

    shutdown: Shutdown,
    block_cache: Arc<RuntimeBlockCache<P, L>>,
    resource_processor: Arc<ResourceProcessor>,

    user_req_tx: Sender<RuntimeQueuedRequest<N, P, L>>,
    model_executor_req_rx: Receiver<ReplayableModelExecutorRequest>,
    model_executor_resp_tx: Sender<ReplayableModelExecutorResponse>,
    request_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    request_slot_reset_rx: Receiver<()>,
}

impl<const N: usize, const L: usize, const P: usize> InferenceRuntime<N, L, P> {
    pub fn new(
        model_runtime_config: RuntimeConfig,
        scheduler_config: SchedulerConfig,
        num_spec_tokens: usize,
        shutdown: Shutdown,
        async_task_handle: &tokio::runtime::Handle,
        resource_processor: Arc<ResourceProcessor>,
    ) -> Self {
        assert!(scheduler_config.max_requests > 0, "runtime requires request capacity");
        assert!(scheduler_config.max_tokens > 0, "runtime requires token capacity");
        assert!(
            scheduler_config.max_tokens_per_request > 0,
            "runtime requires per-request token capacity"
        );
        assert!(
            scheduler_config.max_tokens_per_request <= scheduler_config.max_tokens,
            "runtime per-request token capacity={} exceeds batch token capacity={}",
            scheduler_config.max_tokens_per_request,
            scheduler_config.max_tokens
        );
        assert!(scheduler_config.max_compute_slots > 0, "runtime requires compute slots");
        assert!(
            model_runtime_config.max_queued_requests > 0,
            "runtime requires user-request queue capacity"
        );
        assert!(
            model_runtime_config.max_running_requests > 0,
            "runtime requires running-request capacity"
        );
        let min_initial_tokens = usize::max(1, L - 1);
        assert!(
            model_runtime_config.context_window > min_initial_tokens,
            "runtime context window={} must exceed the minimum initial token count={min_initial_tokens}",
            model_runtime_config.context_window
        );
        assert!(
            u32::try_from(model_runtime_config.context_window).is_ok(),
            "runtime context window must fit u32"
        );
        assert!(
            scheduler_config.max_requests <= model_runtime_config.max_running_requests,
            "scheduler batch request capacity={} exceeds runtime running-request capacity={}",
            scheduler_config.max_requests,
            model_runtime_config.max_running_requests
        );
        assert_eq!(
            N,
            model_runtime_config.num_tokens_per_cache_block(),
            "runtime service compile-time NUM_TOKENS_PER_CACHE_BLOCK={} must match runtime logical cache block={}",
            N,
            model_runtime_config.num_tokens_per_cache_block()
        );
        assert_eq!(
            L,
            model_runtime_config.num_cache_lanes(),
            "runtime service compile-time NUM_CACHE_LANE={} must match model cache lane count={}",
            L,
            model_runtime_config.num_cache_lanes()
        );

        let (req_slot_allocator, request_slot_reset_rx) =
            RequestSlotAllocator::new(model_runtime_config.max_running_requests as u64);
        let request_slot_reset_notifier = req_slot_allocator.reset_notifier();
        let page_id_allocator = Arc::new(U32IDAllocator::new(model_runtime_config.num_pages as u64));
        let block_cache = {
            let block_cache_vec = std::array::from_fn(|cache_lane| {
                let kv_block_allocator = TPKVBlockAllocator::new(
                    model_runtime_config.num_pages_per_kv_block(cache_lane),
                    page_id_allocator.clone(),
                );
                let state_block_allocator = TPStateBlockAllocator::new(
                    model_runtime_config.num_pages_per_state_block(cache_lane),
                    page_id_allocator.clone(),
                );
                let capacity = model_runtime_config.block_cache_capacity(cache_lane);
                Arc::new(SingleLaneTrieBlockCache::new(
                    kv_block_allocator,
                    state_block_allocator,
                    capacity,
                    shutdown.clone(),
                ))
            });
            Arc::new(MultiLaneTrieBlockCache::new(block_cache_vec))
        };

        let (user_req_tx, user_req_rx) = sync_bounded(model_runtime_config.max_queued_requests);
        let model_executor_channel_capacity = scheduler_config
            .max_compute_slots
            .checked_add(1)
            .expect("model executor channel capacity must fit usize");
        let (model_executor_req_tx, model_executor_req_rx) = sync_bounded(model_executor_channel_capacity);
        let (model_executor_resp_tx, model_executor_resp_rx) = sync_bounded(model_executor_channel_capacity);
        let (async_task_req_tx, async_task_req_rx) = async_bounded(model_runtime_config.max_running_requests);
        let (async_task_resp_tx, async_task_resp_rx) =
            sync_bounded::<Box<dyn AsyncTaskResp>>(model_runtime_config.max_running_requests);

        {
            let schedule_queue: ScheduleQueue<RuntimeRequest<N, P, L>, DeviceRequest, DeviceResponse> =
                ScheduleQueue::new(async_task_req_tx);
            let batcher = FIFOBatcher::new();
            let scheduler = InstrumentedScheduler::new(
                SimpleScheduler::new(
                    schedule_queue,
                    batcher,
                    scheduler_config.max_requests,
                    scheduler_config.max_tokens,
                    scheduler_config.max_tokens_per_request,
                    scheduler_config.max_compute_slots,
                ),
                num_spec_tokens,
            );
            let event_loop = EventLoop::new(
                user_req_rx,
                async_task_resp_rx,
                model_executor_req_tx,
                model_executor_resp_rx,
                scheduler,
                req_slot_allocator,
                page_id_allocator,
                model_runtime_config.executor_hibernation_mode,
                model_runtime_config.executor_hibernation_timeout,
                shutdown.clone(),
            );
            let scheduler_shutdown = shutdown.clone();

            let scheduler_thread = std::thread::Builder::new()
                .name("inference-runtime-event-loop".to_string())
                .spawn(move || {
                    let _shutdown_guard = ShutdownGuard::new(scheduler_shutdown);
                    event_loop.event_loop()
                })
                .expect("inference runtime scheduler thread should start");
            drop(scheduler_thread);

            let async_task_pool = AsyncTaskPool::new(
                async_task_req_rx,
                async_task_resp_tx,
                shutdown.clone(),
                model_runtime_config.max_running_requests,
            );
            let async_task_shutdown = shutdown.clone();
            let async_task_join_handle = async_task_handle.spawn(async move {
                let _shutdown_guard = ShutdownGuard::new(async_task_shutdown);
                async_task_pool.event_loop().await
            });
            drop(async_task_join_handle);
        };

        Self {
            model_runtime_config,
            scheduler_config,

            shutdown,
            block_cache,
            resource_processor,

            user_req_tx,
            model_executor_req_rx,
            model_executor_resp_tx,
            request_slot_reset_notifier,
            request_slot_reset_rx,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize_req(
        &self,
        request_id: RawRequestID,
        history_tokens: Vec<Token>,
        prompt_tokens: Vec<Token>,
        sampled_tokens: Vec<Token>,
        input_positions: Option<RequestInputPositions>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling_config: SamplingConfig,
    ) -> Result<(RuntimeQueuedRequest<N, P, L>, ExternalRequest)> {
        let (resources, resource_placements): (Vec<_>, Vec<_>) = resource_entries.into_iter().unzip();
        let num_initial_tokens = history_tokens.len() + prompt_tokens.len() + sampled_tokens.len();
        assert!(
            input_positions
                .as_ref()
                .is_none_or(|positions| positions.initial().len() == num_initial_tokens),
            "explicit request input positions must match the initial token count"
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
        if let Some(input_positions) = &input_positions {
            let max_continuation_index = self.model_runtime_config.context_window - 1 - num_initial_tokens;
            assert!(
                input_positions
                    .continuation_start()
                    .iter()
                    .all(|&position| max_continuation_index <= u32::MAX as usize - position as usize),
                "explicit request continuation positions must fit u32 through the context window"
            );
        }
        if sampled_tokens.len() >= sampling_config.max_sampled_tokens {
            return Err(Error::invalid_argument(format!(
                "decode request initial sampled token count={} must be less than max_sampled_tokens={}",
                sampled_tokens.len(),
                sampling_config.max_sampled_tokens
            )));
        }
        validate_resources(&resources, &resource_placements, num_initial_tokens)?;
        let req_status = AtomicRequestStatus::new();
        let decoder_kv_blocks = TrieDecoderBlocks::new(
            self.block_cache.clone(),
            resources,
            resource_placements,
            history_tokens,
            prompt_tokens,
            sampled_tokens,
        );
        let (token_prob_tx, token_prob_rx) = async_unbounded();
        let queued_request = QueuedRequest::new(
            request_id,
            req_status.clone(),
            decoder_kv_blocks,
            input_positions,
            self.resource_processor.clone(),
            token_prob_tx,
            sampling_config,
            self.model_runtime_config.context_window,
        );
        let external_request = ExternalRequest::new(request_id, req_status, token_prob_rx);
        Ok((queued_request, external_request))
    }

    pub fn submit_req(&self, queued_request: RuntimeQueuedRequest<N, P, L>) -> Result<()> {
        let request_id = queued_request.req_id();
        match self.user_req_tx.try_send(queued_request) {
            Ok(()) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.queued",
                    request_id,
                    "decode request queued"
                );
                Ok(())
            },
            Err(TrySendError::Full(_)) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.queue_full",
                    request_id,
                    "request queue is full"
                );
                Err(Error::resource_exhausted("decode queue is full"))
            },
            Err(TrySendError::Disconnected(_)) => {
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

    pub fn model_executor_request_rx(&self) -> Receiver<ReplayableModelExecutorRequest> {
        self.model_executor_req_rx.clone()
    }

    pub fn model_executor_response_tx(&self) -> Sender<ReplayableModelExecutorResponse> {
        self.model_executor_resp_tx.clone()
    }

    pub fn request_slot_reset_notifier(&self) -> Arc<DedupNotifier<RawRequestSlot>> {
        self.request_slot_reset_notifier.clone()
    }

    pub fn request_slot_reset_rx(&self) -> Receiver<()> {
        self.request_slot_reset_rx.clone()
    }

    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }
}

impl<const N: usize, const L: usize, const P: usize> Drop for InferenceRuntime<N, L, P> {
    fn drop(&mut self) {
        self.shutdown.shutdown();
    }
}

pub fn serve_replay_model<const N: usize, const L: usize, M>(
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    http_service: HTTPService,
    resource_processor: Arc<ResourceProcessor>,
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    model: M,
) -> Result<()>
where
    M: ReplayableModel,
{
    let shutdown = Shutdown::new();
    let server_tokio_runtime = tokio::runtime::Runtime::new()
        .map_err(|error| log_err_unavailable!("unable to initialize RPC async runtime: {error}"))?;
    let model_name = model.model_name().to_string();
    let default_stop_sequences = model.default_stop_sequences();
    let num_spec_tokens = model.num_spec_tokens();
    let runtime = Arc::new(InferenceRuntime::<N, L, NUM_TRIE_PARTITION>::new(
        model_runtime_config,
        scheduler_config,
        num_spec_tokens,
        shutdown.clone(),
        server_tokio_runtime.handle(),
        resource_processor,
    ));
    let inference = Arc::new(Inference::new(runtime.clone(), default_stop_sequences));
    let server_shutdown = shutdown.clone();
    let server_thread = std::thread::Builder::new()
        .name("inference-rpc-servers".to_string())
        .spawn(move || {
            let _shutdown_guard = ShutdownGuard::new(server_shutdown.clone());
            server_tokio_runtime.block_on(rpc::run(
                grpc_listen_addr,
                http_listen_addr,
                model_name,
                http_service,
                inference,
                server_shutdown,
            ))
        })
        .map_err(|error| log_err_unavailable!("unable to start RPC server thread: {error}"))?;

    let executor = ReplayableModelEventLoop::new(
        runtime.model_executor_request_rx(),
        runtime.model_executor_response_tx(),
        runtime.request_slot_reset_notifier(),
        runtime.request_slot_reset_rx(),
        shutdown,
        model,
        default_model_state_snapshot_path(),
    );
    executor.event_loop();
    runtime.shutdown();

    server_thread
        .join()
        .map_err(|_| log_err_internal!("RPC server thread panicked"))?
}

fn default_model_state_snapshot_path() -> PathBuf {
    static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(0);

    let snapshot_id = NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("psi-dec-model-state-{}-{snapshot_id}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use inference_runtime_core::Error;
    use inference_runtime_core::channel::Shutdown;
    use inference_runtime_core::compute::BatchDeviceResponse;
    use inference_runtime_core::compute::DeviceResponse;
    use inference_runtime_core::compute::ExecutorHibernationPlan;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::compute::ReplayableModelExecutorRequest;
    use inference_runtime_core::compute::ReplayableModelExecutorResponse;
    use inference_runtime_core::compute::SampledTokens;
    use inference_runtime_core::config::CacheLaneRuntimeConfig;
    use inference_runtime_core::config::DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT;
    use inference_runtime_core::config::ExecutorHibernationMode;
    use inference_runtime_core::config::RuntimeConfig;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::config::SchedulerConfig;
    use inference_runtime_core::runtime::CompletionReason;
    use inference_runtime_core::runtime::RequestSlotAllocationResult;
    use inference_runtime_core::runtime::RequestSlotAllocator;
    use inference_runtime_core::runtime::RequestStatus;
    use inference_runtime_core::runtime::Token;
    use inference_runtime_core::runtime::scheduler::CommitResult;
    use inference_runtime_core::runtime::scheduler::ComputePhase;
    use inference_runtime_core::runtime::scheduler::PrepareResult;
    use inference_runtime_core::runtime::scheduler::UserRequest;
    use inference_runtime_core::runtime::tasks::ResourceProcessor;
    use ordered_float::NotNan;
    use tokio_stream::StreamExt;

    use super::InferenceRuntime;
    use crate::api::Inference;
    use crate::api::decode::DecodeEvent;
    use crate::api::decode::DecodeRequest;

    #[test]
    fn test_runtime_accepts_a_logical_cache_block_larger_than_one_physical_kv_page() {
        let shutdown = Shutdown::new();
        let async_task_runtime = tokio::runtime::Runtime::new().expect("test Tokio runtime should initialize");
        let runtime_config = RuntimeConfig {
            max_queued_requests: 1,
            max_running_requests: 1,
            executor_hibernation_timeout: DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            executor_hibernation_mode: ExecutorHibernationMode::Selected,
            context_window: 4096,
            num_tokens_per_cache_block: 1024,
            num_kv_heads: 1,
            kv_head_dim: 1,
            kv_dtype_bytes: 1,
            num_pages: 64,
            page_bytes: 32,
            cache_lanes: vec![CacheLaneRuntimeConfig {
                num_pages_per_kv_block: 64,
                num_pages_per_state_block: 0,
                block_cache_capacity: 1,
            }],
        };
        assert_eq!(runtime_config.num_tokens_per_page(), 16);
        assert_eq!(runtime_config.num_tokens_per_cache_block(), 1024);

        let runtime = InferenceRuntime::<1024, 1, 4>::new(
            runtime_config,
            SchedulerConfig {
                max_requests: 1,
                max_tokens: 1024,
                max_tokens_per_request: 1024,
                max_compute_slots: 1,
            },
            0,
            shutdown.clone(),
            async_task_runtime.handle(),
            Arc::new(ResourceProcessor::new()),
        );
        assert_eq!(runtime.model_runtime_config.num_tokens_per_cache_block(), 1024);
        shutdown.shutdown();
    }

    #[test]
    fn test_runtime_validates_initial_tokens_for_cache_lanes() {
        let (single_lane, single_lane_shutdown, _single_lane_async_runtime) = test_runtime::<1>();
        assert!(matches!(
            single_lane.initialize_req(
                1,
                vec![],
                vec![],
                vec![],
                None,
                vec![],
                SamplingConfig::default(),
            ),
            Err(Error::InvalidArgument(message)) if message.contains("minimum initial token count is 1")
        ));
        single_lane_shutdown.shutdown();

        let (mtp, mtp_shutdown, _mtp_async_runtime) = test_runtime::<4>();
        assert!(matches!(
            mtp.initialize_req(
                1,
                vec![],
                vec![Token::new(1), Token::new(2)],
                vec![],
                None,
                vec![],
                SamplingConfig::default(),
            ),
            Err(Error::InvalidArgument(message)) if message.contains("minimum initial token count is 3")
        ));
        assert!(
            mtp.initialize_req(
                2,
                vec![],
                vec![Token::new(1), Token::new(2), Token::new(3)],
                vec![],
                None,
                vec![],
                SamplingConfig::default(),
            )
            .is_ok()
        );
        mtp_shutdown.shutdown();
    }

    #[test]
    fn test_runtime_rejects_initial_tokens_without_output_capacity() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 2,
            ..SamplingConfig::default()
        };
        assert!(matches!(
            runtime.initialize_req(
                1,
                vec![Token::new(1)],
                vec![Token::new(2), Token::new(3)],
                vec![Token::new(4)],
                None,
                vec![],
                sampling,
            ),
            Err(Error::InvalidArgument(message))
                if message.contains("initial token count=4") && message.contains("context window=4")
        ));
        shutdown.shutdown();
    }

    #[test]
    fn test_runtime_rejects_exhausted_initial_sampled_limit() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 2,
            ..SamplingConfig::default()
        };
        assert!(matches!(
            runtime.initialize_req(
                1,
                vec![],
                vec![Token::new(1)],
                vec![Token::new(2), Token::new(3)],
                None,
                vec![],
                sampling,
            ),
            Err(Error::InvalidArgument(message))
                if message.contains("initial sampled token count=2")
                    && message.contains("max_sampled_tokens=2")
        ));
        shutdown.shutdown();
    }

    #[test]
    fn test_commit_trims_spec_input_and_context_visible_output() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 8,
            ..SamplingConfig::default()
        };
        let (queued_request, external_request) = runtime
            .initialize_req(
                1,
                vec![],
                vec![Token::new(1), Token::new(2)],
                vec![],
                None,
                vec![],
                sampling,
            )
            .unwrap();
        let (request_slot_allocator, _request_slot_reset_rx) = RequestSlotAllocator::new(1);
        let request_slot = match request_slot_allocator.allocate() {
            RequestSlotAllocationResult::Ok { request_slot } => request_slot,
            RequestSlotAllocationResult::ResourceLimitExceeded => panic!("test request slot should allocate"),
        };
        let mut request = super::RuntimeRequest::from((queued_request, request_slot));
        assert!(request.store_running());

        let first_query = prepare_decode(&mut request, 2);
        assert!(matches!(
            request.commit(decode_response(1, first_query, &[], 3, &[4, 5, 6])),
            CommitResult::Continue
        ));
        let first_visible = external_request.token_prob_rx().try_recv().unwrap();

        let second_query = prepare_decode(&mut request, 2);
        let QueryTokens::Decode { spec_tokens, .. } = &second_query else {
            panic!("test request should prepare Decode")
        };
        assert_eq!(spec_tokens, &[Token::new(4)]);
        assert!(matches!(
            request.commit(decode_response(1, second_query, &[4], 5, &[6, 7, 8])),
            CommitResult::Terminal
        ));
        let second_visible = external_request.token_prob_rx().try_recv().unwrap();
        assert_eq!(
            [first_visible.tokens, second_visible.tokens].concat(),
            vec![Token::new(3), Token::new(4)]
        );
        assert_eq!(
            external_request.status(),
            RequestStatus::Completed(CompletionReason::ContextLimit)
        );
        shutdown.shutdown();
    }

    #[test]
    fn test_commit_completion_priority_for_tied_limits() {
        assert_eq!(
            single_decode_completion(1, vec![vec![Token::new(4)]]),
            CompletionReason::StopSequence
        );
        assert_eq!(single_decode_completion(1, vec![]), CompletionReason::LengthLimit);
    }

    #[test]
    fn test_decode_completes_at_context_window_through_runtime_event_loop() {
        let (runtime, shutdown, async_runtime) = test_runtime_with_context::<1>(4);
        let runtime = Arc::new(runtime);
        let inference = Inference::new(runtime.clone(), vec![]);
        let sampling = SamplingConfig {
            max_sampled_tokens: 8,
            ..SamplingConfig::default()
        };
        let mut response = Box::pin(
            inference
                .decode(
                    DecodeRequest::new(
                        vec![Token::new(1), Token::new(2), Token::new(3)],
                        None,
                        vec![],
                        sampling,
                    )
                    .unwrap(),
                )
                .unwrap(),
        );

        let ReplayableModelExecutorRequest::Batch(batch_request) = runtime
            .model_executor_request_rx()
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
        else {
            panic!("runtime should submit a device batch")
        };
        let mut dev_reqs = batch_request.dev_reqs;
        assert_eq!(dev_reqs.len(), 1);
        let dev_req = dev_reqs.pop().unwrap();
        let dev_resp = decode_response(dev_req.req_id, dev_req.decoder_query_tokens, &[], 4, &[]);
        runtime
            .model_executor_response_tx()
            .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
                batch_request.seq,
                [dev_resp],
            )))
            .unwrap();

        async_runtime.block_on(async {
            let DecodeEvent::TokenProbs(token_probs) = response.next().await.unwrap().unwrap() else {
                panic!("decode response should contain the sampled token")
            };
            assert_eq!(token_probs.tokens, vec![Token::new(4)]);
            assert!(matches!(
                response.next().await.unwrap().unwrap(),
                DecodeEvent::Completed {
                    reason: CompletionReason::ContextLimit,
                    num_output_tokens: 1,
                }
            ));
            assert!(response.next().await.is_none());
        });
        shutdown.shutdown();
    }

    #[test]
    fn test_model_executor_hibernates_when_idle_and_starts_before_batch() {
        let (runtime, shutdown, _async_runtime) =
            test_runtime_with_executor_hibernation_timeout::<1>(Duration::from_millis(20));
        let request_rx = runtime.model_executor_request_rx();
        let response_tx = runtime.model_executor_response_tx();

        let ReplayableModelExecutorRequest::Stop(stopped_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("idle runtime must stop the model executor")
        };
        assert_eq!(stopped_plan, ExecutorHibernationPlan::selected(vec![], vec![]));
        response_tx.send(ReplayableModelExecutorResponse::Stopped).unwrap();

        let (queued_request, _external_request) = runtime
            .initialize_req(
                1,
                vec![],
                vec![Token::new(1)],
                vec![],
                None,
                vec![],
                SamplingConfig::default(),
            )
            .unwrap();
        runtime.submit_req(queued_request).unwrap();
        let ReplayableModelExecutorRequest::Start(started_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("queued request must start the model executor")
        };
        assert_eq!(started_plan, stopped_plan);
        response_tx.send(ReplayableModelExecutorResponse::Started).unwrap();
        assert!(matches!(
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ReplayableModelExecutorRequest::Batch(_)
        ));
        let ReplayableModelExecutorRequest::Stop(active_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("idle runtime must stop the model executor after the batch")
        };
        assert_eq!(
            active_plan,
            ExecutorHibernationPlan::selected(std::iter::once(0..1).collect(), std::iter::once(0..64).collect())
        );

        shutdown.shutdown();
    }

    #[test]
    fn test_model_executor_uses_fixed_all_hibernation_mode() {
        let (runtime, shutdown, _async_runtime) =
            test_runtime_with_executor_hibernation_mode::<1>(Duration::from_millis(20), ExecutorHibernationMode::All);

        let ReplayableModelExecutorRequest::Stop(plan) = runtime
            .model_executor_request_rx()
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
        else {
            panic!("idle runtime must stop the model executor")
        };
        assert_eq!(plan, ExecutorHibernationPlan::All);
        shutdown.shutdown();
    }

    fn test_runtime<const L: usize>() -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_executor_hibernation_timeout(DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT)
    }

    fn test_runtime_with_context<const L: usize>(
        context_window: usize,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_config(
            context_window,
            DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            ExecutorHibernationMode::Selected,
        )
    }

    fn test_runtime_with_executor_hibernation_timeout<const L: usize>(
        executor_hibernation_timeout: Duration,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_executor_hibernation_mode(executor_hibernation_timeout, ExecutorHibernationMode::Selected)
    }

    fn test_runtime_with_executor_hibernation_mode<const L: usize>(
        executor_hibernation_timeout: Duration,
        executor_hibernation_mode: ExecutorHibernationMode,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_config(4096, executor_hibernation_timeout, executor_hibernation_mode)
    }

    fn test_runtime_with_config<const L: usize>(
        context_window: usize,
        executor_hibernation_timeout: Duration,
        executor_hibernation_mode: ExecutorHibernationMode,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        let shutdown = Shutdown::new();
        let async_runtime = tokio::runtime::Runtime::new().expect("test Tokio runtime should initialize");
        let runtime = InferenceRuntime::<1024, L, 4>::new(
            RuntimeConfig {
                max_queued_requests: 1,
                max_running_requests: 1,
                executor_hibernation_timeout,
                executor_hibernation_mode,
                context_window,
                num_tokens_per_cache_block: 1024,
                num_kv_heads: 1,
                kv_head_dim: 1,
                kv_dtype_bytes: 1,
                num_pages: 64 * L,
                page_bytes: 32,
                cache_lanes: (0..L)
                    .map(|_| {
                        CacheLaneRuntimeConfig {
                            num_pages_per_kv_block: 64,
                            num_pages_per_state_block: 0,
                            block_cache_capacity: 1,
                        }
                    })
                    .collect(),
            },
            SchedulerConfig {
                max_requests: 1,
                max_tokens: 1024,
                max_tokens_per_request: 1024,
                max_compute_slots: 1,
            },
            0,
            shutdown.clone(),
            async_runtime.handle(),
            Arc::new(ResourceProcessor::new()),
        );
        (runtime, shutdown, async_runtime)
    }

    fn prepare_decode(request: &mut super::RuntimeRequest<1024, 4, 1>, token_budget: usize) -> QueryTokens {
        match request.prepare(token_budget) {
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode { .. },
            } => dev_req.decoder_query_tokens,
            _ => panic!("test request should prepare Decode"),
        }
    }

    fn single_decode_completion(max_sampled_tokens: usize, stop_sequences: Vec<Vec<Token>>) -> CompletionReason {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens,
            stop_sequences,
            ..SamplingConfig::default()
        };
        let (queued_request, external_request) = runtime
            .initialize_req(
                1,
                vec![],
                vec![Token::new(1), Token::new(2), Token::new(3)],
                vec![],
                None,
                vec![],
                sampling,
            )
            .unwrap();
        let (request_slot_allocator, _request_slot_reset_rx) = RequestSlotAllocator::new(1);
        let request_slot = match request_slot_allocator.allocate() {
            RequestSlotAllocationResult::Ok { request_slot } => request_slot,
            RequestSlotAllocationResult::ResourceLimitExceeded => panic!("test request slot should allocate"),
        };
        let mut request = super::RuntimeRequest::from((queued_request, request_slot));
        assert!(request.store_running());
        let query_tokens = prepare_decode(&mut request, 3);
        assert!(matches!(
            request.commit(decode_response(1, query_tokens, &[], 4, &[])),
            CommitResult::Terminal
        ));
        let RequestStatus::Completed(completion) = external_request.status() else {
            panic!("test request should complete")
        };
        shutdown.shutdown();
        completion
    }

    fn decode_response(
        req_id: usize,
        query_tokens: QueryTokens,
        validated_tokens: &[u32],
        sampled_token: u32,
        spec_tokens: &[u32],
    ) -> DeviceResponse {
        let probability = NotNan::new(1.0).unwrap();
        DeviceResponse {
            req_id,
            sampled_tokens: SampledTokens::Decode {
                epoch: query_tokens.epoch(),
                validated_tokens: validated_tokens.iter().copied().map(Token::new).collect(),
                validated_probs: vec![probability; validated_tokens.len()],
                sampled_token: Token::new(sampled_token),
                sampled_prob: probability,
                spec_tokens: spec_tokens.iter().copied().map(Token::new).collect(),
                spec_probs: vec![probability; spec_tokens.len()],
                spec_confidences: vec![probability; spec_tokens.len()],
            },
            query_tokens,
        }
    }
}
