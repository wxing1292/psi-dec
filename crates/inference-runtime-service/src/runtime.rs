use std::net::SocketAddr;
use std::sync::Arc;

use async_channel::bounded as async_bounded;
use async_channel::unbounded as async_unbounded;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TrySendError;
use crossbeam_channel::bounded as sync_bounded;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ReplayableModelBatchExecutor;
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
use inference_runtime_core::runtime::RequestSlotAllocator;
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

use crate::api::Inference;
use crate::consts::NUM_TRIE_PARTITION;
use crate::executor::ReplayableModelExecutorLoop;
use crate::rpc::run_servers;

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

    user_req_tx: Sender<RuntimeQueuedRequest<N, P, L>>,
    batch_dev_req_rx: Receiver<BatchDeviceRequest>,
    batch_dev_resp_tx: Sender<BatchDeviceResponse>,
    request_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    request_slot_reset_rx: Receiver<()>,
}

impl<const N: usize, const L: usize, const P: usize> InferenceRuntime<N, L, P> {
    pub fn new(
        model_runtime_config: RuntimeConfig,
        scheduler_config: SchedulerConfig,
        shutdown: Shutdown,
        async_task_handle: &tokio::runtime::Handle,
    ) -> Self {
        assert!(scheduler_config.max_requests > 0, "runtime requires request capacity");
        assert!(scheduler_config.max_tokens > 0, "runtime requires token capacity");
        assert!(
            scheduler_config.max_tokens_per_request > 0,
            "runtime requires per-request token capacity"
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
        let block_cache = {
            let page_id_allocator = Arc::new(U32IDAllocator::new(model_runtime_config.num_pages as u64));
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
        let (batch_dev_req_tx, batch_dev_req_rx) = sync_bounded(scheduler_config.max_compute_slots);
        let (batch_dev_resp_tx, batch_dev_resp_rx) = sync_bounded(scheduler_config.max_compute_slots);
        let (swap_out_task_tx, swap_out_task_rx) = async_bounded(model_runtime_config.max_running_requests);
        let (swap_in_task_tx, swap_in_task_rx) =
            sync_bounded::<RuntimeRequest<N, P, L>>(model_runtime_config.max_running_requests);

        {
            let schedule_queue = ScheduleQueue::new(swap_out_task_tx);
            let batcher = FIFOBatcher::new();
            let scheduler = InstrumentedScheduler::new(SimpleScheduler::new(
                schedule_queue,
                batcher,
                scheduler_config.max_requests,
                scheduler_config.max_tokens,
                scheduler_config.max_tokens_per_request,
                scheduler_config.max_compute_slots,
            ));
            let event_loop = EventLoop::new(
                user_req_rx,
                swap_in_task_rx,
                batch_dev_req_tx,
                batch_dev_resp_rx,
                scheduler,
                req_slot_allocator,
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
                swap_out_task_rx,
                swap_in_task_tx,
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

            user_req_tx,
            batch_dev_req_rx,
            batch_dev_resp_tx,
            request_slot_reset_notifier,
            request_slot_reset_rx,
        }
    }

    pub fn initialize_req(
        &self,
        request_id: RawRequestID,
        tokens: Vec<Token>,
        sampling_config: SamplingConfig,
    ) -> (RuntimeQueuedRequest<N, P, L>, ExternalRequest) {
        let req_status = AtomicRequestStatus::new();
        let decoder_kv_blocks = TrieDecoderBlocks::new(self.block_cache.clone(), [], tokens, []);
        let (token_prob_tx, token_prob_rx) = async_unbounded();
        let queued_request = QueuedRequest::new(
            request_id,
            req_status.clone(),
            decoder_kv_blocks,
            token_prob_tx,
            sampling_config,
        );
        let external_request = ExternalRequest::new(request_id, req_status, token_prob_rx);
        (queued_request, external_request)
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

    pub fn batch_device_request_rx(&self) -> Receiver<BatchDeviceRequest> {
        self.batch_dev_req_rx.clone()
    }

    pub fn batch_device_response_tx(&self) -> Sender<BatchDeviceResponse> {
        self.batch_dev_resp_tx.clone()
    }

    pub fn request_slot_reset_notifier(&self) -> Arc<DedupNotifier<RawRequestSlot>> {
        self.request_slot_reset_notifier.clone()
    }

    pub fn request_slot_reset_rx(&self) -> Receiver<()> {
        self.request_slot_reset_rx.clone()
    }

    pub fn shutdown(&self) {
        tracing::info!("inference runtime service: shutdown requested");
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
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    model: M,
    debug_logging: bool,
) -> Result<()>
where
    M: ReplayableModelBatchExecutor,
{
    let shutdown = Shutdown::new();
    let server_tokio_runtime = tokio::runtime::Runtime::new()
        .map_err(|error| log_err_unavailable!("unable to initialize RPC async runtime: {error}"))?;
    let default_stop_sequences = model.default_stop_sequences();
    let runtime = Arc::new(InferenceRuntime::<N, L, NUM_TRIE_PARTITION>::new(
        model_runtime_config,
        scheduler_config,
        shutdown.clone(),
        server_tokio_runtime.handle(),
    ));
    let inference = Arc::new(Inference::new(runtime.clone(), default_stop_sequences));
    let server_shutdown = shutdown.clone();
    let server_thread = std::thread::Builder::new()
        .name("inference-rpc-servers".to_string())
        .spawn(move || {
            let _shutdown_guard = ShutdownGuard::new(server_shutdown.clone());
            server_tokio_runtime.block_on(run_servers(
                grpc_listen_addr,
                http_listen_addr,
                inference,
                server_shutdown,
            ))
        })
        .map_err(|error| log_err_unavailable!("unable to start RPC server thread: {error}"))?;

    let executor = ReplayableModelExecutorLoop::new(
        runtime.batch_device_request_rx(),
        runtime.batch_device_response_tx(),
        runtime.request_slot_reset_notifier(),
        runtime.request_slot_reset_rx(),
        shutdown,
        model,
    )
    .with_debug_logging(debug_logging);
    executor.event_loop();
    runtime.shutdown();

    server_thread
        .join()
        .map_err(|_| log_err_internal!("RPC server thread panicked"))?
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::channel::Shutdown;
    use inference_runtime_core::config::CacheLaneRuntimeConfig;
    use inference_runtime_core::config::RuntimeConfig;
    use inference_runtime_core::config::SchedulerConfig;

    use super::InferenceRuntime;

    #[test]
    fn test_runtime_accepts_a_logical_cache_block_larger_than_one_physical_kv_page() {
        let shutdown = Shutdown::new();
        let async_task_runtime = tokio::runtime::Runtime::new().expect("test Tokio runtime should initialize");
        let runtime_config = RuntimeConfig {
            max_queued_requests: 1,
            max_running_requests: 1,
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
                max_tokens: 1,
                max_tokens_per_request: 1024,
                max_compute_slots: 1,
            },
            shutdown.clone(),
            async_task_runtime.handle(),
        );
        assert_eq!(runtime.model_runtime_config.num_tokens_per_cache_block(), 1024);
        shutdown.shutdown();
    }
}
