use std::sync::Arc;

use async_channel::bounded as async_bounded;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TrySendError;
use crossbeam_channel::bounded as sync_bounded;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::config::ServiceConfig;
use inference_runtime_core::memory::U32IDAllocator;
use inference_runtime_core::runtime::AtomicRequestStatus;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::InternalRequest;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::RequestSlotAllocationResult;
use inference_runtime_core::runtime::RequestSlotAllocator;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::decoder::TPKVBlockAllocator;
use inference_runtime_core::runtime::decoder::TPStateBlockAllocator;
use inference_runtime_core::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use inference_runtime_core::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use inference_runtime_core::runtime::decoder::trie_cache::TrieDecoderBlocks;
use inference_runtime_core::runtime::scheduler::EventLoop;
use inference_runtime_core::runtime::scheduler::FIFOBatcher;
use inference_runtime_core::runtime::scheduler::FIFOScheduler;
use inference_runtime_core::runtime::scheduler::InstrumentedScheduler;
use inference_runtime_core::runtime::scheduler::ScheduleQueue;
use tonic::Status;

pub struct InferenceRuntime<const N: usize, const L: usize, const P: usize> {
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    service_config: ServiceConfig,

    shutdown: Shutdown,
    req_slot_allocator: RequestSlotAllocator,
    block_cache: Arc<MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>>,

    user_req_tx:
        Sender<InternalRequest<N, P, L, MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>>>,
    batch_dev_req_rx: Receiver<BatchDeviceRequest>,
    batch_dev_resp_tx: Sender<BatchDeviceResponse>,
    request_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    request_slot_reset_rx: Receiver<()>,
}

impl<const N: usize, const L: usize, const P: usize> InferenceRuntime<N, L, P> {
    pub fn new(
        model_runtime_config: RuntimeConfig,
        scheduler_config: SchedulerConfig,
        service_config: ServiceConfig,
        shutdown: Shutdown,
    ) -> Self {
        assert!(scheduler_config.max_requests > 0, "runtime requires request capacity");
        assert!(scheduler_config.max_tokens > 0, "runtime requires token capacity");
        assert!(
            scheduler_config.max_tokens_per_request > 0,
            "runtime requires per-request token capacity"
        );
        assert!(scheduler_config.max_compute_slots > 0, "runtime requires compute slots");
        assert!(
            service_config.user_req_queue_capacity > 0,
            "runtime requires user-request queue capacity"
        );
        assert!(
            service_config.batch_req_queue_capacity > 0,
            "runtime requires batch-request queue capacity"
        );
        assert!(
            service_config.batch_resp_queue_capacity > 0,
            "runtime requires batch-response queue capacity"
        );
        assert!(
            service_config.token_prob_channel_capacity > 0,
            "runtime requires token-output queue capacity"
        );
        assert!(
            service_config.batch_req_queue_capacity >= scheduler_config.max_compute_slots,
            "batch-request queue must cover all compute slots"
        );
        assert!(
            service_config.batch_resp_queue_capacity >= scheduler_config.max_compute_slots,
            "batch-response queue must cover all compute slots"
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
            RequestSlotAllocator::new(scheduler_config.max_requests as u64);
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

        let (user_req_tx, user_req_rx) = sync_bounded(service_config.user_req_queue_capacity);
        let (batch_dev_req_tx, batch_dev_req_rx) = sync_bounded(service_config.batch_req_queue_capacity);
        let (batch_dev_resp_tx, batch_dev_resp_rx) = sync_bounded(service_config.batch_resp_queue_capacity);

        {
            let schedule_queue = ScheduleQueue::new();
            let batcher = FIFOBatcher::new(scheduler_config.max_tokens_per_request);
            let scheduler = FIFOScheduler::new(
                schedule_queue,
                batcher,
                scheduler_config.max_requests,
                scheduler_config.max_tokens,
                scheduler_config.max_tokens_per_request,
                scheduler_config.wait_duration,
                scheduler_config.max_compute_slots,
            );
            let scheduler = InstrumentedScheduler::new(scheduler);
            let event_loop = EventLoop::new(
                user_req_rx,
                batch_dev_req_tx,
                batch_dev_resp_rx,
                scheduler,
                service_config.user_req_queue_capacity,
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
        };

        Self {
            model_runtime_config,
            scheduler_config,
            service_config,

            shutdown,
            req_slot_allocator,
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
        request_id: u64,
        tokens: Vec<Token>,
        sampling_config: SamplingConfig,
    ) -> Result<
        (
            InternalRequest<N, P, L, MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>>,
            ExternalRequest,
        ),
        Status,
    > {
        let req_id = request_id as usize;
        let req_slot = match self.req_slot_allocator.allocate() {
            RequestSlotAllocationResult::Ok { request_slot } => request_slot,
            RequestSlotAllocationResult::ResourceLimitExceeded => {
                return Err(Status::resource_exhausted(
                    "inference runtime service: no free request slots".to_string(),
                ));
            },
        };
        let req_status = AtomicRequestStatus::new();
        let decoder_kv_blocks = TrieDecoderBlocks::new(self.block_cache.clone(), [], tokens, []);
        let (token_prob_tx, token_prob_rx) = async_bounded(self.service_config.token_prob_channel_capacity);
        let internal_request = InternalRequest::new(
            req_id,
            req_slot,
            req_status.clone(),
            decoder_kv_blocks,
            token_prob_tx,
            sampling_config,
        );
        let external_request = ExternalRequest::new(req_id, req_status, token_prob_rx);
        Ok((internal_request, external_request))
    }

    pub fn submit_req(
        &self,
        internal_request: InternalRequest<
            N,
            P,
            L,
            MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>,
        >,
    ) -> Result<(), Status> {
        let request_id = internal_request.req_id();
        assert!(
            internal_request.store_running(),
            "runtime submit requires an initialized request"
        );
        match self.user_req_tx.try_send(internal_request) {
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
                Err(Status::resource_exhausted(
                    "inference runtime service: request queue is full",
                ))
            },
            Err(TrySendError::Disconnected(_)) => {
                tracing::debug!(
                    target: "inference-runtime-service::runtime",
                    phase = "request.runtime_stopped",
                    request_id,
                    "runtime is stopped"
                );
                Err(Status::unavailable("inference runtime service: runtime is stopped"))
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

#[cfg(test)]
mod tests {
    use inference_runtime_core::channel::Shutdown;
    use inference_runtime_core::config::CacheLaneRuntimeConfig;
    use inference_runtime_core::config::RuntimeConfig;
    use inference_runtime_core::config::SchedulerConfig;
    use inference_runtime_core::config::ServiceConfig;

    use super::InferenceRuntime;

    #[test]
    fn test_runtime_accepts_a_logical_cache_block_larger_than_one_physical_kv_page() {
        let shutdown = Shutdown::new();
        let runtime_config = RuntimeConfig {
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
                wait_duration: std::time::Duration::ZERO,
                max_compute_slots: 1,
            },
            ServiceConfig {
                user_req_queue_capacity: 1,
                batch_req_queue_capacity: 1,
                batch_resp_queue_capacity: 1,
                token_prob_channel_capacity: 1,
            },
            shutdown.clone(),
        );
        assert_eq!(runtime.model_runtime_config.num_tokens_per_cache_block(), 1024);
        shutdown.shutdown();
    }
}
