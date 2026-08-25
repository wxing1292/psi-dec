use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use crossbeam_channel::after;

use super::executor_hibernate::collect_allocated_id_ranges;
use crate::Result;
use crate::channel::Shutdown;
use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::compute::ExecutorHibernationPlan;
use crate::compute::ReplayableModelExecutorRequest;
use crate::compute::ReplayableModelExecutorResponse;
use crate::config::ExecutorHibernationMode;
use crate::log_err_internal;
use crate::memory::U32IDAllocator;
use crate::runtime::RequestSlot;
use crate::runtime::RequestSlotAllocationResult;
use crate::runtime::RequestSlotAllocator;
use crate::runtime::scheduler::InstrumentedScheduler;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::tasks::AsyncTaskResp;

const SCHEDULER_STATS_INTERVAL: Duration = Duration::from_secs(30);

pub struct EventLoop<QueuedReq, UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S> {
    user_req_rx: Receiver<QueuedReq>,
    async_task_resp_rx: Receiver<Box<dyn AsyncTaskResp>>,
    model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
    model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,

    scheduler: InstrumentedScheduler<S>,
    request_slot_allocator: RequestSlotAllocator,
    page_id_allocator: Arc<U32IDAllocator>,
    model_executor_state: ModelExecutorState,

    scheduler_stats_timer: Receiver<Instant>,

    executor_hibernation_mode: ExecutorHibernationMode,
    executor_hibernation_timeout: Duration,
    executor_hibernation_timer: Receiver<Instant>,
    idle_heartbeat: Instant,

    shutdown: Shutdown,

    phantom_data_user_req: PhantomData<UserReq>,
    phantom_data_device_req: PhantomData<DeviceReq>,
    phantom_data_device_resp: PhantomData<DeviceResp>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ModelExecutorState {
    Started,
    Stopped(ExecutorHibernationPlan),
}

impl<QueuedReq, UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
    EventLoop<QueuedReq, UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
where
    UserReq: From<(QueuedReq, RequestSlot)> + UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    S: Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        user_req_rx: Receiver<QueuedReq>,
        async_task_resp_rx: Receiver<Box<dyn AsyncTaskResp>>,
        model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
        model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,
        scheduler: InstrumentedScheduler<S>,
        request_slot_allocator: RequestSlotAllocator,
        page_id_allocator: Arc<U32IDAllocator>,
        executor_hibernation_mode: ExecutorHibernationMode,
        executor_hibernation_timeout: Duration,
        shutdown: Shutdown,
    ) -> Self {
        assert!(
            !executor_hibernation_timeout.is_zero(),
            "runtime executor hibernation timeout must be positive"
        );
        Self {
            user_req_rx,
            async_task_resp_rx,
            model_executor_req_tx,
            model_executor_resp_rx,
            scheduler,
            request_slot_allocator,
            page_id_allocator,
            model_executor_state: ModelExecutorState::Started,

            scheduler_stats_timer: after(SCHEDULER_STATS_INTERVAL),

            executor_hibernation_mode,
            executor_hibernation_timeout,
            executor_hibernation_timer: after(executor_hibernation_timeout),
            idle_heartbeat: Instant::now(),

            shutdown,

            phantom_data_user_req: PhantomData,
            phantom_data_device_req: PhantomData,
            phantom_data_device_resp: PhantomData,
        }
    }

    pub fn event_loop(mut self) -> Result<()> {
        let _span = tracing::info_span!("runtime").entered();
        tracing::info!("started");

        let shutdown_rx = self.shutdown.sync_rx().clone();
        'event_loop: while !self.shutdown.is_shutdown() {
            let mut select = Select::new();
            let op_recv_req = if self.request_slot_allocator.free() > 0 {
                Some(select.recv(&self.user_req_rx))
            } else {
                None
            };
            let op_recv_async_task_resp = select.recv(&self.async_task_resp_rx);
            let op_recv_model_executor_resp = select.recv(&self.model_executor_resp_rx);
            let op_scheduler_stats_timer = select.recv(&self.scheduler_stats_timer);
            let op_executor_hibernation_timer = match &self.model_executor_state {
                ModelExecutorState::Started => Some(select.recv(&self.executor_hibernation_timer)),
                ModelExecutorState::Stopped(_) => None,
            };
            let op_shutdown = select.recv(&shutdown_rx);
            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if Some(op_index) == op_recv_req => {
                    let queued_req = op
                        .recv(&self.user_req_rx)
                        .map_err(|error| log_err_internal!("user request channel closed, stopping: {error}"))?;
                    let request_slot = match self.request_slot_allocator.allocate() {
                        RequestSlotAllocationResult::Ok { request_slot } => request_slot,
                        RequestSlotAllocationResult::ResourceLimitExceeded => {
                            panic!("available request-slot capacity must allow allocation")
                        },
                    };
                    let user_req = UserReq::from((queued_req, request_slot));
                    user_req.store_running();
                    self.scheduler.enqueue(user_req);
                },
                _ if op_index == op_recv_async_task_resp => {
                    let resp = op
                        .recv(&self.async_task_resp_rx)
                        .map_err(|error| log_err_internal!("async task response channel closed, stopping: {error}"))?;
                    self.scheduler.handle_async_task_resp(resp);
                },
                _ if op_index == op_recv_model_executor_resp => {
                    let model_executor_resp = op.recv(&self.model_executor_resp_rx).map_err(|error| {
                        log_err_internal!("model executor response channel closed, stopping: {error}")
                    })?;
                    match model_executor_resp {
                        ReplayableModelExecutorResponse::Batch(batch_dev_resp) => {
                            self.scheduler.commit(batch_dev_resp);
                            self.idle_heartbeat = Instant::now();
                        },
                        ReplayableModelExecutorResponse::Started | ReplayableModelExecutorResponse::Stopped => {},
                    }
                },
                _ if op_index == op_scheduler_stats_timer => {
                    let _ = op
                        .recv(&self.scheduler_stats_timer)
                        .expect("selected scheduler stats timer must fire");
                    self.scheduler.print_periodical();
                    self.scheduler_stats_timer = after(SCHEDULER_STATS_INTERVAL);
                },
                _ if Some(op_index) == op_executor_hibernation_timer => {
                    let _ = op
                        .recv(&self.executor_hibernation_timer)
                        .expect("selected executor hibernation timer must fire");
                    let idle_duration = self.idle_heartbeat.elapsed();
                    if idle_duration >= self.executor_hibernation_timeout {
                        self.stop_model_executor()?;
                    } else {
                        self.executor_hibernation_timer = after(self.executor_hibernation_timeout - idle_duration);
                    }
                },
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    break 'event_loop;
                },
                _ => unreachable!(),
            }
            self.do_flush()?;
        }

        self.shutdown.shutdown();
        self.scheduler.print_lifetime();
        tracing::info!("stopped");
        Ok(())
    }

    fn do_flush(&mut self) -> Result<()> {
        match &self.model_executor_state {
            ModelExecutorState::Started => {
                while self.scheduler.can_flush() {
                    let batch_dev_req = self.scheduler.prepare();
                    match self
                        .model_executor_req_tx
                        .try_send(ReplayableModelExecutorRequest::Batch(batch_dev_req))
                    {
                        Ok(()) => {
                            self.idle_heartbeat = Instant::now();
                        },
                        Err(error) => {
                            let ReplayableModelExecutorRequest::Batch(batch_dev_req) = error.into_inner() else {
                                unreachable!("runtime scheduler only sends Batch model executor requests")
                            };
                            self.scheduler.cancel(batch_dev_req);
                            return Err(log_err_internal!(
                                "batch device request channel full / closed, stopping"
                            ));
                        },
                    }
                }
            },
            ModelExecutorState::Stopped(_) if self.scheduler.can_flush() => {
                self.start_model_executor()?;
            },
            ModelExecutorState::Stopped(_) => {},
        }
        Ok(())
    }

    fn start_model_executor(&mut self) -> Result<()> {
        let ModelExecutorState::Stopped(plan) = &self.model_executor_state else {
            panic!("runtime can start only a stopped model executor")
        };

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Start(plan.clone()))
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while starting, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Started;
        self.executor_hibernation_timer = after(self.executor_hibernation_timeout);
        self.idle_heartbeat = Instant::now();
        Ok(())
    }

    fn stop_model_executor(&mut self) -> Result<()> {
        debug_assert_eq!(self.model_executor_state, ModelExecutorState::Started);

        let plan = match self.executor_hibernation_mode {
            ExecutorHibernationMode::All => ExecutorHibernationPlan::All,
            ExecutorHibernationMode::Selected => {
                ExecutorHibernationPlan::selected(
                    collect_allocated_id_ranges(self.request_slot_allocator.allocated_ids_bitmap_iter()),
                    collect_allocated_id_ranges(self.page_id_allocator.allocated_ids_bitmap_iter()),
                )
            },
        };

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Stop(plan.clone()))
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while stopping, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Stopped(plan);
        Ok(())
    }
}
