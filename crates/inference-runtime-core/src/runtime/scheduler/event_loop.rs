use std::marker::PhantomData;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use crossbeam_channel::after;

use crate::Result;
use crate::channel::Shutdown;
use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::compute::ReplayableModelExecutorRequest;
use crate::compute::ReplayableModelExecutorResponse;
use crate::log_err_internal;
use crate::runtime::RequestSlot;
use crate::runtime::RequestSlotAllocationResult;
use crate::runtime::RequestSlotAllocator;
use crate::runtime::scheduler::InstrumentedScheduler;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct EventLoop<QueuedReq, UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S> {
    user_req_rx: Receiver<QueuedReq>,
    swap_in_task_rx: Receiver<UserReq>,
    model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
    model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,

    scheduler: InstrumentedScheduler<S>,
    request_slot_allocator: RequestSlotAllocator,
    model_executor_state: ModelExecutorState,
    model_idle_timeout: Duration,
    model_idle_timer: Receiver<Instant>,
    idle_heartbeat: Instant,

    shutdown: Shutdown,

    phantom_data_device_req: PhantomData<DeviceReq>,
    phantom_data_device_resp: PhantomData<DeviceResp>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelExecutorState {
    Started,
    Stopped,
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
        swap_in_task_rx: Receiver<UserReq>,
        model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
        model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,
        scheduler: InstrumentedScheduler<S>,
        request_slot_allocator: RequestSlotAllocator,
        model_idle_timeout: Duration,
        shutdown: Shutdown,
    ) -> Self {
        assert!(
            !model_idle_timeout.is_zero(),
            "runtime model idle timeout must be positive"
        );
        Self {
            user_req_rx,
            swap_in_task_rx,
            model_executor_req_tx,
            model_executor_resp_rx,
            scheduler,
            request_slot_allocator,
            model_executor_state: ModelExecutorState::Started,
            model_idle_timeout,
            model_idle_timer: after(model_idle_timeout),
            idle_heartbeat: Instant::now(),

            shutdown,

            phantom_data_device_req: PhantomData,
            phantom_data_device_resp: PhantomData,
        }
    }

    pub fn event_loop(mut self) -> Result<()> {
        let span = tracing::info_span!("event loop");
        let _enter = span.enter();
        tracing::info!("started");

        let shutdown_rx = self.shutdown.sync_rx().clone();
        'event_loop: while !self.shutdown.is_shutdown() {
            let mut select = Select::new();
            let op_shutdown = select.recv(&shutdown_rx);
            let op_recv_model_executor_resp = select.recv(&self.model_executor_resp_rx);
            let op_recv_swap_in_task = select.recv(&self.swap_in_task_rx);
            let op_recv_req = if self.request_slot_allocator.free() > 0 {
                Some(select.recv(&self.user_req_rx))
            } else {
                None
            };
            let op_model_idle_timer = match self.model_executor_state {
                ModelExecutorState::Started => Some(select.recv(&self.model_idle_timer)),
                ModelExecutorState::Stopped => None,
            };
            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    tracing::info!("received shutdown signal, stopping");
                    break 'event_loop;
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
                _ if op_index == op_recv_swap_in_task => {
                    let user_req = op
                        .recv(&self.swap_in_task_rx)
                        .map_err(|error| log_err_internal!("swap-in request channel closed, stopping: {error}"))?;
                    if user_req.is_terminal() {
                        tracing::debug!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.reservation_wait_terminal",
                            request_id = user_req.id(),
                            "terminal reservation-wait request dropped"
                        );
                        drop(user_req);
                    } else {
                        self.scheduler.swap_in(user_req);
                    }
                },
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
                _ if Some(op_index) == op_model_idle_timer => {
                    let _ = op
                        .recv(&self.model_idle_timer)
                        .expect("selected model idle timer must fire");
                    let idle_duration = self.idle_heartbeat.elapsed();
                    if idle_duration >= self.model_idle_timeout {
                        self.stop_model_executor()?;
                    } else {
                        self.model_idle_timer = after(self.model_idle_timeout - idle_duration);
                    }
                },
                _ => unreachable!(),
            }
            self.do_flush()?;
        }

        self.shutdown.shutdown();
        tracing::info!("\n{}", self.scheduler.stats_table());
        tracing::info!("stopped");
        Ok(())
    }

    fn do_flush(&mut self) -> Result<()> {
        match self.model_executor_state {
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
            ModelExecutorState::Stopped if self.scheduler.can_flush() => {
                self.start_model_executor()?;
            },
            ModelExecutorState::Stopped => {},
        }
        Ok(())
    }

    fn start_model_executor(&mut self) -> Result<()> {
        debug_assert_eq!(self.model_executor_state, ModelExecutorState::Stopped);

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Start)
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while starting, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Started;
        self.model_idle_timer = after(self.model_idle_timeout);
        self.idle_heartbeat = Instant::now();
        Ok(())
    }

    fn stop_model_executor(&mut self) -> Result<()> {
        debug_assert_eq!(self.model_executor_state, ModelExecutorState::Started);

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Stop)
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while stopping, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Stopped;
        Ok(())
    }
}
