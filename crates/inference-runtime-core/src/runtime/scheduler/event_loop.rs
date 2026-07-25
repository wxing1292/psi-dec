use std::marker::PhantomData;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;

use crate::Result;
use crate::channel::Shutdown;
use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
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
    batch_dev_req_tx: Sender<BatchDeviceReq>,
    batch_dev_resp_rx: Receiver<BatchDeviceResp>,

    scheduler: InstrumentedScheduler<S>,
    request_slot_allocator: RequestSlotAllocator,

    shutdown: Shutdown,

    phantom_data_device_req: PhantomData<DeviceReq>,
    phantom_data_device_resp: PhantomData<DeviceResp>,
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
    pub fn new(
        user_req_rx: Receiver<QueuedReq>,
        swap_in_task_rx: Receiver<UserReq>,
        batch_dev_req_tx: Sender<BatchDeviceReq>,
        batch_dev_resp_rx: Receiver<BatchDeviceResp>,
        scheduler: InstrumentedScheduler<S>,
        request_slot_allocator: RequestSlotAllocator,
        shutdown: Shutdown,
    ) -> Self {
        Self {
            user_req_rx,
            swap_in_task_rx,
            batch_dev_req_tx,
            batch_dev_resp_rx,

            scheduler,
            request_slot_allocator,

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
            let op_recv_batch_dev_resp = select.recv(&self.batch_dev_resp_rx);
            let op_recv_swap_in_task = select.recv(&self.swap_in_task_rx);
            let op_recv_req = if self.request_slot_allocator.free() > 0 {
                Some(select.recv(&self.user_req_rx))
            } else {
                None
            };
            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    tracing::info!("received shutdown signal, stopping");
                    break 'event_loop;
                },
                _ if op_index == op_recv_batch_dev_resp => {
                    let batch_dev_resp = op.recv(&self.batch_dev_resp_rx);
                    match batch_dev_resp {
                        Ok(batch_dev_resp) => {
                            self.scheduler.commit(batch_dev_resp);
                            if self.scheduler.can_flush()
                                && do_flush(&mut self.scheduler, &self.batch_dev_req_tx).is_err()
                            {
                                break 'event_loop;
                            }
                        },
                        Err(_) => {
                            tracing::debug!("batch device response channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ if op_index == op_recv_swap_in_task => {
                    let user_req = op.recv(&self.swap_in_task_rx);
                    match user_req {
                        Ok(user_req) => {
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
                        Err(_) => {
                            tracing::debug!("swap-in request channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                    // TODO: drain both ready request receivers before flushing
                    // so swap-in priority does not depend on Select order.
                    if self.scheduler.can_flush() && do_flush(&mut self.scheduler, &self.batch_dev_req_tx).is_err() {
                        break 'event_loop;
                    }
                },
                _ if Some(op_index) == op_recv_req => {
                    let queued_req = op.recv(&self.user_req_rx);
                    match queued_req {
                        Ok(queued_req) => {
                            let request_slot = match self.request_slot_allocator.allocate() {
                                RequestSlotAllocationResult::Ok { request_slot } => request_slot,
                                RequestSlotAllocationResult::ResourceLimitExceeded => {
                                    panic!("available request-slot capacity must allow allocation")
                                },
                            };
                            let user_req = UserReq::from((queued_req, request_slot));
                            user_req.store_running();
                            self.scheduler.enqueue(user_req);
                            if self.scheduler.can_flush()
                                && do_flush(&mut self.scheduler, &self.batch_dev_req_tx).is_err()
                            {
                                break 'event_loop;
                            }
                        },
                        Err(_) => {
                            tracing::debug!("user request channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ => unreachable!(),
            }
        }

        self.shutdown.shutdown();
        tracing::info!("\n{}", self.scheduler.stats_table());
        tracing::info!("stopped");
        Ok(())
    }
}

fn do_flush<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>(
    scheduler: &mut S,
    batch_dev_req_tx: &Sender<BatchDeviceReq>,
) -> Result<()>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    S: Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    let batch_dev_req = scheduler.prepare();
    match batch_dev_req_tx.try_send(batch_dev_req) {
        Ok(()) => Ok(()),
        Err(err) => {
            let batch_dev_req = err.into_inner();
            scheduler.cancel(batch_dev_req);
            Err(log_err_internal!(
                "batch device request channel full / closed, stopping"
            ))
        },
    }
}
