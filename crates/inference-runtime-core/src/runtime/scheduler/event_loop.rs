use std::marker::PhantomData;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use crossbeam_channel::at;

use crate::Result;
use crate::channel::Shutdown;
use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::log_err_internal;
use crate::runtime::scheduler::ScheduleDecision;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct EventLoop<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S> {
    user_req_rx: Receiver<UserReq>,
    batch_dev_req_tx: Sender<BatchDeviceReq>,
    batch_dev_resp_rx: Receiver<BatchDeviceResp>,

    scheduler: S,
    queue_capacity: usize,

    shutdown: Shutdown,

    phantom_data_device_req: PhantomData<DeviceReq>,
    phantom_data_device_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
    EventLoop<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    S: Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    pub fn new(
        user_req_rx: Receiver<UserReq>,
        batch_dev_req_tx: Sender<BatchDeviceReq>,
        batch_dev_resp_rx: Receiver<BatchDeviceResp>,
        scheduler: S,
        queue_capacity: usize,
        shutdown: Shutdown,
    ) -> Self {
        Self {
            user_req_rx,
            batch_dev_req_tx,
            batch_dev_resp_rx,

            scheduler,
            queue_capacity,

            shutdown,

            phantom_data_device_req: PhantomData,
            phantom_data_device_resp: PhantomData,
        }
    }

    pub fn event_loop(mut self) -> Result<()> {
        let span = tracing::info_span!("event loop");
        let _enter = span.enter();
        tracing::info!("started");

        let mut timer: Option<(Instant, Receiver<Instant>)> = None;
        let shutdown_rx = self.shutdown.sync_rx().clone();
        'event_loop: while !self.shutdown.is_shutdown() {
            let decision = self.scheduler.decision();
            match decision {
                ScheduleDecision::Noop => {
                    timer = None;
                },
                ScheduleDecision::Timeout { instant } => {
                    if timer.as_ref().is_none_or(|(deadline, _)| *deadline != instant) {
                        timer = Some((instant, at(instant)));
                    }
                },
                ScheduleDecision::Flush => {
                    timer = None;
                    if do_flush(&mut self.scheduler, &self.batch_dev_req_tx).is_err() {
                        break 'event_loop;
                    }
                },
            }

            let mut select = Select::new();
            let op_shutdown = select.recv(&shutdown_rx);
            let op_recv_batch_dev_resp = select.recv(&self.batch_dev_resp_rx);
            let op_recv_req = if self.scheduler.queue_size() < self.queue_capacity {
                Some(select.recv(&self.user_req_rx))
            } else {
                None
            };
            // TODO: add swap-task completion only with the full async request
            // lifecycle, including reservation waits and KV/state movement.
            let op_timeout = timer.as_ref().map(|(_, timeout)| select.recv(timeout));

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
                        },
                        Err(_) => {
                            tracing::debug!("batch device response channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ if Some(op_index) == op_recv_req => {
                    let user_req = op.recv(&self.user_req_rx);
                    match user_req {
                        Ok(user_req) => {
                            self.scheduler.enqueue(user_req);
                        },
                        Err(_) => {
                            tracing::debug!("user request channel closed, stopping");
                            break 'event_loop;
                        },
                    }
                },
                _ if Some(op_index) == op_timeout => {
                    let timeout = timer
                        .as_ref()
                        .map(|(_, timeout)| timeout)
                        .expect("event loop timeout select arm requires an active timer");
                    let _ = op.recv(timeout);
                    timer = None;
                    if do_flush(&mut self.scheduler, &self.batch_dev_req_tx).is_err() {
                        break 'event_loop;
                    }
                },
                _ => unreachable!(),
            }
        }

        self.shutdown.shutdown();
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
