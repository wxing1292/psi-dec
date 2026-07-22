use std::time::Instant;

use futures_lite::future::Boxed;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::RawRequestID;

mod async_task;
pub use async_task::AsyncTask;
pub use async_task::AsyncTaskPool;
pub use async_task::SwapInTask;
pub use async_task::SwapOutTask;

mod compute_slot;
pub use compute_slot::ComputeSlot;

mod event_loop;
pub use event_loop::EventLoop;

mod fifo_scheduler;
pub use fifo_scheduler::FIFOScheduler;

mod instrumented_scheduler;
pub use instrumented_scheduler::InstrumentedScheduler;

mod fifo_batcher;
pub use fifo_batcher::FIFOBatcher;

mod schedule_queue;
pub use schedule_queue::ScheduleQueue;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleDecision {
    Noop,
    Timeout { instant: Instant },
    Flush,
}

#[mockall::automock]
pub trait Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq);
    fn decision(&mut self) -> ScheduleDecision;

    fn prepare(&mut self) -> BatchDeviceReq;
    fn cancel(&mut self, batch_dev_req: BatchDeviceReq);
    fn commit(&mut self, batch_dev_resp: BatchDeviceResp);

    fn last_compute_slot_seq(&self) -> RawComputeSlotSeq;
    fn next_compute_slot_seq(&self) -> Option<RawComputeSlotSeq>;

    fn run_queue_size(&self) -> usize;
    fn new_queue_size(&self) -> usize;
    fn swap_in_queue_size(&self) -> usize;
    fn swap_out_queue_size(&self) -> usize;
    fn queue_size(&self) -> usize;
}

#[mockall::automock]
pub trait Batcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn prepare(
        &mut self,
        req_budget: usize,
        token_budget: usize,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        seq: RawComputeSlotSeq,
    ) -> Vec<DeviceReq>;
    fn cancel(&mut self, schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>, dev_reqs: Vec<DeviceReq>);
    fn commit(
        &mut self,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        dev_resps: Vec<DeviceResp>,
    );
}

pub enum PrepareResult<DeviceReq> {
    ResourceLimitExceeded,
    Await { wait: Boxed<()> },
    Pending,
    Continue(DeviceReq),
    Terminal,
}

pub enum CommitResult {
    Continue,
    Terminal,
}

pub enum CancelResult {
    Continue,
    Terminal,
}

#[mockall::automock]
pub trait UserRequest<DeviceReq, DeviceResp>: Send + 'static
where
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn id(&self) -> RawRequestID;

    fn request_estimate(&self) -> usize;
    fn token_estimate(&self, token_budget: usize) -> usize;

    fn prepare(&mut self, token_budget: usize) -> PrepareResult<DeviceReq>;
    fn cancel(&mut self, dev_req: DeviceReq) -> CancelResult;
    fn commit(&mut self, dev_resp: DeviceResp) -> CommitResult;
}
