use futures_lite::future::Boxed;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawRequestID;

mod compute_slot;
pub use compute_slot::ComputeSlot;

mod dedup_vec_deque;

mod event_loop;
pub use event_loop::EventLoop;

mod simple_scheduler;
pub use simple_scheduler::SimpleScheduler;

mod instrumented_scheduler;
pub use instrumented_scheduler::InstrumentedScheduler;

mod fifo_batcher;
pub use fifo_batcher::FIFOBatcher;

mod schedule_queue;
pub use schedule_queue::ScheduleQueue;

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
    fn swap_in(&mut self, user_req: UserReq);
    fn can_flush(&self) -> bool;

    fn prepare(&mut self) -> BatchDeviceReq;
    fn cancel(&mut self, batch_dev_req: BatchDeviceReq);
    fn commit(&mut self, batch_dev_resp: BatchDeviceResp);
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
        max_token_per_req: usize,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
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
    Continue { dev_req: DeviceReq, phase: PreparePhase },
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparePhase {
    Prefill,
    Decode,
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

    fn store_running(&self) -> bool;
    fn store_swapped(&self) -> bool;
    fn is_terminal(&self) -> bool;

    fn request_estimate(&self) -> usize;
    fn token_estimate(&self, token_budget: usize) -> usize;

    fn prepare(&mut self, token_budget: usize) -> PrepareResult<DeviceReq>;
    fn cancel(&mut self, dev_req: DeviceReq) -> CancelResult;
    fn commit(&mut self, dev_resp: DeviceResp) -> CommitResult;
}
