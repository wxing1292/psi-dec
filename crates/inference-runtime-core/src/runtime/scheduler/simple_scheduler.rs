use std::collections::VecDeque;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::scheduler::Batcher;
use crate::runtime::scheduler::ComputeSlot;
use crate::runtime::scheduler::ScheduleQueue;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct SimpleScheduler<UserReq, DeviceReq, DeviceResp, B> {
    schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    batcher: B,
    max_req_budget: usize,
    max_token_budget: usize,
    max_token_per_req: usize,

    next_compute_slot_seq: RawComputeSlotSeq,
    free_compute_slots: VecDeque<ComputeSlot>,
    used_compute_slots: VecDeque<ComputeSlot>,
}

impl<UserReq, DeviceReq, DeviceResp, B> SimpleScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    B: Batcher<UserReq, DeviceReq, DeviceResp>,
{
    pub fn new(
        schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        batcher: B,
        max_req_budget: usize,
        max_token_budget: usize,
        max_token_per_req: usize,
        num_compute_slots: usize,
    ) -> Self {
        assert!(
            max_req_budget > 0,
            "simple scheduler requires a positive request budget"
        );
        assert!(
            max_token_budget > 0,
            "simple scheduler requires a positive token budget"
        );
        assert!(
            max_token_per_req > 0,
            "simple scheduler requires a positive per-request token budget"
        );
        assert!(num_compute_slots > 0, "simple scheduler requires compute slots");

        Self {
            schedule_queue,
            batcher,
            max_req_budget,
            max_token_budget,
            max_token_per_req,

            next_compute_slot_seq: 1,
            free_compute_slots: (0..num_compute_slots).map(ComputeSlot::new).collect(),
            used_compute_slots: VecDeque::with_capacity(num_compute_slots),
        }
    }

    pub fn run_queue_size(&self) -> usize {
        self.schedule_queue.run_queue_size()
    }

    pub fn new_queue_size(&self) -> usize {
        self.schedule_queue.new_queue_size()
    }

    pub fn last_compute_slot_seq(&self) -> RawComputeSlotSeq {
        self.next_compute_slot_seq - 1
    }

    pub fn next_compute_slot_seq(&self) -> Option<RawComputeSlotSeq> {
        if self.free_compute_slots.is_empty() {
            None
        } else {
            Some(self.next_compute_slot_seq)
        }
    }
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, B>
    Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>
    for SimpleScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    B: Batcher<UserReq, DeviceReq, DeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq) {
        self.schedule_queue.enqueue(user_req);
    }

    fn swap_in(&mut self, user_req: UserReq) {
        self.schedule_queue.push_back(user_req);
    }

    fn can_flush(&self) -> bool {
        let token_budget = self.max_token_budget.min(self.max_token_per_req);
        !self.free_compute_slots.is_empty() && self.schedule_queue.token_estimate(token_budget) > 0
    }

    fn prepare(&mut self) -> BatchDeviceReq {
        let mut compute_slot = self
            .free_compute_slots
            .pop_front()
            .expect("simple scheduler prepare requires a free compute slot");

        let compute_slot_seq = self.next_compute_slot_seq;
        compute_slot.prepare(compute_slot_seq);
        self.next_compute_slot_seq += 1;
        self.used_compute_slots.push_back(compute_slot);

        let dev_reqs = self.batcher.prepare(
            self.max_req_budget,
            self.max_token_budget,
            self.max_token_per_req,
            &mut self.schedule_queue,
        );
        BatchDeviceReq::from_parts(compute_slot_seq, dev_reqs)
    }

    fn cancel(&mut self, batch_dev_req: BatchDeviceReq) {
        let mut compute_slot = self
            .used_compute_slots
            .pop_back()
            .expect("simple scheduler cancellation requires a matching compute slot");
        let (compute_slot_seq, dev_reqs) = batch_dev_req.into_inner();
        debug_assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "simple scheduler cancellation compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slots.push_front(compute_slot);

        self.batcher.cancel(&mut self.schedule_queue, dev_reqs);
    }

    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) {
        let mut compute_slot = self
            .used_compute_slots
            .pop_front()
            .expect("simple scheduler commit requires a matching compute slot");
        let (compute_slot_seq, dev_resps) = batch_dev_resp.into_inner();
        assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "simple scheduler commit compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slots.push_back(compute_slot);

        self.batcher.commit(&mut self.schedule_queue, dev_resps);
    }
}

#[cfg(test)]
mod tests {
    use async_channel::bounded;

    use super::*;
    use crate::compute::MockBatchDevResp;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockBatcher;
    use crate::runtime::scheduler::MockUserRequest;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestBatcher = MockBatcher<TestUserReq, MockDevReq, MockDevResp>;
    type TestBatchDeviceResp = MockBatchDevResp<MockDevResp>;

    struct TestBatchDevReq {
        seq: RawComputeSlotSeq,
        dev_reqs: Vec<MockDevReq>,
    }

    impl BatchDevReq<MockDevReq> for TestBatchDevReq {
        fn seq(&self) -> RawComputeSlotSeq {
            self.seq
        }

        fn request_cost(&self) -> usize {
            self.dev_reqs.len()
        }

        fn token_cost(&self) -> usize {
            self.dev_reqs.len()
        }

        fn from_parts(seq: RawComputeSlotSeq, dev_reqs: Vec<MockDevReq>) -> Self {
            Self { seq, dev_reqs }
        }

        fn into_inner(self) -> (RawComputeSlotSeq, Vec<MockDevReq>) {
            (self.seq, self.dev_reqs)
        }
    }

    #[test]
    fn test_can_flush_tracks_runnable_work_and_compute_slot() {
        let max_req_budget = 1;
        let max_token_budget = 8;
        let max_token_per_req = 4;

        let mut batcher = TestBatcher::new();
        batcher
            .expect_prepare()
            .once()
            .with(
                mockall::predicate::eq(max_req_budget),
                mockall::predicate::eq(max_token_budget),
                mockall::predicate::eq(max_token_per_req),
                mockall::predicate::always(),
            )
            .return_once(|_, _, _, _| vec![MockDevReq::new()]);
        batcher.expect_cancel().once().return_once(|_, _| {});

        let (swap_out_task_tx, _swap_out_task_rx) = bounded(1);
        let mut scheduler = SimpleScheduler::new(
            ScheduleQueue::new(swap_out_task_tx),
            batcher,
            max_req_budget,
            max_token_budget,
            max_token_per_req,
            1,
        );
        assert_eq!(scheduler.last_compute_slot_seq(), 0);
        assert_eq!(scheduler.next_compute_slot_seq(), Some(1));

        {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            assert!(!scheduler.can_flush());
            let mut user_req = TestUserReq::new();
            user_req
                .expect_token_estimate()
                .with(mockall::predicate::eq(max_token_per_req))
                .times(2)
                .return_const(1usize);
            scheduler.enqueue(user_req);
            assert!(scheduler.can_flush());

            let batch_dev_req = scheduler.prepare();
            assert_eq!(batch_dev_req.seq(), 1);
            assert!(!scheduler.can_flush());

            scheduler.cancel(batch_dev_req);
            assert!(scheduler.can_flush());
        }
        assert_eq!(scheduler.last_compute_slot_seq(), 1);
        assert_eq!(scheduler.next_compute_slot_seq(), Some(2));
    }
}
