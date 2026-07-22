use std::collections::VecDeque;
use std::marker::PhantomData;
use std::time::Duration;
use std::time::Instant;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::scheduler::Batcher;
use crate::runtime::scheduler::ComputeSlot;
use crate::runtime::scheduler::ScheduleDecision;
use crate::runtime::scheduler::ScheduleQueue;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct FIFOScheduler<UserReq, DeviceReq, DeviceResp, B> {
    schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    batcher: B,
    max_req_budget: usize,
    max_token_budget: usize,
    max_token_per_req: usize,
    max_wait_duration: Duration,

    next_compute_slot_seq: RawComputeSlotSeq,
    free_compute_slot: VecDeque<ComputeSlot>,
    used_compute_slot: VecDeque<ComputeSlot>,
    req_budget: usize,
    token_budget: usize,
    flush_instant: Option<Instant>,

    phantom_data_user_req: PhantomData<UserReq>,
    phantom_data_dev_req: PhantomData<DeviceReq>,
    phantom_data_dev_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp, B> FIFOScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(
        schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        batcher: B,
        max_req_budget: usize,
        max_token_budget: usize,
        max_token_per_req: usize,
        max_wait_duration: Duration,
        num_compute_slot: usize,
    ) -> Self {
        Self {
            schedule_queue,
            batcher,
            max_req_budget,
            max_token_budget,
            max_token_per_req,
            max_wait_duration,

            next_compute_slot_seq: 1,
            free_compute_slot: (0..num_compute_slot).map(ComputeSlot::new).collect(),
            used_compute_slot: VecDeque::with_capacity(num_compute_slot),
            req_budget: max_req_budget,
            token_budget: max_token_budget,
            flush_instant: None,

            phantom_data_user_req: PhantomData,
            phantom_data_dev_req: PhantomData,
            phantom_data_dev_resp: PhantomData,
        }
    }

    fn is_full(&self) -> bool {
        self.req_budget == 0 || self.token_budget == 0
    }

    fn consume(&mut self, req_estimate: usize, token_estimate: usize) {
        debug_assert!(0 < req_estimate, "fifo scheduler request estimate must be positive");
        debug_assert!(0 < token_estimate, "fifo scheduler token estimate must be positive");

        self.req_budget = self.req_budget.saturating_sub(req_estimate);
        self.token_budget = self.token_budget.saturating_sub(token_estimate);
        if self.flush_instant.is_none() {
            self.flush_instant = Some(Instant::now() + self.max_wait_duration)
        }
    }

    fn restimate(&mut self) {
        self.reset();

        let req_estimate = self.schedule_queue.request_estimate();
        if req_estimate == 0 {
            return;
        }
        let token_estimate = self.schedule_queue.token_estimate(self.max_token_per_req);
        if token_estimate == 0 {
            return;
        }
        if self.free_compute_slot.is_empty() {
            self.consume(self.max_req_budget, self.max_token_budget);
        } else {
            self.consume(req_estimate, token_estimate);
        }
    }

    fn reset(&mut self) {
        self.req_budget = self.max_req_budget;
        self.token_budget = self.max_token_budget;
        self.flush_instant = None;
    }
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, B>
    Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>
    for FIFOScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    B: Batcher<UserReq, DeviceReq, DeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq) {
        let req_estimate = 1;
        let token_estimate = user_req.token_estimate(self.max_token_per_req);
        self.schedule_queue.enqueue(user_req);
        self.consume(req_estimate, token_estimate);
    }

    fn decision(&mut self) -> ScheduleDecision {
        if self.free_compute_slot.is_empty() {
            return ScheduleDecision::Noop;
        }

        if self.is_full() {
            return ScheduleDecision::Flush;
        }

        if let Some(flush_instant) = self.flush_instant.as_ref() {
            ScheduleDecision::Timeout {
                instant: *flush_instant,
            }
        } else {
            ScheduleDecision::Noop
        }
    }

    fn prepare(&mut self) -> BatchDeviceReq {
        let mut compute_slot = self
            .free_compute_slot
            .pop_front()
            .expect("fifo scheduler prepare requires a free compute slot");

        let compute_slot_seq = self.next_compute_slot_seq;
        compute_slot.prepare(compute_slot_seq);
        self.next_compute_slot_seq += 1;
        self.used_compute_slot.push_back(compute_slot);

        let dev_reqs = self.batcher.prepare(
            self.max_req_budget,
            self.max_token_budget,
            &mut self.schedule_queue,
            compute_slot_seq,
        );
        let batch_dev_req = BatchDeviceReq::from_parts(compute_slot_seq, dev_reqs);
        self.restimate();
        batch_dev_req
    }

    fn cancel(&mut self, batch_dev_req: BatchDeviceReq) {
        let mut compute_slot = self
            .used_compute_slot
            .pop_back()
            .expect("fifo scheduler cancellation requires a matching compute slot");
        let (compute_slot_seq, dev_reqs) = batch_dev_req.into_inner();
        debug_assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "fifo scheduler cancellation compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slot.push_front(compute_slot);

        self.batcher.cancel(&mut self.schedule_queue, dev_reqs);
        self.restimate();
    }

    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) {
        let mut compute_slot = self
            .used_compute_slot
            .pop_front()
            .expect("fifo scheduler commit requires a matching compute slot");
        let (compute_slot_seq, dev_resps) = batch_dev_resp.into_inner();
        assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "fifo scheduler commit compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slot.push_back(compute_slot);

        self.batcher.commit(&mut self.schedule_queue, dev_resps);
        // TODO add statistics about req / resp?
        self.restimate();
    }

    fn last_compute_slot_seq(&self) -> RawComputeSlotSeq {
        self.next_compute_slot_seq - 1
    }

    fn next_compute_slot_seq(&self) -> Option<RawComputeSlotSeq> {
        if !self.free_compute_slot.is_empty() {
            Some(self.next_compute_slot_seq)
        } else {
            None
        }
    }

    delegate::delegate! {
        to self.schedule_queue {
            fn run_queue_size(&self) -> usize;
            fn new_queue_size(&self) -> usize;
            fn swap_in_queue_size(&self) -> usize;
            fn swap_out_queue_size(&self) -> usize;
        }
    }

    fn queue_size(&self) -> usize {
        self.schedule_queue.run_queue_size()
            + self.schedule_queue.new_queue_size()
            + self.schedule_queue.swap_in_queue_size()
            + self.schedule_queue.swap_out_queue_size()
    }
}
