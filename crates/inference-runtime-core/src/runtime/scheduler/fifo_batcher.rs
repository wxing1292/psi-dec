use std::cmp::min;
use std::marker::PhantomData;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::scheduler::Batcher;
use crate::runtime::scheduler::CancelResult;
use crate::runtime::scheduler::CommitResult;
use crate::runtime::scheduler::PrepareResult;
use crate::runtime::scheduler::ScheduleQueue;
use crate::runtime::scheduler::SwapOutTask;
use crate::runtime::scheduler::UserRequest;

pub struct FIFOBatcher<UserReq, DeviceReq, DeviceResp> {
    max_token_per_req: usize,
    running_reqs: Vec<UserReq>,

    phantom_data_dev_req: PhantomData<DeviceReq>,
    phantom_data_dev_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp> FIFOBatcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(max_token_per_req: usize) -> Self {
        Self {
            max_token_per_req,
            running_reqs: Vec::with_capacity(1024),
            phantom_data_dev_req: PhantomData,
            phantom_data_dev_resp: PhantomData,
        }
    }
}

impl<UserReq, DeviceReq, DeviceResp> Batcher<UserReq, DeviceReq, DeviceResp>
    for FIFOBatcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn prepare(
        &mut self,
        mut req_budget: usize,
        mut token_budget: usize,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        seq: RawComputeSlotSeq,
    ) -> Vec<DeviceReq> {
        debug_assert!(0 < req_budget, "fifo batcher requires a positive request budget");
        debug_assert!(0 < token_budget, "fifo batcher requires a positive token budget");
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");

        let mut dev_reqs = Vec::with_capacity(req_budget);
        'prepare_loop: while let Some(mut user_req) = schedule_queue.pop_front() {
            let req_id = user_req.id();
            if req_budget == 0 || token_budget == 0 {
                self.running_reqs.push(user_req);
                break 'prepare_loop;
            }
            let token_estimate = user_req.token_estimate(min(self.max_token_per_req, token_budget));
            if token_estimate == 0 {
                self.running_reqs.push(user_req);
                continue 'prepare_loop;
            }
            match user_req.prepare(token_estimate) {
                PrepareResult::ResourceLimitExceeded => {
                    if let Some(preempted_req) = schedule_queue.pop_back() {
                        // TODO uninit and put to async queue
                        tracing::warn!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.preempted",
                            request_id = preempted_req.id(),
                            "request aborted due to preemption"
                        );
                        drop(preempted_req);
                        schedule_queue.push_front(user_req);
                        continue 'prepare_loop;
                    } else {
                        // TODO uninit and put to async queue
                        tracing::warn!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.resource_limit",
                            request_id = user_req.id(),
                            "request aborted due to insufficient memory"
                        );
                        drop(user_req);
                        break 'prepare_loop;
                    }
                },
                PrepareResult::Await { wait } => {
                    let swap_out_task = SwapOutTask::AwaitReservation { user_req, wait };
                    if let Err(err) = schedule_queue.push_swap_out(swap_out_task) {
                        tracing::warn!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.swap_out_queue_full",
                            request_id = req_id,
                            error = %err,
                            "request dropped due to async swap-out task queue pressure"
                        );
                    }
                    continue 'prepare_loop;
                },
                PrepareResult::Pending => {
                    schedule_queue.insert(user_req);
                },
                PrepareResult::Continue(dev_req) => {
                    let req_cost = dev_req.req_cost();
                    let token_cost = dev_req.token_cost();
                    debug_assert!(
                        req_cost <= req_budget,
                        "prepared request cost exceeds fifo batch request budget"
                    );
                    debug_assert!(
                        token_cost <= token_budget,
                        "prepared request token cost exceeds fifo batch token budget"
                    );
                    req_budget -= req_cost;
                    token_budget -= token_cost;
                    self.running_reqs.push(user_req);
                    dev_reqs.push(dev_req);
                },
                PrepareResult::Terminal => { /* noop */ },
            }
        }
        for user_req in self.running_reqs.drain(..).rev() {
            schedule_queue.push_front(user_req);
        }

        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be drained");
        dev_reqs
    }

    fn cancel(&mut self, schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>, dev_reqs: Vec<DeviceReq>) {
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");
        for dev_req in dev_reqs {
            let req_id = dev_req.id();
            let mut user_req = schedule_queue
                .remove(&req_id)
                .expect("fifo batch cancellation requires a matching request");
            match user_req.cancel(dev_req) {
                CancelResult::Continue => schedule_queue.push_front(user_req),
                CancelResult::Terminal => { /* noop */ },
            }
        }
    }

    fn commit(
        &mut self,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        dev_resps: Vec<DeviceResp>,
    ) {
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");
        for dev_resp in dev_resps {
            let req_id = dev_resp.id();
            let mut user_req = schedule_queue
                .remove(&req_id)
                .expect("fifo batch commit requires a matching request");
            match user_req.commit(dev_resp) {
                CommitResult::Continue => schedule_queue.push_front(user_req),
                CommitResult::Terminal => { /* noop */ },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use async_channel::bounded as async_bounded;
    use mockall::Sequence;
    use mockall::predicate::eq;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::RawRequestID;
    use crate::runtime::scheduler::MockUserRequest;
    use crate::runtime::scheduler::SwapInTask;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestScheduleQueue = ScheduleQueue<TestUserReq, MockDevReq, MockDevResp>;

    #[test]
    fn test_prepare_cancel_schedules_all() {
        let req_budget = 3;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [
            new_test_scheduled_req(1, 8, 1, 8),
            new_test_scheduled_req(2, 8, 1, 8),
            new_test_scheduled_req(3, 8, 1, 8),
        ];

        test_prepare_cancel(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[]);
    }

    #[test]
    fn test_prepare_cancel_schedules_half_token_budget() {
        let req_budget = 3;
        let token_budget = 12;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 4, 1, 4)];

        test_prepare_cancel(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[3]);
    }

    #[test]
    fn test_prepare_cancel_schedules_half_req_budget() {
        let req_budget = 2;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 8, 1, 8)];

        test_prepare_cancel(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[3]);
    }

    #[test]
    fn test_prepare_commit_schedules_all() {
        let req_budget = 3;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [
            new_test_scheduled_req(1, 8, 1, 8),
            new_test_scheduled_req(2, 8, 1, 8),
            new_test_scheduled_req(3, 8, 1, 8),
        ];

        test_prepare_commit(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[]);
    }

    #[test]
    fn test_prepare_commit_schedules_half_token_budget() {
        let req_budget = 3;
        let token_budget = 12;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 4, 1, 4)];

        test_prepare_commit(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[3]);
    }

    #[test]
    fn test_prepare_commit_schedules_half_req_budget() {
        let req_budget = 2;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 8, 1, 8)];

        test_prepare_commit(req_budget, token_budget, max_token_per_req, &scheduled_reqs, &[3]);
    }

    #[test]
    fn test_prepare_await_success() {
        let req_budget = 1;
        let token_budget = 8;
        let max_token_per_req = 8;
        let req_id = 1;

        let mut seq = Sequence::new();
        let mut user_req = mock_user_req(req_id);
        user_req
            .expect_token_estimate()
            .with(eq(token_budget))
            .in_sequence(&mut seq)
            .return_const(token_budget);
        user_req
            .expect_prepare()
            .once()
            .with(eq(token_budget))
            .in_sequence(&mut seq)
            .return_once(|_| {
                PrepareResult::Await {
                    wait: Box::pin(async {}),
                }
            });

        let mut schedule_queue = TestScheduleQueue::new();
        let swap_out_task_rx = schedule_queue.swap_out_task_receiver();
        schedule_queue.push_back(user_req);

        let mut batcher = FIFOBatcher::new(max_token_per_req);
        let dev_reqs = batcher.prepare(req_budget, token_budget, &mut schedule_queue, 1);
        assert!(dev_reqs.is_empty());

        let swap_out_task = swap_out_task_rx
            .try_recv()
            .expect("awaiting request should enqueue swap-out task");
        match swap_out_task {
            SwapOutTask::AwaitReservation { user_req, .. } => assert_eq!(req_id, user_req.id()),
        }
    }

    #[test]
    fn test_prepare_await_fail() {
        let req_budget = 1;
        let token_budget = 8;
        let max_token_per_req = 8;
        let pending_req_id = 1;
        let dropped_req_id = 2;

        let (swap_out_task_tx, swap_out_task_rx) = async_bounded(1);
        let (swap_in_task_tx, swap_in_task_rx) = async_bounded::<SwapInTask<TestUserReq>>(1);
        let pending_user_req = mock_user_req(pending_req_id);
        swap_out_task_tx
            .try_send(SwapOutTask::AwaitReservation {
                user_req: pending_user_req,
                wait: Box::pin(async {}),
            })
            .expect("test setup should fill swap-out queue");

        let mut seq = Sequence::new();
        let mut user_req = mock_user_req(dropped_req_id);
        user_req
            .expect_token_estimate()
            .with(eq(token_budget))
            .in_sequence(&mut seq)
            .return_const(token_budget);
        user_req
            .expect_prepare()
            .once()
            .with(eq(token_budget))
            .in_sequence(&mut seq)
            .return_once(|_| {
                PrepareResult::Await {
                    wait: Box::pin(async {}),
                }
            });

        let mut schedule_queue = TestScheduleQueue::new_with_swap_channels(
            swap_out_task_tx,
            swap_out_task_rx.clone(),
            swap_in_task_tx,
            swap_in_task_rx,
        );
        schedule_queue.push_back(user_req);

        let mut batcher = FIFOBatcher::new(max_token_per_req);
        let dev_reqs = batcher.prepare(req_budget, token_budget, &mut schedule_queue, 1);
        assert!(dev_reqs.is_empty());
        assert_eq!(0, schedule_queue.run_queue_size());
        assert_eq!(0, schedule_queue.new_queue_size());
        assert_eq!(0, schedule_queue.swap_in_queue_size());

        let swap_out_task = swap_out_task_rx
            .try_recv()
            .expect("full swap-out queue should retain the pending task");
        match swap_out_task {
            SwapOutTask::AwaitReservation { user_req, .. } => assert_eq!(pending_req_id, user_req.id()),
        }
        assert!(swap_out_task_rx.try_recv().is_err());
    }

    #[test]
    fn test_prepare_resource_limit_preempts_tail_and_retries_current() {
        let req_budget = 1;
        let token_budget = 8;
        let max_token_per_req = 8;
        let current_req_id = 1;
        let preempted_req_id = 2;

        let mut current_req = mock_user_req(current_req_id);
        current_req
            .expect_token_estimate()
            .times(2)
            .with(eq(token_budget))
            .return_const(token_budget);
        let mut prepare_count = 0;
        current_req
            .expect_prepare()
            .times(2)
            .with(eq(token_budget))
            .returning(move |_| {
                prepare_count += 1;
                if prepare_count == 1 {
                    PrepareResult::ResourceLimitExceeded
                } else {
                    PrepareResult::Terminal
                }
            });
        let preempted_req = mock_user_req(preempted_req_id);

        let mut schedule_queue = TestScheduleQueue::new();
        schedule_queue.push_back(current_req);
        schedule_queue.push_back(preempted_req);

        let mut batcher = FIFOBatcher::new(max_token_per_req);
        let dev_reqs = batcher.prepare(req_budget, token_budget, &mut schedule_queue, 1);

        assert!(dev_reqs.is_empty());
        assert_eq!(schedule_queue.run_queue_size(), 0);
    }

    fn test_prepare_cancel(
        req_budget: usize,
        token_budget: usize,
        max_token_per_req: usize,
        scheduled_reqs: &[TestScheduledReq],
        unscheduled_req_ids: &[RawRequestID],
    ) {
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + unscheduled_req_ids.len());

        for scheduled_req in scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prepare(
                &mut seq,
                &mut user_req,
                scheduled_req.prepare_token_budget,
                scheduled_req.req_id,
                scheduled_req.req_cost,
                scheduled_req.token_cost,
            );
            user_reqs.push(user_req);
        }
        for &req_id in unscheduled_req_ids {
            user_reqs.push(mock_user_req(req_id));
        }
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_cancel(&mut seq, user_req);
        }

        let mut schedule_queue = TestScheduleQueue::new();
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new(max_token_per_req);
        let dev_reqs = batcher.prepare(req_budget, token_budget, &mut schedule_queue, 1);
        assert_eq!(scheduled_reqs.len(), dev_reqs.len());

        batcher.cancel(&mut schedule_queue, dev_reqs);
    }

    fn test_prepare_commit(
        req_budget: usize,
        token_budget: usize,
        max_token_per_req: usize,
        scheduled_reqs: &[TestScheduledReq],
        unscheduled_req_ids: &[RawRequestID],
    ) {
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + unscheduled_req_ids.len());

        for scheduled_req in scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prepare(
                &mut seq,
                &mut user_req,
                scheduled_req.prepare_token_budget,
                scheduled_req.req_id,
                scheduled_req.req_cost,
                scheduled_req.token_cost,
            );
            user_reqs.push(user_req);
        }
        for &req_id in unscheduled_req_ids {
            user_reqs.push(mock_user_req(req_id));
        }
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_commit(&mut seq, user_req);
        }

        let mut schedule_queue = TestScheduleQueue::new();
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new(max_token_per_req);
        let dev_reqs = batcher.prepare(req_budget, token_budget, &mut schedule_queue, 1);
        assert_eq!(scheduled_reqs.len(), dev_reqs.len());

        let dev_resps = scheduled_reqs
            .iter()
            .map(|scheduled_req| mock_dev_resp(scheduled_req.req_id))
            .collect();
        batcher.commit(&mut schedule_queue, dev_resps);
    }

    #[derive(Clone, Copy)]
    struct TestScheduledReq {
        req_id: RawRequestID,
        prepare_token_budget: usize,
        req_cost: usize,
        token_cost: usize,
    }

    fn new_test_scheduled_req(
        req_id: RawRequestID,
        prepare_token_budget: usize,
        req_cost: usize,
        token_cost: usize,
    ) -> TestScheduledReq {
        TestScheduledReq {
            req_id,
            prepare_token_budget,
            req_cost,
            token_cost,
        }
    }

    fn mock_user_req(req_id: RawRequestID) -> TestUserReq {
        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req
    }

    fn expect_prepare(
        seq: &mut Sequence,
        user_req: &mut TestUserReq,
        token_budget: usize,
        req_id: RawRequestID,
        req_cost: usize,
        token_cost: usize,
    ) {
        user_req
            .expect_token_estimate()
            .with(eq(token_budget))
            .in_sequence(seq)
            .return_const(token_budget);
        user_req
            .expect_prepare()
            .once()
            .with(eq(token_budget))
            .in_sequence(seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(req_id);
                dev_req.expect_req_cost().once().return_const(req_cost);
                dev_req.expect_token_cost().once().return_const(token_cost);
                PrepareResult::Continue(dev_req)
            });
    }

    fn expect_cancel(seq: &mut Sequence, user_req: &mut TestUserReq) {
        user_req
            .expect_cancel()
            .once()
            .in_sequence(seq)
            .return_once(|_| CancelResult::Continue);
    }

    fn expect_commit(seq: &mut Sequence, user_req: &mut TestUserReq) {
        user_req
            .expect_commit()
            .once()
            .in_sequence(seq)
            .return_once(|_| CommitResult::Continue);
    }

    fn mock_dev_resp(req_id: RawRequestID) -> MockDevResp {
        let mut dev_resp = MockDevResp::new();
        dev_resp.expect_id().return_const(req_id);
        dev_resp
    }
}
