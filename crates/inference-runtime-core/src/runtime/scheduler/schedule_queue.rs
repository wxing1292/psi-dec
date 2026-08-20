use std::collections::HashMap;
use std::collections::VecDeque;
use std::collections::hash_map::Entry as HashEntry;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use async_channel::Sender;
use async_channel::TrySendError;
use futures_lite::future::Boxed;
use futures_util::Stream;
use futures_util::stream::FuturesUnordered;
use futures_util::task::noop_waker_ref;
use map_macro::hash_map;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawRequestID;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::scheduler::dedup_vec_deque::DedupVecDeque;
use crate::runtime::tasks::AwaitReservation;
use crate::runtime::tasks::SwapOutTask;

pub struct ScheduleQueue<UserReq, DeviceReq, DeviceResp> {
    id_requests: HashMap<RawRequestID, UserReq>,
    run_queue: DedupVecDeque<RawRequestID>,

    new_queue: VecDeque<UserReq>,
    waiting_reqs: FuturesUnordered<AwaitReservation<UserReq, DeviceReq, DeviceResp>>,
    swap_out_task_tx: Sender<SwapOutTask<UserReq, DeviceReq, DeviceResp>>,

    phantom_data_dev_req: PhantomData<DeviceReq>,
    phantom_data_dev_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp> ScheduleQueue<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(swap_out_task_tx: Sender<SwapOutTask<UserReq, DeviceReq, DeviceResp>>) -> Self {
        Self {
            id_requests: hash_map! {},

            new_queue: VecDeque::new(),
            run_queue: DedupVecDeque::new(),
            waiting_reqs: FuturesUnordered::new(),
            swap_out_task_tx,

            phantom_data_dev_req: PhantomData,
            phantom_data_dev_resp: PhantomData,
        }
    }

    pub fn peek_front(&mut self) -> Option<&mut UserReq> {
        while let Some(req_id) = self.run_queue.front() {
            if self.id_requests.contains_key(req_id) {
                return self.id_requests.get_mut(req_id);
            } else {
                self.run_queue.pop_front();
            }
        }

        let user_req = self.new_queue.pop_front()?;
        let req_id = user_req.id();
        self.run_queue.push_back(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => Some(entry.insert(user_req)),
        }
    }

    pub fn pop_front(&mut self) -> Option<UserReq> {
        self.peek_front()?;
        let Some(req_id) = self.run_queue.pop_front() else {
            unreachable!()
        };
        self.id_requests.remove(&req_id)
    }

    pub fn push_front(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        self.run_queue.push_front(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn pop_back(&mut self) -> Option<UserReq> {
        while let Some(req_id) = self.run_queue.pop_back() {
            match self.id_requests.remove(&req_id) {
                Some(user_req) => return Some(user_req),
                None => continue,
            };
        }
        None
    }

    pub fn push_back(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        self.run_queue.push_back(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn enqueue(&mut self, user_req: UserReq) {
        self.new_queue.push_back(user_req);
    }

    pub fn push_waiting_reqs(&mut self, user_req: UserReq, wait: Boxed<()>) {
        self.waiting_reqs.push(AwaitReservation::new(user_req, wait));
    }

    pub fn pop_ready_reqs(&mut self) -> Option<UserReq> {
        let mut cx = Context::from_waker(noop_waker_ref());
        match Pin::new(&mut self.waiting_reqs).poll_next(&mut cx) {
            Poll::Ready(Some(user_req)) => Some(user_req),
            Poll::Ready(None) | Poll::Pending => None,
        }
    }

    pub fn push_swap_out(
        &self,
        swap_out_task: SwapOutTask<UserReq, DeviceReq, DeviceResp>,
    ) -> Result<(), TrySendError<SwapOutTask<UserReq, DeviceReq, DeviceResp>>> {
        self.swap_out_task_tx.try_send(swap_out_task)
    }

    pub fn insert(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn remove(&mut self, req_id: &RawRequestID) -> Option<UserReq> {
        self.run_queue.remove(req_id);
        self.id_requests.remove(req_id)
    }

    pub fn get_ref(&self, req_id: &RawRequestID) -> Option<&UserReq> {
        self.id_requests.get(req_id)
    }

    pub fn request_estimate(&self) -> usize {
        self.run_queue.len() + self.new_queue.len()
    }

    pub fn token_estimate(&self, max_token_per_req: usize) -> usize {
        self.run_queue.iter().fold(0, |sum, req_id| {
            sum + self
                .id_requests
                .get(req_id)
                .map(|user_req| user_req.token_estimate().token_consumption(max_token_per_req))
                .unwrap_or(0)
        }) + self.new_queue.iter().fold(0, |sum, user_req| {
            sum + user_req.token_estimate().token_consumption(max_token_per_req)
        })
    }

    pub fn run_queue_size(&self) -> usize {
        self.run_queue.len()
    }

    pub fn new_queue_size(&self) -> usize {
        self.new_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use std::future;

    use async_channel::bounded;
    use event_listener::Event;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockUserRequest;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestScheduleQueue = ScheduleQueue<TestUserReq, MockDevReq, MockDevResp>;

    #[test]
    fn test_pop_ready_reqs_polls_notified_wait() {
        let req_id = 7;
        let event = Event::new();
        let wait = event.listen();

        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req.expect_is_terminal().once().return_const(false);

        let (swap_out_task_tx, _swap_out_task_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(swap_out_task_tx);
        schedule_queue.push_waiting_reqs(user_req, Box::pin(wait));

        event.notify(usize::MAX);

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(req_id));
    }

    #[test]
    fn test_pop_ready_reqs_returns_later_ready_req() {
        let mut pending_req = TestUserReq::new();
        pending_req.expect_is_terminal().once().return_const(false);

        let ready_req_id = 7;
        let mut ready_req = TestUserReq::new();
        ready_req.expect_id().return_const(ready_req_id);
        ready_req.expect_is_terminal().once().return_const(false);

        let (swap_out_task_tx, _swap_out_task_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(swap_out_task_tx);
        schedule_queue.push_waiting_reqs(pending_req, Box::pin(future::pending()));
        schedule_queue.push_waiting_reqs(ready_req, Box::pin(async {}));

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(ready_req_id));
        assert!(schedule_queue.pop_ready_reqs().is_none());
    }

    #[test]
    fn test_pop_ready_reqs_returns_terminal_req() {
        let req_id: RawRequestID = 7;
        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req.expect_is_terminal().once().return_const(true);

        let (swap_out_task_tx, _swap_out_task_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(swap_out_task_tx);
        schedule_queue.push_waiting_reqs(user_req, Box::pin(async {}));

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(req_id));
    }
}
