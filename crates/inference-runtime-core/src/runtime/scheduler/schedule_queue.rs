use std::collections::HashMap;
use std::collections::VecDeque;
use std::collections::hash_map::Entry as HashEntry;
use std::marker::PhantomData;

use async_channel::Sender;
use async_channel::TrySendError;
use map_macro::hash_map;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawRequestID;
use crate::runtime::scheduler::SwapOutTask;
use crate::runtime::scheduler::UserRequest;

pub struct ScheduleQueue<UserReq, DeviceReq, DeviceResp> {
    id_requests: HashMap<RawRequestID, UserReq>,
    run_queue: VecDeque<RawRequestID>,

    new_queue: VecDeque<UserReq>,
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
            run_queue: VecDeque::new(),
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
        self.id_requests.remove(req_id)
    }

    pub fn get(&mut self, req_id: &RawRequestID) -> Option<&mut UserReq> {
        self.id_requests.get_mut(req_id)
    }

    pub fn request_estimate(&self) -> usize {
        self.run_queue.len() + self.new_queue.len()
    }

    pub fn token_estimate(&self, max_token_per_req: usize) -> usize {
        self.run_queue.iter().fold(0, |sum, req_id| {
            sum + self
                .id_requests
                .get(req_id)
                .map(|user_req| user_req.token_estimate(max_token_per_req))
                .unwrap_or(0)
        }) + self
            .new_queue
            .iter()
            .fold(0, |sum, user_req| sum + user_req.token_estimate(max_token_per_req))
    }

    pub fn run_queue_size(&self) -> usize {
        self.run_queue.len()
    }

    pub fn new_queue_size(&self) -> usize {
        self.new_queue.len()
    }
}
