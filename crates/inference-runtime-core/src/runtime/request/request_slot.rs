use std::sync::Arc;

use crossbeam_channel::Receiver;

use crate::channel::DedupNotifier;
use crate::memory::U32IDAllocator;
use crate::runtime::RawRequestSlot;

#[derive(Debug)]
pub struct RequestSlot {
    allocator: Arc<U32IDAllocator>,
    reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    request_slot: RawRequestSlot,
}

impl RequestSlot {
    pub fn new(
        allocator: Arc<U32IDAllocator>,
        reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
        request_slot: RawRequestSlot,
    ) -> Self {
        Self {
            allocator,
            reset_notifier,
            request_slot,
        }
    }

    pub fn raw(&self) -> RawRequestSlot {
        self.request_slot
    }
}

impl Drop for RequestSlot {
    fn drop(&mut self) {
        self.reset_notifier.send_one(self.request_slot);
        self.allocator.free_one(self.request_slot);
    }
}

pub enum RequestSlotAllocationResult {
    Ok { request_slot: RequestSlot },
    ResourceLimitExceeded,
}

#[derive(Clone, Debug)]
pub struct RequestSlotAllocator {
    allocator: Arc<U32IDAllocator>,
    reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
}

impl RequestSlotAllocator {
    pub fn new(max_requests: u64) -> (Self, Receiver<()>) {
        let (reset_notifier, reset_rx) = DedupNotifier::new();
        (
            Self {
                allocator: Arc::new(U32IDAllocator::new(max_requests)),
                reset_notifier,
            },
            reset_rx,
        )
    }

    pub fn reset_notifier(&self) -> Arc<DedupNotifier<RawRequestSlot>> {
        self.reset_notifier.clone()
    }

    pub fn allocate(&self) -> RequestSlotAllocationResult {
        match self.allocator.alloc_one() {
            Ok(request_slot) => {
                RequestSlotAllocationResult::Ok {
                    request_slot: RequestSlot::new(self.allocator.clone(), self.reset_notifier.clone(), request_slot),
                }
            },
            Err(_) => RequestSlotAllocationResult::ResourceLimitExceeded,
        }
    }
}
