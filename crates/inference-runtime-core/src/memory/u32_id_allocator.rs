use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use crossbeam_queue::SegQueue;

use crate::Error;
use crate::Result;

#[derive(Clone, Debug)]
pub struct U32IDAllocator {
    inner: Arc<U32IDAllocatorInner>,
}

impl U32IDAllocator {
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: Arc::new(U32IDAllocatorInner::new(capacity)),
        }
    }

    delegate::delegate! {
        to self.inner {
            pub fn alloc_one(&self) -> Result<u32>;
            pub fn alloc_many(&self, count: usize) -> Result<Vec<u32>>;
            pub fn free_one(&self, page_id: u32);
            pub fn free_many<I>(&self, page_ids: I)
            where
                I: IntoIterator<Item = u32>;
            pub fn used(&self) -> usize;
            pub fn free(&self) -> usize;
            pub fn capacity(&self) -> usize;
        }
    }
}

#[derive(Debug)]
struct U32IDAllocatorInner {
    counter: AtomicU32,
    capacity: u64,

    free_ids: SegQueue<u32>,
}

impl U32IDAllocatorInner {
    fn new(capacity: u64) -> Self {
        assert!(capacity <= u32::MAX as u64);
        Self {
            counter: AtomicU32::new(0),
            capacity,
            free_ids: SegQueue::new(),
        }
    }

    fn alloc_one(&self) -> Result<u32> {
        loop {
            if let Some(id) = self.free_ids.pop() {
                return Ok(id);
            }

            let id = self.counter.load(Ordering::Relaxed);
            if id as u64 >= self.capacity {
                return Err(Error::resource_exhausted("not enough ID"));
            }
            if self
                .counter
                .compare_exchange_weak(id, id + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(id);
            }
        }
    }

    fn alloc_many(&self, count: usize) -> Result<Vec<u32>> {
        assert!(count <= u32::MAX as usize);

        let mut ids = Vec::with_capacity(count);

        while ids.len() < count
            && let Some(id) = self.free_ids.pop()
        {
            ids.push(id);
        }
        if ids.len() >= count {
            return Ok(ids);
        }
        let count = (count - ids.len()) as u32;
        match self.counter.try_update(Ordering::SeqCst, Ordering::SeqCst, |start_id| {
            let end_id = start_id as u64 + count as u64;
            if end_id <= self.capacity {
                Some(end_id as u32)
            } else {
                None
            }
        }) {
            Ok(start_id) => {
                (start_id..start_id + count).for_each(|id| {
                    ids.push(id);
                });
                Ok(ids)
            },
            Err(_) => {
                ids.into_iter().for_each(|id| self.free_ids.push(id));
                Err(Error::resource_exhausted("not enough ID"))
            },
        }
    }

    fn free_one(&self, page_id: u32) {
        self.free_ids.push(page_id);
    }

    fn free_many<I>(&self, page_ids: I)
    where
        I: IntoIterator<Item = u32>,
    {
        for id in page_ids {
            self.free_ids.push(id);
        }
    }

    fn used(&self) -> usize {
        self.capacity() - self.free()
    }

    fn free(&self) -> usize {
        self.free_ids.len() + self.capacity.saturating_sub(self.counter.load(Ordering::SeqCst) as u64) as usize
    }

    fn capacity(&self) -> usize {
        self.capacity as usize
    }
}
