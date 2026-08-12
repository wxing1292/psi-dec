use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
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
            pub fn allocated_ids_bitmap_iter(&self) -> impl ExactSizeIterator<Item = u64> + '_;
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
    allocated_ids_bitmap: Box<[AtomicU64]>,
}

impl U32IDAllocatorInner {
    fn new(capacity: u64) -> Self {
        assert!(capacity <= u32::MAX as u64);
        let num_bitmap_words = usize::try_from(capacity.div_ceil(u64::BITS.into()))
            .expect("U32 ID allocator bitmap length must fit usize");
        Self {
            counter: AtomicU32::new(0),
            capacity,
            free_ids: SegQueue::new(),
            allocated_ids_bitmap: std::iter::repeat_with(|| AtomicU64::new(0))
                .take(num_bitmap_words)
                .collect(),
        }
    }

    fn allocated_ids_bitmap_iter(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.allocated_ids_bitmap
            .iter()
            .map(|bitmap| bitmap.load(Ordering::Relaxed))
    }

    fn alloc_one(&self) -> Result<u32> {
        loop {
            if let Some(id) = self.free_ids.pop() {
                self.mark_allocated(id);
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
                self.mark_allocated(id);
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
            ids.iter().copied().for_each(|id| self.mark_allocated(id));
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
                ids.iter().copied().for_each(|id| self.mark_allocated(id));
                Ok(ids)
            },
            Err(_) => {
                ids.into_iter().for_each(|id| self.free_ids.push(id));
                Err(Error::resource_exhausted("not enough ID"))
            },
        }
    }

    fn free_one(&self, page_id: u32) {
        self.mark_free(page_id);
        self.free_ids.push(page_id);
    }

    fn free_many<I>(&self, page_ids: I)
    where
        I: IntoIterator<Item = u32>,
    {
        for id in page_ids {
            self.mark_free(id);
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

    fn mark_allocated(&self, id: u32) {
        let (bitmap, mask) = self.allocated_id_bit(id);
        let previous = bitmap.fetch_or(mask, Ordering::Relaxed);
        debug_assert_eq!(previous & mask, 0, "U32 ID allocator cannot allocate an allocated ID");
    }

    fn mark_free(&self, id: u32) {
        let (bitmap, mask) = self.allocated_id_bit(id);
        let previous = bitmap.fetch_and(!mask, Ordering::Relaxed);
        debug_assert_ne!(previous & mask, 0, "U32 ID allocator cannot free an unallocated ID");
    }

    fn allocated_id_bit(&self, id: u32) -> (&AtomicU64, u64) {
        assert!((id as u64) < self.capacity, "U32 ID allocator ID is out of range");
        let index = id as usize >> 6;
        let bit = id & 63;
        (&self.allocated_ids_bitmap[index], 1_u64 << bit)
    }
}

#[cfg(test)]
mod tests {
    use super::U32IDAllocator;

    #[test]
    fn test_allocate_free_one() {
        let allocator = U32IDAllocator::new(65);
        assert_eq!(allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(), vec![0, 0]);

        let first = allocator.alloc_one().unwrap();
        let second = allocator.alloc_one().unwrap();
        assert_eq!((first, second), (0, 1));
        assert_eq!(allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(), vec![0b11, 0]);

        allocator.free_one(first);
        assert_eq!(allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(), vec![0b10, 0]);
        allocator.free_one(second);
        assert_eq!(allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(), vec![0, 0]);
    }

    #[test]
    fn test_allocate_free_many() {
        let allocator = U32IDAllocator::new(130);
        let ids = allocator.alloc_many(66).unwrap();
        assert_eq!(
            allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(),
            vec![u64::MAX, 0b11, 0]
        );

        allocator.free_many([1, 63, 64, 65]);
        assert_eq!(
            allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(),
            vec![!(0b10 | (1_u64 << 63)), 0, 0]
        );

        assert_eq!(allocator.alloc_many(4).unwrap().len(), 4);
        assert_eq!(
            allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(),
            vec![u64::MAX, 0b11, 0]
        );
        allocator.free_many(ids);
        assert_eq!(allocator.allocated_ids_bitmap_iter().collect::<Vec<_>>(), vec![0, 0, 0]);
    }
}
