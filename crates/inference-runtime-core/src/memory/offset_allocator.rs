use std::slice;
use std::sync::Arc;
use std::sync::Mutex;

use offset_allocator::Allocation;
use offset_allocator::Allocator;

use crate::Error;
use crate::Result;
use crate::memory::BlockAllocator;
use crate::memory::BlockStorage;
use crate::memory::BlockStorageType;

#[derive(Clone)]
pub struct OffsetAllocator<S>
where
    S: BlockStorage,
{
    inner: Arc<Mutex<OffsetAllocatorInner<S>>>,
}

pub struct OffsetAllocatorInner<S>
where
    S: BlockStorage,
{
    allocator: Allocator,
    storage: S,
}

#[derive(Clone)]
pub struct OffsetAllocation {
    base_ptr: *const u8,
    allocation: Allocation,
    len: usize,
}

impl<S> OffsetAllocator<S>
where
    S: BlockStorage,
{
    pub fn new(allocator: Allocator, storage: S) -> Self {
        Self {
            inner: Arc::new(Mutex::new(OffsetAllocatorInner::new(allocator, storage))),
        }
    }
}

impl<S> BlockAllocator for OffsetAllocator<S>
where
    S: BlockStorage,
{
    type BlockSegment = OffsetAllocation;

    fn storage_type(&self) -> BlockStorageType {
        let guard = self.inner.lock().unwrap();
        guard.storage_type()
    }

    fn alloc_segment(&self, size: usize) -> Result<Self::BlockSegment> {
        let mut guard = self.inner.lock().unwrap();
        guard.alloc(size)
    }

    fn free_segment(&self, storage_segment: Self::BlockSegment) {
        let mut guard = self.inner.lock().unwrap();
        guard.free(storage_segment)
    }
}

impl<S> OffsetAllocatorInner<S>
where
    S: BlockStorage,
{
    pub fn new(allocator: Allocator, storage: S) -> Self {
        Self { allocator, storage }
    }

    fn storage_type(&self) -> BlockStorageType {
        self.storage.storage_type()
    }

    fn alloc(&mut self, len: usize) -> Result<OffsetAllocation> {
        assert!(len < u32::MAX as usize);
        if let Some(allocation) = self.allocator.allocate(len as u32) {
            Ok(OffsetAllocation {
                base_ptr: unsafe { self.storage.as_ref() },
                allocation,
                len,
            })
        } else {
            Err(Error::resource_exhausted("not enough mem"))
        }
    }

    fn free(&mut self, allocation: OffsetAllocation) {
        self.allocator.free(allocation.allocation);
    }
}

impl OffsetAllocation {
    pub fn ptr(&self) -> *mut u8 {
        unsafe { self.base_ptr.add(self.allocation.offset as usize) as *mut u8 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr() as *const u8, self.len) }
    }

    pub fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr(), self.len) }
    }
}

unsafe impl Send for OffsetAllocation {}
unsafe impl Sync for OffsetAllocation {}
