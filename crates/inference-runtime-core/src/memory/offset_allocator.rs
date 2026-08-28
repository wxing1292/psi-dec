use std::fmt;
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

pub struct OffsetAllocator<S>
where
    S: BlockStorage,
{
    allocator: Arc<Mutex<Allocator>>,
    storage: Arc<S>,
}

#[derive(Clone)]
pub struct OffsetAllocation {
    base_ptr: *const u8,
    allocation: Allocation,
    len: usize,
}

impl fmt::Debug for OffsetAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OffsetAllocation")
            .field("offset_bytes", &self.offset_bytes())
            .field("len_bytes", &self.len_bytes())
            .finish()
    }
}

impl<S> OffsetAllocator<S>
where
    S: BlockStorage,
{
    pub fn new(allocator: Allocator, storage: S) -> Self {
        Self {
            allocator: Arc::new(Mutex::new(allocator)),
            storage: Arc::new(storage),
        }
    }

    pub const fn storage(&self) -> &Arc<S> {
        &self.storage
    }
}

impl<S> Clone for OffsetAllocator<S>
where
    S: BlockStorage,
{
    fn clone(&self) -> Self {
        Self {
            allocator: Arc::clone(&self.allocator),
            storage: Arc::clone(&self.storage),
        }
    }
}

impl<S> BlockAllocator for OffsetAllocator<S>
where
    S: BlockStorage,
{
    type BlockSegment = OffsetAllocation;

    fn storage_type(&self) -> BlockStorageType {
        self.storage.storage_type()
    }

    fn alloc_segment(&self, len: usize) -> Result<Self::BlockSegment> {
        assert!(len < u32::MAX as usize);
        let Some(allocation) = self.allocator.lock().unwrap().allocate(len as u32) else {
            return Err(Error::resource_exhausted("not enough mem"));
        };
        Ok(OffsetAllocation {
            base_ptr: unsafe { BlockStorage::as_ref(self.storage.as_ref()) },
            allocation,
            len,
        })
    }

    fn free_segment(&self, storage_segment: Self::BlockSegment) {
        self.allocator.lock().unwrap().free(storage_segment.allocation);
    }
}

impl OffsetAllocation {
    pub fn offset_bytes(&self) -> u64 {
        u64::from(self.allocation.offset)
    }

    pub fn len_bytes(&self) -> u64 {
        self.len as u64
    }

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
