use crate::Result;

mod u32_id_allocator;
pub use u32_id_allocator::U32IDAllocator;

mod offset_allocator;
pub use offset_allocator::OffsetAllocation;
pub use offset_allocator::OffsetAllocator;

mod shared_mem;
pub use shared_mem::SharedMem;

mod device_kv_block;
pub use device_kv_block::DeviceBlock;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DeviceID(u32);

#[derive(Clone)]
pub enum BlockStorageType {
    Device { device_id: DeviceID },
    HostPinned { allocation: OffsetAllocation },
}

pub trait BlockStorage: Send + Sync + 'static {
    fn storage_type(&self) -> BlockStorageType;

    fn address(&self) -> usize;
    /// # Safety
    /// Caller must ensure the returned pointer is only dereferenced while the
    /// underlying storage is valid and with correct aliasing/lifetime rules.
    unsafe fn as_ref(&self) -> *const u8;
    /// # Safety
    /// Caller must ensure unique mutable access and that writes stay within the
    /// storage bounds of this block.
    unsafe fn as_mut(&mut self) -> *mut u8;
    fn size(&self) -> usize;
}

pub trait BlockAllocator: Send + Sync + 'static {
    type BlockSegment: Send + Sync + 'static;

    fn storage_type(&self) -> BlockStorageType;

    fn alloc_segment(&self, size: usize) -> Result<Self::BlockSegment>;
    fn free_segment(&self, storage_segment: Self::BlockSegment);
}
