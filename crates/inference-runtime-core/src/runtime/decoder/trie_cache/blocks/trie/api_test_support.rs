use std::sync::Arc;

use offset_allocator::Allocator;

use crate::memory::BlockAllocator;
use crate::memory::BlockStorage;
use crate::memory::BlockStorageType;
use crate::memory::DeviceID;
use crate::memory::OffsetAllocator;
use crate::runtime::resource::Resource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourceURI;
use crate::runtime::resource::SymbolicResource;

const RESOURCE_TOKEN_BYTES: usize = 4;

struct TestResourceStorage(Box<[u8]>);

impl BlockStorage for TestResourceStorage {
    fn storage_type(&self) -> BlockStorageType {
        BlockStorageType::Device {
            device_id: DeviceID::default(),
        }
    }

    fn address(&self) -> usize {
        self.0.as_ptr() as usize
    }

    unsafe fn as_ref(&self) -> *const u8 {
        self.0.as_ptr()
    }

    unsafe fn as_mut(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }

    fn size(&self) -> usize {
        self.0.len()
    }
}

pub fn concrete_resource(
    resource_id: ResourceID,
    num_resource_tokens: u32,
) -> (
    Resource,
    Arc<impl BlockAllocator<BlockSegment = crate::memory::OffsetAllocation>>,
) {
    let allocation_bytes = num_resource_tokens as usize * RESOURCE_TOKEN_BYTES;
    let storage = TestResourceStorage(vec![0; allocation_bytes].into_boxed_slice());
    let allocator = Arc::new(OffsetAllocator::new(Allocator::new(allocation_bytes as u32), storage));
    let source = allocator.alloc_segment(allocation_bytes).unwrap();
    let resource = SymbolicResource::new(resource_id, ResourceURI::new("test://resource".to_string())).into_concrete(
        allocator.clone(),
        source,
        num_resource_tokens,
    );
    (Resource::Concrete(resource), allocator)
}
