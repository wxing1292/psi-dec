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

const RESOURCE_STORAGE_BYTES: usize = 64;
const RESOURCE_ALLOCATION_BYTES: usize = 8;
const NUM_RESOURCE_TOKENS: u32 = 2;

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

pub fn concrete_resource(resource_id: ResourceID) -> (Resource, impl Send) {
    let storage = TestResourceStorage(vec![0; RESOURCE_STORAGE_BYTES].into_boxed_slice());
    let allocator = OffsetAllocator::new(Allocator::new(RESOURCE_STORAGE_BYTES as u32), storage);
    let source = allocator.alloc_segment(RESOURCE_ALLOCATION_BYTES).unwrap();
    let resource = SymbolicResource::new(resource_id, ResourceURI::new("test://resource".to_string()))
        .into_concrete(source, NUM_RESOURCE_TOKENS);
    (Resource::Concrete(resource), allocator)
}
