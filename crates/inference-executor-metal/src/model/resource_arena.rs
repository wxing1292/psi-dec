use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_runtime_core::memory::BlockStorage;
use inference_runtime_core::memory::BlockStorageType;
use inference_runtime_core::memory::DeviceID;
use inference_runtime_core::memory::OffsetAllocator;
use offset_allocator::Allocator;

pub type MetalResourceArena = OffsetAllocator<MetalBufferStorage>;

pub struct MetalBufferStorage(Buffer);

impl MetalBufferStorage {
    pub const fn buffer(&self) -> &Buffer {
        &self.0
    }
}

pub fn new_resource_arena(device: &Device, capacity_bytes: usize) -> MetalResourceArena {
    assert!(capacity_bytes > 0, "Metal resource arena capacity must be positive");
    let capacity = u32::try_from(capacity_bytes).expect("Metal resource arena capacity must fit u32");
    OffsetAllocator::new(
        Allocator::new(capacity),
        MetalBufferStorage(Buffer::new_uninit(device, capacity_bytes)),
    )
}

impl BlockStorage for MetalBufferStorage {
    fn storage_type(&self) -> BlockStorageType {
        BlockStorageType::Device {
            device_id: DeviceID::default(),
        }
    }

    fn address(&self) -> usize {
        self.0.contents() as usize
    }

    unsafe fn as_ref(&self) -> *const u8 {
        self.0.contents().cast()
    }

    unsafe fn as_mut(&mut self) -> *mut u8 {
        self.0.contents().cast()
    }

    fn size(&self) -> usize {
        self.0.len_bytes()
    }
}

// MTLBuffer is a thread-safe immutable allocation handle. Each allocation has one mutable range owner.
// The resource processor writes the range before it transfers the concrete Resource to the decoder.
unsafe impl Send for MetalBufferStorage {}
unsafe impl Sync for MetalBufferStorage {}

#[cfg(test)]
mod tests {
    use inference_runtime_core::memory::BlockAllocator;

    use super::*;

    #[test]
    fn test_alloc_segment_storage() {
        let device = Device::system_default();
        let arena = new_resource_arena(&device, 64);
        let mut allocation = arena.alloc_segment(16).unwrap();
        allocation.slice_mut().fill(7);

        let mut actual = vec![0; 16];
        arena.storage().buffer().read_bytes(0, &mut actual);
        assert_eq!(actual, vec![7; 16]);
        assert_eq!(allocation.offset_bytes(), 0);
        assert_eq!(allocation.len_bytes(), 16);
    }
}
