use std::ptr::NonNull;
use std::slice;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::MTL4CommandAllocator;
use objc2_metal::MTL4CommandBuffer;
use objc2_metal::MTL4CommandEncoder;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::MTL4ComputeCommandEncoder;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal::MTLDevice;
use objc2_metal::MTLEvent;
use objc2_metal::MTLResourceOptions;
use objc2_metal::MTLSharedEvent;

#[test]
fn test_fill() {
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device must exist");
    let queue = device
        .newMTL4CommandQueue()
        .expect("MTL4CommandQueue allocation failed");
    let allocator = device
        .newCommandAllocator()
        .expect("MTL4CommandAllocator allocation failed");
    let command_buffer = device.newCommandBuffer().expect("MTL4CommandBuffer allocation failed");
    let event = device.newSharedEvent().expect("MTLSharedEvent allocation failed");

    let len_bytes = 64;
    let buffer = device
        .newBufferWithLength_options(
            len_bytes,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        )
        .expect("MTLBuffer allocation failed");

    command_buffer.beginCommandBufferWithAllocator(&allocator);
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("MTL4ComputeCommandEncoder allocation failed");
    unsafe {
        encoder.fillBuffer_range_value(
            &buffer,
            NSRange {
                location: 0,
                length: len_bytes,
            },
            0x5a,
        );
    }
    encoder.endEncoding();
    command_buffer.endCommandBuffer();

    let mut command_buffer_ptr =
        NonNull::new(Retained::as_ptr(&command_buffer).cast_mut()).expect("command buffer pointer must not be null");
    unsafe {
        queue.commit_count(NonNull::from(&mut command_buffer_ptr), 1);
    }
    let event_ref: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
    queue.signalEvent_value(event_ref, 1);
    assert!(event.waitUntilSignaledValue_timeoutMS(1, 10_000));

    let actual = unsafe { slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), len_bytes) };
    assert_eq!(actual, &[0x5a; 64]);
    allocator.reset();
}
