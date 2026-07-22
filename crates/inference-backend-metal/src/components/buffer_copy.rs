use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const BUFFER_COPY_SOURCE: &str = include_str!("metal/buffer_copy.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferCopy32Shape {
    pub num_values: u32,
}

impl BufferCopy32Shape {
    pub fn validate(self) {
        assert!(self.num_values > 0, "32-bit buffer copy requires num_values > 0");
    }

    fn bytes(self) -> usize {
        self.num_values as usize * size_of::<f32>()
    }
}

#[derive(Clone, Copy)]
pub struct BufferCopy32Buffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
    pub input_offset_bytes: usize,
    pub output_offset_bytes: usize,
}

pub struct F32BufferCopyKernel {
    kernel: Kernel,
}

pub struct U32BufferCopyKernel {
    kernel: Kernel,
}

impl F32BufferCopyKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: Kernel::new(device, BUFFER_COPY_SOURCE, "f32_buffer_copy"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: BufferCopy32Shape,
        buffers: BufferCopy32Buffers<'a>,
    ) -> BufferCopy32Invocation<'a> {
        BufferCopy32Invocation {
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

impl U32BufferCopyKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: Kernel::new(device, BUFFER_COPY_SOURCE, "u32_buffer_copy"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: BufferCopy32Shape,
        buffers: BufferCopy32Buffers<'a>,
    ) -> BufferCopy32Invocation<'a> {
        BufferCopy32Invocation {
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

pub struct BufferCopy32Invocation<'a> {
    kernel: &'a Kernel,
    shape: BufferCopy32Shape,
    buffers: BufferCopy32Buffers<'a>,
}

impl Operator for BufferCopy32Invocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_eq!(
            self.buffers.input_offset_bytes % size_of::<f32>(),
            0,
            "32-bit buffer copy input offset must be 4-byte aligned"
        );
        assert_eq!(
            self.buffers.output_offset_bytes % size_of::<f32>(),
            0,
            "32-bit buffer copy output offset must be 4-byte aligned"
        );
        assert!(
            self.buffers.input_offset_bytes + self.shape.bytes() <= self.buffers.input.len_bytes(),
            "32-bit buffer copy input range is out of bounds"
        );
        assert!(
            self.buffers.output_offset_bytes + self.shape.bytes() <= self.buffers.output.len_bytes(),
            "32-bit buffer copy output range is out of bounds"
        );
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, self.buffers.input_offset_bytes);
        builder.set_buffer_write(1, self.buffers.output, self.buffers.output_offset_bytes);
        builder.set_u32(2, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}
