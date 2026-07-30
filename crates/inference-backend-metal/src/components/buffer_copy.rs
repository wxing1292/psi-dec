use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const BUFFER_COPY_SOURCE: &str = include_str!("metal/buffer_copy.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct F32BufferCopyShape {
    pub num_values: u32,
}

impl F32BufferCopyShape {
    pub fn validate(self) {
        assert!(self.num_values > 0, "f32 buffer copy requires num_values > 0");
    }

    fn bytes(self) -> usize {
        (self.num_values as usize)
            .checked_mul(size_of::<f32>())
            .expect("f32 buffer copy byte length must fit usize")
    }
}

#[derive(Clone, Copy)]
pub struct F32BufferCopyBuffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
    pub input_offset_bytes: usize,
    pub output_offset_bytes: usize,
}

pub struct F32BufferCopyKernel {
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
        shape: F32BufferCopyShape,
        buffers: F32BufferCopyBuffers<'a>,
    ) -> F32BufferCopyInvocation<'a> {
        F32BufferCopyInvocation {
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

pub struct F32BufferCopyInvocation<'a> {
    kernel: &'a Kernel,
    shape: F32BufferCopyShape,
    buffers: F32BufferCopyBuffers<'a>,
}

impl Operator for F32BufferCopyInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_eq!(
            self.buffers.input_offset_bytes % size_of::<f32>(),
            0,
            "f32 buffer copy input offset must be 4-byte aligned"
        );
        assert_eq!(
            self.buffers.output_offset_bytes % size_of::<f32>(),
            0,
            "f32 buffer copy output offset must be 4-byte aligned"
        );
        let input_end_bytes = self
            .buffers
            .input_offset_bytes
            .checked_add(self.shape.bytes())
            .expect("f32 buffer copy input range must fit usize");
        let output_end_bytes = self
            .buffers
            .output_offset_bytes
            .checked_add(self.shape.bytes())
            .expect("f32 buffer copy output range must fit usize");
        assert!(
            input_end_bytes <= self.buffers.input.len_bytes(),
            "f32 buffer copy input range is out of bounds"
        );
        assert!(
            output_end_bytes <= self.buffers.output.len_bytes(),
            "f32 buffer copy output range is out of bounds"
        );
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, self.buffers.input_offset_bytes);
        builder.set_buffer_write(1, self.buffers.output, self.buffers.output_offset_bytes);
        builder.set_u32(2, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct U32BufferCopyShape {
    pub num_values: u32,
}

impl U32BufferCopyShape {
    pub fn validate(self) {
        assert!(self.num_values > 0, "u32 buffer copy requires num_values > 0");
    }

    fn bytes(self) -> usize {
        (self.num_values as usize)
            .checked_mul(size_of::<u32>())
            .expect("u32 buffer copy byte length must fit usize")
    }
}

#[derive(Clone, Copy)]
pub struct U32BufferCopyBuffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
    pub input_offset_bytes: usize,
    pub output_offset_bytes: usize,
}

pub struct U32BufferCopyKernel {
    kernel: Kernel,
}

impl U32BufferCopyKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: Kernel::new(device, BUFFER_COPY_SOURCE, "u32_buffer_copy"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: U32BufferCopyShape,
        buffers: U32BufferCopyBuffers<'a>,
    ) -> U32BufferCopyInvocation<'a> {
        U32BufferCopyInvocation {
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

pub struct U32BufferCopyInvocation<'a> {
    kernel: &'a Kernel,
    shape: U32BufferCopyShape,
    buffers: U32BufferCopyBuffers<'a>,
}

impl Operator for U32BufferCopyInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_eq!(
            self.buffers.input_offset_bytes % size_of::<u32>(),
            0,
            "u32 buffer copy input offset must be 4-byte aligned"
        );
        assert_eq!(
            self.buffers.output_offset_bytes % size_of::<u32>(),
            0,
            "u32 buffer copy output offset must be 4-byte aligned"
        );
        let input_end_bytes = self
            .buffers
            .input_offset_bytes
            .checked_add(self.shape.bytes())
            .expect("u32 buffer copy input range must fit usize");
        let output_end_bytes = self
            .buffers
            .output_offset_bytes
            .checked_add(self.shape.bytes())
            .expect("u32 buffer copy output range must fit usize");
        assert!(
            input_end_bytes <= self.buffers.input.len_bytes(),
            "u32 buffer copy input range is out of bounds"
        );
        assert!(
            output_end_bytes <= self.buffers.output.len_bytes(),
            "u32 buffer copy output range is out of bounds"
        );
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, self.buffers.input_offset_bytes);
        builder.set_buffer_write(1, self.buffers.output, self.buffers.output_offset_bytes);
        builder.set_u32(2, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}
