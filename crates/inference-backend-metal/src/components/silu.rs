use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const SILU_MUL_SOURCE: &str = include_str!("metal/silu.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiluShape {
    pub num_values: u32,
    pub dtype: Dtype,
}

impl SiluShape {
    pub fn f32(num_values: u32) -> Self {
        Self {
            num_values,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_values: u32) -> Self {
        Self {
            num_values,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_values > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn bytes(self) -> usize {
        self.num_values as usize * self.dtype.item_size()
    }
}

#[derive(Clone, Copy)]
pub struct SiluBuffers<'a> {
    pub gate: &'a Buffer,
    pub up: &'a Buffer,
    pub output: &'a Buffer,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SiluBufferOffsets {
    pub gate_offset_bytes: usize,
    pub up_offset_bytes: usize,
    pub output_offset_bytes: usize,
}

pub struct SiluKernel {
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
}

impl SiluKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, SILU_MUL_SOURCE, "silu_mul_f32"),
            bf16_kernel: Kernel::new(device, SILU_MUL_SOURCE, "silu_mul_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: SiluShape, buffers: SiluBuffers<'a>) -> SiluInvocation<'a> {
        self.invoke_with_offsets(shape, buffers, SiluBufferOffsets::default())
    }

    pub fn invoke_with_offsets<'a>(
        &'a self,
        shape: SiluShape,
        buffers: SiluBuffers<'a>,
        offsets: SiluBufferOffsets,
    ) -> SiluInvocation<'a> {
        SiluInvocation {
            kernel: self.kernel(shape),
            shape,
            buffers,
            offsets,
        }
    }

    fn kernel(&self, shape: SiluShape) -> &Kernel {
        match shape.dtype {
            Dtype::Float32 => &self.f32_kernel,
            Dtype::Bfloat16 => &self.bf16_kernel,
            dtype => panic!("unsupported SiLU mul dtype {dtype:?}"),
        }
    }
}

pub struct SiluInvocation<'a> {
    kernel: &'a Kernel,
    shape: SiluShape,
    buffers: SiluBuffers<'a>,
    offsets: SiluBufferOffsets,
}

impl Operator for SiluInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl SiluInvocation<'_> {
    fn validate(&self) {
        self.shape.validate();
        assert!(self.buffers.gate.len_bytes() >= self.offsets.gate_offset_bytes + self.shape.bytes());
        assert!(self.buffers.up.len_bytes() >= self.offsets.up_offset_bytes + self.shape.bytes());
        assert!(self.buffers.output.len_bytes() >= self.offsets.output_offset_bytes + self.shape.bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.gate, self.offsets.gate_offset_bytes);
        builder.set_buffer_read(1, self.buffers.up, self.offsets.up_offset_bytes);
        builder.set_buffer_write(2, self.buffers.output, self.offsets.output_offset_bytes);
        builder.set_u32(3, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}
