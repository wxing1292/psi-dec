use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const BUFFER_CAST_SOURCE: &str = include_str!("metal/buffer_cast.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferCastShape {
    pub num_values: u32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl BufferCastShape {
    pub fn bf16_to_f32(num_values: u32) -> Self {
        Self {
            num_values,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
        }
    }

    pub fn f32_to_bf16(num_values: u32) -> Self {
        Self {
            num_values,
            input_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_values > 0);
        assert!(
            matches!(
                (self.input_dtype, self.output_dtype),
                (Dtype::Bfloat16, Dtype::Float32) | (Dtype::Float32, Dtype::Bfloat16)
            ),
            "unsupported buffer cast dtype combination: input={:?}, output={:?}",
            self.input_dtype,
            self.output_dtype
        );
    }

    pub fn input_bytes(self) -> usize {
        self.num_values as usize * self.input_dtype.item_size()
    }

    pub fn output_bytes(self) -> usize {
        self.num_values as usize * self.output_dtype.item_size()
    }
}

#[derive(Clone, Copy)]
pub struct BufferCastBuffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct BufferCastKernel {
    bf16_to_f32: Kernel,
    f32_to_bf16: Kernel,
}

impl BufferCastKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            bf16_to_f32: Kernel::new(device, BUFFER_CAST_SOURCE, "bf16_to_f32"),
            f32_to_bf16: Kernel::new(device, BUFFER_CAST_SOURCE, "f32_to_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: BufferCastShape, buffers: BufferCastBuffers<'a>) -> BufferCastInvocation<'a> {
        BufferCastInvocation {
            kernel: self.kernel(shape),
            shape,
            buffers,
        }
    }

    fn kernel(&self, shape: BufferCastShape) -> &Kernel {
        match (shape.input_dtype, shape.output_dtype) {
            (Dtype::Bfloat16, Dtype::Float32) => &self.bf16_to_f32,
            (Dtype::Float32, Dtype::Bfloat16) => &self.f32_to_bf16,
            (input_dtype, output_dtype) => {
                panic!("unsupported buffer cast dtype combination: input={input_dtype:?}, output={output_dtype:?}")
            },
        }
    }
}

pub struct BufferCastInvocation<'a> {
    kernel: &'a Kernel,
    shape: BufferCastShape,
    buffers: BufferCastBuffers<'a>,
}

impl Operator for BufferCastInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert!(self.buffers.input.len_bytes() >= self.shape.input_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes());

        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_write(1, self.buffers.output, 0);
        builder.set_u32(2, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}
