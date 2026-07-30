use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const BUFFER_CAST_SOURCE: &str = include_str!("metal/buffer_cast.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferCastConfig {
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl BufferCastConfig {
    pub fn bf16_to_f32() -> Self {
        Self {
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
        }
    }

    pub fn f32_to_bf16() -> Self {
        Self {
            input_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
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

    pub fn input_bytes(self, shape: BufferCastShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.input_dtype.item_size())
            .expect("buffer cast input byte length must fit usize")
    }

    pub fn output_bytes(self, shape: BufferCastShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.output_dtype.item_size())
            .expect("buffer cast output byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferCastShape {
    pub num_values: u32,
}

impl BufferCastShape {
    pub fn validate(self) {
        assert!(self.num_values > 0);
    }
}

#[derive(Clone, Copy)]
pub struct BufferCastBuffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct BufferCastKernel {
    config: BufferCastConfig,
    kernel: Kernel,
}

impl BufferCastKernel {
    pub fn new(device: &Device, config: BufferCastConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, BUFFER_CAST_SOURCE, buffer_cast_function_name(config)),
        }
    }

    pub fn invoke<'a>(&'a self, shape: BufferCastShape, buffers: BufferCastBuffers<'a>) -> BufferCastInvocation<'a> {
        BufferCastInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape,
            buffers,
        }
    }
}

pub struct BufferCastInvocation<'a> {
    kernel: &'a Kernel,
    config: BufferCastConfig,
    shape: BufferCastShape,
    buffers: BufferCastBuffers<'a>,
}

impl Operator for BufferCastInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.config.validate();
        self.shape.validate();
        assert!(self.buffers.input.len_bytes() >= self.config.input_bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.output_bytes(self.shape));

        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_write(1, self.buffers.output, 0);
        builder.set_u32(2, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

fn buffer_cast_function_name(config: BufferCastConfig) -> &'static str {
    match (config.input_dtype, config.output_dtype) {
        (Dtype::Bfloat16, Dtype::Float32) => "bf16_to_f32",
        (Dtype::Float32, Dtype::Bfloat16) => "f32_to_bf16",
        (input_dtype, output_dtype) => {
            panic!("unsupported buffer cast dtype combination: input={input_dtype:?}, output={output_dtype:?}")
        },
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::Stream;

    #[test]
    fn test_bf16_to_f32() {
        let device = Device::system_default();
        let input_values = [-2.5_f32, -0.0, 0.75, 17.0];
        let input_bits = input_values.map(|value| bf16::from_f32(value).to_bits());
        let input = Buffer::from_slice(&device, &input_bits);
        let output = Buffer::new_zeroed(&device, input_values.len() * size_of::<f32>());
        run(
            &device,
            BufferCastConfig::bf16_to_f32(),
            input_values.len() as u32,
            &input,
            &output,
        );
        assert_eq!(output.read_typed::<f32>(0, input_values.len()), input_values);
    }

    #[test]
    fn test_f32_to_bf16() {
        let device = Device::system_default();
        let input_values = [-2.5_f32, -0.0, 0.75, 17.0];
        let expected = input_values.map(|value| bf16::from_f32(value).to_bits());
        let input = Buffer::from_slice(&device, &input_values);
        let output = Buffer::new_zeroed(&device, expected.len() * size_of::<u16>());
        run(
            &device,
            BufferCastConfig::f32_to_bf16(),
            input_values.len() as u32,
            &input,
            &output,
        );
        assert_eq!(output.read_typed::<u16>(0, expected.len()), expected);
    }

    fn run(device: &Device, config: BufferCastConfig, num_values: u32, input: &Buffer, output: &Buffer) {
        let stream = Stream::new(device);
        let kernel = BufferCastKernel::new(device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(BufferCastShape { num_values }, BufferCastBuffers { input, output }));
        stream.submit_replay(&builder.build()).wait();
    }
}
