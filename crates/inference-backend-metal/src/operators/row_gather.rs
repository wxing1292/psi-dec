//! Indexed row gather for dense Metal buffers.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const ROW_GATHER_SOURCE: &str = include_str!("metal/row_gather.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowGatherConfig {
    pub num_cols: u32,
    pub dtype: Dtype,
}

impl RowGatherConfig {
    fn validate(self) {
        assert!(self.num_cols > 0);
        assert!(matches!(self.dtype, Dtype::Bfloat16 | Dtype::Float32));
    }

    fn row_bytes(self) -> usize {
        (self.num_cols as usize)
            .checked_mul(self.dtype.item_size())
            .expect("row gather byte length per row must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowGatherShape {
    pub num_rows: u32,
}

impl RowGatherShape {
    fn validate(self, config: RowGatherConfig) {
        config.validate();
        assert!(self.num_rows > 0);
        self.num_values(config);
    }

    fn row_indices_bytes(self) -> usize {
        (self.num_rows as usize)
            .checked_mul(size_of::<u32>())
            .expect("row gather index byte length must fit usize")
    }

    fn output_bytes(self, config: RowGatherConfig) -> usize {
        (self.num_values(config) as usize)
            .checked_mul(config.dtype.item_size())
            .expect("row gather output byte length must fit usize")
    }

    fn num_values(self, config: RowGatherConfig) -> u32 {
        self.num_rows
            .checked_mul(config.num_cols)
            .expect("row gather value count must fit the shader u32 index domain")
    }
}

#[derive(Clone, Copy)]
pub struct RowGatherBuffers<'a> {
    pub input: &'a Buffer,
    /// Each active index must select a complete row from `input`.
    pub row_indices: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct RowGatherKernel {
    config: RowGatherConfig,
    kernel: Kernel,
}

impl RowGatherKernel {
    pub fn new(device: &Device, config: RowGatherConfig) -> Self {
        config.validate();
        let function_name = match config.dtype {
            Dtype::Bfloat16 => "row_gather_bf16",
            Dtype::Float32 => "row_gather_f32",
            _ => unreachable!("validated row gather dtype"),
        };
        Self {
            config,
            kernel: Kernel::new(device, ROW_GATHER_SOURCE, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: RowGatherShape, buffers: RowGatherBuffers<'a>) -> RowGatherInvocation<'a> {
        RowGatherInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

pub struct RowGatherInvocation<'a> {
    config: RowGatherConfig,
    kernel: &'a Kernel,
    shape: RowGatherShape,
    buffers: RowGatherBuffers<'a>,
}

impl Operator for RowGatherInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.row_indices, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.config.num_cols);
        builder.set_u32(4, self.shape.num_rows);
        builder.dispatch_1d(self.shape.num_values(self.config) as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

impl RowGatherInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.input.len_bytes() >= self.config.row_bytes());
        assert!(self.buffers.row_indices.len_bytes() >= self.shape.row_indices_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes(self.config));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Stream;

    #[test]
    fn test_bf16() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = RowGatherKernel::new(
            &device,
            RowGatherConfig {
                num_cols: 2,
                dtype: Dtype::Bfloat16,
            },
        );
        let input = Buffer::from_slice(
            &device,
            &[0x3f80_u16, 0x4000_u16, 0x4040_u16, 0x4080_u16, 0x40a0_u16, 0x40c0_u16],
        );
        let row_indices = Buffer::from_slice(&device, &[2_u32, 0]);
        let output = Buffer::new_zeroed(&device, 4 * size_of::<u16>());

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            RowGatherShape { num_rows: 2 },
            RowGatherBuffers {
                input: &input,
                row_indices: &row_indices,
                output: &output,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let values = output.read_typed::<u16>(0, 4);
        assert_eq!(values, vec![0x40a0, 0x40c0, 0x3f80, 0x4000]);
    }
}
