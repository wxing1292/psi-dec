//! Indexed row gather for dense Metal buffers.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

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
    pub num_total_rows: u32,
}

impl RowGatherShape {
    fn validate(self, config: RowGatherConfig) {
        config.validate();
        assert!(self.num_total_rows > 0);
        self.num_values(config);
    }

    fn row_indices_bytes(self) -> usize {
        (self.num_total_rows as usize)
            .checked_mul(size_of::<u32>())
            .expect("row gather index byte length must fit usize")
    }

    fn output_bytes(self, config: RowGatherConfig) -> usize {
        (self.num_values(config) as usize)
            .checked_mul(config.dtype.item_size())
            .expect("row gather output byte length must fit usize")
    }

    fn num_values(self, config: RowGatherConfig) -> u32 {
        self.num_total_rows
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
            num_active_rows_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: RowGatherShape,
        num_active_rows_key: ReplayParameterKey,
        buffers: RowGatherBuffers<'a>,
    ) -> RowGatherInvocation<'a> {
        RowGatherInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape: capacity_shape,
            buffers,
            num_active_rows_key: Some(num_active_rows_key),
        }
    }
}

pub struct RowGatherInvocation<'a> {
    config: RowGatherConfig,
    kernel: &'a Kernel,
    shape: RowGatherShape,
    buffers: RowGatherBuffers<'a>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

impl Operator for RowGatherInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.row_indices, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        recorder.set_u32(3, self.config.num_cols);
        match self.num_active_rows_key {
            Some(key) => recorder.bind_u32(4, key, 1, self.shape.num_total_rows),
            None => recorder.set_u32(4, self.shape.num_total_rows),
        }
        recorder.dispatch_1d(self.shape.num_values(self.config) as usize, NUM_THREADS_PER_THREADBLOCK);
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
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.row_gather.num_active_rows");
    const OUTPUT_CANARY: u8 = 0xa5;

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
            RowGatherShape { num_total_rows: 2 },
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

    #[test]
    fn test_bucketed_bf16_and_f32_preserve_inactive_tail_across_grow_and_shrink() {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            assert_bucketed_grow_and_shrink(dtype);
        }
    }

    fn assert_bucketed_grow_and_shrink(dtype: Dtype) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = RowGatherConfig { num_cols: 3, dtype };
        let capacity_shape = RowGatherShape { num_total_rows: 4 };
        let row_bytes = config.row_bytes();
        let input_values = (0..4 * row_bytes)
            .map(|index| u8::try_from(index + 1).unwrap())
            .collect::<Vec<_>>();
        let input = Buffer::from_slice(&device, &input_values);
        let row_indices = Buffer::from_slice(&device, &[2_u32, 0, 1, u32::MAX]);
        let output = Buffer::from_slice(&device, &vec![OUTPUT_CANARY; 5 * row_bytes]);
        let kernel = RowGatherKernel::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            capacity_shape,
            NUM_ACTIVE_ROWS,
            RowGatherBuffers {
                input: &input,
                row_indices: &row_indices,
                output: &output,
            },
        ));
        let replay = builder.build();

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        let expected_active = gathered_bytes(&input_values, row_bytes, &[2, 0, 1]);
        assert_eq!(output.read_typed::<u8>(0, 3 * row_bytes), expected_active);
        assert_eq!(
            output.read_typed::<u8>(3 * row_bytes, 2 * row_bytes),
            vec![OUTPUT_CANARY; 2 * row_bytes]
        );

        row_indices.write_typed(3, &[3_u32]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 4))
            .wait();
        let expected_full = gathered_bytes(&input_values, row_bytes, &[2, 0, 1, 3]);
        let full_output = output.read_typed::<u8>(0, 4 * row_bytes);
        assert_eq!(full_output, expected_full);
        assert_eq!(
            output.read_typed::<u8>(4 * row_bytes, row_bytes),
            vec![OUTPUT_CANARY; row_bytes]
        );

        row_indices.write_typed(3, &[u32::MAX]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u8>(0, 3 * row_bytes), expected_active);
        assert_eq!(
            output.read_typed::<u8>(3 * row_bytes, row_bytes),
            full_output[3 * row_bytes..]
        );
        assert_eq!(
            output.read_typed::<u8>(4 * row_bytes, row_bytes),
            vec![OUTPUT_CANARY; row_bytes]
        );
    }

    fn gathered_bytes(input: &[u8], row_bytes: usize, row_indices: &[usize]) -> Vec<u8> {
        row_indices
            .iter()
            .flat_map(|&row_index| {
                input[row_index * row_bytes..(row_index + 1) * row_bytes]
                    .iter()
                    .copied()
            })
            .collect()
    }
}
