use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const ROW_GATHER_SOURCE: &str = include_str!("metal/row_gather.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowGatherShape {
    pub num_rows: u32,
    pub num_cols: u32,
    pub dtype: Dtype,
}

impl RowGatherShape {
    pub fn bf16(num_rows: u32, num_cols: u32) -> Self {
        Self {
            num_rows,
            num_cols,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn f32(num_rows: u32, num_cols: u32) -> Self {
        Self {
            num_rows,
            num_cols,
            dtype: Dtype::Float32,
        }
    }

    pub fn validate(self) {
        assert!(self.num_rows > 0);
        assert!(self.num_cols > 0);
        assert!(matches!(self.dtype, Dtype::Bfloat16 | Dtype::Float32));
    }

    pub fn min_input_bytes(self) -> usize {
        self.num_cols as usize * self.dtype.item_size()
    }

    pub fn row_indices_bytes(self) -> usize {
        self.num_rows as usize * size_of::<u32>()
    }

    pub fn output_bytes(self) -> usize {
        self.num_rows as usize * self.num_cols as usize * self.dtype.item_size()
    }

    fn num_values(self) -> u32 {
        self.num_rows * self.num_cols
    }
}

#[derive(Clone, Copy)]
pub struct RowGatherBuffers<'a> {
    pub input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct RowGatherKernel {
    bf16_kernel: Kernel,
    f32_kernel: Kernel,
}

impl RowGatherKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            bf16_kernel: Kernel::new(device, ROW_GATHER_SOURCE, "row_gather_bf16"),
            f32_kernel: Kernel::new(device, ROW_GATHER_SOURCE, "row_gather_f32"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: RowGatherShape, buffers: RowGatherBuffers<'a>) -> RowGatherInvocation<'a> {
        RowGatherInvocation {
            kernel: self.kernel(shape),
            shape,
            buffers,
        }
    }

    fn kernel(&self, shape: RowGatherShape) -> &Kernel {
        match shape.dtype {
            Dtype::Bfloat16 => &self.bf16_kernel,
            Dtype::Float32 => &self.f32_kernel,
            dtype => panic!("unsupported row gather dtype {dtype:?}"),
        }
    }
}

pub struct RowGatherInvocation<'a> {
    kernel: &'a Kernel,
    shape: RowGatherShape,
    buffers: RowGatherBuffers<'a>,
}

impl Operator for RowGatherInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl RowGatherInvocation<'_> {
    fn validate(&self) {
        self.shape.validate();
        assert!(self.buffers.input.len_bytes() >= self.shape.min_input_bytes());
        assert!(self.buffers.row_indices.len_bytes() >= self.shape.row_indices_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.row_indices, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_cols);
        builder.set_u32(4, self.shape.num_rows);
        builder.dispatch_1d(self.shape.num_values() as usize, NUM_THREADS_PER_THREADBLOCK);
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
        let kernel = RowGatherKernel::new(&device);
        let input = Buffer::from_slice(
            &device,
            &[0x3f80_u16, 0x4000_u16, 0x4040_u16, 0x4080_u16, 0x40a0_u16, 0x40c0_u16],
        );
        let row_indices = Buffer::from_slice(&device, &[2_u32, 0]);
        let output = Buffer::new_zeroed(&device, 4 * size_of::<u16>());

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            RowGatherShape::bf16(2, 2),
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
