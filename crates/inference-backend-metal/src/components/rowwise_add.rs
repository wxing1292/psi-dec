use super::assert_u32_count_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const ROWWISE_ADD_SOURCE: &str = include_str!("metal/rowwise_add.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowwiseAddConfig {
    pub row_width: u32,
    pub dtype: Dtype,
}

impl RowwiseAddConfig {
    pub fn validate(self) {
        assert!(self.row_width > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    fn num_values(self, shape: RowwiseAddShape) -> usize {
        checked_product(
            "rowwise add value count",
            &[shape.num_rows as usize, self.row_width as usize],
        )
    }

    fn lhs_values(self, shape: RowwiseAddShape) -> usize {
        let rows = shape
            .lhs_row_offset
            .checked_add(shape.num_rows)
            .expect("rowwise add LHS row range must fit u32");
        checked_product("rowwise add LHS value count", &[rows as usize, self.row_width as usize])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowwiseAddShape {
    pub num_rows: u32,
    pub lhs_row_offset: u32,
}

impl RowwiseAddShape {
    pub fn validate(self, config: RowwiseAddConfig) {
        config.validate();
        assert!(self.num_rows > 0);
        assert_u32_count_domain(config.num_values(self), "rowwise add");
        assert_u32_count_domain(config.lhs_values(self), "rowwise add LHS");
    }
}

#[derive(Clone, Copy)]
pub struct RowwiseAddBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct RowwiseAddKernel {
    config: RowwiseAddConfig,
    kernel: Kernel,
}

impl RowwiseAddKernel {
    pub fn new(device: &Device, config: RowwiseAddConfig) -> Self {
        config.validate();
        let function_name = match config.dtype {
            Dtype::Float32 => "rowwise_add_f32",
            Dtype::Bfloat16 => "rowwise_add_bf16",
            dtype => panic!("unsupported rowwise add dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source(config), function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: RowwiseAddShape, buffers: RowwiseAddBuffers<'a>) -> RowwiseAddInvocation<'a> {
        RowwiseAddInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn source(config: RowwiseAddConfig) -> String {
    ROWWISE_ADD_SOURCE.replacen(
        "using namespace metal;",
        &format!(
            "using namespace metal;\nconstant uint row_width = {}u;",
            config.row_width
        ),
        1,
    )
}

pub struct RowwiseAddInvocation<'a> {
    config: RowwiseAddConfig,
    kernel: &'a Kernel,
    shape: RowwiseAddShape,
    buffers: RowwiseAddBuffers<'a>,
}

impl Operator for RowwiseAddInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_rows);
        builder.set_u32(4, self.shape.lhs_row_offset);
        builder.dispatch_1d(self.config.num_values(self.shape), NUM_THREADS_PER_THREADBLOCK);
    }
}

impl RowwiseAddInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.lhs.len_bytes() >= bytes(self.config.lhs_values(self.shape), self.config.dtype));
        let active_bytes = bytes(self.config.num_values(self.shape), self.config.dtype);
        assert!(self.buffers.rhs.len_bytes() >= active_bytes);
        assert!(self.buffers.output.len_bytes() >= active_bytes);
    }
}

fn bytes(num_values: usize, dtype: Dtype) -> usize {
    num_values
        .checked_mul(dtype.item_size())
        .expect("rowwise add byte length must fit usize")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::Stream;

    #[test]
    fn shape_models_contiguous_rows_from_an_lhs_offset() {
        let config = RowwiseAddConfig {
            row_width: 151_936,
            dtype: Dtype::Bfloat16,
        };
        let shape = RowwiseAddShape {
            num_rows: 4,
            lhs_row_offset: 8,
        };
        shape.validate(config);
        assert_eq!(config.num_values(shape), 607_744);
        assert_eq!(config.lhs_values(shape), 1_823_232);
    }

    #[test]
    fn f32_kernel_adds_rows_from_the_requested_lhs_offset() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = RowwiseAddKernel::new(
            &device,
            RowwiseAddConfig {
                row_width: 3,
                dtype: Dtype::Float32,
            },
        );
        let lhs = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]);
        let rhs = Buffer::from_slice(&device, &[0.5_f32, -1.0, 2.0, 1.0, 2.0, 3.0]);
        let output = Buffer::new_zeroed_elements(&device, 6, Dtype::Float32);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            RowwiseAddShape {
                num_rows: 2,
                lhs_row_offset: 1,
            },
            RowwiseAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        stream.submit_replay(&builder.build()).wait();
        assert_eq!(output.read_typed::<f32>(0, 6), [10.5, 19.0, 32.0, 101.0, 202.0, 303.0]);
    }

    #[test]
    fn bf16_kernel_adds_rows_from_the_requested_lhs_offset() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = RowwiseAddKernel::new(
            &device,
            RowwiseAddConfig {
                row_width: 3,
                dtype: Dtype::Bfloat16,
            },
        );
        let lhs = bf16_buffer(&device, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]);
        let rhs = bf16_buffer(&device, &[0.5, -1.0, 2.0, 1.0, 2.0, 3.0]);
        let output = Buffer::new_zeroed_elements(&device, 6, Dtype::Bfloat16);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            RowwiseAddShape {
                num_rows: 2,
                lhs_row_offset: 1,
            },
            RowwiseAddBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        stream.submit_replay(&builder.build()).wait();
        let actual = output
            .read_typed::<u16>(0, 6)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = [10.5, 19.0, 32.0, 101.0, 202.0, 303.0].map(|value| bf16::from_f32(value).to_f32());
        assert_eq!(actual, expected);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }
}
