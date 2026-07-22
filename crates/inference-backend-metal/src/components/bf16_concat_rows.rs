use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;

const BF16_CONCAT_ROWS_SOURCE: &str = include_str!("metal/bf16_concat_rows.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16ConcatRowsShape {
    pub num_rows: u32,
    pub num_cols: u32,
}

impl Bf16ConcatRowsShape {
    pub fn validate(self) {
        assert!(self.num_rows > 0, "bf16 row concat requires num_rows > 0");
        assert!(self.num_cols > 0, "bf16 row concat requires num_cols > 0");
        u32::try_from(self.output_elements_u64())
            .expect("bf16 row concat output elements exceeds the shader u32 count domain");
    }

    fn input_elements_u64(self) -> u64 {
        u64::from(self.num_rows)
            .checked_mul(u64::from(self.num_cols))
            .expect("bf16 row concat input element count must fit u64")
    }

    fn output_elements_u64(self) -> u64 {
        self.input_elements_u64()
            .checked_mul(2)
            .expect("bf16 row concat output element count must fit u64")
    }

    fn input_bytes_u64(self) -> u64 {
        self.input_elements_u64()
            .checked_mul(size_of::<u16>().try_into().expect("bf16 item size must fit u64"))
            .expect("bf16 row concat input byte length must fit u64")
    }

    fn output_bytes_u64(self) -> u64 {
        self.output_elements_u64()
            .checked_mul(size_of::<u16>().try_into().expect("bf16 item size must fit u64"))
            .expect("bf16 row concat output byte length must fit u64")
    }

    fn num_values(self) -> usize {
        self.output_elements_u64()
            .try_into()
            .expect("bf16 row concat dispatch count must fit host usize")
    }
}

#[derive(Clone, Copy)]
pub struct Bf16ConcatRowsBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Bf16ConcatRowsKernel {
    kernel: Kernel,
}

impl Bf16ConcatRowsKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: Kernel::new(device, BF16_CONCAT_ROWS_SOURCE, "bf16_concat_rows"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Bf16ConcatRowsShape,
        buffers: Bf16ConcatRowsBuffers<'a>,
    ) -> Bf16ConcatRowsInvocation<'a> {
        Bf16ConcatRowsInvocation {
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

pub struct Bf16ConcatRowsInvocation<'a> {
    kernel: &'a Kernel,
    shape: Bf16ConcatRowsShape,
    buffers: Bf16ConcatRowsBuffers<'a>,
}

impl Operator for Bf16ConcatRowsInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        validate_buffers(self.shape, self.buffers);
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_rows);
        builder.set_u32(4, self.shape.num_cols);
        builder.dispatch_1d(self.shape.num_values(), NUM_THREADS_PER_THREADBLOCK);
    }
}

fn validate_buffers(shape: Bf16ConcatRowsShape, buffers: Bf16ConcatRowsBuffers<'_>) {
    assert!(
        buffers.lhs.len_bytes_u64() >= shape.input_bytes_u64(),
        "bf16 row concat lhs buffer is too small"
    );
    assert!(
        buffers.rhs.len_bytes_u64() >= shape.input_bytes_u64(),
        "bf16 row concat rhs buffer is too small"
    );
    assert!(
        buffers.output.len_bytes_u64() >= shape.output_bytes_u64(),
        "bf16 row concat output buffer is too small"
    );
    assert_ne!(
        buffers.output.as_raw_ptr(),
        buffers.lhs.as_raw_ptr(),
        "bf16 row concat output must not alias lhs"
    );
    assert_ne!(
        buffers.output.as_raw_ptr(),
        buffers.rhs.as_raw_ptr(),
        "bf16 row concat output must not alias rhs"
    );
}

#[cfg(test)]
mod tests {
    use super::Bf16ConcatRowsBuffers;
    use super::Bf16ConcatRowsShape;
    use super::validate_buffers;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;

    #[test]
    #[should_panic(expected = "bf16 row concat output elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_doubled_output_count_overflow() {
        Bf16ConcatRowsShape {
            num_rows: u32::MAX,
            num_cols: 1,
        }
        .validate();
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output must not alias lhs")]
    fn test_buffers_reject_output_lhs_alias_without_dispatch() {
        let device = Device::system_default();
        let shared = Buffer::new_zeroed_elements(&device, 4, Dtype::Bfloat16);
        let rhs = Buffer::new_zeroed_elements(&device, 2, Dtype::Bfloat16);
        validate_buffers(
            Bf16ConcatRowsShape {
                num_rows: 1,
                num_cols: 2,
            },
            Bf16ConcatRowsBuffers {
                lhs: &shared,
                rhs: &rhs,
                output: &shared,
            },
        );
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output must not alias rhs")]
    fn test_buffers_reject_output_rhs_alias_without_dispatch() {
        let device = Device::system_default();
        let lhs = Buffer::new_zeroed_elements(&device, 2, Dtype::Bfloat16);
        let shared = Buffer::new_zeroed_elements(&device, 4, Dtype::Bfloat16);
        validate_buffers(
            Bf16ConcatRowsShape {
                num_rows: 1,
                num_cols: 2,
            },
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &shared,
                output: &shared,
            },
        );
    }
}
