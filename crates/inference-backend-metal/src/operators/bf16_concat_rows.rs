//! Row-wise concatenation for two BF16 matrices.

use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const BF16_CONCAT_ROWS_SOURCE: &str = include_str!("metal/bf16_concat_rows.metal");
const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16ConcatRowsConfig {
    pub num_cols: u32,
}

impl Bf16ConcatRowsConfig {
    fn validate(self) {
        assert!(self.num_cols > 0, "bf16 row concat requires num_cols > 0");
    }

    fn validate_shape(self, shape: Bf16ConcatRowsShape) {
        self.validate();
        shape.validate();
        u32::try_from(self.output_elements_u64(shape))
            .expect("bf16 row concat output elements exceeds the shader u32 count domain");
    }

    fn input_elements_u64(self, shape: Bf16ConcatRowsShape) -> u64 {
        u64::from(shape.num_rows)
            .checked_mul(u64::from(self.num_cols))
            .expect("bf16 row concat input element count must fit u64")
    }

    fn output_elements_u64(self, shape: Bf16ConcatRowsShape) -> u64 {
        self.input_elements_u64(shape)
            .checked_mul(2)
            .expect("bf16 row concat output element count must fit u64")
    }

    fn input_bytes_u64(self, shape: Bf16ConcatRowsShape) -> u64 {
        self.input_elements_u64(shape)
            .checked_mul(size_of::<u16>().try_into().expect("bf16 item size must fit u64"))
            .expect("bf16 row concat input byte length must fit u64")
    }

    fn output_bytes_u64(self, shape: Bf16ConcatRowsShape) -> u64 {
        self.output_elements_u64(shape)
            .checked_mul(size_of::<u16>().try_into().expect("bf16 item size must fit u64"))
            .expect("bf16 row concat output byte length must fit u64")
    }

    fn num_values(self, shape: Bf16ConcatRowsShape) -> usize {
        self.output_elements_u64(shape)
            .try_into()
            .expect("bf16 row concat dispatch count must fit host usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16ConcatRowsShape {
    pub num_rows: u32,
}

impl Bf16ConcatRowsShape {
    fn validate(self) {
        assert!(self.num_rows > 0, "bf16 row concat requires num_rows > 0");
    }
}

#[derive(Clone, Copy)]
pub struct Bf16ConcatRowsBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Bf16ConcatRowsKernel {
    config: Bf16ConcatRowsConfig,
    kernel: Kernel,
}

impl Bf16ConcatRowsKernel {
    pub fn new(device: &Device, config: Bf16ConcatRowsConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, BF16_CONCAT_ROWS_SOURCE, "bf16_concat_rows"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Bf16ConcatRowsShape,
        buffers: Bf16ConcatRowsBuffers<'a>,
    ) -> Bf16ConcatRowsInvocation<'a> {
        Bf16ConcatRowsInvocation {
            kernel: self,
            shape,
            buffers,
            num_active_rows_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: Bf16ConcatRowsShape,
        num_active_rows_key: ReplayParameterKey,
        buffers: Bf16ConcatRowsBuffers<'a>,
    ) -> Bf16ConcatRowsInvocation<'a> {
        Bf16ConcatRowsInvocation {
            kernel: self,
            shape: capacity_shape,
            buffers,
            num_active_rows_key: Some(num_active_rows_key),
        }
    }
}

pub struct Bf16ConcatRowsInvocation<'a> {
    kernel: &'a Bf16ConcatRowsKernel,
    shape: Bf16ConcatRowsShape,
    buffers: Bf16ConcatRowsBuffers<'a>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

impl Operator for Bf16ConcatRowsInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        config.validate_shape(self.shape);
        validate_buffers(config, self.shape, self.buffers);
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        match self.num_active_rows_key {
            Some(key) => builder.bind_u32(3, key, 1, self.shape.num_rows),
            None => builder.set_u32(3, self.shape.num_rows),
        }
        builder.set_u32(4, config.num_cols);
        builder.dispatch_1d(config.num_values(self.shape), NUM_THREADS_PER_THREADBLOCK);
    }
}

fn validate_buffers(config: Bf16ConcatRowsConfig, shape: Bf16ConcatRowsShape, buffers: Bf16ConcatRowsBuffers<'_>) {
    assert!(
        buffers.lhs.len_bytes_u64() >= config.input_bytes_u64(shape),
        "bf16 row concat lhs buffer is too small"
    );
    assert!(
        buffers.rhs.len_bytes_u64() >= config.input_bytes_u64(shape),
        "bf16 row concat rhs buffer is too small"
    );
    assert!(
        buffers.output.len_bytes_u64() >= config.output_bytes_u64(shape),
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
    use std::panic::AssertUnwindSafe;

    use super::*;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.bf16_concat_rows.num_active_rows");
    const INPUT_POISON: u16 = 0xffff;
    const OUTPUT_CANARY: u16 = 0x7fc1;

    #[test]
    fn test_exact_replay_has_no_parameter() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernel = Bf16ConcatRowsKernel::new(&device, Bf16ConcatRowsConfig { num_cols: 2 });
        let lhs = Buffer::from_slice(&device, &[1_u16, 2, 3, 4]);
        let rhs = Buffer::from_slice(&device, &[11_u16, 12, 13, 14]);
        let output = Buffer::from_slice(&device, &[OUTPUT_CANARY; 8]);

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke(
            Bf16ConcatRowsShape { num_rows: 2 },
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 0);
        stream.submit_replay(&replay).wait();

        assert_eq!(output.read_typed::<u16>(0, 8), vec![1, 2, 11, 12, 3, 4, 13, 14]);
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_tail_across_grow_and_shrink() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Bf16ConcatRowsConfig { num_cols: 3 };
        let capacity_shape = Bf16ConcatRowsShape { num_rows: 4 };
        let active_shape = Bf16ConcatRowsShape { num_rows: 3 };
        let lhs = Buffer::from_slice(
            &device,
            &[1_u16, 2, 3, 4, 5, 6, 7, 8, 9, INPUT_POISON, INPUT_POISON, INPUT_POISON],
        );
        let rhs = Buffer::from_slice(
            &device,
            &[
                101_u16,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                INPUT_POISON,
                INPUT_POISON,
                INPUT_POISON,
            ],
        );
        let output = Buffer::from_slice(&device, &[OUTPUT_CANARY; 30]);
        let kernel = Bf16ConcatRowsKernel::new(&device, config);

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke_bucketed(
            capacity_shape,
            NUM_ACTIVE_ROWS,
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 1);

        let expected_active = reference_concat(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[101, 102, 103, 104, 105, 106, 107, 108, 109],
            3,
        );
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(0, 18), expected_active);
        assert_eq!(output.read_typed::<u16>(18, 12), vec![OUTPUT_CANARY; 12]);

        let exact_output = Buffer::from_slice(&device, &[OUTPUT_CANARY; 18]);
        let mut exact_recorder = stream.create_replay_program();
        exact_recorder.record(kernel.invoke(
            active_shape,
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &exact_output,
            },
        ));
        let exact_replay = exact_recorder.build();
        assert_eq!(exact_replay.stats().parameter_count, 0);
        stream.submit_replay(&exact_replay).wait();
        assert_eq!(exact_output.read_typed::<u16>(0, 18), expected_active);

        lhs.write_typed(9, &[10_u16, 11, 12]);
        rhs.write_typed(9, &[110_u16, 111, 112]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 4))
            .wait();
        let expected_full = reference_concat(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            3,
        );
        let full_output = output.read_typed::<u16>(0, 24);
        assert_eq!(full_output, expected_full);
        assert_eq!(output.read_typed::<u16>(24, 6), vec![OUTPUT_CANARY; 6]);

        lhs.write_typed(9, &[INPUT_POISON; 3]);
        rhs.write_typed(9, &[INPUT_POISON; 3]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(0, 18), expected_active);
        assert_eq!(output.read_typed::<u16>(18, 6), full_output[18..]);
        assert_eq!(output.read_typed::<u16>(24, 6), vec![OUTPUT_CANARY; 6]);
    }

    #[test]
    fn test_bucketed_replay_validates_arguments_and_total_capacity_buffers() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Bf16ConcatRowsConfig { num_cols: 3 };
        let shape = Bf16ConcatRowsShape { num_rows: 4 };
        let kernel = Bf16ConcatRowsKernel::new(&device, config);
        let lhs = Buffer::new_zeroed_elements(&device, 12, Dtype::Bfloat16);
        let rhs = Buffer::new_zeroed_elements(&device, 12, Dtype::Bfloat16);
        let output = Buffer::new_zeroed_elements(&device, 24, Dtype::Bfloat16);
        let short_lhs = Buffer::new_zeroed_elements(&device, 11, Dtype::Bfloat16);
        let short_rhs = Buffer::new_zeroed_elements(&device, 11, Dtype::Bfloat16);
        let short_output = Buffer::new_zeroed_elements(&device, 23, Dtype::Bfloat16);

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke_bucketed(
            shape,
            NUM_ACTIVE_ROWS,
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 1);

        assert_panics(|| {
            let _ = stream.submit_replay(&replay);
        });
        assert_panics(|| {
            let arguments = ReplayArguments::new().with_i32(NUM_ACTIVE_ROWS, 3);
            let _ = stream.submit_replay_with_arguments(&replay, &arguments);
        });
        for invalid_num_active_rows in [0, 5] {
            assert_panics(|| {
                let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, invalid_num_active_rows);
                let _ = stream.submit_replay_with_arguments(&replay, &arguments);
            });
        }

        for buffers in [
            Bf16ConcatRowsBuffers {
                lhs: &short_lhs,
                rhs: &rhs,
                output: &output,
            },
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &short_rhs,
                output: &output,
            },
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &short_output,
            },
        ] {
            assert_panics(|| {
                let mut recorder = stream.create_replay_program();
                recorder.record(kernel.invoke_bucketed(shape, NUM_ACTIVE_ROWS, buffers));
            });
        }
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_doubled_output_count_overflow() {
        Bf16ConcatRowsConfig { num_cols: 1 }.validate_shape(Bf16ConcatRowsShape { num_rows: u32::MAX });
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output must not alias lhs")]
    fn test_buffers_reject_output_lhs_alias_without_dispatch() {
        let device = Device::system_default();
        let shared = Buffer::new_zeroed_elements(&device, 4, Dtype::Bfloat16);
        let rhs = Buffer::new_zeroed_elements(&device, 2, Dtype::Bfloat16);
        validate_buffers(
            Bf16ConcatRowsConfig { num_cols: 2 },
            Bf16ConcatRowsShape { num_rows: 1 },
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
            Bf16ConcatRowsConfig { num_cols: 2 },
            Bf16ConcatRowsShape { num_rows: 1 },
            Bf16ConcatRowsBuffers {
                lhs: &lhs,
                rhs: &shared,
                output: &shared,
            },
        );
    }

    fn reference_concat(lhs: &[u16], rhs: &[u16], num_cols: usize) -> Vec<u16> {
        lhs.chunks_exact(num_cols)
            .zip(rhs.chunks_exact(num_cols))
            .flat_map(|(lhs_row, rhs_row)| lhs_row.iter().chain(rhs_row).copied())
            .collect()
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(AssertUnwindSafe(f)).is_err());
    }
}
