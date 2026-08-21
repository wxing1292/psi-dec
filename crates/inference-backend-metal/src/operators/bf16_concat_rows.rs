//! Bucketed row-wise concatenation for two BF16 matrices.

use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const SOURCE: &str = include_str!("metal/bf16_concat_rows.metal");
const NUM_BFLOATS_PER_VECTOR: u32 = 4;
const BFLOAT4_ALIGNMENT_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        config.validate();
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_columns: u32,
}

impl Config {
    fn validate(self) {
        assert!(self.num_columns > 0, "bf16 row concat requires num_columns > 0");
        assert!(
            self.num_columns.is_multiple_of(NUM_BFLOATS_PER_VECTOR),
            "bf16 row concat requires num_columns to be divisible by {NUM_BFLOATS_PER_VECTOR}"
        );
    }

    fn validate_num_total_rows(self, num_total_rows: u32) {
        self.validate();
        assert!(num_total_rows > 0, "bf16 row concat requires num_total_rows > 0");
        u32::try_from(self.output_elements_u64(num_total_rows))
            .expect("bf16 row concat output elements exceeds the shader u32 count domain");
    }

    fn input_elements_u64(self, num_total_rows: u32) -> u64 {
        (num_total_rows as u64)
            .checked_mul(self.num_columns as u64)
            .expect("bf16 row concat input element count must fit u64")
    }

    fn output_elements_u64(self, num_total_rows: u32) -> u64 {
        self.input_elements_u64(num_total_rows)
            .checked_mul(2)
            .expect("bf16 row concat output element count must fit u64")
    }

    fn input_bytes_u64(self, num_total_rows: u32) -> u64 {
        self.input_elements_u64(num_total_rows)
            .checked_mul(size_of::<u16>() as u64)
            .expect("bf16 row concat input byte length must fit u64")
    }

    fn output_bytes_u64(self, num_total_rows: u32) -> u64 {
        self.output_elements_u64(num_total_rows)
            .checked_mul(size_of::<u16>() as u64)
            .expect("bf16 row concat output byte length must fit u64")
    }

    fn num_threads(self, num_total_rows: u32) -> usize {
        self.output_elements_u64(num_total_rows)
            .checked_div(NUM_BFLOATS_PER_VECTOR as u64)
            .expect("validated BF16 row width must contain complete bfloat4 vectors")
            .try_into()
            .expect("bf16 row concat dispatch count must fit host usize")
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Kernel {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Kernel {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        Self {
            constants,
            kernel: CompiledKernel::new(device, SOURCE, "bf16_concat_rows_bfloat4"),
        }
    }

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        num_total_rows: u32,
        num_active_rows_key: ReplayParameterKey,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        Invocation {
            kernel: self,
            num_total_rows,
            buffers,
            num_active_rows_key,
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Kernel,
    num_total_rows: u32,
    buffers: Buffers<'a>,
    num_active_rows_key: ReplayParameterKey,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let constants = self.kernel.constants;
        let config = constants.config;
        config.validate_num_total_rows(self.num_total_rows);
        validate_buffers(config, self.num_total_rows, self.buffers);
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.lhs, 0);
        recorder.set_buffer_read(1, self.buffers.rhs, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        recorder.bind_u32(3, self.num_active_rows_key, 1, self.num_total_rows);
        recorder.set_u32(4, config.num_columns);
        recorder.dispatch_1d(
            config.num_threads(self.num_total_rows),
            constants.thread_block.required_threads as usize,
        );
    }
}

fn validate_buffers(config: Config, num_total_rows: u32, buffers: Buffers<'_>) {
    assert!(
        buffers.lhs.len_bytes_u64() >= config.input_bytes_u64(num_total_rows),
        "bf16 row concat lhs buffer is too small"
    );
    assert!(
        buffers.rhs.len_bytes_u64() >= config.input_bytes_u64(num_total_rows),
        "bf16 row concat rhs buffer is too small"
    );
    assert!(
        buffers.output.len_bytes_u64() >= config.output_bytes_u64(num_total_rows),
        "bf16 row concat output buffer is too small"
    );
    validate_bfloat4_alignment("lhs", buffers.lhs);
    validate_bfloat4_alignment("rhs", buffers.rhs);
    validate_bfloat4_alignment("output", buffers.output);
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

fn validate_bfloat4_alignment(name: &str, buffer: &Buffer) {
    assert!(
        (buffer.contents() as usize).is_multiple_of(BFLOAT4_ALIGNMENT_BYTES),
        "bf16 row concat {name} buffer must be {BFLOAT4_ALIGNMENT_BYTES}-byte aligned"
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
    fn test_constants_have_explicit_thread_block_scope() {
        let config = Config { num_columns: 4 };
        assert_eq!(
            KernelConstants::current(config),
            KernelConstants {
                config,
                thread_block: ThreadBlockConstants { required_threads: 256 },
            }
        );
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_tail_across_grow_and_shrink() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config { num_columns: 4 };
        let num_total_rows = 4;
        let lhs = Buffer::from_slice(
            &device,
            &[
                1_u16,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                INPUT_POISON,
                INPUT_POISON,
                INPUT_POISON,
                INPUT_POISON,
            ],
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
                110,
                111,
                112,
                INPUT_POISON,
                INPUT_POISON,
                INPUT_POISON,
                INPUT_POISON,
            ],
        );
        let output = Buffer::from_slice(&device, &[OUTPUT_CANARY; 40]);
        let kernel = Kernel::new(&device, config);

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke_bucketed(
            num_total_rows,
            NUM_ACTIVE_ROWS,
            Buffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &output,
            },
        ));
        let replay = recorder.build();

        let expected_active = reference_concat(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            4,
        );
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(0, 24), expected_active);
        assert_eq!(output.read_typed::<u16>(24, 16), vec![OUTPUT_CANARY; 16]);

        lhs.write_typed(12, &[13_u16, 14, 15, 16]);
        rhs.write_typed(12, &[113_u16, 114, 115, 116]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 4))
            .wait();
        let expected_full = reference_concat(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            ],
            4,
        );
        let full_output = output.read_typed::<u16>(0, 32);
        assert_eq!(full_output, expected_full);
        assert_eq!(output.read_typed::<u16>(32, 8), vec![OUTPUT_CANARY; 8]);

        lhs.write_typed(12, &[INPUT_POISON; 4]);
        rhs.write_typed(12, &[INPUT_POISON; 4]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(0, 24), expected_active);
        assert_eq!(output.read_typed::<u16>(24, 8), full_output[24..]);
        assert_eq!(output.read_typed::<u16>(32, 8), vec![OUTPUT_CANARY; 8]);
    }

    #[test]
    fn test_config_requires_complete_bfloat4_columns() {
        let device = Device::system_default();
        for num_columns in [0, 1, 2, 3, 5, 6, 7] {
            assert_panics(|| {
                let _ = Kernel::new(&device, Config { num_columns });
            });
        }
        let _ = Kernel::new(&device, Config { num_columns: 4 });
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output elements exceeds the shader u32 count domain")]
    fn test_num_total_rows_rejects_doubled_output_count_overflow() {
        Config { num_columns: 4 }.validate_num_total_rows(u32::MAX);
    }

    #[test]
    #[should_panic(expected = "bf16 row concat output must not alias lhs")]
    fn test_buffers_reject_output_lhs_alias_without_dispatch() {
        let device = Device::system_default();
        let shared = Buffer::new_zeroed_elements(&device, 8, Dtype::Bfloat16);
        let rhs = Buffer::new_zeroed_elements(&device, 4, Dtype::Bfloat16);
        validate_buffers(
            Config { num_columns: 4 },
            1,
            Buffers {
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
        let lhs = Buffer::new_zeroed_elements(&device, 4, Dtype::Bfloat16);
        let shared = Buffer::new_zeroed_elements(&device, 8, Dtype::Bfloat16);
        validate_buffers(
            Config { num_columns: 4 },
            1,
            Buffers {
                lhs: &lhs,
                rhs: &shared,
                output: &shared,
            },
        );
    }

    fn reference_concat(lhs: &[u16], rhs: &[u16], num_columns: usize) -> Vec<u16> {
        lhs.chunks_exact(num_columns)
            .zip(rhs.chunks_exact(num_columns))
            .flat_map(|(lhs_row, rhs_row)| lhs_row.iter().chain(rhs_row).copied())
            .collect()
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(AssertUnwindSafe(f)).is_err());
    }
}
