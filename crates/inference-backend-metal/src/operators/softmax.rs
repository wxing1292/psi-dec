use std::collections::HashSet;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel as CompiledKernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::operators::mlx_headers::find_mlx_metal_header_root;
use crate::operators::mlx_headers::read_mlx_metal_header;

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
        let num_threads_needed = config.num_values_per_row.div_ceil(4);
        let required_threads = num_threads_needed.div_ceil(32) * 32;
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_values_per_row: u32,
    pub dtype: Dtype,
}

impl Config {
    fn validate(self) {
        assert!(self.num_values_per_row > 0);
        assert!(self.num_values_per_row <= 4096);
        match self.dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 softmax is not implemented"),
            dtype => panic!("unsupported softmax dtype {dtype:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_rows: u32,
}

impl Shape {
    fn validate(self) {
        assert!(self.num_total_rows > 0);
    }

    fn bytes(self, config: Config) -> usize {
        self.validate();
        config.validate();
        (self.num_total_rows as usize)
            .checked_mul(config.num_values_per_row as usize)
            .and_then(|num_values| num_values.checked_mul(config.dtype.item_size()))
            .expect("softmax byte length must fit usize")
    }
}

pub struct Kernel {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
}

impl Kernel {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        let source = softmax_source();
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, "block_softmax_bfloat16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate();
        Invocation {
            kernel: self,
            shape,
            buffers,
            num_active_rows_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: Shape,
        num_active_rows_key: ReplayParameterKey,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        capacity_shape.validate();
        Invocation {
            kernel: self,
            shape: capacity_shape,
            buffers,
            num_active_rows_key: Some(num_active_rows_key),
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Kernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        shape.validate();
        let constants = self.kernel.constants;
        let config = constants.config;
        let bytes = shape.bytes(config);
        assert!(bytes <= self.buffers.input.len_bytes());
        assert!(bytes <= self.buffers.output.len_bytes());

        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_write(1, self.buffers.output, 0);
        recorder.set_i32(2, config.num_values_per_row as i32);
        match self.num_active_rows_key {
            Some(key) => recorder.bind_u32(3, key, 1, shape.num_total_rows),
            None => recorder.set_u32(3, shape.num_total_rows),
        }

        let required_threads = constants.thread_block.required_threads as usize;
        recorder.dispatch_1d(shape.num_total_rows as usize * required_threads, required_threads);
    }
}

fn softmax_source() -> String {
    let root = mlx_metal_header_root();
    let mut included = HashSet::new();
    let mut source = String::new();
    source.push_str(
        "#include <metal_stdlib>\n#include <metal_common>\n#include <metal_simdgroup>\nusing namespace metal;\n",
    );
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/defines.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/utils.h",
        &mut included,
    ));
    source.push_str(&read_mlx_metal_header(
        &root,
        "mlx/backend/metal/kernels/softmax.h",
        &mut included,
    ));
    source.push_str(BUCKETED_SOFTMAX_SOURCE);
    source
}

// This kernel is the MLX single-row softmax with one fixed-capacity replay guard.
// The guard is uniform across the whole threadgroup and occurs before all input
// reads and threadgroup barriers.
const BUCKETED_SOFTMAX_SOURCE: &str = r#"
template <typename T, typename AccT = T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_single_row_bucketed(
    const device T* in,
    device T* out,
    constant int& axis_size,
    constant uint& num_active_rows,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (gid >= num_active_rows) {
    return;
  }

  int lid = _lid;
  constexpr int SIMD_SIZE = 32;
  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  AccT ld[N_READS];

  in += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = AccT(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] =
          ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : Limits<AccT>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AccT maxval = Limits<AccT>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];

  AccT normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    AccT exp_x = softmax_exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = T(ld[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = T(ld[i] * normalizer);
      }
    }
  }
}

template [[host_name("block_softmax_bfloat16")]] [[kernel]]
decltype(softmax_single_row_bucketed<bfloat16_t, bfloat16_t>)
softmax_single_row_bucketed<bfloat16_t, bfloat16_t>;
"#;

fn mlx_metal_header_root() -> PathBuf {
    find_mlx_metal_header_root("softmax.h", |_| true, "softmax")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::Buffers;
    use super::Config;
    use super::Kernel;
    use super::KernelConstants;
    use super::Shape;
    use super::ThreadBlockConstants;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.softmax.num_active_rows");

    #[test]
    fn test_constants_have_explicit_thread_block_scope() {
        for (num_values_per_row, required_threads) in [(1, 32), (129, 64), (4096, 1024)] {
            let config = Config {
                num_values_per_row,
                dtype: Dtype::Bfloat16,
            };
            assert_eq!(
                KernelConstants::current(config),
                KernelConstants {
                    config,
                    thread_block: ThreadBlockConstants { required_threads },
                }
            );
        }
    }

    #[test]
    #[should_panic(expected = "F32 softmax is not implemented")]
    fn test_f32_is_explicit_future_work() {
        Config {
            num_values_per_row: 4,
            dtype: Dtype::Float32,
        }
        .validate();
    }

    #[test]
    fn test_reference() {
        let config = Config {
            num_values_per_row: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape { num_total_rows: 2 };
        let (device, kernel) = create_softmax_kernel(config);
        let stream = Stream::new(&device);
        let input_values = [-2.0, -1.0, 0.0, 1.0, 4.0, 2.0, 0.0, -2.0];
        let input = bf16_buffer(&device, &input_values);
        let output = Buffer::new_zeroed(&device, shape.bytes(config));

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                input: &input,
                output: &output,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let actual = read_bf16_values(&output, input_values.len());
        let expected = cpu_softmax_bf16_rows(
            &input_values,
            shape.num_total_rows as usize,
            config.num_values_per_row as usize,
        );
        assert_close(&actual, &expected, 0.01);
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_tail_across_grow_and_shrink() {
        let config = Config {
            num_values_per_row: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape { num_total_rows: 4 };
        let (device, kernel) = create_softmax_kernel(config);
        let stream = Stream::new(&device);
        let all_input_values = [
            -2.0, -1.0, 0.0, 1.0, 4.0, 2.0, 0.0, -2.0, 0.5, -0.5, 1.5, 2.5, -3.0, 3.0, 2.0, 1.0,
        ];
        let mut three_row_input_values = all_input_values;
        three_row_input_values[12..].fill(f32::NAN);
        let input = bf16_buffer(&device, &three_row_input_values);
        let sentinel = bf16::from_f32(-777.0).to_f32();
        let output = bf16_buffer(&device, &[sentinel; 16]);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            shape,
            NUM_ACTIVE_ROWS,
            Buffers {
                input: &input,
                output: &output,
            },
        ));
        let replay = builder.build();

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        let expected_three = cpu_softmax_bf16_rows(&all_input_values[..12], 3, 4);
        let first = read_bf16_values(&output, 16);
        assert_close(&first[..12], &expected_three, 0.01);
        assert_eq!(&first[12..], &[sentinel; 4]);

        write_bf16_values(&input, &all_input_values);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 4))
            .wait();
        let expected_four = cpu_softmax_bf16_rows(&all_input_values, 4, 4);
        let full = read_bf16_values(&output, 16);
        assert_close(&full, &expected_four, 0.01);

        write_bf16_values(&input, &three_row_input_values);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        let shrunk = read_bf16_values(&output, 16);
        assert_close(&shrunk[..12], &expected_three, 0.01);
        assert_eq!(&shrunk[12..], &full[12..]);
    }

    fn create_softmax_kernel(config: Config) -> (Device, Kernel) {
        let device = Device::system_default();
        let kernel = Kernel::new(&device, config);
        (device, kernel)
    }

    fn cpu_softmax_bf16_rows(values: &[f32], num_rows: usize, num_values_per_row: usize) -> Vec<f32> {
        assert_eq!(values.len(), num_rows * num_values_per_row);
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks_exact(num_values_per_row) {
            let row: Vec<f32> = row.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (*value - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            output.extend(exps.into_iter().map(|value| bf16::from_f32(value / sum).to_f32()));
        }
        output
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        buffer.write_typed(0, &bits);
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "value mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
