use std::mem::size_of;

use super::top_k;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayArguments;

const DSPARK_MARKOV_SAMPLING_SOURCE: &str = include_str!("../metal/dspark_markov_sampling.metal");
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MapThreadBlockConstants {
    max_vocab_tokens: u32,
    required_threads: u32,
    simdgroup_width: u32,
    results_per_simdgroup: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MapKernelConstants {
    thread_block: MapThreadBlockConstants,
}

impl MapKernelConstants {
    fn current() -> Self {
        Self {
            thread_block: MapThreadBlockConstants {
                max_vocab_tokens: 64,
                required_threads: 128,
                simdgroup_width: 32,
                results_per_simdgroup: 4,
            },
        }
    }

    fn validate(self) {
        let thread_block = self.thread_block;
        assert!(thread_block.required_threads >= thread_block.max_vocab_tokens);
        assert!(
            thread_block
                .required_threads
                .is_multiple_of(thread_block.simdgroup_width)
        );
        assert!(
            thread_block
                .max_vocab_tokens
                .is_multiple_of(self.results_per_thread_block_iteration())
        );
    }

    fn num_simdgroups(self) -> u32 {
        self.thread_block.required_threads / self.thread_block.simdgroup_width
    }

    fn results_per_thread_block_iteration(self) -> u32 {
        self.num_simdgroups() * self.thread_block.results_per_simdgroup
    }

    fn partial_candidate_layout(self) -> top_k::PartialCandidateLayout {
        top_k::PartialCandidateLayout::new(self.thread_block.max_vocab_tokens)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MapConfig {
    pub vocab_size: u32,
    pub rank: u32,
    pub w1_group_size: u32,
    pub w1_bits: u32,
    pub w2_group_size: u32,
    pub w2_bits: u32,
    pub io_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
    pub confidence: ConfidenceConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfidenceConfig {
    pub hidden_dim: u32,
}

impl MapConfig {
    pub fn validate(self) {
        assert!(self.vocab_size > 0);
        assert!(self.rank > 0);
        validate_affine_layout(self.rank, self.w1_group_size, self.w1_bits);
        validate_affine_layout(self.rank, self.w2_group_size, self.w2_bits);
        validate_boundary_dtype("I/O", self.io_dtype);
        validate_boundary_dtype("scale/bias", self.scale_bias_dtype);
        self.confidence.validate(self.rank);
        let constants = MapKernelConstants::current();
        constants.validate();
        let _ = self.thread_block_memory_bytes(constants);
    }

    fn thread_block_memory_bytes(self, constants: MapKernelConstants) -> usize {
        (self.rank as usize)
            .checked_mul(self.io_dtype.item_size())
            .and_then(|bytes| {
                bytes.checked_add(
                    constants.thread_block.max_vocab_tokens as usize * (size_of::<f32>() + size_of::<i32>()),
                )
            })
            .expect("DSpark Markov threadblock memory size must fit usize")
    }

    fn weight_bytes(self, bits: u32) -> usize {
        (self.vocab_size as usize)
            .checked_mul(self.rank as usize)
            .and_then(|values| values.checked_mul(bits as usize))
            .expect("DSpark Markov weight bit length must fit usize")
            / 8
    }

    fn affine_param_bytes(self, group_size: u32) -> usize {
        (self.vocab_size as usize)
            .checked_mul((self.rank / group_size) as usize)
            .and_then(|values| values.checked_mul(self.scale_bias_dtype.item_size()))
            .expect("DSpark Markov affine parameter byte length must fit usize")
    }
}

impl ConfidenceConfig {
    pub fn validate(self, rank: u32) {
        assert!(self.hidden_dim > 0);
        let _ = self.input_dim(rank);
    }

    fn input_dim(self, rank: u32) -> u32 {
        self.hidden_dim
            .checked_add(rank)
            .expect("DSpark confidence input dimension must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MapShape {
    pub sampling: top_k::Shape,
    pub base_logits_row_offset: u32,
}

impl MapShape {
    pub fn validate(self, config: MapConfig) {
        config.validate();
        self.sampling.validate();
        assert_eq!(
            self.sampling.vocab_size, config.vocab_size,
            "DSpark Markov sampling vocabulary must match the kernel config"
        );
        let _ = self
            .base_logits_row_offset
            .checked_add(self.sampling.num_total_sampling_inputs)
            .expect("DSpark Markov base-logit row range must fit u32");
    }
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub input_token_ids: &'a Buffer,
    pub base_logits: &'a Buffer,
    pub w1_weight: &'a Buffer,
    pub w1_scales: &'a Buffer,
    pub w1_biases: &'a Buffer,
    pub w2_weight: &'a Buffer,
    pub w2_scales: &'a Buffer,
    pub w2_biases: &'a Buffer,
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub confidence: ConfidenceBuffers<'a>,
}

#[derive(Clone, Copy)]
pub struct ConfidenceBuffers<'a> {
    pub hidden: &'a Buffer,
    pub weight: &'a Buffer,
    pub bias: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MapCompute {
    config: MapConfig,
    constants: MapKernelConstants,
    kernel: CompiledKernel,
}

impl MapCompute {
    pub fn new(device: &Device, config: MapConfig) -> Self {
        config.validate();
        let constants = MapKernelConstants::current();
        constants.validate();
        let required_thread_block_memory_bytes = config.thread_block_memory_bytes(constants);
        let max_thread_block_memory_bytes = device.max_threadblock_memory_length();
        assert!(
            required_thread_block_memory_bytes <= max_thread_block_memory_bytes,
            "DSpark Markov requires {required_thread_block_memory_bytes} bytes of thread-block memory, but the device \
             supports {max_thread_block_memory_bytes}"
        );
        let kernel = CompiledKernel::new(device, &source(config, constants), "dspark_markov_top_k_map");
        let max_total_threads = kernel.max_total_threads_per_threadblock();
        let required_threads = constants.thread_block.required_threads;
        assert!(
            required_threads as usize <= max_total_threads,
            "DSpark Markov requires {required_threads} threads per threadblock, but the pipeline supports \
             {max_total_threads}"
        );
        let thread_execution_width = kernel.thread_execution_width();
        assert_eq!(
            thread_execution_width, constants.thread_block.simdgroup_width as usize,
            "DSpark Markov pipeline must use the configured SIMD width"
        );
        assert!(
            (required_threads as usize).is_multiple_of(thread_execution_width),
            "DSpark Markov threadblock size must contain complete SIMDgroups"
        );
        let static_thread_block_memory_bytes = kernel.static_threadblock_memory_length();
        assert!(
            static_thread_block_memory_bytes <= max_thread_block_memory_bytes,
            "DSpark Markov pipeline uses {static_thread_block_memory_bytes} bytes of thread-block memory, but the \
             device supports {max_thread_block_memory_bytes}"
        );
        Self {
            config,
            constants,
            kernel,
        }
    }

    pub fn invoke_replay<'a>(&'a self, shape: MapShape, buffers: MapBuffers<'a>) -> MapInvocation<'a> {
        MapInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }

    pub fn partial_candidate_layout(&self) -> top_k::PartialCandidateLayout {
        self.constants.partial_candidate_layout()
    }

    pub fn candidate_count(&self, shape: top_k::Shape) -> usize {
        shape.validate();
        assert_eq!(
            shape.vocab_size, self.config.vocab_size,
            "DSpark Markov sampling vocabulary must match the kernel config"
        );
        let layout = self.partial_candidate_layout();
        (shape.num_total_sampling_inputs as usize)
            .checked_mul(shape.vocab_size.div_ceil(layout.vocab_partition_size()) as usize)
            .and_then(|count| count.checked_mul(shape.top_k as usize))
            .expect("DSpark Markov partial candidate count must fit usize")
    }

    pub fn add_replay_arguments(
        &self,
        shape: top_k::Shape,
        num_active_sampling_inputs: u32,
        arguments: &mut ReplayArguments,
    ) {
        shape.validate();
        assert_eq!(
            shape.vocab_size, self.config.vocab_size,
            "DSpark Markov sampling vocabulary must match the kernel config"
        );
        assert!(
            num_active_sampling_inputs > 0 && num_active_sampling_inputs <= shape.num_total_sampling_inputs,
            "DSpark Markov active sampling inputs must fit the recorded capacity"
        );
        let num_partitions = shape
            .vocab_size
            .div_ceil(self.partial_candidate_layout().vocab_partition_size());
        let required_threads = self.constants.thread_block.required_threads;
        let num_threads_per_row = num_partitions
            .checked_mul(required_threads)
            .expect("DSpark Markov threads per request must fit u32");
        let num_active_threads = num_active_sampling_inputs
            .checked_mul(num_threads_per_row)
            .expect("DSpark Markov active thread count must fit u32");
        let num_total_threads = shape
            .num_total_sampling_inputs
            .checked_mul(num_threads_per_row)
            .expect("DSpark Markov total thread count must fit u32");
        assert!(num_active_threads <= num_total_threads);
        arguments.set_u32(top_k::MAP_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

pub struct MapInvocation<'a> {
    kernel: &'a MapCompute,
    shape: MapShape,
    buffers: MapBuffers<'a>,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let sampling = self.shape.sampling;
        let layout = self.kernel.partial_candidate_layout();
        let num_partitions = sampling.vocab_size.div_ceil(layout.vocab_partition_size());
        let required_threads = self.kernel.constants.thread_block.required_threads;
        let num_threads_per_row = num_partitions
            .checked_mul(required_threads)
            .expect("DSpark Markov threads per request must fit u32");
        let num_total_threads = sampling
            .num_total_sampling_inputs
            .checked_mul(num_threads_per_row)
            .expect("DSpark Markov total thread count must fit u32");

        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.input_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.base_logits, 0);
        recorder.set_buffer_read(2, self.buffers.w1_weight, 0);
        recorder.set_buffer_read(3, self.buffers.w1_scales, 0);
        recorder.set_buffer_read(4, self.buffers.w1_biases, 0);
        recorder.set_buffer_read(5, self.buffers.w2_weight, 0);
        recorder.set_buffer_read(6, self.buffers.w2_scales, 0);
        recorder.set_buffer_read(7, self.buffers.w2_biases, 0);
        recorder.set_buffer_write(8, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_write(9, self.buffers.tile_logits, 0);
        recorder.bind_u32(
            10,
            top_k::MAP_NUM_ACTIVE_THREADS_KEY,
            num_threads_per_row,
            num_total_threads,
        );
        recorder.set_u32(11, sampling.top_k);
        recorder.set_u32(12, num_partitions);
        recorder.set_u32(13, self.shape.base_logits_row_offset);
        recorder.set_buffer_read(14, self.buffers.confidence.hidden, 0);
        recorder.set_buffer_read(15, self.buffers.confidence.weight, 0);
        recorder.set_buffer_read(16, self.buffers.confidence.bias, 0);
        recorder.set_buffer_write(17, self.buffers.confidence.output, 0);
        recorder.dispatch_1d(num_total_threads as usize, required_threads as usize);
    }
}

impl MapInvocation<'_> {
    fn validate(&self) {
        let config = self.kernel.config;
        self.shape.validate(config);
        let sampling = self.shape.sampling;
        let candidate_count = self.kernel.candidate_count(sampling);
        let base_rows = self
            .shape
            .base_logits_row_offset
            .checked_add(sampling.num_total_sampling_inputs)
            .expect("DSpark Markov base-logit row count must fit u32");
        let base_logits_bytes = (base_rows as usize)
            .checked_mul(config.vocab_size as usize)
            .and_then(|values| values.checked_mul(config.io_dtype.item_size()))
            .expect("DSpark Markov base-logit byte length must fit usize");
        assert!(
            self.buffers.input_token_ids.len_bytes() >= sampling.num_total_sampling_inputs as usize * size_of::<i32>()
        );
        assert!(self.buffers.base_logits.len_bytes() >= base_logits_bytes);
        assert_eq!(self.buffers.w1_weight.len_bytes(), config.weight_bytes(config.w1_bits));
        assert_eq!(
            self.buffers.w1_scales.len_bytes(),
            config.affine_param_bytes(config.w1_group_size)
        );
        assert_eq!(
            self.buffers.w1_biases.len_bytes(),
            config.affine_param_bytes(config.w1_group_size)
        );
        assert_eq!(self.buffers.w2_weight.len_bytes(), config.weight_bytes(config.w2_bits));
        assert_eq!(
            self.buffers.w2_scales.len_bytes(),
            config.affine_param_bytes(config.w2_group_size)
        );
        assert_eq!(
            self.buffers.w2_biases.len_bytes(),
            config.affine_param_bytes(config.w2_group_size)
        );
        assert!(
            self.buffers.tile_token_ids.len_bytes()
                >= candidate_count
                    .checked_mul(size_of::<i32>())
                    .expect("DSpark Markov tile-token byte length must fit usize")
        );
        assert!(
            self.buffers.tile_logits.len_bytes()
                >= candidate_count
                    .checked_mul(size_of::<f32>())
                    .expect("DSpark Markov tile-logit byte length must fit usize")
        );
        let confidence = config.confidence;
        let buffers = self.buffers.confidence;
        let hidden_bytes = (base_rows as usize)
            .checked_mul(confidence.hidden_dim as usize)
            .and_then(|values| values.checked_mul(config.io_dtype.item_size()))
            .expect("DSpark confidence hidden byte length must fit usize");
        let output_bytes = (base_rows as usize)
            .checked_mul(size_of::<f32>())
            .expect("DSpark confidence output byte length must fit usize");
        assert!(buffers.hidden.len_bytes() >= hidden_bytes);
        assert_eq!(
            buffers.weight.len_bytes(),
            confidence.input_dim(config.rank) as usize * config.io_dtype.item_size()
        );
        assert_eq!(buffers.bias.len_bytes(), config.io_dtype.item_size());
        assert!(buffers.output.len_bytes() >= output_bytes);
    }
}

fn validate_affine_layout(rank: u32, group_size: u32, bits: u32) {
    assert!(matches!(group_size, 32 | 64 | 128));
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    assert_eq!(rank % group_size, 0);
    assert_eq!(
        rank.checked_mul(bits)
            .expect("DSpark Markov packed row width must fit u32")
            % 8,
        0
    );
}

fn validate_boundary_dtype(name: &str, dtype: Dtype) {
    match dtype {
        Dtype::Bfloat16 => {},
        Dtype::Float32 => todo!("F32 DSpark Markov {name} is not supported"),
        dtype => panic!("unsupported DSpark Markov {name} dtype {dtype:?}"),
    }
}

fn source(config: MapConfig, constants: MapKernelConstants) -> String {
    let thread_block = constants.thread_block;
    let values_per_simd_lane = config.rank.div_ceil(thread_block.simdgroup_width);
    let w2_lane_group_aligned = config.rank.is_multiple_of(thread_block.simdgroup_width)
        && config.w2_group_size.is_multiple_of(values_per_simd_lane);
    format!(
        "#define DSPARK_MARKOV_THREADBLOCK_SIZE {}u\n#define DSPARK_MARKOV_SIMD_WIDTH {}u\n#define \
         DSPARK_MARKOV_RESULTS_PER_SIMDGROUP {}u\n#define DSPARK_MARKOV_VOCAB_SIZE {}u\n#define DSPARK_MARKOV_RANK \
         {}u\n#define DSPARK_MARKOV_W1_GROUP_SIZE {}u\n#define DSPARK_MARKOV_W1_BITS {}u\n#define \
         DSPARK_MARKOV_W2_GROUP_SIZE {}u\n#define DSPARK_MARKOV_W2_BITS {}u\n#define \
         DSPARK_MARKOV_W2_LANE_GROUP_ALIGNED {}\n#define DSPARK_MARKOV_VOCAB_TILE_SIZE {}u\n#define \
         DSPARK_CONFIDENCE_HIDDEN_DIM {}u\n{DSPARK_MARKOV_SAMPLING_SOURCE}",
        thread_block.required_threads,
        thread_block.simdgroup_width,
        thread_block.results_per_simdgroup,
        config.vocab_size,
        config.rank,
        config.w1_group_size,
        config.w1_bits,
        config.w2_group_size,
        config.w2_bits,
        u8::from(w2_lane_group_aligned),
        thread_block.max_vocab_tokens,
        config.confidence.hidden_dim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MapConfig {
        MapConfig {
            vocab_size: 64,
            rank: 64,
            w1_group_size: 64,
            w1_bits: 4,
            w2_group_size: 64,
            w2_bits: 4,
            io_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
            confidence: ConfidenceConfig { hidden_dim: 32 },
        }
    }

    #[test]
    fn test_constants_define_map_task_and_partial_candidate_layout() {
        let constants = MapKernelConstants::current();
        assert_eq!(constants.thread_block.max_vocab_tokens, 64);
        assert_eq!(constants.thread_block.required_threads, 128);
        assert_eq!(constants.thread_block.simdgroup_width, 32);
        assert_eq!(constants.thread_block.results_per_simdgroup, 4);
        assert_eq!(constants.partial_candidate_layout().vocab_partition_size(), 64);
    }

    #[test]
    #[should_panic(expected = "F32 DSpark Markov I/O is not supported")]
    fn test_f32_workload_contract_is_explicit_future_work() {
        MapConfig {
            io_dtype: Dtype::Float32,
            ..config()
        }
        .validate();
    }
}
