use std::mem::size_of;

use super::top_k::TOP_K_TILE_NUM_ACTIVE_THREADS_KEY;
use super::top_k::TopKSampleShape;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayArguments;

const DSPARK_MARKOV_SAMPLING_SOURCE: &str = include_str!("../metal/dspark_markov_sampling.metal");
const DSPARK_MARKOV_SIMD_WIDTH: u32 = 32;
const DSPARK_MARKOV_RESULTS_PER_SIMDGROUP: u32 = 4;
const DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK: u32 = 128;
const DSPARK_MARKOV_VOCAB_TILE_SIZE: u32 = 64;
const DSPARK_MARKOV_NUM_SIMDGROUPS: u32 = DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK / DSPARK_MARKOV_SIMD_WIDTH;
const DSPARK_MARKOV_RESULTS_PER_WAVE: u32 = DSPARK_MARKOV_NUM_SIMDGROUPS * DSPARK_MARKOV_RESULTS_PER_SIMDGROUP;
const _: () = {
    assert!(DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK >= DSPARK_MARKOV_VOCAB_TILE_SIZE);
    assert!(DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK.is_multiple_of(DSPARK_MARKOV_SIMD_WIDTH));
    assert!(DSPARK_MARKOV_VOCAB_TILE_SIZE.is_multiple_of(DSPARK_MARKOV_RESULTS_PER_WAVE));
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkMarkovTopKMapConfig {
    pub vocab_size: u32,
    pub rank: u32,
    pub w1_group_size: u32,
    pub w1_bits: u32,
    pub w2_group_size: u32,
    pub w2_bits: u32,
    pub io_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
    pub confidence: DSparkConfidenceConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkConfidenceConfig {
    pub hidden_dim: u32,
}

impl DSparkMarkovTopKMapConfig {
    pub fn validate(self) {
        assert!(self.vocab_size > 0);
        assert!(self.rank > 0);
        validate_affine_layout(self.rank, self.w1_group_size, self.w1_bits);
        validate_affine_layout(self.rank, self.w2_group_size, self.w2_bits);
        validate_boundary_dtype("I/O", self.io_dtype);
        validate_boundary_dtype("scale/bias", self.scale_bias_dtype);
        self.confidence.validate(self.rank);
        let _ = self.threadblock_memory_bytes();
    }

    fn threadblock_memory_bytes(self) -> usize {
        (self.rank as usize)
            .checked_mul(self.io_dtype.item_size())
            .and_then(|bytes| {
                bytes.checked_add(DSPARK_MARKOV_VOCAB_TILE_SIZE as usize * (size_of::<f32>() + size_of::<i32>()))
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

impl DSparkConfidenceConfig {
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
pub struct DSparkMarkovTopKMapShape {
    pub sampling: TopKSampleShape,
    pub base_logits_row_offset: u32,
}

impl DSparkMarkovTopKMapShape {
    pub fn validate(self, config: DSparkMarkovTopKMapConfig) {
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
pub struct DSparkMarkovTopKMapBuffers<'a> {
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
    pub confidence: DSparkConfidenceBuffers<'a>,
}

#[derive(Clone, Copy)]
pub struct DSparkConfidenceBuffers<'a> {
    pub hidden: &'a Buffer,
    pub weight: &'a Buffer,
    pub bias: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct DSparkMarkovTopKMapKernel {
    config: DSparkMarkovTopKMapConfig,
    kernel: Kernel,
}

impl DSparkMarkovTopKMapKernel {
    pub fn new(device: &Device, config: DSparkMarkovTopKMapConfig) -> Self {
        config.validate();
        let required_threadblock_memory_bytes = config.threadblock_memory_bytes();
        let max_threadblock_memory_bytes = device.max_threadblock_memory_length();
        assert!(
            required_threadblock_memory_bytes <= max_threadblock_memory_bytes,
            "DSpark Markov requires {required_threadblock_memory_bytes} bytes of threadblock memory, but the device \
             supports {max_threadblock_memory_bytes}"
        );
        let kernel = Kernel::new(device, &source(config), "dspark_markov_top_k_map");
        let max_total_threads = kernel.max_total_threads_per_threadblock();
        assert!(
            DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK as usize <= max_total_threads,
            "DSpark Markov requires {DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK} threads per threadblock, but the \
             pipeline supports {max_total_threads}"
        );
        let thread_execution_width = kernel.thread_execution_width();
        assert_eq!(
            thread_execution_width, DSPARK_MARKOV_SIMD_WIDTH as usize,
            "DSpark Markov pipeline must use the configured SIMD width"
        );
        assert!(
            (DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK as usize).is_multiple_of(thread_execution_width),
            "DSpark Markov threadblock size must contain complete SIMDgroups"
        );
        let static_threadblock_memory_bytes = kernel.static_threadblock_memory_length();
        assert!(
            static_threadblock_memory_bytes <= max_threadblock_memory_bytes,
            "DSpark Markov pipeline uses {static_threadblock_memory_bytes} bytes of threadblock memory, but the \
             device supports {max_threadblock_memory_bytes}"
        );
        Self { config, kernel }
    }

    pub fn invoke_replay<'a>(
        &'a self,
        shape: DSparkMarkovTopKMapShape,
        buffers: DSparkMarkovTopKMapBuffers<'a>,
    ) -> DSparkMarkovTopKMapInvocation<'a> {
        DSparkMarkovTopKMapInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }

    pub fn vocab_tile_size(&self) -> u32 {
        DSPARK_MARKOV_VOCAB_TILE_SIZE
    }

    pub fn candidate_count(&self, shape: TopKSampleShape) -> usize {
        shape.validate();
        assert_eq!(
            shape.vocab_size, self.config.vocab_size,
            "DSpark Markov sampling vocabulary must match the kernel config"
        );
        (shape.num_total_sampling_inputs as usize)
            .checked_mul(shape.vocab_size.div_ceil(DSPARK_MARKOV_VOCAB_TILE_SIZE) as usize)
            .and_then(|count| count.checked_mul(shape.top_k as usize))
            .expect("DSpark Markov tile candidate count must fit usize")
    }

    pub fn add_replay_arguments(
        &self,
        shape: TopKSampleShape,
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
        if shape.num_total_sampling_inputs <= 1 {
            return;
        }
        let num_tiles = shape.vocab_size.div_ceil(DSPARK_MARKOV_VOCAB_TILE_SIZE);
        let num_threads_per_row = num_tiles
            .checked_mul(DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK)
            .expect("DSpark Markov threads per request must fit u32");
        let num_active_threads = num_active_sampling_inputs
            .checked_mul(num_threads_per_row)
            .expect("DSpark Markov active thread count must fit u32");
        let num_total_threads = shape
            .num_total_sampling_inputs
            .checked_mul(num_threads_per_row)
            .expect("DSpark Markov total thread count must fit u32");
        assert!(num_active_threads <= num_total_threads);
        arguments.set_u32(TOP_K_TILE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

pub struct DSparkMarkovTopKMapInvocation<'a> {
    kernel: &'a DSparkMarkovTopKMapKernel,
    shape: DSparkMarkovTopKMapShape,
    buffers: DSparkMarkovTopKMapBuffers<'a>,
}

impl Operator for DSparkMarkovTopKMapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let sampling = self.shape.sampling;
        let num_tiles = sampling.vocab_size.div_ceil(DSPARK_MARKOV_VOCAB_TILE_SIZE);
        let num_threads_per_row = num_tiles
            .checked_mul(DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK)
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
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(10, num_total_threads);
        } else {
            recorder.bind_u32(
                10,
                TOP_K_TILE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.set_u32(11, sampling.top_k);
        recorder.set_u32(12, num_tiles);
        recorder.set_u32(13, self.shape.base_logits_row_offset);
        recorder.set_buffer_read(14, self.buffers.confidence.hidden, 0);
        recorder.set_buffer_read(15, self.buffers.confidence.weight, 0);
        recorder.set_buffer_read(16, self.buffers.confidence.bias, 0);
        recorder.set_buffer_write(17, self.buffers.confidence.output, 0);
        recorder.dispatch_1d(
            num_total_threads as usize,
            DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

impl DSparkMarkovTopKMapInvocation<'_> {
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

fn source(config: DSparkMarkovTopKMapConfig) -> String {
    let values_per_simd_lane = config.rank.div_ceil(DSPARK_MARKOV_SIMD_WIDTH);
    let w2_lane_group_aligned = config.rank.is_multiple_of(DSPARK_MARKOV_SIMD_WIDTH)
        && config.w2_group_size.is_multiple_of(values_per_simd_lane);
    format!(
        "#define DSPARK_MARKOV_THREADBLOCK_SIZE {}u\n#define DSPARK_MARKOV_SIMD_WIDTH {}u\n#define \
         DSPARK_MARKOV_RESULTS_PER_SIMDGROUP {}u\n#define DSPARK_MARKOV_VOCAB_SIZE {}u\n#define DSPARK_MARKOV_RANK \
         {}u\n#define DSPARK_MARKOV_W1_GROUP_SIZE {}u\n#define DSPARK_MARKOV_W1_BITS {}u\n#define \
         DSPARK_MARKOV_W2_GROUP_SIZE {}u\n#define DSPARK_MARKOV_W2_BITS {}u\n#define \
         DSPARK_MARKOV_W2_LANE_GROUP_ALIGNED {}\n#define DSPARK_MARKOV_VOCAB_TILE_SIZE {}u\n#define \
         DSPARK_CONFIDENCE_HIDDEN_DIM {}u\n{DSPARK_MARKOV_SAMPLING_SOURCE}",
        DSPARK_MARKOV_NUM_THREADS_PER_THREADBLOCK,
        DSPARK_MARKOV_SIMD_WIDTH,
        DSPARK_MARKOV_RESULTS_PER_SIMDGROUP,
        config.vocab_size,
        config.rank,
        config.w1_group_size,
        config.w1_bits,
        config.w2_group_size,
        config.w2_bits,
        u8::from(w2_lane_group_aligned),
        DSPARK_MARKOV_VOCAB_TILE_SIZE,
        config.confidence.hidden_dim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DSparkMarkovTopKMapConfig {
        DSparkMarkovTopKMapConfig {
            vocab_size: 64,
            rank: 64,
            w1_group_size: 64,
            w1_bits: 4,
            w2_group_size: 64,
            w2_bits: 4,
            io_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
            confidence: DSparkConfidenceConfig { hidden_dim: 32 },
        }
    }

    #[test]
    #[should_panic(expected = "F32 DSpark Markov I/O is not supported")]
    fn test_f32_workload_contract_is_explicit_future_work() {
        DSparkMarkovTopKMapConfig {
            io_dtype: Dtype::Float32,
            ..config()
        }
        .validate();
    }
}
