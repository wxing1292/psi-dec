use super::MAX_TOP_K;
use super::SAMPLING_NUM_THREADS_PER_THREADBLOCK;
use super::SAMPLING_SOURCE;
use super::checked_bytes;
use super::checked_num_threads;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;

const TOP_K_REDUCTION_LIMIT: u32 = 32;
const TOP_K_VOCAB_TILE_SIZE: u32 = 256;
pub const TOP_K_TILE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.tile_num_active_threads");
const TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.merge_num_active_threads");

#[derive(Clone, Copy, Debug)]
pub struct TopKSampleShape {
    pub num_total_sampling_inputs: u32,
    pub vocab_size: u32,
    pub top_k: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKSamplingOperation {
    Sample,
    WriteDistribution,
    SampleAndWriteDistribution,
}

impl TopKSampleShape {
    pub fn validate(self) {
        assert!(self.num_total_sampling_inputs > 0);
        assert!(self.vocab_size > 0);
        assert!(self.top_k > 0);
        assert!(self.top_k <= self.vocab_size);
        assert!(self.top_k <= MAX_TOP_K);
        assert!(
            i32::try_from(self.vocab_size).is_ok(),
            "Metal top-k vocab index must fit i32"
        );
        checked_product(
            "Metal top-k logits element count",
            &[self.num_total_sampling_inputs as usize, self.vocab_size as usize],
        );
        checked_product(
            "Metal top-k tile candidate count",
            &[
                self.num_total_sampling_inputs as usize,
                self.vocab_size.div_ceil(vocab_tile_size()) as usize,
                self.top_k as usize,
            ],
        );
    }
}

fn vocab_tile_size() -> u32 {
    TOP_K_VOCAB_TILE_SIZE
}

fn num_tiles(shape: TopKSampleShape, vocab_tile_size: u32) -> u32 {
    shape.validate();
    assert!(vocab_tile_size > 0);
    shape.vocab_size.div_ceil(vocab_tile_size)
}

fn tile_top_k(shape: TopKSampleShape) -> u32 {
    shape.validate();
    shape.top_k
}

fn tile_count(shape: TopKSampleShape, vocab_tile_size: u32) -> usize {
    checked_product(
        "Metal top-k tile candidate count",
        &[
            shape.num_total_sampling_inputs as usize,
            num_tiles(shape, vocab_tile_size) as usize,
            tile_top_k(shape) as usize,
        ],
    )
}

#[derive(Clone, Copy)]
pub struct TopKTileBuffers<'a> {
    pub logits: &'a Buffer,
    pub logits_offset_bytes: usize,
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct TopKSampleBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub token_ids: &'a Buffer,
    pub token_probs: &'a Buffer,
    pub runtime_params: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct TopKWriteDistributionBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub distribution_token_ids: &'a Buffer,
    pub distribution_probs: &'a Buffer,
    pub runtime_params: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

#[derive(Clone, Copy)]
pub struct TopKSampleAndWriteDistributionBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
    pub distribution_token_ids: &'a Buffer,
    pub distribution_probs: &'a Buffer,
    pub runtime_params: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

fn assert_tile_buffers_fit(shape: TopKSampleShape, buffers: TopKTileBuffers<'_>, logits_item_size: usize) {
    let logits_bytes = checked_product(
        "Metal top-k logits byte length",
        &[
            shape.num_total_sampling_inputs as usize,
            shape.vocab_size as usize,
            logits_item_size,
        ],
    );
    assert!(
        buffers.logits.len_bytes()
            >= buffers
                .logits_offset_bytes
                .checked_add(logits_bytes)
                .expect("Metal top-k logits region must fit usize"),
        "top-k logits buffer is too short for total sampling inputs"
    );
    let candidates = tile_count(shape, vocab_tile_size());
    assert!(
        buffers.tile_token_ids.len_bytes() >= checked_bytes("Metal top-k tile token", candidates, size_of::<i32>()),
        "top-k tile token buffer is too short"
    );
    assert!(
        buffers.tile_logits.len_bytes() >= checked_bytes("Metal top-k tile logit", candidates, size_of::<f32>()),
        "top-k tile logits buffer is too short"
    );
}

fn assert_merge_inputs_fit(
    shape: TopKSampleShape,
    tile_token_ids: &Buffer,
    tile_logits: &Buffer,
    runtime_params: &Buffer,
    vocab_tile_size: u32,
) {
    let candidates = tile_count(shape, vocab_tile_size);
    assert!(
        tile_token_ids.len_bytes() >= checked_bytes("Metal top-k merge token", candidates, size_of::<i32>()),
        "top-k tile token buffer is too short"
    );
    assert!(
        tile_logits.len_bytes() >= checked_bytes("Metal top-k merge logit", candidates, size_of::<f32>()),
        "top-k tile logits buffer is too short"
    );
    assert!(
        runtime_params.len_bytes()
            >= checked_product(
                "Metal top-k runtime parameter byte length",
                &[shape.num_total_sampling_inputs as usize, 6, size_of::<u32>()],
            ),
        "top-k runtime parameter buffer is too short"
    );
}

pub struct TopKTileKernels {
    f32_reduction: Kernel,
    f32_bitonic: Kernel,
    bf16_reduction: Kernel,
    bf16_bitonic: Kernel,
}

impl TopKTileKernels {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            f32_reduction: Kernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles"),
            f32_bitonic: Kernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bitonic"),
            bf16_reduction: Kernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bf16"),
            bf16_bitonic: Kernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bf16_bitonic"),
        }
    }

    pub fn invoke_replay<'a>(
        &'a self,
        shape: TopKSampleShape,
        logits_dtype: Dtype,
        operation: TopKSamplingOperation,
        buffers: TopKTileBuffers<'a>,
    ) -> TopKTileInvocation<'a> {
        shape.validate();
        let kind = selected_top_k_tile_kernel(shape, operation);
        let (kernel, logits_item_size) = match (logits_dtype, kind) {
            (Dtype::Float32, TopKTileKernelKind::Reduction) => (&self.f32_reduction, size_of::<f32>()),
            (Dtype::Float32, TopKTileKernelKind::Bitonic) => (&self.f32_bitonic, size_of::<f32>()),
            (Dtype::Bfloat16, TopKTileKernelKind::Reduction) => (&self.bf16_reduction, size_of::<u16>()),
            (Dtype::Bfloat16, TopKTileKernelKind::Bitonic) => (&self.bf16_bitonic, size_of::<u16>()),
            (dtype, _) => panic!("unsupported top-k logits dtype {dtype:?}"),
        };
        TopKTileInvocation {
            kernel,
            logits_item_size,
            shape,
            buffers,
        }
    }

    pub fn candidate_count(&self, shape: TopKSampleShape) -> usize {
        tile_count(shape, vocab_tile_size())
    }

    pub fn add_replay_arguments(
        &self,
        shape: TopKSampleShape,
        num_active_sampling_inputs: u32,
        arguments: &mut ReplayArguments,
    ) {
        shape.validate();
        assert!(
            num_active_sampling_inputs > 0 && num_active_sampling_inputs <= shape.num_total_sampling_inputs,
            "top-k active sampling inputs must fit the recorded capacity"
        );
        if shape.num_total_sampling_inputs <= 1 {
            return;
        }
        let num_tiles = num_tiles(shape, vocab_tile_size());
        let num_threads_per_row = checked_num_threads(num_tiles, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
        let num_active_threads = checked_num_threads(num_active_sampling_inputs, num_threads_per_row);
        let num_total_threads = checked_num_threads(shape.num_total_sampling_inputs, num_threads_per_row);
        assert!(num_active_threads <= num_total_threads);
        arguments.set_u32(TOP_K_TILE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TopKTileKernelKind {
    Reduction,
    Bitonic,
}

fn selected_top_k_tile_kernel(shape: TopKSampleShape, operation: TopKSamplingOperation) -> TopKTileKernelKind {
    shape.validate();
    match operation {
        TopKSamplingOperation::Sample if shape.top_k <= TOP_K_REDUCTION_LIMIT => TopKTileKernelKind::Reduction,
        TopKSamplingOperation::Sample
        | TopKSamplingOperation::WriteDistribution
        | TopKSamplingOperation::SampleAndWriteDistribution => TopKTileKernelKind::Bitonic,
    }
}

pub struct TopKTileInvocation<'a> {
    kernel: &'a Kernel,
    logits_item_size: usize,
    shape: TopKSampleShape,
    buffers: TopKTileBuffers<'a>,
}

impl Operator for TopKTileInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_tile_buffers_fit(self.shape, self.buffers, self.logits_item_size);
        let vocab_tile_size = vocab_tile_size();
        let num_tiles = num_tiles(self.shape, vocab_tile_size);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.logits, self.buffers.logits_offset_bytes);
        recorder.set_buffer_write(1, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_write(2, self.buffers.tile_logits, 0);
        recorder.set_u32(4, self.shape.vocab_size);
        recorder.set_u32(5, self.shape.top_k);
        recorder.set_u32(6, vocab_tile_size);
        recorder.set_u32(7, num_tiles);
        let num_threads_per_row = checked_num_threads(num_tiles, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(3, num_total_threads);
        } else {
            recorder.bind_u32(
                3,
                TOP_K_TILE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(
            num_total_threads as usize,
            SAMPLING_NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

pub struct TopKMergeKernels {
    sample: Kernel,
    write_distribution: Kernel,
    sample_and_write_distribution: Kernel,
}

impl TopKMergeKernels {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            sample: Kernel::new(device, SAMPLING_SOURCE, "top_k_sample_tiles"),
            write_distribution: Kernel::new(device, SAMPLING_SOURCE, "top_k_write_distribution_tiles"),
            sample_and_write_distribution: Kernel::new(
                device,
                SAMPLING_SOURCE,
                "top_k_sample_and_write_distribution_tiles",
            ),
        }
    }

    pub fn invoke_sample<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleBuffers<'a>,
    ) -> TopKSampleInvocation<'a> {
        TopKSampleInvocation {
            kernel: &self.sample,
            shape,
            buffers,
        }
    }

    pub fn invoke_write_distribution<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKWriteDistributionBuffers<'a>,
    ) -> TopKWriteDistributionInvocation<'a> {
        TopKWriteDistributionInvocation {
            kernel: &self.write_distribution,
            shape,
            buffers,
        }
    }

    pub fn invoke_sample_and_write_distribution<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleAndWriteDistributionBuffers<'a>,
    ) -> TopKSampleAndWriteDistributionInvocation<'a> {
        self.invoke_sample_and_write_distribution_with_vocab_tile_size(shape, buffers, vocab_tile_size())
    }

    pub fn invoke_sample_and_write_distribution_with_vocab_tile_size<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleAndWriteDistributionBuffers<'a>,
        vocab_tile_size: u32,
    ) -> TopKSampleAndWriteDistributionInvocation<'a> {
        assert!(vocab_tile_size > 0);
        TopKSampleAndWriteDistributionInvocation {
            kernel: &self.sample_and_write_distribution,
            shape,
            buffers,
            vocab_tile_size,
        }
    }

    pub fn add_replay_arguments(
        &self,
        shape: TopKSampleShape,
        num_active_sampling_inputs: u32,
        arguments: &mut ReplayArguments,
    ) {
        add_top_k_merge_replay_argument(shape, num_active_sampling_inputs, arguments);
    }
}

pub struct TopKSampleInvocation<'a> {
    kernel: &'a Kernel,
    shape: TopKSampleShape,
    buffers: TopKSampleBuffers<'a>,
}

impl Operator for TopKSampleInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_merge_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            vocab_tile_size(),
        );
        assert!(
            self.buffers.token_ids.len_bytes()
                >= checked_bytes(
                    "Metal sampled token",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<i32>(),
                ),
            "top-k sampled-token buffer is too short"
        );
        assert!(
            self.buffers.token_probs.len_bytes()
                >= checked_bytes(
                    "Metal sampled probability",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<f32>(),
                ),
            "top-k sampled-probability buffer is too short"
        );
        let vocab_tile_size = vocab_tile_size();
        let num_tiles = num_tiles(self.shape, vocab_tile_size);
        let tile_top_k = tile_top_k(self.shape);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.token_probs, 0);
        recorder.set_buffer_read(4, self.buffers.runtime_params, 0);
        recorder.set_u32(6, self.shape.top_k);
        recorder.set_u32(7, num_tiles);
        recorder.set_u32(8, tile_top_k);
        recorder.set_u32(9, vocab_tile_size);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(5, num_total_threads);
        } else {
            recorder.bind_u32(
                5,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(
            num_total_threads as usize,
            SAMPLING_NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

pub struct TopKWriteDistributionInvocation<'a> {
    kernel: &'a Kernel,
    shape: TopKSampleShape,
    buffers: TopKWriteDistributionBuffers<'a>,
}

impl Operator for TopKWriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_merge_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            vocab_tile_size(),
        );
        assert!(
            self.buffers.max_k >= self.shape.top_k,
            "top-k write-distribution slots must cover active top_k"
        );
        assert!(
            self.buffers.num_output_distributions > 0,
            "top-k write-distribution output requires distributions"
        );
        let output_elements = checked_product(
            "Metal top-k write-distribution output element count",
            &[
                self.buffers.num_output_distributions as usize,
                self.buffers.max_k as usize,
            ],
        );
        let vocab_tile_size = vocab_tile_size();
        let num_tiles = num_tiles(self.shape, vocab_tile_size);
        let tile_top_k = tile_top_k(self.shape);
        assert!(
            self.buffers.output_distribution_indices.len_bytes()
                >= checked_bytes(
                    "Metal write-distribution output index",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k write-distribution output-index buffer too short"
        );
        assert!(
            self.buffers.distribution_token_ids.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<i32>())
                    .expect("Metal top-k write-distribution token bytes must fit usize"),
            "top-k write-distribution token buffer too short for declared outputs"
        );
        assert!(
            self.buffers.distribution_probs.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<f32>())
                    .expect("Metal top-k write-distribution probability bytes must fit usize"),
            "top-k write-distribution prob buffer too short for declared outputs"
        );
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.distribution_token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.distribution_probs, 0);
        recorder.set_buffer_read(4, self.buffers.runtime_params, 0);
        recorder.set_buffer_read(5, self.buffers.output_distribution_indices, 0);
        recorder.set_u32(7, self.shape.top_k);
        recorder.set_u32(8, num_tiles);
        recorder.set_u32(9, tile_top_k);
        recorder.set_u32(10, vocab_tile_size);
        recorder.set_u32(11, self.buffers.max_k);
        recorder.set_u32(12, self.buffers.num_output_distributions);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(6, num_total_threads);
        } else {
            recorder.bind_u32(
                6,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(
            num_total_threads as usize,
            SAMPLING_NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

pub struct TopKSampleAndWriteDistributionInvocation<'a> {
    kernel: &'a Kernel,
    shape: TopKSampleShape,
    buffers: TopKSampleAndWriteDistributionBuffers<'a>,
    vocab_tile_size: u32,
}

impl Operator for TopKSampleAndWriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_merge_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            self.vocab_tile_size,
        );
        assert!(
            self.buffers.sampled_token_ids.len_bytes()
                >= checked_bytes(
                    "Metal sampled token",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<i32>(),
                ),
            "top-k sample-and-write-distribution sampled-token buffer is too short"
        );
        assert!(
            self.buffers.sampled_token_probs.len_bytes()
                >= checked_bytes(
                    "Metal sampled probability",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<f32>(),
                ),
            "top-k sample-and-write-distribution sampled-probability buffer is too short"
        );
        assert!(
            self.buffers.max_k >= self.shape.top_k,
            "top-k sample-and-write-distribution slots must cover batch top_k"
        );
        assert!(
            self.buffers.num_output_distributions > 0,
            "top-k sample-and-write-distribution output requires distributions"
        );
        let output_elements = checked_product(
            "Metal top-k sample-and-write-distribution output element count",
            &[
                self.buffers.num_output_distributions as usize,
                self.buffers.max_k as usize,
            ],
        );
        assert!(
            self.buffers.output_distribution_indices.len_bytes()
                >= checked_bytes(
                    "Metal write-distribution output index",
                    self.shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k sample-and-write-distribution output-index buffer too short"
        );
        assert!(
            self.buffers.distribution_token_ids.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<i32>())
                    .expect("Metal top-k sample-and-write-distribution token bytes must fit usize"),
            "top-k sample-and-write-distribution token buffer too short"
        );
        assert!(
            self.buffers.distribution_probs.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<f32>())
                    .expect("Metal top-k sample-and-write-distribution probability bytes must fit usize"),
            "top-k sample-and-write-distribution probability buffer too short"
        );
        let vocab_tile_size = self.vocab_tile_size;
        let num_tiles = num_tiles(self.shape, vocab_tile_size);
        let tile_top_k = tile_top_k(self.shape);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.sampled_token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.sampled_token_probs, 0);
        recorder.set_buffer_write(4, self.buffers.distribution_token_ids, 0);
        recorder.set_buffer_write(5, self.buffers.distribution_probs, 0);
        recorder.set_buffer_read(6, self.buffers.runtime_params, 0);
        recorder.set_buffer_read(7, self.buffers.output_distribution_indices, 0);
        recorder.set_u32(9, self.shape.top_k);
        recorder.set_u32(10, num_tiles);
        recorder.set_u32(11, tile_top_k);
        recorder.set_u32(12, vocab_tile_size);
        recorder.set_u32(13, self.buffers.max_k);
        recorder.set_u32(14, self.buffers.num_output_distributions);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(8, num_total_threads);
        } else {
            recorder.bind_u32(
                8,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(
            num_total_threads as usize,
            SAMPLING_NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

fn add_top_k_merge_replay_argument(
    shape: TopKSampleShape,
    num_active_sampling_inputs: u32,
    arguments: &mut ReplayArguments,
) {
    shape.validate();
    assert!(
        num_active_sampling_inputs > 0 && num_active_sampling_inputs <= shape.num_total_sampling_inputs,
        "top-k active sampling inputs must fit the recorded capacity"
    );
    if shape.num_total_sampling_inputs <= 1 {
        return;
    }
    let num_active_threads = checked_num_threads(num_active_sampling_inputs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
    let num_total_threads = checked_num_threads(shape.num_total_sampling_inputs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
    assert!(num_active_threads <= num_total_threads);
    arguments.set_u32(TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
}

#[cfg(test)]
#[path = "top_k_test.rs"]
mod tests;
