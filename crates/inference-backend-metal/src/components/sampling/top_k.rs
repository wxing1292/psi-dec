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
pub(super) const TOP_K_TILE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
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
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_tile_buffers_fit(self.shape, self.buffers, self.logits_item_size);
        let vocab_tile_size = vocab_tile_size();
        let num_tiles = num_tiles(self.shape, vocab_tile_size);
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.logits, self.buffers.logits_offset_bytes);
        builder.set_buffer_write(1, self.buffers.tile_token_ids, 0);
        builder.set_buffer_write(2, self.buffers.tile_logits, 0);
        builder.set_u32(4, self.shape.vocab_size);
        builder.set_u32(5, self.shape.top_k);
        builder.set_u32(6, vocab_tile_size);
        builder.set_u32(7, num_tiles);
        let num_threads_per_row = checked_num_threads(num_tiles, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            builder.set_u32(3, num_total_threads);
        } else {
            builder.bind_u32(
                3,
                TOP_K_TILE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        builder.dispatch_1d(
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
    fn record(self, builder: &CommandRecorder<'_>) {
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
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        builder.set_buffer_read(1, self.buffers.tile_logits, 0);
        builder.set_buffer_write(2, self.buffers.token_ids, 0);
        builder.set_buffer_write(3, self.buffers.token_probs, 0);
        builder.set_buffer_read(4, self.buffers.runtime_params, 0);
        builder.set_u32(6, self.shape.top_k);
        builder.set_u32(7, num_tiles);
        builder.set_u32(8, tile_top_k);
        builder.set_u32(9, vocab_tile_size);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            builder.set_u32(5, num_total_threads);
        } else {
            builder.bind_u32(
                5,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        builder.dispatch_1d(
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
    fn record(self, builder: &CommandRecorder<'_>) {
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
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        builder.set_buffer_read(1, self.buffers.tile_logits, 0);
        builder.set_buffer_write(2, self.buffers.distribution_token_ids, 0);
        builder.set_buffer_write(3, self.buffers.distribution_probs, 0);
        builder.set_buffer_read(4, self.buffers.runtime_params, 0);
        builder.set_buffer_read(5, self.buffers.output_distribution_indices, 0);
        builder.set_u32(7, self.shape.top_k);
        builder.set_u32(8, num_tiles);
        builder.set_u32(9, tile_top_k);
        builder.set_u32(10, vocab_tile_size);
        builder.set_u32(11, self.buffers.max_k);
        builder.set_u32(12, self.buffers.num_output_distributions);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            builder.set_u32(6, num_total_threads);
        } else {
            builder.bind_u32(
                6,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        builder.dispatch_1d(
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
    fn record(self, builder: &CommandRecorder<'_>) {
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
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        builder.set_buffer_read(1, self.buffers.tile_logits, 0);
        builder.set_buffer_write(2, self.buffers.sampled_token_ids, 0);
        builder.set_buffer_write(3, self.buffers.sampled_token_probs, 0);
        builder.set_buffer_write(4, self.buffers.distribution_token_ids, 0);
        builder.set_buffer_write(5, self.buffers.distribution_probs, 0);
        builder.set_buffer_read(6, self.buffers.runtime_params, 0);
        builder.set_buffer_read(7, self.buffers.output_distribution_indices, 0);
        builder.set_u32(9, self.shape.top_k);
        builder.set_u32(10, num_tiles);
        builder.set_u32(11, tile_top_k);
        builder.set_u32(12, vocab_tile_size);
        builder.set_u32(13, self.buffers.max_k);
        builder.set_u32(14, self.buffers.num_output_distributions);
        let num_threads_per_row = SAMPLING_NUM_THREADS_PER_THREADBLOCK;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            builder.set_u32(8, num_total_threads);
        } else {
            builder.bind_u32(
                8,
                TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        builder.dispatch_1d(
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
mod tests {
    use half::bf16;
    use inference_executor_core::sampling::SamplerConfig;
    use inference_executor_core::sampling::SamplingDomain;
    use inference_executor_core::sampling::reference::rejection_sample_reference;
    use inference_executor_core::sampling::reference::sparse_sample_row_reference;
    use inference_executor_core::sampling::reference::sparse_sample_row_with_domain_reference;

    use super::super::SAMPLING_NUM_THREADS_PER_THREADBLOCK;
    use super::super::checked_num_threads;
    use super::super::rejection::REJECTION_NUM_ACTIVE_THREADS_KEY;
    use super::super::rejection::REJECTION_NUM_DRAFT_DISTRIBUTIONS_KEY;
    use super::super::rejection::REJECTION_NUM_TARGET_DISTRIBUTIONS_KEY;
    use super::TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY;
    use super::TOP_K_TILE_NUM_ACTIVE_THREADS_KEY;
    use super::TOP_K_VOCAB_TILE_SIZE;
    use crate::components::SparseRejectionSampleBuffers;
    use crate::components::SparseRejectionSampleKernel;
    use crate::components::SparseRejectionSampleShape;
    use crate::components::TopKMergeKernels;
    use crate::components::TopKSampleAndWriteDistributionBuffers;
    use crate::components::TopKSampleBuffers;
    use crate::components::TopKSampleShape;
    use crate::components::TopKSamplingOperation;
    use crate::components::TopKTileBuffers;
    use crate::components::TopKTileKernels;
    use crate::components::TopKWriteDistributionBuffers;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    fn top_k_replay_arguments(
        shape: TopKSampleShape,
        num_active_sampling_inputs: u32,
        include_tile: bool,
        include_merge: bool,
    ) -> ReplayArguments {
        assert!(num_active_sampling_inputs > 0);
        assert!(num_active_sampling_inputs <= shape.num_total_sampling_inputs);
        let mut arguments = ReplayArguments::new();
        if shape.num_total_sampling_inputs == 1 {
            return arguments;
        }
        if include_tile {
            let num_tiles = shape.vocab_size.div_ceil(TOP_K_VOCAB_TILE_SIZE);
            let num_threads_per_row = checked_num_threads(num_tiles, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
            let num_active_threads = checked_num_threads(num_active_sampling_inputs, num_threads_per_row);
            let num_total_threads = checked_num_threads(shape.num_total_sampling_inputs, num_threads_per_row);
            assert!(num_active_threads <= num_total_threads);
            arguments.set_u32(TOP_K_TILE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
        }
        if include_merge {
            let num_active_threads =
                checked_num_threads(num_active_sampling_inputs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
            let num_total_threads =
                checked_num_threads(shape.num_total_sampling_inputs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
            assert!(num_active_threads <= num_total_threads);
            arguments.set_u32(TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
        }
        arguments
    }

    fn rejection_replay_arguments(
        shape: SparseRejectionSampleShape,
        num_active_reqs: u32,
        num_active_target_distributions: u32,
        num_active_draft_distributions: u32,
    ) -> ReplayArguments {
        assert!(num_active_reqs > 0 && num_active_reqs <= shape.num_total_reqs);
        assert!(num_active_target_distributions <= shape.num_total_target_distributions);
        assert!(num_active_draft_distributions <= shape.num_total_draft_distributions);
        let expected_num_target_distributions = num_active_draft_distributions
            .checked_add(num_active_reqs)
            .expect("sparse rejection target-distribution count must fit u32");
        assert_eq!(num_active_target_distributions, expected_num_target_distributions);
        let mut arguments = ReplayArguments::new();
        if shape.num_total_reqs > 1 {
            let num_active_threads = checked_num_threads(num_active_reqs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
            let num_total_threads = checked_num_threads(shape.num_total_reqs, SAMPLING_NUM_THREADS_PER_THREADBLOCK);
            assert!(num_active_threads <= num_total_threads);
            arguments.set_u32(REJECTION_NUM_ACTIVE_THREADS_KEY, num_active_threads);
        }
        if shape.num_total_target_distributions > 1 {
            arguments.set_u32(REJECTION_NUM_TARGET_DISTRIBUTIONS_KEY, num_active_target_distributions);
        }
        if shape.num_total_draft_distributions > 0 {
            arguments.set_u32(REJECTION_NUM_DRAFT_DISTRIBUTIONS_KEY, num_active_draft_distributions);
        }
        arguments
    }

    fn sampling_runtime_params(
        device: &Device,
        rows: u32,
        temperature: f32,
        top_p: f32,
        seed: u32,
        top_k: u32,
    ) -> Buffer {
        let params = Buffer::new_zeroed(device, rows as usize * 6 * size_of::<u32>());
        let config = SamplerConfig {
            temperature,
            top_k: top_k as usize,
            top_p,
            seed,
        };
        for row in 0..rows as usize {
            write_sampling_runtime_params(&params, row, &config, row as u32, SamplingDomain::Target);
        }
        params
    }

    fn write_sampling_runtime_params(
        params: &Buffer,
        row: usize,
        config: &SamplerConfig,
        sample_position: u32,
        domain: SamplingDomain,
    ) {
        let offset = row * 6;
        params.write_typed(offset, &[config.temperature, config.top_p]);
        params.write_typed(
            offset + 2,
            &[config.seed(), sample_position, config.top_k as u32, domain as u32],
        );
    }

    #[test]
    fn test_tile_kernel_selection() {
        let reduction_shape = TopKSampleShape {
            num_total_sampling_inputs: 1,
            vocab_size: 256,
            top_k: 20,
        };
        let bitonic_shape = TopKSampleShape {
            top_k: 64,
            ..reduction_shape
        };
        assert_eq!(
            super::selected_top_k_tile_kernel(reduction_shape, TopKSamplingOperation::Sample),
            super::TopKTileKernelKind::Reduction
        );
        assert_eq!(
            super::selected_top_k_tile_kernel(bitonic_shape, TopKSamplingOperation::Sample),
            super::TopKTileKernelKind::Bitonic
        );
        assert_eq!(
            super::selected_top_k_tile_kernel(reduction_shape, TopKSamplingOperation::WriteDistribution),
            super::TopKTileKernelKind::Bitonic
        );
        assert_eq!(
            super::selected_top_k_tile_kernel(reduction_shape, TopKSamplingOperation::SampleAndWriteDistribution,),
            super::TopKTileKernelKind::Bitonic
        );
    }

    #[test]
    fn test_greedy() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 8,
            top_k: 4,
        };
        let logits = Buffer::from_slice(
            &device,
            &[
                0.1f32, 2.0, 1.5, 0.2, 3.0, 0.0, 2.5, 1.0, //
                4.0, 3.5, 0.0, 1.0, 2.0, 3.0, 0.5, 1.5,
            ],
        );
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample = TopKMergeKernels::new(&device);
        let runtime_params =
            sampling_runtime_params(&device, shape.num_total_sampling_inputs, 0.0, 1.0, 1, shape.top_k);

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        assert_eq!(token_ids.read_typed::<i32>(0, 2), vec![4, 0]);
        assert_eq!(token_probs.read_typed::<f32>(0, 2), vec![1.0, 1.0]);
    }

    #[test]
    fn test_bucket_inactive_rows() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 4,
            vocab_size: 4,
            top_k: 1,
        };
        let logits = Buffer::from_slice(
            &device,
            &[
                0.0_f32, 4.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0,
            ],
        );
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::from_slice(&device, &[-99_i32; 4]);
        let token_probs = Buffer::from_slice(&device, &[-99.0_f32; 4]);
        let runtime_params = sampling_runtime_params(&device, 4, 0.0, 1.0, 7, 1);
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample = TopKMergeKernels::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(shape, 2, true, true);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        assert_eq!(token_ids.read_typed::<i32>(0, 4), vec![1, 0, -99, -99]);
        assert_eq!(token_probs.read_typed::<f32>(0, 4), vec![1.0, 1.0, -99.0, -99.0]);
    }

    #[test]
    fn test_row_offset() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 1,
            vocab_size: 4,
            top_k: 1,
        };
        let logits = Buffer::from_slice(
            &device,
            &[
                0.0_f32, 9.0, 0.0, 0.0, // ignored row
                0.0, 0.0, 11.0, 0.0, // active row
            ],
        );
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: shape.vocab_size as usize * size_of::<f32>(),
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, false);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        assert_eq!(tile_token_ids.read_typed::<i32>(0, 1), vec![2]);
    }

    #[test]
    fn test_random() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 3,
            vocab_size: 17,
            top_k: 5,
        };
        let random_seed = 0x19A2_7C4D;
        let sample_seed = 0x64E3_10B9;
        let logits_values = generated_logits(
            shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
            random_seed,
        );
        let logits = Buffer::from_slice(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample = TopKMergeKernels::new(&device);
        let temperature = 0.9;
        let top_p = 0.82;
        let runtime_params = sampling_runtime_params(
            &device,
            shape.num_total_sampling_inputs,
            temperature,
            top_p,
            sample_seed,
            shape.top_k,
        );

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let config = SamplerConfig {
            temperature,
            top_k: shape.top_k as usize,
            top_p,
            seed: sample_seed,
        };
        let actual_tokens = token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize);
        let actual_probs = token_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize);
        for row in 0..shape.num_total_sampling_inputs as usize {
            let expected = sparse_sample_row_reference(
                &config,
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                shape.top_k as usize,
                row as u32,
            );
            assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
            assert!(
                (actual_probs[row] - expected.sampled_prob).abs() <= 1.0e-5,
                "row={row} actual_prob={} expected_prob={}",
                actual_probs[row],
                expected.sampled_prob
            );
        }
    }

    #[test]
    fn test_dynamic_params() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 8,
            top_k: 4,
        };
        let row_logits = [0.2f32, 1.0, 1.8, 2.5, 2.1, 0.3, -0.2, 1.4];
        let logits = Buffer::from_slice(&device, &[row_logits, row_logits].concat());
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
        let runtime_params =
            Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * 6 * size_of::<u32>());
        let first = SamplerConfig {
            temperature: 0.8,
            top_k: 1,
            top_p: 1.0,
            seed: 7,
        };
        let first_expected = sparse_sample_row_reference(&first, &row_logits, shape.top_k as usize, 11);
        let second = (8..1_000)
            .map(|seed| {
                SamplerConfig {
                    seed,
                    top_k: shape.top_k as usize,
                    ..first
                }
            })
            .find(|config| {
                sparse_sample_row_reference(config, &row_logits, shape.top_k as usize, 29).sampled_token
                    != first_expected.sampled_token
            })
            .expect("test logits must produce a distinct sample for some seed");
        write_sampling_runtime_params(&runtime_params, 0, &first, 11, SamplingDomain::Target);
        write_sampling_runtime_params(&runtime_params, 1, &second, 29, SamplingDomain::Target);

        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample = TopKMergeKernels::new(&device);
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);

        stream.submit_replay_with_arguments(&replay, &arguments).wait();
        assert_eq!(
            token_ids.read_typed::<i32>(0, 2),
            vec![
                first_expected.sampled_token as i32,
                sparse_sample_row_reference(&second, &row_logits, shape.top_k as usize, 29).sampled_token as i32,
            ]
        );

        let updated = SamplerConfig {
            temperature: 0.0,
            top_k: shape.top_k as usize,
            top_p: 1.0,
            seed: 99,
        };
        write_sampling_runtime_params(&runtime_params, 0, &updated, 41, SamplingDomain::Target);
        write_sampling_runtime_params(&runtime_params, 1, &first, 53, SamplingDomain::Target);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();
        assert_eq!(
            token_ids.read_typed::<i32>(0, 2),
            vec![
                sparse_sample_row_reference(&updated, &row_logits, shape.top_k as usize, 41).sampled_token as i32,
                sparse_sample_row_reference(&first, &row_logits, shape.top_k as usize, 53).sampled_token as i32,
            ]
        );
    }

    #[test]
    fn test_fused_distribution() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 8,
            top_k: 4,
        };
        let logits_values = vec![
            0.2f32, 1.0, 1.8, 2.5, 2.1, 0.3, -0.2, 1.4, //
            2.2, 0.1, 1.7, 0.9, 2.8, 1.2, -0.4, 2.0,
        ];
        let logits = Buffer::from_slice(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let sampled_token_ids = Buffer::new_zeroed(&device, 2 * size_of::<i32>());
        let sampled_token_probs = Buffer::new_zeroed(&device, 2 * size_of::<f32>());
        let distribution_token_ids = Buffer::new_zeroed(&device, 2 * shape.top_k as usize * size_of::<i32>());
        let distribution_probs = Buffer::new_zeroed(&device, 2 * shape.top_k as usize * size_of::<f32>());
        let output_distribution_indices = Buffer::from_slice(&device, &[1_u32, 0]);
        let runtime_params = Buffer::new_zeroed(&device, 2 * 6 * size_of::<u32>());
        let configs = [
            SamplerConfig {
                temperature: 0.8,
                top_k: 1,
                top_p: 1.0,
                seed: 7,
            },
            SamplerConfig {
                temperature: 0.9,
                top_k: 4,
                top_p: 0.8,
                seed: 19,
            },
        ];
        write_sampling_runtime_params(&runtime_params, 0, &configs[0], 11, SamplingDomain::Target);
        write_sampling_runtime_params(&runtime_params, 1, &configs[1], 29, SamplingDomain::Draft);

        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample_and_write_distribution = TopKMergeKernels::new(&device);
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::SampleAndWriteDistribution,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample_and_write_distribution.invoke_sample_and_write_distribution(
            shape,
            TopKSampleAndWriteDistributionBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_token_probs,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                runtime_params: &runtime_params,
                output_distribution_indices: &output_distribution_indices,
                max_k: shape.top_k,
                num_output_distributions: 2,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let expected = configs
            .iter()
            .enumerate()
            .map(|(row, config)| {
                sparse_sample_row_with_domain_reference(
                    config,
                    &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                    config.top_k,
                    [11, 29][row],
                    [SamplingDomain::Target, SamplingDomain::Draft][row],
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sampled_token_ids.read_typed::<i32>(0, 2),
            expected.iter().map(|row| row.sampled_token as i32).collect::<Vec<_>>()
        );
        assert_close(
            &sampled_token_probs.read_typed::<f32>(0, 2),
            &expected.iter().map(|row| row.sampled_prob).collect::<Vec<_>>(),
            1.0e-5,
        );
        let distribution_ids = distribution_token_ids.read_typed::<i32>(0, 8);
        let distribution_probs = distribution_probs.read_typed::<f32>(0, 8);
        assert_eq!(&distribution_ids[0..4], expected[1].prob_token_ids.as_slice());
        assert_close(&distribution_probs[0..4], &expected[1].prob_values, 1.0e-5);
        assert_eq!(distribution_ids[4], expected[0].prob_token_ids[0]);
        assert_close(&distribution_probs[4..5], &expected[0].prob_values[0..1], 1.0e-5);
    }

    #[test]
    fn test_distribution_slots() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 3,
            vocab_size: 17,
            top_k: 5,
        };
        let random_seed = 0xE6B4_2A17;
        let logits_values = generated_logits(
            shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
            random_seed,
        );
        let logits = Buffer::from_slice(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let output_row_offset = 1;
        let max_k = 8;
        let num_output_distributions = shape.num_total_sampling_inputs + output_row_offset;
        let output_distribution_indices = Buffer::from_slice(
            &device,
            &(0..shape.num_total_sampling_inputs)
                .map(|row| row + output_row_offset)
                .collect::<Vec<_>>(),
        );
        let distribution_token_ids = Buffer::new_zeroed(
            &device,
            num_output_distributions as usize * max_k as usize * size_of::<i32>(),
        );
        let distribution_probs = Buffer::new_zeroed(
            &device,
            num_output_distributions as usize * max_k as usize * size_of::<f32>(),
        );
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let write_distribution = TopKMergeKernels::new(&device);
        let temperature = 0.9;
        let top_p = 0.82;
        let runtime_params = sampling_runtime_params(
            &device,
            shape.num_total_sampling_inputs,
            temperature,
            top_p,
            1,
            shape.top_k,
        );

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::WriteDistribution,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(write_distribution.invoke_write_distribution(
            shape,
            TopKWriteDistributionBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                runtime_params: &runtime_params,
                output_distribution_indices: &output_distribution_indices,
                max_k,
                num_output_distributions,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let actual_tokens =
            distribution_token_ids.read_typed::<i32>(0, num_output_distributions as usize * max_k as usize);
        let actual_probs = distribution_probs.read_typed::<f32>(0, num_output_distributions as usize * max_k as usize);
        let config = SamplerConfig {
            temperature,
            top_k: shape.top_k as usize,
            top_p,
            seed: 1,
        };
        for row in 0..shape.num_total_sampling_inputs as usize {
            let expected = sparse_sample_row_reference(
                &config,
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                shape.top_k as usize,
                row as u32,
            );
            let start = (row + output_row_offset as usize) * max_k as usize;
            assert_eq!(
                &actual_tokens[start..start + shape.top_k as usize],
                expected.prob_token_ids.as_slice()
            );
            assert_close(
                &actual_probs[start..start + shape.top_k as usize],
                &expected.prob_values,
                1.0e-5,
            );
        }
    }

    #[test]
    fn test_bf16_reduction() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 257,
            top_k: 20,
        };
        let random_seed = 0x4F91_C3E8;
        let sample_seed = 0xB72D_5A60;
        let logits_values = generated_logits(
            shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
            random_seed,
        );
        let logits = bf16_buffer(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Bfloat16;
        let sample = TopKMergeKernels::new(&device);
        let temperature = 0.7;
        let top_p = 0.8;
        let runtime_params = sampling_runtime_params(
            &device,
            shape.num_total_sampling_inputs,
            temperature,
            top_p,
            sample_seed,
            shape.top_k,
        );

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        assert_sample_matches_bf16_reference(
            shape,
            &logits_values,
            &token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize),
            &token_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize),
            SamplerConfig {
                temperature,
                top_k: shape.top_k as usize,
                top_p,
                seed: sample_seed,
            },
        );
    }

    #[test]
    fn test_f32_bitonic() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 257,
            top_k: 64,
        };
        let random_seed = 0xD03A_8F51;
        let sample_seed = 0x2C76_EB94;
        let logits_values = generated_logits(
            shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
            random_seed,
        );
        let logits = Buffer::from_slice(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let token_ids = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(&device, shape.num_total_sampling_inputs as usize * size_of::<f32>());
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Float32;
        let sample = TopKMergeKernels::new(&device);
        let temperature = 0.7;
        let top_p = 0.8;
        let runtime_params = sampling_runtime_params(
            &device,
            shape.num_total_sampling_inputs,
            temperature,
            top_p,
            sample_seed,
            shape.top_k,
        );

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(sample.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let config = SamplerConfig {
            temperature,
            top_k: shape.top_k as usize,
            top_p,
            seed: sample_seed,
        };
        let actual_tokens = token_ids.read_typed::<i32>(0, shape.num_total_sampling_inputs as usize);
        let actual_probs = token_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize);
        for row in 0..shape.num_total_sampling_inputs as usize {
            let expected = sparse_sample_row_reference(
                &config,
                &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                shape.top_k as usize,
                row as u32,
            );
            assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
            assert!(
                (actual_probs[row] - expected.sampled_prob).abs() <= 1.0e-5,
                "row={row} actual_prob={} expected_prob={}",
                actual_probs[row],
                expected.sampled_prob
            );
        }
    }

    #[test]
    fn test_bf16_bitonic_distribution() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = TopKSampleShape {
            num_total_sampling_inputs: 2,
            vocab_size: 257,
            top_k: 64,
        };
        let random_seed = 0x8A15_6D3F;
        let logits_values = generated_logits(
            shape.num_total_sampling_inputs as usize * shape.vocab_size as usize,
            random_seed,
        );
        let logits = bf16_buffer(&device, &logits_values);
        let tile_token_ids = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<i32>(),
        );
        let tile_logits = Buffer::new_zeroed(
            &device,
            super::tile_count(shape, super::vocab_tile_size()) * size_of::<f32>(),
        );
        let distribution_token_ids = Buffer::new_zeroed(
            &device,
            shape.num_total_sampling_inputs as usize * shape.top_k as usize * size_of::<i32>(),
        );
        let distribution_probs = Buffer::new_zeroed(
            &device,
            shape.num_total_sampling_inputs as usize * shape.top_k as usize * size_of::<f32>(),
        );
        let output_distribution_indices =
            Buffer::from_slice(&device, &(0..shape.num_total_sampling_inputs).collect::<Vec<_>>());
        let topk = TopKTileKernels::new(&device);
        let logits_dtype = crate::metal::Dtype::Bfloat16;
        let write_distribution = TopKMergeKernels::new(&device);
        let temperature = 0.7;
        let top_p = 0.8;
        let runtime_params = sampling_runtime_params(
            &device,
            shape.num_total_sampling_inputs,
            temperature,
            top_p,
            1,
            shape.top_k,
        );

        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            logits_dtype,
            TopKSamplingOperation::WriteDistribution,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(write_distribution.invoke_write_distribution(
            shape,
            TopKWriteDistributionBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                runtime_params: &runtime_params,
                output_distribution_indices: &output_distribution_indices,
                max_k: shape.top_k,
                num_output_distributions: shape.num_total_sampling_inputs,
            },
        ));
        let program = builder.build();
        let arguments = top_k_replay_arguments(shape, shape.num_total_sampling_inputs, true, true);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let actual_tokens = distribution_token_ids
            .read_typed::<i32>(0, shape.num_total_sampling_inputs as usize * shape.top_k as usize);
        let actual_probs =
            distribution_probs.read_typed::<f32>(0, shape.num_total_sampling_inputs as usize * shape.top_k as usize);
        let config = SamplerConfig {
            temperature,
            top_k: shape.top_k as usize,
            top_p,
            seed: 1,
        };
        for row in 0..shape.num_total_sampling_inputs as usize {
            let expected = sparse_sample_row_reference(
                &config,
                &bf16_rounded_logits(
                    &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                ),
                shape.top_k as usize,
                row as u32,
            );
            let start = row * shape.top_k as usize;
            assert_eq!(
                &actual_tokens[start..start + shape.top_k as usize],
                expected.prob_token_ids.as_slice(),
                "row={row}"
            );
            assert_close(
                &actual_probs[start..start + shape.top_k as usize],
                &expected.prob_values,
                1.0e-5,
            );
        }
    }

    #[test]
    fn test_rejection() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = SparseRejectionSampleShape {
            num_total_reqs: 1,
            num_total_target_distributions: 3,
            num_total_draft_distributions: 2,
            top_k: 4,
            max_target_k: 4,
            max_draft_k: 4,
        };
        let target_rows = vec![
            vec![0.0, 0.50, 0.20, 0.30, 0.0, 0.0],
            vec![0.0, 0.10, 0.55, 0.35, 0.0, 0.0],
            vec![0.0, 0.05, 0.10, 0.85, 0.0, 0.0],
        ];
        let draft_rows = vec![
            vec![0.0, 0.25, 0.50, 0.25, 0.0, 0.0],
            vec![0.0, 0.10, 0.20, 0.70, 0.0, 0.0],
        ];
        let draft_tokens = vec![2_u32, 3];
        let target_distribution = write_distributions_from_dense(&target_rows, shape.max_target_k as usize);
        let draft_distribution = write_distributions_from_dense(&draft_rows, shape.max_draft_k as usize);
        let flat_draft_distribution_indices = vec![2_u32, 0];
        let mut mapped_draft_distribution_token_ids = vec![-1_i32; 3 * shape.max_draft_k as usize];
        let mut mapped_draft_distribution_probs = vec![0.0_f32; 3 * shape.max_draft_k as usize];
        for (draft_row, &distribution_row) in flat_draft_distribution_indices.iter().enumerate() {
            let source = draft_row * shape.max_draft_k as usize;
            let destination = distribution_row as usize * shape.max_draft_k as usize;
            mapped_draft_distribution_token_ids[destination..destination + shape.max_draft_k as usize]
                .copy_from_slice(&draft_distribution.0[source..source + shape.max_draft_k as usize]);
            mapped_draft_distribution_probs[destination..destination + shape.max_draft_k as usize]
                .copy_from_slice(&draft_distribution.1[source..source + shape.max_draft_k as usize]);
        }
        let target_distribution_token_ids = Buffer::from_slice(&device, &target_distribution.0);
        let target_distribution_probs = Buffer::from_slice(&device, &target_distribution.1);
        let draft_distribution_token_ids = Buffer::from_slice(&device, &mapped_draft_distribution_token_ids);
        let draft_distribution_probs = Buffer::from_slice(&device, &mapped_draft_distribution_probs);
        let draft_token_ids = Buffer::from_slice(
            &device,
            &draft_tokens.iter().map(|token| *token as i32).collect::<Vec<_>>(),
        );
        let cu_target_distributions = Buffer::from_slice(&device, &[0_u32, shape.num_total_target_distributions]);
        let cu_draft_distributions = Buffer::from_slice(&device, &[0_u32, shape.num_total_draft_distributions]);
        let accepted_token_ids = Buffer::new_zeroed(&device, shape.num_accepted_token_slots() * size_of::<i32>());
        let accepted_probs = Buffer::new_zeroed(&device, shape.num_accepted_token_slots() * size_of::<f32>());
        let num_accepted_tokens = Buffer::new_zeroed(&device, shape.num_total_reqs as usize * size_of::<u32>());
        let sampled_token_ids = Buffer::new_zeroed(&device, shape.num_total_reqs as usize * size_of::<i32>());
        let sampled_probs = Buffer::new_zeroed(&device, shape.num_total_reqs as usize * size_of::<f32>());
        let kernel = SparseRejectionSampleKernel::new(&device);
        let seed = 7;
        let sample_position = 19;
        let runtime_params = Buffer::from_slice(&device, &[seed, sample_position, shape.top_k, 0]);
        let flat_draft_distribution_indices = Buffer::from_slice(&device, &flat_draft_distribution_indices);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            shape,
            SparseRejectionSampleBuffers {
                target_distribution_token_ids: &target_distribution_token_ids,
                target_distribution_probs: &target_distribution_probs,
                draft_distribution_token_ids: &draft_distribution_token_ids,
                draft_distribution_probs: &draft_distribution_probs,
                flat_draft_token_ids: &draft_token_ids,
                cu_target_distributions: &cu_target_distributions,
                cu_draft_distributions: &cu_draft_distributions,
                flat_draft_distribution_indices: &flat_draft_distribution_indices,
                flat_accepted_token_ids: &accepted_token_ids,
                flat_accepted_probs: &accepted_probs,
                num_accepted_tokens: &num_accepted_tokens,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = rejection_replay_arguments(shape, 1, 3, 2);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let expected = rejection_sample_reference(&draft_tokens, &target_rows, &draft_rows, seed, sample_position);
        let accepted_len = num_accepted_tokens.read_typed::<u32>(0, 1)[0] as usize;
        assert_eq!(accepted_len, expected.accepted_tokens.len());
        assert_eq!(
            accepted_token_ids.read_typed::<i32>(0, accepted_len),
            expected
                .accepted_tokens
                .iter()
                .map(|token| *token as i32)
                .collect::<Vec<_>>()
        );
        assert_close(
            &accepted_probs.read_typed::<f32>(0, accepted_len),
            &expected.accepted_probs,
            1.0e-5,
        );
        assert_eq!(
            sampled_token_ids.read_typed::<i32>(0, 1)[0],
            expected.sampled_token as i32
        );
        assert_close(&sampled_probs.read_typed::<f32>(0, 1), &[expected.sampled_prob], 1.0e-5);
    }

    #[test]
    fn test_rejection_all_accept() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = SparseRejectionSampleShape {
            num_total_reqs: 1,
            num_total_target_distributions: 2,
            num_total_draft_distributions: 1,
            top_k: 1,
            max_target_k: 1,
            max_draft_k: 1,
        };
        let target_distribution_token_ids = Buffer::from_slice(&device, &[1_i32, 1]);
        let target_distribution_probs = Buffer::from_slice(&device, &[1.0_f32, 1.0]);
        let draft_distribution_token_ids = Buffer::from_slice(&device, &[1_i32]);
        let draft_distribution_probs = Buffer::from_slice(&device, &[1.0_f32]);
        let draft_token_ids = Buffer::from_slice(&device, &[1_i32]);
        let cu_target_distributions = Buffer::from_slice(&device, &[0_u32, 2]);
        let cu_draft_distributions = Buffer::from_slice(&device, &[0_u32, 1]);
        let flat_draft_distribution_indices = Buffer::from_slice(&device, &[0_u32]);
        let accepted_token_ids = Buffer::new_zeroed(&device, size_of::<i32>());
        let accepted_probs = Buffer::new_zeroed(&device, size_of::<f32>());
        let num_accepted_tokens = Buffer::new_zeroed(&device, size_of::<u32>());
        let sampled_token_ids = Buffer::new_zeroed(&device, size_of::<i32>());
        let sampled_probs = Buffer::new_zeroed(&device, size_of::<f32>());
        let seed = 7_u32;
        let sample_position = 19_u32;
        let runtime_params = Buffer::from_slice(&device, &[seed, sample_position, shape.top_k, 0]);
        let kernel = SparseRejectionSampleKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            shape,
            SparseRejectionSampleBuffers {
                target_distribution_token_ids: &target_distribution_token_ids,
                target_distribution_probs: &target_distribution_probs,
                draft_distribution_token_ids: &draft_distribution_token_ids,
                draft_distribution_probs: &draft_distribution_probs,
                flat_draft_token_ids: &draft_token_ids,
                cu_target_distributions: &cu_target_distributions,
                cu_draft_distributions: &cu_draft_distributions,
                flat_draft_distribution_indices: &flat_draft_distribution_indices,
                flat_accepted_token_ids: &accepted_token_ids,
                flat_accepted_probs: &accepted_probs,
                num_accepted_tokens: &num_accepted_tokens,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_probs,
                runtime_params: &runtime_params,
            },
        ));
        let program = builder.build();
        let arguments = rejection_replay_arguments(shape, 1, 2, 1);
        stream.submit_replay_with_arguments(&program, &arguments).wait();

        let expected = rejection_sample_reference(
            &[1],
            &[vec![0.0, 1.0], vec![0.0, 1.0]],
            &[vec![0.0, 1.0]],
            seed,
            sample_position,
        );
        assert_eq!(
            num_accepted_tokens.read_typed::<u32>(0, 1),
            vec![expected.accepted_tokens.len() as u32]
        );
        assert_eq!(
            accepted_token_ids.read_typed::<i32>(0, 1),
            vec![expected.accepted_tokens[0] as i32]
        );
        assert_eq!(accepted_probs.read_typed::<f32>(0, 1), expected.accepted_probs);
        assert_eq!(
            sampled_token_ids.read_typed::<i32>(0, 1),
            vec![expected.sampled_token as i32]
        );
        assert_eq!(sampled_probs.read_typed::<f32>(0, 1), vec![expected.sampled_prob]);
    }

    #[test]
    fn test_rejection_ragged_zero_drafts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = SparseRejectionSampleShape {
            num_total_reqs: 4,
            num_total_target_distributions: 4,
            num_total_draft_distributions: 0,
            top_k: 4,
            max_target_k: 4,
            max_draft_k: 4,
        };
        let target_distribution_token_ids =
            Buffer::from_slice(&device, &[1_i32, 7, 6, 5, 3, 4, 5, 6, 9, 9, 9, 9, 8, 8, 8, 8]);
        let target_distribution_probs = Buffer::from_slice(
            &device,
            &[
                1.0_f32, 100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0, 98.0, 98.0, 98.0, 98.0,
            ],
        );
        let draft_distribution_token_ids = Buffer::from_slice(&device, &[-1_i32; 4]);
        let draft_distribution_probs = Buffer::from_slice(&device, &[0.0_f32; 4]);
        let draft_token_ids = Buffer::from_slice(&device, &[0_i32]);
        let cu_target_distributions = Buffer::from_slice(&device, &[0_u32, 1, 2, u32::MAX, u32::MAX]);
        let cu_draft_distributions = Buffer::from_slice(&device, &[0_u32, 0, 0, u32::MAX, u32::MAX]);
        let flat_draft_distribution_indices = Buffer::from_slice(&device, &[0_u32]);
        let accepted_token_ids = Buffer::new_zeroed(&device, size_of::<i32>());
        let accepted_probs = Buffer::new_zeroed(&device, size_of::<f32>());
        let num_accepted_tokens = Buffer::from_slice(&device, &[99_u32; 4]);
        let sampled_token_ids = Buffer::from_slice(&device, &[-99_i32; 4]);
        let sampled_probs = Buffer::from_slice(&device, &[-99.0_f32; 4]);
        let runtime_params = Buffer::from_slice(&device, &[7_u32, 11, 1, 0, 19, 23, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0]);
        let kernel = SparseRejectionSampleKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            shape,
            SparseRejectionSampleBuffers {
                target_distribution_token_ids: &target_distribution_token_ids,
                target_distribution_probs: &target_distribution_probs,
                draft_distribution_token_ids: &draft_distribution_token_ids,
                draft_distribution_probs: &draft_distribution_probs,
                flat_draft_token_ids: &draft_token_ids,
                cu_target_distributions: &cu_target_distributions,
                cu_draft_distributions: &cu_draft_distributions,
                flat_draft_distribution_indices: &flat_draft_distribution_indices,
                flat_accepted_token_ids: &accepted_token_ids,
                flat_accepted_probs: &accepted_probs,
                num_accepted_tokens: &num_accepted_tokens,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_probs,
                runtime_params: &runtime_params,
            },
        ));
        let replay = builder.build();
        let arguments = rejection_replay_arguments(shape, 2, 2, 0);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        assert_eq!(num_accepted_tokens.read_typed::<u32>(0, 4), vec![0, 0, 99, 99]);
        assert_eq!(sampled_token_ids.read_typed::<i32>(0, 4), vec![1, 3, -99, -99]);
        assert_eq!(sampled_probs.read_typed::<f32>(0, 4), vec![1.0, 1.0, -99.0, -99.0]);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn bf16_rounded_logits(values: &[f32]) -> Vec<f32> {
        values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
    }

    fn assert_sample_matches_bf16_reference(
        shape: TopKSampleShape,
        logits_values: &[f32],
        actual_tokens: &[i32],
        actual_probs: &[f32],
        config: SamplerConfig,
    ) {
        for row in 0..shape.num_total_sampling_inputs as usize {
            let expected = sparse_sample_row_reference(
                &config,
                &bf16_rounded_logits(
                    &logits_values[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
                ),
                shape.top_k as usize,
                row as u32,
            );
            assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
            assert!(
                (actual_probs[row] - expected.sampled_prob).abs() <= 1.0e-5,
                "row={row} actual_prob={} expected_prob={}",
                actual_probs[row],
                expected.sampled_prob
            );
        }
    }

    fn generated_logits(count: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 4_194_304.0) - 2.0
            })
            .collect()
    }

    fn generated_prob_rows(num_rows: usize, vocab_size: usize, random_seed: u32) -> Vec<Vec<f32>> {
        let mut values = generated_logits(num_rows * vocab_size, random_seed)
            .into_iter()
            .map(|value| value.abs() + 0.05)
            .collect::<Vec<_>>();
        for row in values.chunks_mut(vocab_size) {
            let sum = row.iter().sum::<f32>();
            for value in row {
                *value /= sum;
            }
        }
        values.chunks(vocab_size).map(|row| row.to_vec()).collect()
    }

    fn write_distributions_from_dense(rows: &[Vec<f32>], row_stride: usize) -> (Vec<i32>, Vec<f32>) {
        let mut token_ids = vec![-1; rows.len() * row_stride];
        let mut probs = vec![0.0; rows.len() * row_stride];
        for (row_index, row) in rows.iter().enumerate() {
            let mut write_distribution = row
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, prob)| *prob > 0.0)
                .map(|(token, prob)| (token as i32, prob))
                .collect::<Vec<_>>();
            write_distribution
                .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap().then_with(|| left.0.cmp(&right.0)));
            assert!(write_distribution.len() <= row_stride);
            let base = row_index * row_stride;
            for (slot, (token, prob)) in write_distribution.into_iter().enumerate() {
                token_ids[base + slot] = token;
                probs[base + slot] = prob;
            }
        }
        (token_ids, probs)
    }

    fn generated_draft_tokens(count: usize, vocab_size: usize, random_seed: u32) -> Vec<u32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as usize % vocab_size) as u32
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
