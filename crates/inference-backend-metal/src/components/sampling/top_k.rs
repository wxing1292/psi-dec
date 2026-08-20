use super::MAX_TOP_K;
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
pub const TOP_K_MAP_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.tile_num_active_threads");
pub const TOP_K_TILE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey = TOP_K_MAP_NUM_ACTIVE_THREADS_KEY;
const TOP_K_REDUCE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.merge_num_active_threads");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKPartialCandidateLayout {
    vocab_partition_size: u32,
}

impl TopKPartialCandidateLayout {
    pub fn new(vocab_partition_size: u32) -> Self {
        assert!(vocab_partition_size > 0);
        Self { vocab_partition_size }
    }

    pub fn vocab_partition_size(self) -> u32 {
        self.vocab_partition_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TopKMapThreadBlockSpecialization {
    max_vocab_tokens: u32,
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TopKMapAlgorithm {
    Reduction,
    Bitonic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TopKMapKernelSpecialization {
    logits_dtype: Dtype,
    algorithm: TopKMapAlgorithm,
    thread_block: TopKMapThreadBlockSpecialization,
}

impl TopKMapKernelSpecialization {
    fn partial_candidate_layout(self) -> TopKPartialCandidateLayout {
        TopKPartialCandidateLayout::new(self.thread_block.max_vocab_tokens)
    }
}

struct TopKMapPlanner {
    reduction_limit: u32,
    thread_block: TopKMapThreadBlockSpecialization,
}

impl TopKMapPlanner {
    fn new() -> Self {
        Self {
            reduction_limit: TOP_K_REDUCTION_LIMIT,
            thread_block: TopKMapThreadBlockSpecialization {
                max_vocab_tokens: TOP_K_VOCAB_TILE_SIZE,
                required_threads: 256,
            },
        }
    }

    fn select(
        &self,
        shape: TopKSampleShape,
        logits_dtype: Dtype,
        operation: TopKSamplingOperation,
    ) -> TopKMapKernelSpecialization {
        shape.validate();
        let algorithm = match operation {
            TopKSamplingOperation::Sample if shape.top_k <= self.reduction_limit => TopKMapAlgorithm::Reduction,
            TopKSamplingOperation::Sample
            | TopKSamplingOperation::WriteDistribution
            | TopKSamplingOperation::SampleAndWriteDistribution => TopKMapAlgorithm::Bitonic,
        };
        TopKMapKernelSpecialization {
            logits_dtype,
            algorithm,
            thread_block: self.thread_block,
        }
    }

    fn partial_candidate_layout(&self) -> TopKPartialCandidateLayout {
        TopKPartialCandidateLayout::new(self.thread_block.max_vocab_tokens)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TopKReduceThreadBlockSpecialization {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TopKReduceKernelSpecialization {
    thread_block: TopKReduceThreadBlockSpecialization,
}

impl TopKReduceKernelSpecialization {
    fn current() -> Self {
        Self {
            thread_block: TopKReduceThreadBlockSpecialization { required_threads: 256 },
        }
    }
}

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
            "Metal top-k partial candidate count",
            &[
                self.num_total_sampling_inputs as usize,
                self.vocab_size
                    .div_ceil(standard_partial_candidate_layout().vocab_partition_size()) as usize,
                self.top_k as usize,
            ],
        );
    }
}

fn standard_partial_candidate_layout() -> TopKPartialCandidateLayout {
    TopKMapPlanner::new().partial_candidate_layout()
}

fn num_vocab_partitions(shape: TopKSampleShape, layout: TopKPartialCandidateLayout) -> u32 {
    shape.validate();
    shape.vocab_size.div_ceil(layout.vocab_partition_size())
}

fn num_candidates_per_partition(shape: TopKSampleShape) -> u32 {
    shape.validate();
    shape.top_k
}

fn partial_candidate_count(shape: TopKSampleShape, layout: TopKPartialCandidateLayout) -> usize {
    checked_product(
        "Metal top-k partial candidate count",
        &[
            shape.num_total_sampling_inputs as usize,
            num_vocab_partitions(shape, layout) as usize,
            num_candidates_per_partition(shape) as usize,
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

pub type TopKMapBuffers<'a> = TopKTileBuffers<'a>;

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

fn assert_map_buffers_fit(
    shape: TopKSampleShape,
    layout: TopKPartialCandidateLayout,
    buffers: TopKMapBuffers<'_>,
    logits_item_size: usize,
) {
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
    let candidates = partial_candidate_count(shape, layout);
    assert!(
        buffers.tile_token_ids.len_bytes() >= checked_bytes("Metal top-k partial token", candidates, size_of::<i32>()),
        "top-k partial-token buffer is too short"
    );
    assert!(
        buffers.tile_logits.len_bytes() >= checked_bytes("Metal top-k partial logit", candidates, size_of::<f32>()),
        "top-k partial-logit buffer is too short"
    );
}

fn assert_reduce_inputs_fit(
    shape: TopKSampleShape,
    tile_token_ids: &Buffer,
    tile_logits: &Buffer,
    runtime_params: &Buffer,
    layout: TopKPartialCandidateLayout,
) {
    let candidates = partial_candidate_count(shape, layout);
    assert!(
        tile_token_ids.len_bytes() >= checked_bytes("Metal top-k reduce token", candidates, size_of::<i32>()),
        "top-k partial-token buffer is too short"
    );
    assert!(
        tile_logits.len_bytes() >= checked_bytes("Metal top-k reduce logit", candidates, size_of::<f32>()),
        "top-k partial-logit buffer is too short"
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

pub struct TopKMapKernels {
    planner: TopKMapPlanner,
    f32_reduction: Kernel,
    f32_bitonic: Kernel,
    bf16_reduction: Kernel,
    bf16_bitonic: Kernel,
}

impl TopKMapKernels {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            planner: TopKMapPlanner::new(),
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
        buffers: TopKMapBuffers<'a>,
    ) -> TopKMapInvocation<'a> {
        let specialization = self.planner.select(shape, logits_dtype, operation);
        let (kernel, logits_item_size) = match (specialization.logits_dtype, specialization.algorithm) {
            (Dtype::Float32, TopKMapAlgorithm::Reduction) => (&self.f32_reduction, size_of::<f32>()),
            (Dtype::Float32, TopKMapAlgorithm::Bitonic) => (&self.f32_bitonic, size_of::<f32>()),
            (Dtype::Bfloat16, TopKMapAlgorithm::Reduction) => (&self.bf16_reduction, size_of::<u16>()),
            (Dtype::Bfloat16, TopKMapAlgorithm::Bitonic) => (&self.bf16_bitonic, size_of::<u16>()),
            (dtype, _) => panic!("unsupported top-k logits dtype {dtype:?}"),
        };
        TopKMapInvocation {
            kernel,
            logits_item_size,
            specialization,
            shape,
            buffers,
        }
    }

    pub fn candidate_count(&self, shape: TopKSampleShape) -> usize {
        partial_candidate_count(shape, self.partial_candidate_layout())
    }

    pub fn partial_candidate_layout(&self) -> TopKPartialCandidateLayout {
        self.planner.partial_candidate_layout()
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
        let num_partitions = num_vocab_partitions(shape, self.partial_candidate_layout());
        let num_threads_per_row = checked_num_threads(num_partitions, self.planner.thread_block.required_threads);
        let num_active_threads = checked_num_threads(num_active_sampling_inputs, num_threads_per_row);
        let num_total_threads = checked_num_threads(shape.num_total_sampling_inputs, num_threads_per_row);
        assert!(num_active_threads <= num_total_threads);
        arguments.set_u32(TOP_K_MAP_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

pub type TopKTileKernels = TopKMapKernels;

pub struct TopKMapInvocation<'a> {
    kernel: &'a Kernel,
    logits_item_size: usize,
    specialization: TopKMapKernelSpecialization,
    shape: TopKSampleShape,
    buffers: TopKMapBuffers<'a>,
}

pub type TopKTileInvocation<'a> = TopKMapInvocation<'a>;

impl Operator for TopKMapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        let layout = self.specialization.partial_candidate_layout();
        assert_map_buffers_fit(self.shape, layout, self.buffers, self.logits_item_size);
        let num_partitions = num_vocab_partitions(self.shape, layout);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.logits, self.buffers.logits_offset_bytes);
        recorder.set_buffer_write(1, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_write(2, self.buffers.tile_logits, 0);
        recorder.set_u32(4, self.shape.vocab_size);
        recorder.set_u32(5, self.shape.top_k);
        recorder.set_u32(6, layout.vocab_partition_size());
        recorder.set_u32(7, num_partitions);
        let required_threads = self.specialization.thread_block.required_threads;
        let num_threads_per_row = checked_num_threads(num_partitions, required_threads);
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(3, num_total_threads);
        } else {
            recorder.bind_u32(
                3,
                TOP_K_MAP_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(num_total_threads as usize, required_threads as usize);
    }
}

pub struct TopKReduceKernels {
    specialization: TopKReduceKernelSpecialization,
    sample: Kernel,
    write_distribution: Kernel,
    sample_and_write_distribution: Kernel,
}

impl TopKReduceKernels {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            specialization: TopKReduceKernelSpecialization::current(),
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
        self.invoke_sample_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_sample_with_layout<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleBuffers<'a>,
        partial_candidate_layout: TopKPartialCandidateLayout,
    ) -> TopKSampleInvocation<'a> {
        TopKSampleInvocation {
            kernel: &self.sample,
            specialization: self.specialization,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    pub fn invoke_write_distribution<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKWriteDistributionBuffers<'a>,
    ) -> TopKWriteDistributionInvocation<'a> {
        self.invoke_write_distribution_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_write_distribution_with_layout<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKWriteDistributionBuffers<'a>,
        partial_candidate_layout: TopKPartialCandidateLayout,
    ) -> TopKWriteDistributionInvocation<'a> {
        TopKWriteDistributionInvocation {
            kernel: &self.write_distribution,
            specialization: self.specialization,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    pub fn invoke_sample_and_write_distribution<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleAndWriteDistributionBuffers<'a>,
    ) -> TopKSampleAndWriteDistributionInvocation<'a> {
        self.invoke_sample_and_write_distribution_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_sample_and_write_distribution_with_layout<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleAndWriteDistributionBuffers<'a>,
        partial_candidate_layout: TopKPartialCandidateLayout,
    ) -> TopKSampleAndWriteDistributionInvocation<'a> {
        TopKSampleAndWriteDistributionInvocation {
            kernel: &self.sample_and_write_distribution,
            specialization: self.specialization,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    #[deprecated(note = "use invoke_sample_and_write_distribution_with_layout")]
    pub fn invoke_sample_and_write_distribution_with_vocab_tile_size<'a>(
        &'a self,
        shape: TopKSampleShape,
        buffers: TopKSampleAndWriteDistributionBuffers<'a>,
        vocab_tile_size: u32,
    ) -> TopKSampleAndWriteDistributionInvocation<'a> {
        self.invoke_sample_and_write_distribution_with_layout(
            shape,
            buffers,
            TopKPartialCandidateLayout::new(vocab_tile_size),
        )
    }

    pub fn add_replay_arguments(
        &self,
        shape: TopKSampleShape,
        num_active_sampling_inputs: u32,
        arguments: &mut ReplayArguments,
    ) {
        add_top_k_reduce_replay_argument(
            shape,
            num_active_sampling_inputs,
            self.specialization.thread_block.required_threads,
            arguments,
        );
    }
}

pub type TopKMergeKernels = TopKReduceKernels;

pub struct TopKSampleInvocation<'a> {
    kernel: &'a Kernel,
    specialization: TopKReduceKernelSpecialization,
    partial_candidate_layout: TopKPartialCandidateLayout,
    shape: TopKSampleShape,
    buffers: TopKSampleBuffers<'a>,
}

impl Operator for TopKSampleInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_reduce_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            self.partial_candidate_layout,
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
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = num_candidates_per_partition(self.shape);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.token_probs, 0);
        recorder.set_buffer_read(4, self.buffers.runtime_params, 0);
        recorder.set_u32(6, self.shape.top_k);
        recorder.set_u32(7, num_partitions);
        recorder.set_u32(8, num_candidates_per_partition);
        recorder.set_u32(9, self.partial_candidate_layout.vocab_partition_size());
        let num_threads_per_row = self.specialization.thread_block.required_threads;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(5, num_total_threads);
        } else {
            recorder.bind_u32(
                5,
                TOP_K_REDUCE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

pub struct TopKWriteDistributionInvocation<'a> {
    kernel: &'a Kernel,
    specialization: TopKReduceKernelSpecialization,
    partial_candidate_layout: TopKPartialCandidateLayout,
    shape: TopKSampleShape,
    buffers: TopKWriteDistributionBuffers<'a>,
}

impl Operator for TopKWriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_reduce_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            self.partial_candidate_layout,
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
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = num_candidates_per_partition(self.shape);
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
        recorder.set_u32(8, num_partitions);
        recorder.set_u32(9, num_candidates_per_partition);
        recorder.set_u32(10, self.partial_candidate_layout.vocab_partition_size());
        recorder.set_u32(11, self.buffers.max_k);
        recorder.set_u32(12, self.buffers.num_output_distributions);
        let num_threads_per_row = self.specialization.thread_block.required_threads;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(6, num_total_threads);
        } else {
            recorder.bind_u32(
                6,
                TOP_K_REDUCE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

pub struct TopKSampleAndWriteDistributionInvocation<'a> {
    kernel: &'a Kernel,
    specialization: TopKReduceKernelSpecialization,
    partial_candidate_layout: TopKPartialCandidateLayout,
    shape: TopKSampleShape,
    buffers: TopKSampleAndWriteDistributionBuffers<'a>,
}

impl Operator for TopKSampleAndWriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.shape.validate();
        assert_reduce_inputs_fit(
            self.shape,
            self.buffers.tile_token_ids,
            self.buffers.tile_logits,
            self.buffers.runtime_params,
            self.partial_candidate_layout,
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
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = num_candidates_per_partition(self.shape);
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
        recorder.set_u32(10, num_partitions);
        recorder.set_u32(11, num_candidates_per_partition);
        recorder.set_u32(12, self.partial_candidate_layout.vocab_partition_size());
        recorder.set_u32(13, self.buffers.max_k);
        recorder.set_u32(14, self.buffers.num_output_distributions);
        let num_threads_per_row = self.specialization.thread_block.required_threads;
        let num_total_threads = checked_num_threads(self.shape.num_total_sampling_inputs, num_threads_per_row);
        if num_threads_per_row == num_total_threads {
            recorder.set_u32(8, num_total_threads);
        } else {
            recorder.bind_u32(
                8,
                TOP_K_REDUCE_NUM_ACTIVE_THREADS_KEY,
                num_threads_per_row,
                num_total_threads,
            );
        }
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

fn add_top_k_reduce_replay_argument(
    shape: TopKSampleShape,
    num_active_sampling_inputs: u32,
    required_threads: u32,
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
    let num_active_threads = checked_num_threads(num_active_sampling_inputs, required_threads);
    let num_total_threads = checked_num_threads(shape.num_total_sampling_inputs, required_threads);
    assert!(num_active_threads <= num_total_threads);
    arguments.set_u32(TOP_K_REDUCE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
}

#[cfg(test)]
#[path = "top_k_test.rs"]
mod tests;
