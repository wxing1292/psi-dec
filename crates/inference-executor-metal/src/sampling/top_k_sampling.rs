use std::mem::size_of;

use inference_backend_metal::components::SAMPLING_NUM_THREADS_PER_THREADBLOCK;
use inference_backend_metal::components::TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY;
use inference_backend_metal::components::TOP_K_REDUCTION_LIMIT;
use inference_backend_metal::components::TOP_K_TILE_NUM_ACTIVE_THREADS_KEY;
use inference_backend_metal::components::TOP_K_VOCAB_TILE_SIZE;
use inference_backend_metal::components::TopKSampleAndSparseDistributionBuffers;
use inference_backend_metal::components::TopKSampleAndSparseDistributionKernel;
use inference_backend_metal::components::TopKSampleBuffers;
use inference_backend_metal::components::TopKSampleKernel;
use inference_backend_metal::components::TopKSampleShape;
use inference_backend_metal::components::TopKSparseDistributionBuffers;
use inference_backend_metal::components::TopKSparseDistributionKernel;
use inference_backend_metal::components::TopKTileBf16BitonicKernel;
use inference_backend_metal::components::TopKTileBf16Kernel;
use inference_backend_metal::components::TopKTileBitonicKernel;
use inference_backend_metal::components::TopKTileBuffers;
use inference_backend_metal::components::TopKTileKernel;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingLogitsDtype;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::sampling::RuntimeParamRows;

struct TopKSamplingKernels {
    topk: TopKTileKernel,
    topk_bitonic: TopKTileBitonicKernel,
    topk_bf16: TopKTileBf16Kernel,
    topk_bf16_bitonic: TopKTileBf16BitonicKernel,
    sample: TopKSampleKernel,
    sample_sparse_distribution: TopKSampleAndSparseDistributionKernel,
    sparse_distribution: TopKSparseDistributionKernel,
}

impl TopKSamplingKernels {
    pub fn new(device: &Device) -> Self {
        Self {
            topk: TopKTileKernel::new(device),
            topk_bitonic: TopKTileBitonicKernel::new(device),
            topk_bf16: TopKTileBf16Kernel::new(device),
            topk_bf16_bitonic: TopKTileBf16BitonicKernel::new(device),
            sample: TopKSampleKernel::new(device),
            sample_sparse_distribution: TopKSampleAndSparseDistributionKernel::new(device),
            sparse_distribution: TopKSparseDistributionKernel::new(device),
        }
    }

    fn record_sample<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        buffers: TopKSampleForwardBuffers<'a>,
    ) {
        let shape = component_shape(shape);
        let tile_buffers = TopKTileBuffers {
            logits: buffers.logits,
            logits_offset_bytes: buffers.logits_offset_bytes,
            tile_token_ids: buffers.tile_token_ids,
            tile_logits: buffers.tile_logits,
        };
        if uses_reduction_tile_pipeline(shape.top_k) {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.topk.invoke_replay(shape, tile_buffers)));
        } else {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_bitonic.invoke_replay(shape, tile_buffers)));
        }
        self.record_sample_from_topk(recorder, shape, buffers);
    }

    fn record_sparse_distribution_bf16<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        buffers: TopKSampleForwardBuffers<'a>,
        sparse_distribution_output: TopKSamplingSparseDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_bf16_bitonic.invoke_replay(
            shape,
            TopKTileBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.tile_token_ids,
                tile_logits: buffers.tile_logits,
            },
        )));
        self.record_sparse_distribution_from_topk(recorder, shape, buffers, sparse_distribution_output);
    }

    fn record_sample_and_sparse_distribution_bf16<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        buffers: TopKSampleForwardBuffers<'a>,
        sparse_distribution_output: TopKSamplingSparseDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_bf16_bitonic.invoke_replay(
            shape,
            TopKTileBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.tile_token_ids,
                tile_logits: buffers.tile_logits,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.sample_sparse_distribution.invoke_replay(
            shape,
            TopKSampleAndSparseDistributionBuffers {
                tile_token_ids: buffers.tile_token_ids,
                tile_logits: buffers.tile_logits,
                sampled_token_ids: buffers.token_ids,
                sampled_token_probs: buffers.token_probs,
                distribution_token_ids: sparse_distribution_output.token_ids,
                distribution_probs: sparse_distribution_output.probs,
                runtime_params: buffers.runtime_params,
                output_distribution_indices: sparse_distribution_output.output_distribution_indices,
                max_k: sparse_distribution_output.max_k,
                num_output_distributions: sparse_distribution_output.num_output_distributions,
            },
        )));
    }

    fn record_sample_bf16<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        buffers: TopKSampleForwardBuffers<'a>,
    ) {
        let shape = component_shape(shape);
        let tile_buffers = TopKTileBuffers {
            logits: buffers.logits,
            logits_offset_bytes: buffers.logits_offset_bytes,
            tile_token_ids: buffers.tile_token_ids,
            tile_logits: buffers.tile_logits,
        };
        if uses_reduction_tile_pipeline(shape.top_k) {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_bf16.invoke_replay(shape, tile_buffers)));
        } else {
            recorder.record_with_barrier_before(ReplayOp::opaque(
                self.topk_bf16_bitonic.invoke_replay(shape, tile_buffers),
            ));
        }
        self.record_sample_from_topk(recorder, shape, buffers);
    }

    fn record_sample_from_topk<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSampleShape,
        buffers: TopKSampleForwardBuffers<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.sample.invoke_replay(
            shape,
            TopKSampleBuffers {
                tile_token_ids: buffers.tile_token_ids,
                tile_logits: buffers.tile_logits,
                token_ids: buffers.token_ids,
                token_probs: buffers.token_probs,
                runtime_params: buffers.runtime_params,
            },
        )));
    }

    fn record_sparse_distribution_from_topk<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSampleShape,
        buffers: TopKSampleForwardBuffers<'a>,
        sparse_distribution_output: TopKSamplingSparseDistributionOutput<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.sparse_distribution.invoke_replay(
            shape,
            TopKSparseDistributionBuffers {
                tile_token_ids: buffers.tile_token_ids,
                tile_logits: buffers.tile_logits,
                distribution_token_ids: sparse_distribution_output.token_ids,
                distribution_probs: sparse_distribution_output.probs,
                runtime_params: buffers.runtime_params,
                output_distribution_indices: sparse_distribution_output.output_distribution_indices,
                max_k: sparse_distribution_output.max_k,
                num_output_distributions: sparse_distribution_output.num_output_distributions,
            },
        )));
    }
}

#[derive(Clone, Copy)]
struct TopKSampleForwardBuffers<'a> {
    logits: &'a Buffer,
    logits_offset_bytes: usize,
    tile_token_ids: &'a Buffer,
    tile_logits: &'a Buffer,
    token_ids: &'a Buffer,
    token_probs: &'a Buffer,
    runtime_params: &'a Buffer,
}

struct TopKSamplingScratch {
    tile_token_ids: Buffer,
    tile_logits: Buffer,
    runtime_params: Buffer,
}

impl TopKSamplingScratch {
    fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        let max_shape = bounds.max_shape();
        Self {
            tile_token_ids: Buffer::new_zeroed_elements(device, max_shape.tile_count(), Dtype::Int32),
            tile_logits: Buffer::new_zeroed_elements(device, max_shape.tile_count(), Dtype::Float32),
            runtime_params: Buffer::new_zeroed_elements(
                device,
                (max_shape.num_total_sampling_inputs as usize)
                    .checked_mul(6)
                    .expect("top-k sampling runtime parameter capacity must fit usize"),
                Dtype::Uint32,
            ),
        }
    }

    fn set_configs(
        &self,
        bounds: TopKSamplingBounds,
        configs: &[SamplerConfig],
        sample_positions: &[u32],
        domain: SamplingDomain,
    ) {
        assert_eq!(
            configs.len(),
            sample_positions.len(),
            "top-k sampling runtime configs must have one logical position per input"
        );
        assert!(
            configs.len() <= self.runtime_params.len_bytes() / (6 * size_of::<u32>()),
            "top-k sampling runtime inputs exceed capacity"
        );
        for (row, (config, &sample_position)) in configs.iter().zip(sample_positions).enumerate() {
            self.runtime_params.write_typed(
                row * 6,
                &[
                    config.temperature.to_bits(),
                    config.top_p.to_bits(),
                    config.seed(),
                    sample_position,
                    bounds
                        .active_top_k(config)
                        .expect("top-k sampling config should fit sampler bounds"),
                    u32::from(domain),
                ],
            );
        }
    }
}

pub struct TopKSamplingOutputBuffers {
    pub token_ids: Buffer,
    pub token_probs: Buffer,
}

impl TopKSamplingOutputBuffers {
    pub fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        let max_shape = bounds.max_shape();
        Self {
            token_ids: Buffer::new_zeroed_elements(device, max_shape.num_total_sampling_inputs as usize, Dtype::Int32),
            token_probs: Buffer::new_zeroed_elements(
                device,
                max_shape.num_total_sampling_inputs as usize,
                Dtype::Float32,
            ),
        }
    }

    pub fn as_output(&self) -> TopKSamplingOutput<'_> {
        TopKSamplingOutput {
            sampled_token_ids: &self.token_ids,
            sampled_token_probs: &self.token_probs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingSparseDistributionOutput<'a> {
    pub token_ids: &'a Buffer,
    pub probs: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

pub struct TopKSampling {
    kernels: TopKSamplingKernels,
    bounds: TopKSamplingBounds,
    scratch: TopKSamplingScratch,
    runtime_param_rows: RuntimeParamRows,
}

impl TopKSampling {
    pub fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        bounds.validate();
        Self {
            kernels: TopKSamplingKernels::new(device),
            bounds,
            scratch: TopKSamplingScratch::new(device, bounds),
            runtime_param_rows: RuntimeParamRows::default(),
        }
    }

    pub fn set_configs(&self, configs: &[SamplerConfig], sample_positions: &[u32], domain: SamplingDomain) {
        self.scratch.set_configs(self.bounds, configs, sample_positions, domain);
        self.runtime_param_rows.set(configs.len(), "top-k sampling");
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        self.validate_input(shape);
        self.kernels.record_sample(
            recorder,
            shape,
            TopKSampleForwardBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                tile_token_ids: &self.scratch.tile_token_ids,
                tile_logits: &self.scratch.tile_logits,
                token_ids: output.sampled_token_ids,
                token_probs: output.sampled_token_probs,
                runtime_params: &self.scratch.runtime_params,
            },
        );
    }

    pub fn record_bf16<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        self.validate_input(shape);
        self.kernels.record_sample_bf16(
            recorder,
            shape,
            TopKSampleForwardBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                tile_token_ids: &self.scratch.tile_token_ids,
                tile_logits: &self.scratch.tile_logits,
                token_ids: output.sampled_token_ids,
                token_probs: output.sampled_token_probs,
                runtime_params: &self.scratch.runtime_params,
            },
        );
    }

    pub fn record_sparse_distribution_bf16<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingSparseDistributionOutput<'a>,
    ) {
        self.validate_input(shape);
        self.kernels.record_sparse_distribution_bf16(
            recorder,
            shape,
            TopKSampleForwardBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                tile_token_ids: &self.scratch.tile_token_ids,
                tile_logits: &self.scratch.tile_logits,
                token_ids: &self.scratch.tile_token_ids,
                token_probs: &self.scratch.tile_logits,
                runtime_params: &self.scratch.runtime_params,
            },
            output,
        );
    }

    pub fn record_bf16_with_sparse_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        inputs: TopKSamplingInputs<'a>,
        sample_output: TopKSamplingOutput<'a>,
        sparse_distribution_output: TopKSamplingSparseDistributionOutput<'a>,
    ) {
        self.validate_input(shape);
        self.kernels.record_sample_and_sparse_distribution_bf16(
            recorder,
            shape,
            TopKSampleForwardBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                tile_token_ids: &self.scratch.tile_token_ids,
                tile_logits: &self.scratch.tile_logits,
                token_ids: sample_output.sampled_token_ids,
                token_probs: sample_output.sampled_token_probs,
                runtime_params: &self.scratch.runtime_params,
            },
            sparse_distribution_output,
        );
    }

    pub fn active_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        self.bounds
            .active_shape(configs)
            .expect("top-k sampling config should fit sampler bounds")
    }

    pub fn add_replay_arguments(&self, shape: TopKSamplingShape, arguments: &mut ReplayArguments) {
        self.validate_input(shape);
        self.runtime_param_rows
            .consume(shape.num_active_sampling_inputs, "top-k sampling");
        if shape.num_total_sampling_inputs > 1 {
            let num_tiles = shape.vocab_size.div_ceil(TOP_K_VOCAB_TILE_SIZE);
            let tile_num_active_threads = shape
                .num_active_sampling_inputs
                .checked_mul(num_tiles)
                .and_then(|threads| threads.checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK))
                .expect("top-k tile active thread count must fit u32");
            let tile_num_total_threads = shape
                .num_total_sampling_inputs
                .checked_mul(num_tiles)
                .and_then(|threads| threads.checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK))
                .expect("top-k tile total thread count must fit u32");
            assert!(tile_num_active_threads <= tile_num_total_threads);
            assert_eq!(tile_num_active_threads % SAMPLING_NUM_THREADS_PER_THREADBLOCK, 0);
            arguments.set_u32(TOP_K_TILE_NUM_ACTIVE_THREADS_KEY, tile_num_active_threads);

            let merge_num_active_threads = shape
                .num_active_sampling_inputs
                .checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK)
                .expect("top-k merge active thread count must fit u32");
            let merge_num_total_threads = shape
                .num_total_sampling_inputs
                .checked_mul(SAMPLING_NUM_THREADS_PER_THREADBLOCK)
                .expect("top-k merge total thread count must fit u32");
            assert!(merge_num_active_threads <= merge_num_total_threads);
            assert_eq!(merge_num_active_threads % SAMPLING_NUM_THREADS_PER_THREADBLOCK, 0);
            arguments.set_u32(TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY, merge_num_active_threads);
        }
    }

    fn validate_input(&self, shape: TopKSamplingShape) {
        assert!(
            shape.num_total_sampling_inputs <= self.bounds.max_sampling_inputs,
            "top-k sampling total inputs exceed capacity"
        );
        assert!(shape.num_active_sampling_inputs <= shape.num_total_sampling_inputs);
        assert_eq!(
            shape.vocab_size, self.bounds.vocab_size,
            "top-k sampling vocab must match capacity"
        );
        assert!(
            shape.top_k > 0 && shape.top_k <= self.bounds.top_k,
            "top-k sampling width exceeds capacity"
        );
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits_dtype: TopKSamplingLogitsDtype,
    pub inputs: TopKSamplingInputs<'a>,
    pub output: TopKSamplingOutput<'a>,
}

impl Layer for TopKSampling {
    type Input<'a> = TopKSamplingInput<'a>;
    type Output<'a> = TopKSamplingOutput<'a>;

    type InputShape = TopKSamplingShape;
    type OutputShape = TopKSamplingShape;

    fn input_shape(&self) -> Self::InputShape {
        self.bounds.max_shape()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.bounds.max_shape()
    }
}

impl ReplayLayer for TopKSampling {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match input.logits_dtype {
            TopKSamplingLogitsDtype::Float32 => {
                Self::record(self, recorder, input.shape, input.inputs, input.output);
            },
            TopKSamplingLogitsDtype::Bfloat16 => {
                Self::record_bf16(self, recorder, input.shape, input.inputs, input.output);
            },
        }
        input.output
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingInputs<'a> {
    pub logits: &'a Buffer,
    pub logits_offset_bytes: usize,
}

#[derive(Clone, Copy)]
pub struct TopKSamplingOutput<'a> {
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
}

fn component_shape(shape: TopKSamplingShape) -> TopKSampleShape {
    TopKSampleShape {
        num_total_sampling_inputs: shape.num_total_sampling_inputs,
        vocab_size: shape.vocab_size,
        top_k: shape.top_k,
    }
}

fn uses_reduction_tile_pipeline(top_k: u32) -> bool {
    top_k <= TOP_K_REDUCTION_LIMIT
}

#[cfg(test)]
mod tests {
    use super::uses_reduction_tile_pipeline;

    #[test]
    fn test_pipeline_boundary() {
        assert!(uses_reduction_tile_pipeline(1));
        assert!(uses_reduction_tile_pipeline(32));
        assert!(!uses_reduction_tile_pipeline(33));
        assert!(!uses_reduction_tile_pipeline(256));
    }
}
