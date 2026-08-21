use std::mem::size_of;

use inference_backend_metal::components::sampling::top_k;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingLogitsDtype;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::sampling::RuntimeParamRows;

struct TopKSamplingCompute {
    map: top_k::MapCompute,
    reduce: top_k::ReduceCompute,
}

impl TopKSamplingCompute {
    fn new(device: &Device) -> Self {
        Self {
            map: top_k::MapCompute::new(device),
            reduce: top_k::ReduceCompute::new(device),
        }
    }

    fn record_sample<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        let shape = component_shape(shape);
        let map_buffers = top_k::MapBuffers {
            logits: buffers.logits,
            logits_offset_bytes: buffers.logits_offset_bytes,
            tile_token_ids: buffers.partial_token_ids,
            tile_logits: buffers.partial_logits,
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::Sample,
            map_buffers,
        )));
        self.record_reduce_sample(recorder, shape, buffers, output);
    }

    fn record_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::WriteDistribution,
            top_k::MapBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
            },
        )));
        self.record_reduce_write_distribution(recorder, shape, buffers, output);
    }

    fn record_sample_and_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        sample_output: TopKSamplingOutput<'a>,
        sparse_output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::SampleAndWriteDistribution,
            top_k::MapBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.reduce.invoke_sample_and_write_distribution_with_layout(
                shape,
                top_k::SampleAndWriteDistributionBuffers {
                    tile_token_ids: buffers.partial_token_ids,
                    tile_logits: buffers.partial_logits,
                    sampled_token_ids: sample_output.sampled_token_ids,
                    sampled_token_probs: sample_output.sampled_token_probs,
                    distribution_token_ids: sparse_output.token_ids,
                    distribution_probs: sparse_output.probs,
                    runtime_params: buffers.runtime_params,
                    output_distribution_indices: sparse_output.output_distribution_indices,
                    max_k: sparse_output.max_k,
                    num_output_distributions: sparse_output.num_output_distributions,
                },
                self.map.partial_candidate_layout(),
            ),
        ));
    }

    fn record_reduce_sample<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: top_k::Shape,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.reduce.invoke_sample_with_layout(
            shape,
            top_k::SampleBuffers {
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
                token_ids: output.sampled_token_ids,
                token_probs: output.sampled_token_probs,
                runtime_params: buffers.runtime_params,
            },
            self.map.partial_candidate_layout(),
        )));
    }

    fn record_reduce_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: top_k::Shape,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.reduce.invoke_write_distribution_with_layout(
            shape,
            top_k::WriteDistributionBuffers {
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
                distribution_token_ids: output.token_ids,
                distribution_probs: output.probs,
                runtime_params: buffers.runtime_params,
                output_distribution_indices: output.output_distribution_indices,
                max_k: output.max_k,
                num_output_distributions: output.num_output_distributions,
            },
            self.map.partial_candidate_layout(),
        )));
    }
}

#[derive(Clone, Copy)]
struct TopKSamplingComputeBuffers<'a> {
    logits: &'a Buffer,
    logits_offset_bytes: usize,
    partial_token_ids: &'a Buffer,
    partial_logits: &'a Buffer,
    runtime_params: &'a Buffer,
}

struct TopKSamplingScratch {
    partial_token_ids: Buffer,
    partial_logits: Buffer,
}

impl TopKSamplingScratch {
    fn new(device: &Device, bounds: TopKSamplingBounds, map: &top_k::MapCompute) -> Self {
        let max_shape = bounds.max_shape();
        let candidate_count = map.candidate_count(component_shape(max_shape));
        Self {
            partial_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            partial_logits: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32),
        }
    }
}

pub struct TopKSamplingRuntimeParams {
    bounds: TopKSamplingBounds,
    buffer: Buffer,
    rows: RuntimeParamRows,
}

impl TopKSamplingRuntimeParams {
    pub fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        bounds.validate();
        Self {
            bounds,
            buffer: Buffer::new_zeroed_elements(
                device,
                (bounds.max_sampling_inputs as usize)
                    .checked_mul(6)
                    .expect("top-k sampling runtime parameter capacity must fit usize"),
                Dtype::Uint32,
            ),
            rows: RuntimeParamRows::default(),
        }
    }

    pub fn set_configs(&self, configs: &[SamplerConfig], sample_positions: &[u32], domain: SamplingDomain) {
        assert_eq!(
            configs.len(),
            sample_positions.len(),
            "top-k sampling runtime configs must have one logical position per input"
        );
        assert!(
            configs.len() <= self.buffer.len_bytes() / (6 * size_of::<u32>()),
            "top-k sampling runtime inputs exceed capacity"
        );
        for (row, (config, &sample_position)) in configs.iter().zip(sample_positions).enumerate() {
            self.buffer.write_typed(
                row * 6,
                &[
                    config.temperature.to_bits(),
                    config.top_p.to_bits(),
                    config.seed(),
                    sample_position,
                    self.bounds
                        .active_top_k(config)
                        .expect("top-k sampling config should fit sampler bounds"),
                    u32::from(domain),
                ],
            );
        }
        self.rows.set(configs.len() as u32);
    }

    pub fn active_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        self.bounds
            .active_shape(configs)
            .expect("top-k sampling config should fit sampler bounds")
    }

    pub fn max_sampling_inputs(&self) -> u32 {
        self.bounds.max_sampling_inputs
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn consume(&self, shape: TopKSamplingShape) {
        self.validate(shape);
        self.rows.consume(shape.num_active_sampling_inputs, "top-k sampling");
    }

    fn validate(&self, shape: TopKSamplingShape) {
        validate_shape(self.bounds, shape);
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
pub struct TopKSamplingWriteDistributionOutput<'a> {
    pub token_ids: &'a Buffer,
    pub probs: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

pub struct TopKSampling {
    compute: TopKSamplingCompute,
    runtime_params: TopKSamplingRuntimeParams,
    scratch: TopKSamplingScratch,
}

impl TopKSampling {
    pub fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        bounds.validate();
        let compute = TopKSamplingCompute::new(device);
        let scratch = TopKSamplingScratch::new(device, bounds, &compute.map);
        Self {
            compute,
            runtime_params: TopKSamplingRuntimeParams::new(device, bounds),
            scratch,
        }
    }

    pub fn set_configs(&self, configs: &[SamplerConfig], sample_positions: &[u32], domain: SamplingDomain) {
        self.runtime_params.set_configs(configs, sample_positions, domain);
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_sample(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                runtime_params: self.runtime_params.buffer(),
            },
            output,
        );
    }

    pub fn record_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_write_distribution(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                runtime_params: self.runtime_params.buffer(),
            },
            output,
        );
    }

    pub fn record_with_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        inputs: TopKSamplingInputs<'a>,
        sample_output: TopKSamplingOutput<'a>,
        write_distribution_output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_sample_and_write_distribution(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                runtime_params: self.runtime_params.buffer(),
            },
            sample_output,
            write_distribution_output,
        );
    }

    pub fn active_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        self.runtime_params.active_shape(configs)
    }

    pub fn max_sampling_inputs(&self) -> u32 {
        self.runtime_params.max_sampling_inputs()
    }

    pub fn add_replay_arguments(&self, shape: TopKSamplingShape, arguments: &mut ReplayArguments) {
        self.validate(shape);
        self.runtime_params.consume(shape);
        let component_shape = component_shape(shape);
        self.compute
            .map
            .add_replay_arguments(component_shape, shape.num_active_sampling_inputs, arguments);
        self.compute
            .reduce
            .add_replay_arguments(component_shape, shape.num_active_sampling_inputs, arguments);
    }

    fn validate(&self, shape: TopKSamplingShape) {
        self.runtime_params.validate(shape);
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits_dtype: TopKSamplingLogitsDtype,
    pub inputs: TopKSamplingInputs<'a>,
    pub output: TopKSamplingOutput<'a>,
}

impl ReplayLayer for TopKSampling {
    type Input<'a> = TopKSamplingInput<'a>;
    type Output<'a> = TopKSamplingOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        Self::record(
            self,
            recorder,
            input.shape,
            input.logits_dtype,
            input.inputs,
            input.output,
        );
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

fn component_shape(shape: TopKSamplingShape) -> top_k::Shape {
    top_k::Shape {
        num_total_sampling_inputs: shape.num_total_sampling_inputs,
        vocab_size: shape.vocab_size,
        top_k: shape.top_k,
    }
}

fn component_logits_dtype(dtype: TopKSamplingLogitsDtype) -> Dtype {
    match dtype {
        TopKSamplingLogitsDtype::Float32 => Dtype::Float32,
        TopKSamplingLogitsDtype::Bfloat16 => Dtype::Bfloat16,
    }
}

fn validate_shape(bounds: TopKSamplingBounds, shape: TopKSamplingShape) {
    assert!(
        shape.num_total_sampling_inputs <= bounds.max_sampling_inputs,
        "top-k sampling total inputs exceed capacity"
    );
    assert!(shape.num_active_sampling_inputs <= shape.num_total_sampling_inputs);
    assert_eq!(
        shape.vocab_size, bounds.vocab_size,
        "top-k sampling vocab must match capacity"
    );
    assert!(
        shape.top_k > 0 && shape.top_k <= bounds.top_k,
        "top-k sampling width exceeds capacity"
    );
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use inference_backend_metal::metal::Device;
    use inference_executor_core::sampling::SamplerConfig;
    use inference_executor_core::sampling::SamplingDomain;
    use inference_executor_core::sampling::TopKSamplingBounds;

    use super::TopKSamplingRuntimeParams;

    #[test]
    fn test_runtime_parameter_api_set_handles_mixed_configs_and_domains() {
        let device = Device::system_default();
        let bounds = TopKSamplingBounds {
            max_sampling_inputs: 4,
            vocab_size: 128,
            top_k: 16,
        };
        let runtime_params = TopKSamplingRuntimeParams::new(&device, bounds);
        let configs = [
            SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: 7,
            },
            SamplerConfig {
                temperature: 0.75,
                top_k: 8,
                top_p: 0.9,
                seed: 11,
            },
            SamplerConfig {
                temperature: 1.25,
                top_k: 16,
                top_p: 0.5,
                seed: 13,
            },
        ];
        let sample_positions = [2, 5, 9];

        runtime_params.set_configs(&configs, &sample_positions, SamplingDomain::Target);
        let target_shape = runtime_params.active_shape(&configs);
        assert_eq!(target_shape.num_active_sampling_inputs, 3);
        assert_eq!(target_shape.num_total_sampling_inputs, 3);
        assert_eq!(target_shape.top_k, 16);
        assert_eq!(
            runtime_params.buffer().read_typed::<u32>(0, 18),
            vec![
                0.0_f32.to_bits(),
                1.0_f32.to_bits(),
                7,
                2,
                1,
                u32::from(SamplingDomain::Target),
                0.75_f32.to_bits(),
                0.9_f32.to_bits(),
                11,
                5,
                8,
                u32::from(SamplingDomain::Target),
                1.25_f32.to_bits(),
                0.5_f32.to_bits(),
                13,
                9,
                16,
                u32::from(SamplingDomain::Target),
            ]
        );
        runtime_params.consume(target_shape);

        runtime_params.set_configs(&configs[..1], &[12], SamplingDomain::Draft);
        let draft_shape = runtime_params.active_shape(&configs[..1]);
        assert_eq!(
            runtime_params.buffer().read_typed::<u32>(0, 6),
            vec![
                0.0_f32.to_bits(),
                1.0_f32.to_bits(),
                7,
                12,
                1,
                u32::from(SamplingDomain::Draft),
            ]
        );
        runtime_params.consume(draft_shape);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| runtime_params.consume(draft_shape))).is_err());
    }
}
