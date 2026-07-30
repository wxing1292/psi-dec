use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const RMS_NORM_SOURCE: &str = include_str!("metal/rms_norm.metal");

const RMS_NUM_THREADS_PER_THREADBLOCK: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RMSNormConfig {
    pub hidden_dim: u32,
    pub eps: f32,
    pub io_dtype: Dtype,
}

impl RMSNormConfig {
    pub fn f32(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Float32,
        }
    }

    pub fn bf16(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_values(self, shape: RMSNormShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_total_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("RMSNorm value count must fit usize")
    }

    pub fn bytes(self, shape: RMSNormShape) -> usize {
        self.num_values(shape)
            .checked_mul(self.io_dtype.item_size())
            .expect("RMSNorm byte length must fit usize")
    }

    pub fn weight_bytes(self) -> usize {
        (self.hidden_dim as usize)
            .checked_mul(self.io_dtype.item_size())
            .expect("RMSNorm weight byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RMSNormShape {
    pub num_total_tokens: u32,
}

impl RMSNormShape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
    }
}

#[derive(Clone, Copy)]
pub struct RMSNormBuffers<'a> {
    pub input: &'a Buffer,
    pub weight: &'a Buffer,
    pub output: &'a Buffer,
}

/// RMSNorm data flow:
///
/// ```text
/// buffers.input ----\
///                    +--> RMSNorm --> buffers.output
/// buffers.weight ---/
/// ```
pub struct RMSNormKernel {
    config: RMSNormConfig,
    kernel: Kernel,
}

impl RMSNormKernel {
    pub fn new(device: &Device, config: RMSNormConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, RMS_NORM_SOURCE, rms_norm_function_name(config)),
        }
    }

    pub fn invoke<'a>(&'a self, shape: RMSNormShape, buffers: RMSNormBuffers<'a>) -> RMSNormInvocation<'a> {
        RMSNormInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape,
            buffers,
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active token count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: RMSNormShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: RMSNormBuffers<'a>,
    ) -> RMSNormInvocation<'a> {
        RMSNormInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape: capacity_shape,
            buffers,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }
}

pub struct RMSNormInvocation<'a> {
    kernel: &'a Kernel,
    config: RMSNormConfig,
    shape: RMSNormShape,
    buffers: RMSNormBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub(super) struct RMSNormReplayInvocation {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: RMSNormConfig,
    pub(super) shape: RMSNormShape,
    pub(super) buffers: RMSNormOwnedBuffers,
    pub(super) num_active_tokens_key: Option<ReplayParameterKey>,
}

pub(super) struct RMSNormReplayOp {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: RMSNormConfig,
    pub(super) shape: RMSNormShape,
    pub(super) buffers: RMSNormOwnedBuffers,
    pub(super) num_active_tokens_key: Option<ReplayParameterKey>,
}

#[derive(Clone)]
pub(super) struct RMSNormOwnedBuffers {
    pub(super) input: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) input_len_bytes: usize,
    pub(super) weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) weight_len_bytes: usize,
    pub(super) output: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) output_len_bytes: usize,
}

impl Operator for RMSNormInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.weight, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        record_num_active_tokens(builder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(4, self.config.hidden_dim);
        builder.set_f32(5, self.config.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * RMS_NUM_THREADS_PER_THREADBLOCK,
            RMS_NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl Operator for RMSNormReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.input, 0);
        builder.set_retained_buffer_read(1, &self.buffers.weight, 0);
        builder.set_retained_buffer_write(2, &self.buffers.output, 0);
        record_num_active_tokens(builder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(4, self.config.hidden_dim);
        builder.set_f32(5, self.config.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * RMS_NUM_THREADS_PER_THREADBLOCK,
            RMS_NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl RMSNormInvocation<'_> {
    pub(super) fn into_replay_op(self) -> RMSNormReplayOp {
        RMSNormReplayOp {
            pipeline: self.kernel.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers: RMSNormOwnedBuffers {
                input: self.buffers.input.as_raw_retained(),
                input_len_bytes: self.buffers.input.len_bytes(),
                weight: self.buffers.weight.as_raw_retained(),
                weight_len_bytes: self.buffers.weight.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }

    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        assert!(self.buffers.input.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.weight.len_bytes() >= self.config.weight_bytes());
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }
}

impl RMSNormReplayInvocation {
    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        assert!(self.buffers.input_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.weight_len_bytes >= self.config.weight_bytes());
        assert!(self.buffers.output_len_bytes >= self.config.bytes(self.shape));
    }
}

impl RMSNormReplayOp {
    pub(super) fn into_replay(self) -> RMSNormReplayInvocation {
        RMSNormReplayInvocation {
            pipeline: self.pipeline,
            config: self.config,
            shape: self.shape,
            buffers: self.buffers,
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }
}

fn record_num_active_tokens(
    builder: &CommandRecorder,
    binding_index: usize,
    token_capacity: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => builder.bind_u32(binding_index, key, 1, token_capacity),
        None => builder.set_u32(binding_index, token_capacity),
    }
}

fn rms_norm_function_name(config: RMSNormConfig) -> &'static str {
    match config.io_dtype {
        Dtype::Float32 => "rms_norm_f32",
        Dtype::Bfloat16 => "rms_norm_bf16",
        dtype => panic!("unsupported RMSNorm dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use inference_executor_core::reference::rms_norm_reference;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.rms_norm.num_active_tokens");

    #[test]
    fn test_bucketed_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let num_active_tokens = 2_u32;
        let num_total_tokens = 4_u32;
        let hidden_dim = 8_u32;
        let config = RMSNormConfig::f32(hidden_dim, 1.0e-6);
        let shape = RMSNormShape { num_total_tokens };
        let kernel = RMSNormKernel::new(&device, config);
        let input_values = (0..config.num_values(shape))
            .map(|index| index as f32 * 0.03125 - 0.5)
            .collect::<Vec<_>>();
        let weight_values = (0..hidden_dim)
            .map(|index| 0.75 + index as f32 * 0.03125)
            .collect::<Vec<_>>();
        let input = Buffer::from_slice(&device, &input_values);
        let weight = Buffer::from_slice(&device, &weight_values);
        let sentinel = -321.0_f32;
        let output = Buffer::from_slice(&device, &vec![sentinel; config.num_values(shape)]);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            shape,
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &input,
                weight: &weight,
                output: &output,
            },
        ));
        let replay = builder.build();
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();

        let active_values = num_active_tokens as usize * hidden_dim as usize;
        let expected = rms_norm_reference(
            &input_values[..active_values],
            &weight_values,
            None,
            num_active_tokens as usize,
            hidden_dim as usize,
            1.0e-6,
        );
        assert_close(&output.read_typed::<f32>(0, active_values), &expected, 1.0e-5);
        assert_eq!(
            output.read_typed::<f32>(active_values, config.num_values(shape) - active_values),
            vec![sentinel; config.num_values(shape) - active_values]
        );
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "RMSNorm mismatch at {index}: expected={expected} actual={actual} tolerance={tolerance}"
            );
        }
    }
}
