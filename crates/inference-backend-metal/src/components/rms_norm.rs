use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::components::residual_rms_norm::DuplicateResidualRMSNormOwnedBuffers;
use crate::components::residual_rms_norm::ResidualRMSNormOwnedBuffers;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const RMS_NORM_SOURCE: &str = include_str!("metal/rms_norm.metal");

const RMS_NUM_THREADS_PER_THREADBLOCK: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RMSNormShape {
    pub num_total_tokens: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl RMSNormShape {
    pub fn f32(num_total_tokens: u32, hidden_dim: u32) -> Self {
        Self {
            num_total_tokens,
            hidden_dim,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_total_tokens: u32, hidden_dim: u32) -> Self {
        Self {
            num_total_tokens,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
        assert!(self.hidden_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_values(self) -> usize {
        self.num_total_tokens as usize * self.hidden_dim as usize
    }

    pub fn bytes(self) -> usize {
        self.num_values() * self.dtype.item_size()
    }

    pub fn weight_bytes(self) -> usize {
        self.hidden_dim as usize * self.dtype.item_size()
    }
}

#[derive(Clone, Copy)]
pub struct RMSNormBuffers<'a> {
    pub input: &'a Buffer,
    pub weight: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct RMSNormKernel {
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
}

impl RMSNormKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, RMS_NORM_SOURCE, "rms_norm_f32"),
            bf16_kernel: Kernel::new(device, RMS_NORM_SOURCE, "rms_norm_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: RMSNormShape, buffers: RMSNormBuffers<'a>, eps: f32) -> RMSNormInvocation<'a> {
        RMSNormInvocation {
            kernel: self.kernel(shape),
            shape,
            buffers,
            eps,
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active token count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: RMSNormShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: RMSNormBuffers<'a>,
        eps: f32,
    ) -> RMSNormInvocation<'a> {
        RMSNormInvocation {
            kernel: self.kernel(capacity_shape),
            shape: capacity_shape,
            buffers,
            eps,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    fn kernel(&self, shape: RMSNormShape) -> &Kernel {
        match shape.dtype {
            Dtype::Float32 => &self.f32_kernel,
            Dtype::Bfloat16 => &self.bf16_kernel,
            dtype => panic!("unsupported RMSNorm dtype {dtype:?}"),
        }
    }
}

pub struct RMSNormInvocation<'a> {
    kernel: &'a Kernel,
    shape: RMSNormShape,
    buffers: RMSNormBuffers<'a>,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct RMSNormReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shape: RMSNormShape,
    buffers: RMSNormOwnedBuffers,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct RMSNormReplayOp {
    shape: RMSNormShape,
    buffers: RMSNormOwnedBuffers,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

#[derive(Clone)]
struct RMSNormOwnedBuffers {
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    input_len_bytes: usize,
    weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_len_bytes: usize,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    output_len_bytes: usize,
}

impl Operator for RMSNormInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl Operator for RMSNormReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl RMSNormInvocation<'_> {
    pub fn into_replay_op(self) -> RMSNormReplayOp {
        RMSNormReplayOp {
            shape: self.shape,
            buffers: RMSNormOwnedBuffers {
                input: self.buffers.input.as_raw_retained(),
                input_len_bytes: self.buffers.input.len_bytes(),
                weight: self.buffers.weight.as_raw_retained(),
                weight_len_bytes: self.buffers.weight.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
            eps: self.eps,
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }

    fn validate(&self) {
        self.shape.validate();
        assert!(self.eps > 0.0);
        assert!(self.buffers.input.len_bytes() >= self.shape.bytes());
        assert!(self.buffers.weight.len_bytes() >= self.shape.weight_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.weight, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        record_num_active_tokens(builder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(4, self.shape.hidden_dim);
        builder.set_f32(5, self.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * RMS_NUM_THREADS_PER_THREADBLOCK,
            RMS_NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl RMSNormReplayInvocation {
    fn validate(&self) {
        self.shape.validate();
        assert!(self.eps > 0.0);
        assert!(self.buffers.input_len_bytes >= self.shape.bytes());
        assert!(self.buffers.weight_len_bytes >= self.shape.weight_bytes());
        assert!(self.buffers.output_len_bytes >= self.shape.bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.input, 0);
        builder.set_retained_buffer_read(1, &self.buffers.weight, 0);
        builder.set_retained_buffer_write(2, &self.buffers.output, 0);
        record_num_active_tokens(builder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(4, self.shape.hidden_dim);
        builder.set_f32(5, self.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * RMS_NUM_THREADS_PER_THREADBLOCK,
            RMS_NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl RMSNormReplayOp {
    pub fn into_replay(self) -> RMSNormReplayInvocation {
        let device = Device::from_raw_retained(self.buffers.input.device());
        RMSNormReplayInvocation {
            pipeline: Kernel::new(&device, RMS_NORM_SOURCE, rms_norm_function_name(self.shape)).as_raw_retained(),
            shape: self.shape,
            buffers: self.buffers,
            eps: self.eps,
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }

    pub fn shape(&self) -> RMSNormShape {
        self.shape
    }

    pub fn input_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffers.input
    }

    pub fn into_residual_rms_norm_buffers(
        self,
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
    ) -> (ResidualRMSNormOwnedBuffers, f32, Option<ReplayParameterKey>) {
        (
            ResidualRMSNormOwnedBuffers::new(
                lhs,
                lhs_len_bytes,
                rhs,
                rhs_len_bytes,
                self.buffers.weight,
                self.buffers.weight_len_bytes,
                residual_output,
                residual_output_len_bytes,
                self.buffers.output,
                self.buffers.output_len_bytes,
            ),
            self.eps,
            self.num_active_tokens_key,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn into_duplicate_residual_rms_norm_buffers(
        self,
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
        duplicate_residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        duplicate_residual_output_len_bytes: usize,
    ) -> (DuplicateResidualRMSNormOwnedBuffers, f32, Option<ReplayParameterKey>) {
        (
            DuplicateResidualRMSNormOwnedBuffers::new(
                lhs,
                lhs_len_bytes,
                rhs,
                rhs_len_bytes,
                self.buffers.weight,
                self.buffers.weight_len_bytes,
                residual_output,
                residual_output_len_bytes,
                duplicate_residual_output,
                duplicate_residual_output_len_bytes,
                self.buffers.output,
                self.buffers.output_len_bytes,
            ),
            self.eps,
            self.num_active_tokens_key,
        )
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

fn rms_norm_function_name(shape: RMSNormShape) -> &'static str {
    match shape.dtype {
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
        let kernel = RMSNormKernel::new(&device);
        let num_active_tokens = 2_u32;
        let num_total_tokens = 4_u32;
        let hidden_dim = 8_u32;
        let shape = RMSNormShape::f32(num_total_tokens, hidden_dim);
        let input_values = (0..shape.num_values())
            .map(|index| index as f32 * 0.03125 - 0.5)
            .collect::<Vec<_>>();
        let weight_values = (0..hidden_dim)
            .map(|index| 0.75 + index as f32 * 0.03125)
            .collect::<Vec<_>>();
        let input = Buffer::from_slice(&device, &input_values);
        let weight = Buffer::from_slice(&device, &weight_values);
        let sentinel = -321.0_f32;
        let output = Buffer::from_slice(&device, &vec![sentinel; shape.num_values()]);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_bucketed(
            shape,
            NUM_ACTIVE_TOKENS,
            RMSNormBuffers {
                input: &input,
                weight: &weight,
                output: &output,
            },
            1.0e-6,
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
            output.read_typed::<f32>(active_values, shape.num_values() - active_values),
            vec![sentinel; shape.num_values() - active_values]
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
