use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const RESIDUAL_RMS_NORM_SOURCE: &str = include_str!("metal/residual_rms_norm.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualRMSNormShape {
    pub num_total_tokens: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl ResidualRMSNormShape {
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
pub struct ResidualRMSNormBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub weight: &'a Buffer,
    pub residual_output: &'a Buffer,
    pub norm_output: &'a Buffer,
}

pub struct ResidualRMSNormKernel {
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
    bf16_vec4_kernel: Kernel,
}

impl ResidualRMSNormKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, RESIDUAL_RMS_NORM_SOURCE, "residual_rms_norm_f32"),
            bf16_kernel: Kernel::new(device, RESIDUAL_RMS_NORM_SOURCE, "residual_rms_norm_bf16"),
            bf16_vec4_kernel: Kernel::new(device, RESIDUAL_RMS_NORM_SOURCE, "residual_rms_norm_bf16_vec4"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: ResidualRMSNormShape,
        buffers: ResidualRMSNormBuffers<'a>,
        eps: f32,
    ) -> ResidualRMSNormInvocation<'a> {
        ResidualRMSNormInvocation {
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
        capacity_shape: ResidualRMSNormShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: ResidualRMSNormBuffers<'a>,
        eps: f32,
    ) -> ResidualRMSNormInvocation<'a> {
        ResidualRMSNormInvocation {
            kernel: self.kernel(capacity_shape),
            shape: capacity_shape,
            buffers,
            eps,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    pub fn invoke_bf16_scalar<'a>(
        &'a self,
        shape: ResidualRMSNormShape,
        buffers: ResidualRMSNormBuffers<'a>,
        eps: f32,
    ) -> ResidualRMSNormInvocation<'a> {
        assert_eq!(shape.dtype, Dtype::Bfloat16);
        ResidualRMSNormInvocation {
            kernel: &self.bf16_kernel,
            shape,
            buffers,
            eps,
            num_active_tokens_key: None,
        }
    }

    pub fn invoke_bf16_vectorized<'a>(
        &'a self,
        shape: ResidualRMSNormShape,
        buffers: ResidualRMSNormBuffers<'a>,
        eps: f32,
    ) -> ResidualRMSNormInvocation<'a> {
        assert_eq!(shape.dtype, Dtype::Bfloat16);
        assert_eq!(
            shape.hidden_dim % 4,
            0,
            "vectorized residual RMSNorm requires hidden_dim divisible by 4"
        );
        ResidualRMSNormInvocation {
            kernel: &self.bf16_vec4_kernel,
            shape,
            buffers,
            eps,
            num_active_tokens_key: None,
        }
    }

    pub fn invoke_owned(
        &self,
        shape: ResidualRMSNormShape,
        buffers: ResidualRMSNormOwnedBuffers,
        eps: f32,
    ) -> ResidualRMSNormReplayInvocation {
        ResidualRMSNormReplayInvocation {
            pipeline: self.kernel(shape).as_raw_retained(),
            shape,
            buffers,
            eps,
            num_active_tokens_key: None,
        }
    }

    fn kernel(&self, shape: ResidualRMSNormShape) -> &Kernel {
        match shape.dtype {
            Dtype::Float32 => &self.f32_kernel,
            Dtype::Bfloat16 if shape.hidden_dim.is_multiple_of(4) => &self.bf16_vec4_kernel,
            Dtype::Bfloat16 => &self.bf16_kernel,
            dtype => panic!("unsupported residual RMSNorm dtype {dtype:?}"),
        }
    }
}

pub struct ResidualRMSNormInvocation<'a> {
    kernel: &'a Kernel,
    shape: ResidualRMSNormShape,
    buffers: ResidualRMSNormBuffers<'a>,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct ResidualRMSNormReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shape: ResidualRMSNormShape,
    buffers: ResidualRMSNormOwnedBuffers,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct DuplicateResidualRMSNormReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shape: ResidualRMSNormShape,
    buffers: DuplicateResidualRMSNormOwnedBuffers,
    duplicate_row_stride: u32,
    duplicate_column_offset: u32,
    eps: f32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

#[derive(Clone)]
pub struct ResidualRMSNormOwnedBuffers {
    lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    lhs_len_bytes: usize,
    rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    rhs_len_bytes: usize,
    weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_len_bytes: usize,
    residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    residual_output_len_bytes: usize,
    norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_output_len_bytes: usize,
}

pub struct DuplicateResidualRMSNormOwnedBuffers {
    lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    lhs_len_bytes: usize,
    rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    rhs_len_bytes: usize,
    weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_len_bytes: usize,
    residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    residual_output_len_bytes: usize,
    duplicate_residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    duplicate_residual_output_len_bytes: usize,
    norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_output_len_bytes: usize,
}

impl ResidualRMSNormOwnedBuffers {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        weight: Retained<ProtocolObject<dyn MTLBuffer>>,
        weight_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
        norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        norm_output_len_bytes: usize,
    ) -> Self {
        Self {
            lhs,
            lhs_len_bytes,
            rhs,
            rhs_len_bytes,
            weight,
            weight_len_bytes,
            residual_output,
            residual_output_len_bytes,
            norm_output,
            norm_output_len_bytes,
        }
    }

    fn from_buffers(buffers: ResidualRMSNormBuffers<'_>) -> Self {
        Self::new(
            buffers.lhs.as_raw_retained(),
            buffers.lhs.len_bytes(),
            buffers.rhs.as_raw_retained(),
            buffers.rhs.len_bytes(),
            buffers.weight.as_raw_retained(),
            buffers.weight.len_bytes(),
            buffers.residual_output.as_raw_retained(),
            buffers.residual_output.len_bytes(),
            buffers.norm_output.as_raw_retained(),
            buffers.norm_output.len_bytes(),
        )
    }
}

impl DuplicateResidualRMSNormOwnedBuffers {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        weight: Retained<ProtocolObject<dyn MTLBuffer>>,
        weight_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
        duplicate_residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        duplicate_residual_output_len_bytes: usize,
        norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        norm_output_len_bytes: usize,
    ) -> Self {
        Self {
            lhs,
            lhs_len_bytes,
            rhs,
            rhs_len_bytes,
            weight,
            weight_len_bytes,
            residual_output,
            residual_output_len_bytes,
            duplicate_residual_output,
            duplicate_residual_output_len_bytes,
            norm_output,
            norm_output_len_bytes,
        }
    }
}

impl Operator for ResidualRMSNormInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_read(2, self.buffers.weight, 0);
        builder.set_buffer_write(3, self.buffers.residual_output, 0);
        builder.set_buffer_write(4, self.buffers.norm_output, 0);
        record_num_active_tokens(builder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(6, self.shape.hidden_dim);
        builder.set_f32(7, self.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * NUM_THREADS_PER_THREADBLOCK,
            NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl Operator for ResidualRMSNormReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        builder.set_retained_buffer_read(2, &self.buffers.weight, 0);
        builder.set_retained_buffer_write(3, &self.buffers.residual_output, 0);
        builder.set_retained_buffer_write(4, &self.buffers.norm_output, 0);
        record_num_active_tokens(builder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(6, self.shape.hidden_dim);
        builder.set_f32(7, self.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * NUM_THREADS_PER_THREADBLOCK,
            NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl Operator for DuplicateResidualRMSNormReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        builder.set_retained_buffer_read(2, &self.buffers.weight, 0);
        builder.set_retained_buffer_write(3, &self.buffers.residual_output, 0);
        builder.set_retained_buffer_write(4, &self.buffers.duplicate_residual_output, 0);
        builder.set_retained_buffer_write(5, &self.buffers.norm_output, 0);
        record_num_active_tokens(builder, 6, self.shape.num_total_tokens, self.num_active_tokens_key);
        builder.set_u32(7, self.shape.hidden_dim);
        if self.shape.hidden_dim.is_multiple_of(4) {
            builder.set_u32(8, self.duplicate_row_stride / 4);
            builder.set_u32(9, self.duplicate_column_offset / 4);
        } else {
            builder.set_u32(8, self.duplicate_row_stride);
            builder.set_u32(9, self.duplicate_column_offset);
        }
        builder.set_f32(10, self.eps);
        builder.dispatch_1d(
            self.shape.num_total_tokens as usize * NUM_THREADS_PER_THREADBLOCK,
            NUM_THREADS_PER_THREADBLOCK,
        );
    }
}

impl ResidualRMSNormInvocation<'_> {
    fn validate(&self) {
        self.shape.validate();
        assert!(self.eps > 0.0);
        assert!(self.buffers.lhs.len_bytes() >= self.shape.bytes());
        assert!(self.buffers.rhs.len_bytes() >= self.shape.bytes());
        assert!(self.buffers.weight.len_bytes() >= self.shape.weight_bytes());
        assert!(self.buffers.residual_output.len_bytes() >= self.shape.bytes());
        assert!(self.buffers.norm_output.len_bytes() >= self.shape.bytes());
    }
}

impl ResidualRMSNormReplayInvocation {
    pub fn new(shape: ResidualRMSNormShape, buffers: ResidualRMSNormOwnedBuffers, eps: f32) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        Self {
            pipeline: Kernel::new(
                &device,
                RESIDUAL_RMS_NORM_SOURCE,
                residual_rms_norm_function_name(shape),
            )
            .as_raw_retained(),
            shape,
            buffers,
            eps,
            num_active_tokens_key: None,
        }
    }

    pub fn new_bucketed(
        capacity_shape: ResidualRMSNormShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: ResidualRMSNormOwnedBuffers,
        eps: f32,
    ) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        Self {
            pipeline: Kernel::new(
                &device,
                RESIDUAL_RMS_NORM_SOURCE,
                residual_rms_norm_function_name(capacity_shape),
            )
            .as_raw_retained(),
            shape: capacity_shape,
            buffers,
            eps,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    fn validate(&self) {
        self.shape.validate();
        assert!(self.eps > 0.0);
        assert!(self.buffers.lhs_len_bytes >= self.shape.bytes());
        assert!(self.buffers.rhs_len_bytes >= self.shape.bytes());
        assert!(self.buffers.weight_len_bytes >= self.shape.weight_bytes());
        assert!(self.buffers.residual_output_len_bytes >= self.shape.bytes());
        assert!(self.buffers.norm_output_len_bytes >= self.shape.bytes());
    }
}

impl DuplicateResidualRMSNormReplayInvocation {
    pub fn new(
        shape: ResidualRMSNormShape,
        buffers: DuplicateResidualRMSNormOwnedBuffers,
        duplicate_row_stride: u32,
        duplicate_column_offset: u32,
        eps: f32,
    ) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        Self {
            pipeline: Kernel::new(
                &device,
                RESIDUAL_RMS_NORM_SOURCE,
                duplicate_residual_rms_norm_function_name(shape),
            )
            .as_raw_retained(),
            shape,
            buffers,
            duplicate_row_stride,
            duplicate_column_offset,
            eps,
            num_active_tokens_key: None,
        }
    }

    pub fn new_bucketed(
        capacity_shape: ResidualRMSNormShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: DuplicateResidualRMSNormOwnedBuffers,
        duplicate_row_stride: u32,
        duplicate_column_offset: u32,
        eps: f32,
    ) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        Self {
            pipeline: Kernel::new(
                &device,
                RESIDUAL_RMS_NORM_SOURCE,
                duplicate_residual_rms_norm_function_name(capacity_shape),
            )
            .as_raw_retained(),
            shape: capacity_shape,
            buffers,
            duplicate_row_stride,
            duplicate_column_offset,
            eps,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    fn validate(&self) {
        self.shape.validate();
        assert_eq!(self.shape.dtype, Dtype::Bfloat16);
        assert!(self.eps > 0.0);
        assert!(self.duplicate_row_stride >= self.shape.hidden_dim);
        assert!(self.duplicate_column_offset <= self.duplicate_row_stride - self.shape.hidden_dim);
        if self.shape.hidden_dim.is_multiple_of(4) {
            assert!(self.duplicate_row_stride.is_multiple_of(4));
            assert!(self.duplicate_column_offset.is_multiple_of(4));
        }
        assert!(self.buffers.lhs_len_bytes >= self.shape.bytes());
        assert!(self.buffers.rhs_len_bytes >= self.shape.bytes());
        assert!(self.buffers.weight_len_bytes >= self.shape.weight_bytes());
        assert!(self.buffers.residual_output_len_bytes >= self.shape.bytes());
        assert!(self.buffers.norm_output_len_bytes >= self.shape.bytes());
        let last_row_start = (self.shape.num_total_tokens as usize - 1)
            .checked_mul(self.duplicate_row_stride as usize)
            .expect("duplicate residual last-row offset must fit usize");
        let required_values = last_row_start
            .checked_add(self.duplicate_column_offset as usize)
            .and_then(|value| value.checked_add(self.shape.hidden_dim as usize))
            .expect("duplicate residual value count must fit usize");
        let required_bytes = required_values
            .checked_mul(Dtype::Bfloat16.item_size())
            .expect("duplicate residual byte count must fit usize");
        assert!(self.buffers.duplicate_residual_output_len_bytes >= required_bytes);
        for other in [
            &self.buffers.lhs,
            &self.buffers.rhs,
            &self.buffers.weight,
            &self.buffers.residual_output,
            &self.buffers.norm_output,
        ] {
            assert!(
                !std::ptr::eq(
                    Retained::as_ptr(&self.buffers.duplicate_residual_output),
                    Retained::as_ptr(other),
                ),
                "duplicate residual output must not alias another fused residual/RMSNorm buffer"
            );
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

fn residual_rms_norm_function_name(shape: ResidualRMSNormShape) -> &'static str {
    match shape.dtype {
        Dtype::Float32 => "residual_rms_norm_f32",
        Dtype::Bfloat16 if shape.hidden_dim.is_multiple_of(4) => "residual_rms_norm_bf16_vec4",
        Dtype::Bfloat16 => "residual_rms_norm_bf16",
        dtype => panic!("unsupported residual RMSNorm dtype {dtype:?}"),
    }
}

fn duplicate_residual_rms_norm_function_name(shape: ResidualRMSNormShape) -> &'static str {
    match shape.dtype {
        Dtype::Bfloat16 if shape.hidden_dim.is_multiple_of(4) => "duplicate_residual_rms_norm_bf16_vec4",
        Dtype::Bfloat16 => "duplicate_residual_rms_norm_bf16",
        dtype => panic!("unsupported duplicate residual RMSNorm dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use half::bf16;

    use super::*;
    use crate::components::RMSNormBuffers;
    use crate::components::RMSNormKernel;
    use crate::components::RMSNormShape;
    use crate::components::ResidualBuffers;
    use crate::components::ResidualKernel;
    use crate::components::ResidualShape;
    use crate::metal::Stream;

    #[test]
    fn test_bf16_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let residual = ResidualKernel::new(&device);
        let rms_norm = RMSNormKernel::new(&device);
        let fused = ResidualRMSNormKernel::new(&device);
        let tokens = 3;
        let hidden_dim = 128;
        let num_values = tokens * hidden_dim;
        let lhs = bf16_buffer(&device, num_values, 13, -0.75);
        let rhs = bf16_buffer(&device, num_values, 17, -0.25);
        let weight = bf16_buffer(&device, hidden_dim, 5, 0.001);
        let unfused_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let unfused_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_scalar_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_scalar_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_vec4_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_vec4_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());

        let mut builder = stream.create_replay_program();
        builder.record(residual.invoke(
            ResidualShape::bf16(num_values as u32),
            ResidualBuffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &unfused_residual,
            },
        ));
        builder.record_with_barrier_before(rms_norm.invoke(
            RMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            RMSNormBuffers {
                input: &unfused_residual,
                weight: &weight,
                output: &unfused_norm,
            },
            1.0e-6,
        ));
        stream.submit_replay(&builder.build()).wait();

        let mut builder = stream.create_replay_program();
        builder.record(fused.invoke_bf16_scalar(
            ResidualRMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            ResidualRMSNormBuffers {
                lhs: &lhs,
                rhs: &rhs,
                weight: &weight,
                residual_output: &fused_scalar_residual,
                norm_output: &fused_scalar_norm,
            },
            1.0e-6,
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            unfused_residual.read_typed::<u16>(0, num_values),
            fused_scalar_residual.read_typed::<u16>(0, num_values)
        );
        assert_eq!(
            unfused_norm.read_typed::<u16>(0, num_values),
            fused_scalar_norm.read_typed::<u16>(0, num_values)
        );

        let mut builder = stream.create_replay_program();
        builder.record(fused.invoke(
            ResidualRMSNormShape::bf16(tokens as u32, hidden_dim as u32),
            ResidualRMSNormBuffers {
                lhs: &lhs,
                rhs: &rhs,
                weight: &weight,
                residual_output: &fused_vec4_residual,
                norm_output: &fused_vec4_norm,
            },
            1.0e-6,
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            unfused_residual.read_typed::<u16>(0, num_values),
            fused_vec4_residual.read_typed::<u16>(0, num_values)
        );
        assert_eq!(
            unfused_norm.read_typed::<u16>(0, num_values),
            fused_vec4_norm.read_typed::<u16>(0, num_values)
        );
    }

    fn bf16_buffer(device: &Device, len: usize, step: usize, base: f32) -> Buffer {
        let values = (0..len)
            .map(|index| bf16::from_f32(base + ((index * step) % 23) as f32 * 0.03125).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &values)
    }
}
