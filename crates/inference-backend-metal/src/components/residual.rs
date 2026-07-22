use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::components::ResidualRMSNormShape;
use crate::components::residual_rms_norm::DuplicateResidualRMSNormReplayInvocation;
use crate::components::residual_rms_norm::ResidualRMSNormReplayInvocation;
use crate::components::rms_norm::RMSNormReplayOp;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const RESIDUAL_ADD_SOURCE: &str = include_str!("metal/residual.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualShape {
    pub num_values: u32,
    pub lhs_dtype: Dtype,
    pub rhs_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl ResidualShape {
    pub fn f32(num_values: u32) -> Self {
        Self {
            num_values,
            lhs_dtype: Dtype::Float32,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_values: u32) -> Self {
        Self {
            num_values,
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn bf16_f32_to_bf16(num_values: u32) -> Self {
        Self {
            num_values,
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_values > 0);
        assert!(
            matches!(
                (self.lhs_dtype, self.rhs_dtype, self.output_dtype),
                (Dtype::Float32, Dtype::Float32, Dtype::Float32)
                    | (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16)
                    | (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16)
            ),
            "unsupported residual dtype combination: lhs={:?}, rhs={:?}, output={:?}",
            self.lhs_dtype,
            self.rhs_dtype,
            self.output_dtype
        );
    }

    pub fn lhs_bytes(self) -> usize {
        self.num_values as usize * self.lhs_dtype.item_size()
    }

    pub fn rhs_bytes(self) -> usize {
        self.num_values as usize * self.rhs_dtype.item_size()
    }

    pub fn output_bytes(self) -> usize {
        self.num_values as usize * self.output_dtype.item_size()
    }
}

#[derive(Clone, Copy)]
pub struct ResidualBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct ResidualKernel {
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
    bf16_f32_to_bf16_kernel: Kernel,
}

impl ResidualKernel {
    pub fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, RESIDUAL_ADD_SOURCE, "residual_add_f32"),
            bf16_kernel: Kernel::new(device, RESIDUAL_ADD_SOURCE, "residual_add_bf16"),
            bf16_f32_to_bf16_kernel: Kernel::new(device, RESIDUAL_ADD_SOURCE, "residual_add_bf16_f32_to_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: ResidualShape, buffers: ResidualBuffers<'a>) -> ResidualInvocation<'a> {
        ResidualInvocation {
            kernel: self.kernel(shape),
            shape,
            buffers,
        }
    }

    fn kernel(&self, shape: ResidualShape) -> &Kernel {
        match (shape.lhs_dtype, shape.rhs_dtype, shape.output_dtype) {
            (Dtype::Float32, Dtype::Float32, Dtype::Float32) => &self.f32_kernel,
            (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16) => &self.bf16_kernel,
            (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16) => &self.bf16_f32_to_bf16_kernel,
            (lhs_dtype, rhs_dtype, output_dtype) => {
                panic!(
                    "unsupported residual add dtype combination: lhs={lhs_dtype:?}, rhs={rhs_dtype:?}, \
                     output={output_dtype:?}"
                )
            },
        }
    }
}

pub struct ResidualInvocation<'a> {
    kernel: &'a Kernel,
    shape: ResidualShape,
    buffers: ResidualBuffers<'a>,
}

pub struct ResidualReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shape: ResidualShape,
    buffers: ResidualOwnedBuffers,
}

pub struct ResidualReplayOp {
    shape: ResidualShape,
    buffers: ResidualOwnedBuffers,
}

pub struct DuplicateResidualReplayOp {
    residual: ResidualReplayOp,
    duplicate_output: DuplicateResidualOwnedOutput,
}

#[derive(Clone, Copy)]
pub struct DuplicateResidualOutput<'a> {
    pub buffer: &'a Buffer,
    pub row_stride: u32,
    pub column_offset: u32,
}

struct DuplicateResidualOwnedOutput {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_len_bytes: usize,
    row_stride: u32,
    column_offset: u32,
}

#[derive(Clone)]
struct ResidualOwnedBuffers {
    lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    lhs_len_bytes: usize,
    rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    rhs_len_bytes: usize,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    output_len_bytes: usize,
}

impl Operator for ResidualInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl Operator for ResidualReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl ResidualInvocation<'_> {
    pub fn into_replay_op(self) -> ResidualReplayOp {
        ResidualReplayOp {
            shape: self.shape,
            buffers: ResidualOwnedBuffers {
                lhs: self.buffers.lhs.as_raw_retained(),
                lhs_len_bytes: self.buffers.lhs.len_bytes(),
                rhs: self.buffers.rhs.as_raw_retained(),
                rhs_len_bytes: self.buffers.rhs.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
        }
    }

    pub fn into_duplicate_replay_op(self, duplicate_output: DuplicateResidualOutput<'_>) -> DuplicateResidualReplayOp {
        DuplicateResidualReplayOp {
            residual: self.into_replay_op(),
            duplicate_output: DuplicateResidualOwnedOutput {
                buffer: duplicate_output.buffer.as_raw_retained(),
                buffer_len_bytes: duplicate_output.buffer.len_bytes(),
                row_stride: duplicate_output.row_stride,
                column_offset: duplicate_output.column_offset,
            },
        }
    }

    fn validate(&self) {
        self.shape.validate();
        assert!(self.buffers.lhs.len_bytes() >= self.shape.lhs_bytes());
        assert!(self.buffers.rhs.len_bytes() >= self.shape.rhs_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

impl ResidualReplayOp {
    pub fn into_replay(self) -> ResidualReplayInvocation {
        let device = Device::from_raw_retained(self.buffers.lhs.device());
        let pipeline = Kernel::new(&device, RESIDUAL_ADD_SOURCE, residual_function_name(self.shape)).as_raw_retained();
        ResidualReplayInvocation {
            pipeline,
            shape: self.shape,
            buffers: self.buffers,
        }
    }

    pub fn fuse_rms_norm(self, rms_norm: RMSNormReplayOp) -> ResidualRMSNormReplayInvocation {
        let rms_shape = rms_norm.shape();
        let residual_values = rms_shape
            .num_total_tokens
            .checked_mul(rms_shape.hidden_dim)
            .expect("residual RMSNorm value count must fit u32");
        assert_eq!(self.shape.num_values, residual_values);
        assert_eq!(self.shape.lhs_dtype, rms_shape.dtype);
        assert_eq!(self.shape.rhs_dtype, rms_shape.dtype);
        assert_eq!(self.shape.output_dtype, rms_shape.dtype);
        assert!(
            std::ptr::eq(
                Retained::as_ptr(&self.buffers.output),
                Retained::as_ptr(rms_norm.input_buffer()),
            ),
            "residual output must be the fused RMSNorm input"
        );

        let (buffers, eps, num_active_tokens_key) = rms_norm.into_residual_rms_norm_buffers(
            self.buffers.lhs,
            self.buffers.lhs_len_bytes,
            self.buffers.rhs,
            self.buffers.rhs_len_bytes,
            self.buffers.output,
            self.buffers.output_len_bytes,
        );
        let shape = ResidualRMSNormShape {
            num_total_tokens: rms_shape.num_total_tokens,
            hidden_dim: rms_shape.hidden_dim,
            dtype: rms_shape.dtype,
        };
        match num_active_tokens_key {
            Some(key) => ResidualRMSNormReplayInvocation::new_bucketed(shape, key, buffers, eps),
            None => ResidualRMSNormReplayInvocation::new(shape, buffers, eps),
        }
    }
}

impl DuplicateResidualReplayOp {
    pub fn fuse_rms_norm(self, rms_norm: RMSNormReplayOp) -> DuplicateResidualRMSNormReplayInvocation {
        let rms_shape = rms_norm.shape();
        let residual_values = rms_shape
            .num_total_tokens
            .checked_mul(rms_shape.hidden_dim)
            .expect("duplicate residual RMSNorm value count must fit u32");
        assert_eq!(self.residual.shape.num_values, residual_values);
        assert_eq!(self.residual.shape.lhs_dtype, Dtype::Bfloat16);
        assert_eq!(self.residual.shape.rhs_dtype, Dtype::Bfloat16);
        assert_eq!(self.residual.shape.output_dtype, Dtype::Bfloat16);
        assert_eq!(rms_shape.dtype, Dtype::Bfloat16);
        assert!(
            std::ptr::eq(
                Retained::as_ptr(&self.residual.buffers.output),
                Retained::as_ptr(rms_norm.input_buffer()),
            ),
            "residual output must be the fused RMSNorm input"
        );

        let duplicate_row_stride = self.duplicate_output.row_stride;
        let duplicate_column_offset = self.duplicate_output.column_offset;
        let (buffers, eps, num_active_tokens_key) = rms_norm.into_duplicate_residual_rms_norm_buffers(
            self.residual.buffers.lhs,
            self.residual.buffers.lhs_len_bytes,
            self.residual.buffers.rhs,
            self.residual.buffers.rhs_len_bytes,
            self.residual.buffers.output,
            self.residual.buffers.output_len_bytes,
            self.duplicate_output.buffer,
            self.duplicate_output.buffer_len_bytes,
        );
        let shape = ResidualRMSNormShape {
            num_total_tokens: rms_shape.num_total_tokens,
            hidden_dim: rms_shape.hidden_dim,
            dtype: rms_shape.dtype,
        };
        match num_active_tokens_key {
            Some(key) => {
                DuplicateResidualRMSNormReplayInvocation::new_bucketed(
                    shape,
                    key,
                    buffers,
                    duplicate_row_stride,
                    duplicate_column_offset,
                    eps,
                )
            },
            None => {
                DuplicateResidualRMSNormReplayInvocation::new(
                    shape,
                    buffers,
                    duplicate_row_stride,
                    duplicate_column_offset,
                    eps,
                )
            },
        }
    }
}

impl ResidualReplayInvocation {
    fn validate(&self) {
        self.shape.validate();
        assert!(self.buffers.lhs_len_bytes >= self.shape.lhs_bytes());
        assert!(self.buffers.rhs_len_bytes >= self.shape.rhs_bytes());
        assert!(self.buffers.output_len_bytes >= self.shape.output_bytes());
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        builder.set_retained_buffer_write(2, &self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

fn residual_function_name(shape: ResidualShape) -> &'static str {
    match (shape.lhs_dtype, shape.rhs_dtype, shape.output_dtype) {
        (Dtype::Float32, Dtype::Float32, Dtype::Float32) => "residual_add_f32",
        (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16) => "residual_add_bf16",
        (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16) => "residual_add_bf16_f32_to_bf16",
        (lhs_dtype, rhs_dtype, output_dtype) => {
            panic!(
                "unsupported residual add dtype combination: lhs={lhs_dtype:?}, rhs={rhs_dtype:?}, \
                 output={output_dtype:?}"
            )
        },
    }
}
