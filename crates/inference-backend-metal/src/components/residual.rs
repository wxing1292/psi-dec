use std::ops::Range;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::components::ResidualRMSNormShape;
use crate::components::residual_rms_norm::ResidualCaptureRMSNormReplayInvocation;
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

pub struct ResidualCaptureReplayOp {
    residual: ResidualReplayOp,
    capture: OwnedResidualCaptureTarget,
}

/// Destination for capturing every complete row produced by a fused residual add.
///
/// The target is currently supported only for BF16 residual-add/RMSNorm replay
/// fusion. Each complete residual row is written into the selected destination
/// columns. The capture must be immediately followed by its RMSNorm fusion
/// partner, the destination range width must equal that RMSNorm's hidden
/// dimension, and the destination buffer must not alias any fused
/// residual/RMSNorm buffer.
///
/// Capture rows whose width is a multiple of four use the vec4 kernel and
/// therefore require four-element-aligned destination rows and column starts.
/// Other layouts in that variant are unsupported and panic instead of falling
/// back to a more generic kernel.
///
/// The target alone does not know the residual row width, replay token
/// capacity, or fused buffers. Those remaining invariants, including
/// destination capacity, are asserted when the replay fusion is constructed
/// or recorded.
#[derive(Clone, Copy)]
pub struct ResidualCaptureTarget<'a> {
    buffer: &'a Buffer,
    row_width: u32,
    column_start: u32,
    column_end: u32,
}

struct OwnedResidualCaptureTarget {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_len_bytes: usize,
    row_width: u32,
    column_start: u32,
    column_end: u32,
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

impl<'a> ResidualCaptureTarget<'a> {
    /// Selects the destination columns for every complete residual row.
    ///
    /// `row_width` and `columns` are tensor coordinates, not byte offsets.
    /// This constructor verifies that the range is non-empty and contained in
    /// a destination row. When the range width is a multiple of four, the
    /// destination row width and column start must also be multiples of four;
    /// unsupported layouts panic rather than falling back to the scalar
    /// variant. The range width must equal the fused BF16 RMSNorm hidden
    /// dimension; replay fusion asserts that delayed invariant together with
    /// the immediate-fusion, no-alias, and capacity contracts. Replay fusion
    /// owns the dtype-specific byte and vector-width lowering.
    pub fn columns(buffer: &'a Buffer, row_width: u32, columns: Range<u32>) -> Self {
        assert!(row_width > 0, "residual capture row width must be positive");
        assert!(
            columns.start < columns.end,
            "residual capture column range must be non-empty"
        );
        assert!(
            columns.end <= row_width,
            "residual capture columns must be within the row"
        );
        let column_width = columns.end - columns.start;
        if column_width.is_multiple_of(4) {
            assert!(
                row_width.is_multiple_of(4) && columns.start.is_multiple_of(4),
                "unsupported residual capture layout: a capture width divisible by four requires aligned row width \
                 and column start"
            );
        }
        Self {
            buffer,
            row_width,
            column_start: columns.start,
            column_end: columns.end,
        }
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

    pub fn into_capture_replay_op(self, capture: ResidualCaptureTarget<'_>) -> ResidualCaptureReplayOp {
        ResidualCaptureReplayOp {
            residual: self.into_replay_op(),
            capture: OwnedResidualCaptureTarget {
                buffer: capture.buffer.as_raw_retained(),
                buffer_len_bytes: capture.buffer.len_bytes(),
                row_width: capture.row_width,
                column_start: capture.column_start,
                column_end: capture.column_end,
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

impl ResidualCaptureReplayOp {
    pub fn fuse_rms_norm(self, rms_norm: RMSNormReplayOp) -> ResidualCaptureRMSNormReplayInvocation {
        let rms_shape = rms_norm.shape();
        let residual_values = rms_shape
            .num_total_tokens
            .checked_mul(rms_shape.hidden_dim)
            .expect("residual capture RMSNorm value count must fit u32");
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

        assert_eq!(
            self.capture.column_end - self.capture.column_start,
            rms_shape.hidden_dim,
            "residual capture column width must match hidden dimension"
        );
        let capture_row_width = self.capture.row_width;
        let capture_column_start = self.capture.column_start;
        let (buffers, eps, num_active_tokens_key) = rms_norm.into_residual_capture_rms_norm_buffers(
            self.residual.buffers.lhs,
            self.residual.buffers.lhs_len_bytes,
            self.residual.buffers.rhs,
            self.residual.buffers.rhs_len_bytes,
            self.residual.buffers.output,
            self.residual.buffers.output_len_bytes,
            self.capture.buffer,
            self.capture.buffer_len_bytes,
        );
        let shape = ResidualRMSNormShape {
            num_total_tokens: rms_shape.num_total_tokens,
            hidden_dim: rms_shape.hidden_dim,
            dtype: rms_shape.dtype,
        };
        match num_active_tokens_key {
            Some(key) => {
                ResidualCaptureRMSNormReplayInvocation::new_bucketed(
                    shape,
                    key,
                    buffers,
                    capture_row_width,
                    capture_column_start,
                    eps,
                )
            },
            None => {
                ResidualCaptureRMSNormReplayInvocation::new(
                    shape,
                    buffers,
                    capture_row_width,
                    capture_column_start,
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
