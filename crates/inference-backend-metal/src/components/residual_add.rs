use std::ops::Range;

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

const RESIDUAL_ADD_SOURCE: &str = include_str!("metal/residual_add.metal");

const NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualAddConfig {
    pub lhs_dtype: Dtype,
    pub rhs_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl ResidualAddConfig {
    pub fn f32() -> Self {
        Self {
            lhs_dtype: Dtype::Float32,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Float32,
        }
    }

    pub fn bf16() -> Self {
        Self {
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn bf16_f32_to_bf16() -> Self {
        Self {
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(
            matches!(
                (self.lhs_dtype, self.rhs_dtype, self.output_dtype),
                (Dtype::Float32, Dtype::Float32, Dtype::Float32)
                    | (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16)
                    | (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16)
            ),
            "unsupported residual-add dtype combination: lhs={:?}, rhs={:?}, output={:?}",
            self.lhs_dtype,
            self.rhs_dtype,
            self.output_dtype
        );
    }

    pub fn lhs_bytes(self, shape: ResidualAddShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.lhs_dtype.item_size())
            .expect("residual-add lhs byte length must fit usize")
    }

    pub fn rhs_bytes(self, shape: ResidualAddShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.rhs_dtype.item_size())
            .expect("residual-add rhs byte length must fit usize")
    }

    pub fn output_bytes(self, shape: ResidualAddShape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.output_dtype.item_size())
            .expect("residual-add output byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualAddShape {
    pub num_values: u32,
}

impl ResidualAddShape {
    pub fn validate(self) {
        assert!(self.num_values > 0);
    }
}

#[derive(Clone, Copy)]
pub struct ResidualAddBuffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

/// Elementwise residual-add data flow:
///
/// ```text
/// buffers.lhs ----\
///                  +--> output = lhs + rhs --> buffers.output
/// buffers.rhs ----/
/// ```
pub struct ResidualAddKernel {
    config: ResidualAddConfig,
    kernel: Kernel,
}

impl ResidualAddKernel {
    pub fn new(device: &Device, config: ResidualAddConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Kernel::new(device, RESIDUAL_ADD_SOURCE, residual_add_function_name(config)),
        }
    }

    pub fn invoke<'a>(&'a self, shape: ResidualAddShape, buffers: ResidualAddBuffers<'a>) -> ResidualAddInvocation<'a> {
        ResidualAddInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape,
            buffers,
        }
    }
}

pub struct ResidualAddInvocation<'a> {
    kernel: &'a Kernel,
    config: ResidualAddConfig,
    shape: ResidualAddShape,
    buffers: ResidualAddBuffers<'a>,
}

pub(super) struct ResidualAddReplayInvocation {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: ResidualAddConfig,
    pub(super) shape: ResidualAddShape,
    pub(super) buffers: ResidualAddOwnedBuffers,
}

pub(super) struct ResidualAddReplayOp {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: ResidualAddConfig,
    pub(super) shape: ResidualAddShape,
    pub(super) buffers: ResidualAddOwnedBuffers,
}

pub(super) struct ResidualAddCaptureReplayOp {
    pub(super) residual: ResidualAddReplayOp,
    pub(super) capture: OwnedResidualAddCaptureTarget,
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
/// BF16 RMSNorm requires a hidden dimension that is divisible by four. The
/// capture uses the same vec4 kernel. The destination range width, row width,
/// and column start must therefore be divisible by four.
///
/// The target alone does not know the residual row width, replay token
/// capacity, or fused buffers. Those remaining invariants, including
/// destination capacity, are asserted when the replay fusion is constructed
/// or recorded.
#[derive(Clone, Copy)]
pub struct ResidualAddCaptureTarget<'a> {
    buffer: &'a Buffer,
    row_width: u32,
    column_start: u32,
    column_end: u32,
}

pub(super) struct OwnedResidualAddCaptureTarget {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) buffer_len_bytes: usize,
    pub(super) row_width: u32,
    pub(super) column_start: u32,
    pub(super) column_end: u32,
}

#[derive(Clone)]
pub(super) struct ResidualAddOwnedBuffers {
    pub(super) lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) lhs_len_bytes: usize,
    pub(super) rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) rhs_len_bytes: usize,
    pub(super) output: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) output_len_bytes: usize,
}

impl Operator for ResidualAddInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.lhs, 0);
        builder.set_buffer_read(1, self.buffers.rhs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

impl Operator for ResidualAddReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        builder.set_retained_buffer_write(2, &self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_values);
        builder.dispatch_1d(self.shape.num_values as usize, NUM_THREADS_PER_THREADBLOCK);
    }
}

impl<'a> ResidualAddCaptureTarget<'a> {
    /// Selects the destination columns for every complete residual row.
    ///
    /// `row_width` and `columns` are tensor coordinates, not byte offsets.
    /// This constructor verifies that the range is non-empty and contained in
    /// a destination row. The range width, destination row width, and column
    /// start must be multiples of four. The range width must equal the fused
    /// BF16 RMSNorm hidden dimension; replay fusion asserts that delayed
    /// invariant together with the immediate-fusion, no-alias, and capacity
    /// contracts. Replay fusion owns the dtype-specific byte and vector-width
    /// lowering.
    pub fn columns(buffer: &'a Buffer, row_width: u32, columns: Range<u32>) -> Self {
        assert!(row_width > 0, "residual-add capture row width must be positive");
        assert!(
            columns.start < columns.end,
            "residual-add capture column range must be non-empty"
        );
        assert!(
            columns.end <= row_width,
            "residual-add capture columns must be within the row"
        );
        let column_width = columns.end - columns.start;
        assert!(
            column_width.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture width must be divisible by four"
        );
        assert!(
            row_width.is_multiple_of(4) && columns.start.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture requires aligned row width and column start"
        );
        Self {
            buffer,
            row_width,
            column_start: columns.start,
            column_end: columns.end,
        }
    }
}

impl ResidualAddInvocation<'_> {
    pub(super) fn into_replay_op(self) -> ResidualAddReplayOp {
        ResidualAddReplayOp {
            pipeline: self.kernel.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers: ResidualAddOwnedBuffers {
                lhs: self.buffers.lhs.as_raw_retained(),
                lhs_len_bytes: self.buffers.lhs.len_bytes(),
                rhs: self.buffers.rhs.as_raw_retained(),
                rhs_len_bytes: self.buffers.rhs.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
        }
    }

    pub(super) fn into_capture_replay_op(self, capture: ResidualAddCaptureTarget<'_>) -> ResidualAddCaptureReplayOp {
        ResidualAddCaptureReplayOp {
            residual: self.into_replay_op(),
            capture: OwnedResidualAddCaptureTarget {
                buffer: capture.buffer.as_raw_retained(),
                buffer_len_bytes: capture.buffer.len_bytes(),
                row_width: capture.row_width,
                column_start: capture.column_start,
                column_end: capture.column_end,
            },
        }
    }

    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        assert!(self.buffers.lhs.len_bytes() >= self.config.lhs_bytes(self.shape));
        assert!(self.buffers.rhs.len_bytes() >= self.config.rhs_bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.output_bytes(self.shape));
    }
}

impl ResidualAddReplayOp {
    pub(super) fn into_replay(self) -> ResidualAddReplayInvocation {
        ResidualAddReplayInvocation {
            pipeline: self.pipeline,
            config: self.config,
            shape: self.shape,
            buffers: self.buffers,
        }
    }
}

impl ResidualAddReplayInvocation {
    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        assert!(self.buffers.lhs_len_bytes >= self.config.lhs_bytes(self.shape));
        assert!(self.buffers.rhs_len_bytes >= self.config.rhs_bytes(self.shape));
        assert!(self.buffers.output_len_bytes >= self.config.output_bytes(self.shape));
    }
}

fn residual_add_function_name(config: ResidualAddConfig) -> &'static str {
    match (config.lhs_dtype, config.rhs_dtype, config.output_dtype) {
        (Dtype::Float32, Dtype::Float32, Dtype::Float32) => "residual_add_f32",
        (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16) => "residual_add_bf16",
        (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16) => "residual_add_bf16_f32_to_bf16",
        (lhs_dtype, rhs_dtype, output_dtype) => {
            panic!(
                "unsupported residual-add dtype combination: lhs={lhs_dtype:?}, rhs={rhs_dtype:?}, \
                 output={output_dtype:?}"
            )
        },
    }
}
