use std::ops::Range;

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

/// Two-dimensional row-major residual shape.
///
/// Both fields count elements. They do not count bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualAddRowShape {
    pub num_total_rows: u32,
    pub num_columns: u32,
}

impl ResidualAddRowShape {
    pub fn validate(self) {
        assert!(self.num_total_rows > 0);
        assert!(self.num_columns > 0);
        self.num_total_rows
            .checked_mul(self.num_columns)
            .expect("residual-add row value count must fit u32");
    }

    pub fn num_values(self) -> u32 {
        self.validate();
        self.num_total_rows * self.num_columns
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
            row_shape: None,
            num_active_rows_key: None,
        }
    }

    /// Records an exact number of complete rows.
    pub fn invoke_rows<'a>(
        &'a self,
        shape: ResidualAddRowShape,
        buffers: ResidualAddBuffers<'a>,
    ) -> ResidualAddInvocation<'a> {
        ResidualAddInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape: ResidualAddShape {
                num_values: shape.num_values(),
            },
            buffers,
            row_shape: Some(shape),
            num_active_rows_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active row count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: ResidualAddRowShape,
        num_active_rows_key: ReplayParameterKey,
        buffers: ResidualAddBuffers<'a>,
    ) -> ResidualAddInvocation<'a> {
        ResidualAddInvocation {
            kernel: &self.kernel,
            config: self.config,
            shape: ResidualAddShape {
                num_values: capacity_shape.num_values(),
            },
            buffers,
            row_shape: Some(capacity_shape),
            num_active_rows_key: Some(num_active_rows_key),
        }
    }
}

pub struct ResidualAddInvocation<'a> {
    kernel: &'a Kernel,
    config: ResidualAddConfig,
    shape: ResidualAddShape,
    buffers: ResidualAddBuffers<'a>,
    row_shape: Option<ResidualAddRowShape>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

pub(super) struct ResidualAddReplayInvocation {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: ResidualAddConfig,
    pub(super) shape: ResidualAddShape,
    pub(super) buffers: ResidualAddOwnedBuffers,
    pub(super) row_shape: Option<ResidualAddRowShape>,
    pub(super) num_active_rows_key: Option<ReplayParameterKey>,
}

pub(super) struct ResidualAddReplayOp {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) config: ResidualAddConfig,
    pub(super) shape: ResidualAddShape,
    pub(super) buffers: ResidualAddOwnedBuffers,
    pub(super) row_shape: Option<ResidualAddRowShape>,
    pub(super) num_active_rows_key: Option<ReplayParameterKey>,
}

pub(super) struct ResidualAddCaptureReplayOp {
    pub(super) residual: ResidualAddReplayOp,
    pub(super) capture: OwnedResidualAddCaptureTarget,
}

pub(super) struct ResidualAddCaptureReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residual: ResidualAddReplayOp,
    capture: OwnedResidualAddCaptureTarget,
}

/// Destination for capturing every complete row produced by a residual add.
///
/// The target is supported for BF16 replay. Each complete residual row is
/// written into the selected destination columns. An adjacent compatible
/// RMSNorm can fuse with this operation, but fusion is not required.
///
/// The BF16 capture kernel uses vec4 loads and stores. The source column count,
/// destination range width, destination column count, and destination column
/// start must therefore be divisible by four.
///
/// The residual invocation must carry an explicit `ResidualAddRowShape`.
/// Recording checks that its column count equals the selected column count.
/// Recording also checks destination capacity and buffer aliasing.
#[derive(Clone, Copy)]
pub struct ResidualAddCaptureTarget<'a> {
    buffer: &'a Buffer,
    num_destination_columns: u32,
    column_start: u32,
    column_end: u32,
}

pub(super) struct OwnedResidualAddCaptureTarget {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) buffer_len_bytes: usize,
    pub(super) num_destination_columns: u32,
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
        record_shape(builder, self.shape, self.row_shape, self.num_active_rows_key);
        dispatch_values(builder, self.shape.num_values);
    }
}

impl Operator for ResidualAddReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        builder.set_retained_buffer_write(2, &self.buffers.output, 0);
        record_shape(builder, self.shape, self.row_shape, self.num_active_rows_key);
        dispatch_values(builder, self.shape.num_values);
    }
}

impl Operator for ResidualAddCaptureReplayInvocation {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_retained_pipeline_state(&self.pipeline);
        builder.set_retained_buffer_read(0, &self.residual.buffers.lhs, 0);
        builder.set_retained_buffer_read(1, &self.residual.buffers.rhs, 0);
        builder.set_retained_buffer_write(2, &self.residual.buffers.output, 0);
        builder.set_retained_buffer_write(3, &self.capture.buffer, 0);
        let row_shape = self.row_shape();
        match self.residual.num_active_rows_key {
            Some(key) => builder.bind_u32(4, key, 1, row_shape.num_total_rows),
            None => builder.set_u32(4, row_shape.num_total_rows),
        }
        builder.set_u32(5, row_shape.num_columns / 4);
        builder.set_u32(6, self.capture.num_destination_columns / 4);
        builder.set_u32(7, self.capture.column_start / 4);
        let num_vectors = self.residual.shape.num_values as usize / 4;
        builder.dispatch_threadblocks(
            (num_vectors.div_ceil(NUM_THREADS_PER_THREADBLOCK), 1, 1),
            (NUM_THREADS_PER_THREADBLOCK, 1, 1),
        );
    }
}

impl<'a> ResidualAddCaptureTarget<'a> {
    /// Selects the destination columns for every complete residual row.
    ///
    /// `num_destination_columns` and `columns` are element counts and tensor
    /// coordinates, not byte offsets.
    /// This constructor verifies that the range is non-empty and contained in
    /// a destination row. The range width, destination column count, and column
    /// start must be multiples of four. The range width must equal the
    /// residual source column count. Recording asserts this invariant together with the
    /// no-alias and capacity contracts.
    pub fn columns(buffer: &'a Buffer, num_destination_columns: u32, columns: Range<u32>) -> Self {
        assert!(
            num_destination_columns > 0,
            "residual-add capture destination must have columns"
        );
        assert!(
            columns.start < columns.end,
            "residual-add capture column range must be non-empty"
        );
        assert!(
            columns.end <= num_destination_columns,
            "residual-add capture columns must be within the row"
        );
        let column_width = columns.end - columns.start;
        assert!(
            column_width.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture width must be divisible by four"
        );
        assert!(
            num_destination_columns.is_multiple_of(4) && columns.start.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture requires aligned column count and column start"
        );
        Self {
            buffer,
            num_destination_columns,
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
            row_shape: self.row_shape,
            num_active_rows_key: self.num_active_rows_key,
        }
    }

    pub(super) fn into_capture_replay_op(self, capture: ResidualAddCaptureTarget<'_>) -> ResidualAddCaptureReplayOp {
        assert!(
            self.row_shape.is_some(),
            "residual-add capture requires an explicit row shape"
        );
        ResidualAddCaptureReplayOp {
            residual: self.into_replay_op(),
            capture: OwnedResidualAddCaptureTarget {
                buffer: capture.buffer.as_raw_retained(),
                buffer_len_bytes: capture.buffer.len_bytes(),
                num_destination_columns: capture.num_destination_columns,
                column_start: capture.column_start,
                column_end: capture.column_end,
            },
        }
    }

    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        if let Some(row_shape) = self.row_shape {
            row_shape.validate();
            assert_eq!(self.shape.num_values, row_shape.num_values());
        }
        assert!(self.num_active_rows_key.is_none() || self.row_shape.is_some());
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
            row_shape: self.row_shape,
            num_active_rows_key: self.num_active_rows_key,
        }
    }
}

impl ResidualAddCaptureReplayOp {
    pub(super) fn into_replay(self) -> ResidualAddCaptureReplayInvocation {
        let device = Device::from_raw_retained(self.residual.buffers.lhs.device());
        ResidualAddCaptureReplayInvocation {
            pipeline: Kernel::new(&device, RESIDUAL_ADD_SOURCE, "residual_add_capture_bf16_vec4").as_raw_retained(),
            residual: self.residual,
            capture: self.capture,
        }
    }
}

impl ResidualAddReplayInvocation {
    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        if let Some(row_shape) = self.row_shape {
            row_shape.validate();
            assert_eq!(self.shape.num_values, row_shape.num_values());
        }
        assert!(self.num_active_rows_key.is_none() || self.row_shape.is_some());
        assert!(self.buffers.lhs_len_bytes >= self.config.lhs_bytes(self.shape));
        assert!(self.buffers.rhs_len_bytes >= self.config.rhs_bytes(self.shape));
        assert!(self.buffers.output_len_bytes >= self.config.output_bytes(self.shape));
    }
}

impl ResidualAddCaptureReplayInvocation {
    fn row_shape(&self) -> ResidualAddRowShape {
        self.residual
            .row_shape
            .expect("residual-add capture requires an explicit row shape")
    }

    fn validate(&self) {
        self.residual.config.validate();
        self.residual.shape.validate();
        let row_shape = self.row_shape();
        row_shape.validate();
        assert_eq!(self.residual.shape.num_values, row_shape.num_values());
        assert_eq!(self.residual.config, ResidualAddConfig::bf16());
        assert_eq!(
            self.capture.column_end - self.capture.column_start,
            row_shape.num_columns,
            "residual-add capture column count must match the residual source column count"
        );
        assert!(row_shape.num_columns.is_multiple_of(4));
        let required_values = (row_shape.num_total_rows as usize - 1)
            .checked_mul(self.capture.num_destination_columns as usize)
            .and_then(|value| value.checked_add(self.capture.column_end as usize))
            .expect("residual-add capture value count must fit usize");
        let required_bytes = required_values
            .checked_mul(Dtype::Bfloat16.item_size())
            .expect("residual-add capture byte count must fit usize");
        assert!(self.capture.buffer_len_bytes >= required_bytes);
        assert!(self.residual.buffers.lhs_len_bytes >= self.residual.config.lhs_bytes(self.residual.shape));
        assert!(self.residual.buffers.rhs_len_bytes >= self.residual.config.rhs_bytes(self.residual.shape));
        assert!(self.residual.buffers.output_len_bytes >= self.residual.config.output_bytes(self.residual.shape));
        for other in [
            &self.residual.buffers.lhs,
            &self.residual.buffers.rhs,
            &self.residual.buffers.output,
        ] {
            assert!(
                !std::ptr::eq(Retained::as_ptr(&self.capture.buffer), Retained::as_ptr(other)),
                "residual-add capture output must not alias a residual-add buffer"
            );
        }
    }
}

fn record_shape(
    builder: &CommandRecorder,
    shape: ResidualAddShape,
    row_shape: Option<ResidualAddRowShape>,
    num_active_rows_key: Option<ReplayParameterKey>,
) {
    match (row_shape, num_active_rows_key) {
        (Some(row_shape), Some(key)) => {
            builder.bind_u32(3, key, 1, row_shape.num_total_rows);
            builder.set_u32(4, row_shape.num_columns);
        },
        (Some(row_shape), None) => {
            builder.set_u32(3, row_shape.num_total_rows);
            builder.set_u32(4, row_shape.num_columns);
        },
        (None, None) => {
            builder.set_u32(3, shape.num_values);
            builder.set_u32(4, 1);
        },
        (None, Some(_)) => panic!("residual-add replay parameter requires a row shape"),
    }
}

fn dispatch_values(builder: &CommandRecorder, num_values: u32) {
    builder.dispatch_threadblocks(
        ((num_values as usize).div_ceil(NUM_THREADS_PER_THREADBLOCK), 1, 1),
        (NUM_THREADS_PER_THREADBLOCK, 1, 1),
    );
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
