use inference_backend_metal::components::residual_add;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct ResidualAdd {
    compute: residual_add::Compute,
}

impl ResidualAdd {
    pub fn new(device: &Device) -> Self {
        Self {
            compute: residual_add::Compute::new(device, residual_add::Config::bf16()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_rows: u32,
        num_columns: u32,
        num_active_rows: ReplayU32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = residual_add::RowShape {
            num_total_rows,
            num_columns,
        };
        let buffers = residual_add::Buffers {
            lhs,
            rhs,
            output: residual_output,
        };
        let invocation = match num_active_rows {
            ReplayU32::Fixed(num_active_rows) => {
                assert_eq!(num_active_rows, num_total_rows);
                self.compute.invoke_rows(shape, buffers)
            },
            ReplayU32::Parameter(key) => self.compute.invoke_bucketed(shape, key, buffers),
        };
        recorder.record_with_barrier_before(ReplayOp::residual_add(invocation));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_capture<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_rows: u32,
        num_columns: u32,
        num_active_rows: ReplayU32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        capture: residual_add::CaptureTarget<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = residual_add::RowShape {
            num_total_rows,
            num_columns,
        };
        let buffers = residual_add::Buffers {
            lhs,
            rhs,
            output: residual_output,
        };
        let invocation = match num_active_rows {
            ReplayU32::Fixed(num_active_rows) => {
                assert_eq!(num_active_rows, num_total_rows);
                self.compute.invoke_rows(shape, buffers)
            },
            ReplayU32::Parameter(key) => self.compute.invoke_bucketed(shape, key, buffers),
        };
        recorder.record_with_barrier_before(ReplayOp::residual_add_with_capture(invocation, capture));
    }
}
