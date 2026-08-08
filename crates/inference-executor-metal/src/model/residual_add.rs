use inference_backend_metal::components::ResidualAddBuffers;
use inference_backend_metal::components::ResidualAddCaptureTarget;
use inference_backend_metal::components::ResidualAddConfig;
use inference_backend_metal::components::ResidualAddKernel;
use inference_backend_metal::components::ResidualAddRowShape;
use inference_backend_metal::components::ResidualAddShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct ResidualAdd {
    compute: ResidualAddKernel,
}

impl ResidualAdd {
    pub fn new(device: &Device) -> Self {
        Self {
            compute: ResidualAddKernel::new(device, ResidualAddConfig::bf16()),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_values: u32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let invocation = self.compute.invoke(
            ResidualAddShape { num_values },
            ResidualAddBuffers {
                lhs,
                rhs,
                output: residual_output,
            },
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add(invocation));
    }

    /// Records a fixed-capacity residual add with a submission-time active token count.
    #[allow(clippy::too_many_arguments)]
    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        hidden_dim: u32,
        num_active_tokens_key: ReplayParameterKey,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let invocation = self.compute.invoke_bucketed(
            ResidualAddRowShape {
                num_total_rows: num_total_tokens,
                num_columns: hidden_dim,
            },
            num_active_tokens_key,
            ResidualAddBuffers {
                lhs,
                rhs,
                output: residual_output,
            },
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add(invocation));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_capture<'a, R>(
        &'a self,
        recorder: &mut R,
        num_rows: u32,
        num_columns: u32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        capture: ResidualAddCaptureTarget<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let invocation = self.compute.invoke_rows(
            ResidualAddRowShape {
                num_total_rows: num_rows,
                num_columns,
            },
            ResidualAddBuffers {
                lhs,
                rhs,
                output: residual_output,
            },
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add_with_capture(invocation, capture));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_bucketed_with_capture<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        hidden_dim: u32,
        num_active_tokens_key: ReplayParameterKey,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        capture: ResidualAddCaptureTarget<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let invocation = self.compute.invoke_bucketed(
            ResidualAddRowShape {
                num_total_rows: num_total_tokens,
                num_columns: hidden_dim,
            },
            num_active_tokens_key,
            ResidualAddBuffers {
                lhs,
                rhs,
                output: residual_output,
            },
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add_with_capture(invocation, capture));
    }
}
