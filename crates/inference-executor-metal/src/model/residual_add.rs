use inference_backend_metal::components::ResidualAddBuffers;
use inference_backend_metal::components::ResidualAddCaptureTarget;
use inference_backend_metal::components::ResidualAddConfig;
use inference_backend_metal::components::ResidualAddKernel;
use inference_backend_metal::components::ResidualAddShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
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

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_capture<'a, R>(
        &'a self,
        recorder: &mut R,
        num_values: u32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        capture: ResidualAddCaptureTarget<'a>,
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
        recorder.record_with_barrier_before(ReplayOp::residual_add_with_capture(invocation, capture));
    }
}
