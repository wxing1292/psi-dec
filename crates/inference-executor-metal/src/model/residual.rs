use inference_backend_metal::components::ResidualBuffers;
use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::components::ResidualKernel;
use inference_backend_metal::components::ResidualShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct Residual {
    op: ResidualKernel,
}

impl Residual {
    pub fn new(device: &Device) -> Self {
        Self {
            op: ResidualKernel::new(device),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_values: u32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        residual_capture: Option<ResidualCaptureTarget<'a>>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let invocation = self.op.invoke(
            ResidualShape::bf16(num_values),
            ResidualBuffers {
                lhs,
                rhs,
                output: residual_output,
            },
        );
        let op = match residual_capture {
            Some(capture) => ReplayOp::residual_add_with_capture(invocation, capture),
            None => ReplayOp::residual_add(invocation),
        };
        recorder.record_with_barrier_before(op);
    }
}
