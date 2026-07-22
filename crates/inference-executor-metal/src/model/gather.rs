use inference_backend_metal::components::RowGatherBuffers;
use inference_backend_metal::components::RowGatherKernel;
use inference_backend_metal::components::RowGatherShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct Gather {
    op: RowGatherKernel,
}

impl Gather {
    pub fn new(device: &Device) -> Self {
        Self {
            op: RowGatherKernel::new(device),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_rows: u32,
        hidden_dim: u32,
        input: &'a Buffer,
        row_indices: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.op.invoke(
            RowGatherShape::bf16(num_rows, hidden_dim),
            RowGatherBuffers {
                input,
                row_indices,
                output,
            },
        )));
    }
}
