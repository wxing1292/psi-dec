use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::operators::RowGatherBuffers;
use inference_backend_metal::operators::RowGatherConfig;
use inference_backend_metal::operators::RowGatherKernel;
use inference_backend_metal::operators::RowGatherShape;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct Gather {
    op: RowGatherKernel,
}

impl Gather {
    pub fn new(device: &Device, hidden_dim: u32) -> Self {
        Self {
            op: RowGatherKernel::new(
                device,
                RowGatherConfig {
                    num_cols: hidden_dim,
                    dtype: Dtype::Bfloat16,
                },
            ),
        }
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_rows: u32,
        input: &'a Buffer,
        row_indices: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.op.invoke(
            RowGatherShape {
                num_total_rows: num_rows,
            },
            RowGatherBuffers {
                input,
                row_indices,
                output,
            },
        )));
    }

    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_rows: u32,
        num_active_rows_key: ReplayParameterKey,
        input: &'a Buffer,
        row_indices: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.op.invoke_bucketed(
            RowGatherShape { num_total_rows },
            num_active_rows_key,
            RowGatherBuffers {
                input,
                row_indices,
                output,
            },
        )));
    }
}
