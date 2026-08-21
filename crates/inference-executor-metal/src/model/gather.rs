use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::operators::row_gather;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct Gather {
    compute: row_gather::Kernel,
}

impl Gather {
    pub fn new(device: &Device, hidden_dim: u32) -> Self {
        Self {
            compute: row_gather::Kernel::new(
                device,
                row_gather::Config {
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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke(
            row_gather::Shape {
                num_total_rows: num_rows,
            },
            row_gather::Buffers {
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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke_bucketed(
            row_gather::Shape { num_total_rows },
            num_active_rows_key,
            row_gather::Buffers {
                input,
                row_indices,
                output,
            },
        )));
    }
}
