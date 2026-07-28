use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormInvocation;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct RmsNorm {
    hidden_dim: usize,
    eps: f32,
    weight: Buffer,
    kernel: RMSNormKernel,
}

impl RmsNorm {
    pub fn new(device: &Device, hidden_dim: usize, eps: f32, weight: Buffer) -> Self {
        assert!(hidden_dim > 0, "RMS norm hidden dimension must be positive");
        assert!(eps.is_finite() && eps > 0.0, "RMS norm epsilon must be positive");
        Self {
            hidden_dim,
            eps,
            weight,
            kernel: RMSNormKernel::new(device),
        }
    }

    fn invocation<'a>(&'a self, num_tokens: u32, input: &'a Buffer, output: &'a Buffer) -> RMSNormInvocation<'a> {
        self.kernel.invoke(
            RMSNormShape::bf16(
                num_tokens,
                self.hidden_dim
                    .try_into()
                    .expect("RMS norm hidden dimension must fit u32"),
            ),
            RMSNormBuffers {
                input,
                weight: &self.weight,
                output,
            },
            self.eps,
        )
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, num_tokens: u32, input: &'a Buffer, output: &'a Buffer)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::rms_norm(self.invocation(num_tokens, input, output)));
    }

    pub fn record_with_barrier<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.invocation(num_tokens, input, output)));
    }

    pub fn record_opaque<'a, R>(&'a self, recorder: &mut R, num_tokens: u32, input: &'a Buffer, output: &'a Buffer)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::opaque(self.invocation(num_tokens, input, output)));
    }

    pub fn record_opaque_with_barrier<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.invocation(num_tokens, input, output)));
    }
}
