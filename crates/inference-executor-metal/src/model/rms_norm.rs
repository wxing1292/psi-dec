use inference_backend_metal::components::rms_norm;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;

pub struct RMSNorm {
    weight: Option<Buffer>,
    compute: rms_norm::Compute,
}

impl RMSNorm {
    pub fn new(device: &Device, hidden_dim: usize, eps: f32) -> Self {
        assert!(hidden_dim > 0, "RMS norm hidden dimension must be positive");
        assert!(eps.is_finite() && eps > 0.0, "RMS norm epsilon must be positive");
        let hidden_dim = hidden_dim.try_into().expect("RMS norm hidden dimension must fit u32");
        Self {
            weight: None,
            compute: rms_norm::Compute::new(device, rms_norm::Config::bf16(hidden_dim, eps)),
        }
    }

    pub fn load_weights(&mut self, weight: Buffer) {
        assert!(self.weight.is_none(), "RMS norm weights are already loaded");
        self.weight = Some(weight);
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weight.is_some(), "RMS norm weights are not loaded");
        self.weight.take();
    }

    fn weight(&self) -> &Buffer {
        self.weight
            .as_ref()
            .expect("RMS norm weights must be loaded before execution")
    }

    fn invocation<'a>(&'a self, num_tokens: u32, input: &'a Buffer, output: &'a Buffer) -> rms_norm::Invocation<'a> {
        self.compute.invoke(
            rms_norm::Shape {
                num_total_tokens: num_tokens,
            },
            rms_norm::Buffers {
                input,
                weight: self.weight(),
                output,
            },
        )
    }

    fn bucketed_invocation<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        input: &'a Buffer,
        output: &'a Buffer,
    ) -> rms_norm::Invocation<'a> {
        self.compute.invoke_bucketed(
            rms_norm::Shape { num_total_tokens },
            num_active_tokens_key,
            rms_norm::Buffers {
                input,
                weight: self.weight(),
                output,
            },
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

    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        input: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::rms_norm(self.bucketed_invocation(
            num_total_tokens,
            num_active_tokens_key,
            input,
            output,
        )));
    }

    pub fn record_bucketed_with_barrier<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        input: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.bucketed_invocation(
            num_total_tokens,
            num_active_tokens_key,
            input,
            output,
        )));
    }
}
