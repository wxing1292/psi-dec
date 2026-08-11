use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormConfig;
use inference_backend_metal::components::RMSNormInvocation;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;
use crate::model::residency_digest::ModelResidencyHasher;

pub struct RMSNorm {
    weight: Option<Buffer>,
    compute: RMSNormKernel,
}

impl RMSNorm {
    pub fn new(device: &Device, hidden_dim: usize, eps: f32) -> Self {
        assert!(hidden_dim > 0, "RMS norm hidden dimension must be positive");
        assert!(eps.is_finite() && eps > 0.0, "RMS norm epsilon must be positive");
        let hidden_dim = hidden_dim.try_into().expect("RMS norm hidden dimension must fit u32");
        Self {
            weight: None,
            compute: RMSNormKernel::new(device, RMSNormConfig::bf16(hidden_dim, eps)),
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

    pub fn hash_weights(&self, hasher: &mut ModelResidencyHasher, name: &str) {
        hasher.buffer(name, self.weight());
    }

    fn weight(&self) -> &Buffer {
        self.weight
            .as_ref()
            .expect("RMS norm weights must be loaded before execution")
    }

    fn invocation<'a>(&'a self, num_tokens: u32, input: &'a Buffer, output: &'a Buffer) -> RMSNormInvocation<'a> {
        self.compute.invoke(
            RMSNormShape {
                num_total_tokens: num_tokens,
            },
            RMSNormBuffers {
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
    ) -> RMSNormInvocation<'a> {
        self.compute.invoke_bucketed(
            RMSNormShape { num_total_tokens },
            num_active_tokens_key,
            RMSNormBuffers {
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
