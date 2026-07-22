use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPKernels;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::DenseMLPReplayShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratchBindings;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMLPMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub dtype: Dtype,
}

impl DenseMLPMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }
}

pub struct DenseMLP {
    core: DenseMLPCore,
    config: DenseMLPMetalConfig,
    kernels: QuantizedDenseMLPKernels,
}

#[derive(Clone, Copy)]
pub struct DenseMLPReplayInput<'a> {
    pub shape: DenseMLPReplayShape,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: DenseMLPScratchBindings<'a>,
    pub weights: QuantizedDenseMLPWeights<'a>,
}

impl DenseMLP {
    pub fn new(device: &Device, core: DenseMLPCore, config: DenseMLPMetalConfig) -> Self {
        core.validate();
        config.validate();
        let kernels = QuantizedDenseMLPKernels::new(device, backend_config(&core, config));
        Self { core, config, kernels }
    }
}

impl Layer for DenseMLP {
    type Input<'a> = DenseMLPReplayInput<'a>;
    type Output<'a> = &'a Buffer;

    type InputShape = DenseMLPCore;
    type OutputShape = DenseMLPCore;

    fn input_shape(&self) -> Self::InputShape {
        self.core.clone()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.core.clone()
    }
}

impl ReplayLayer for DenseMLP {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        input.shape.validate();
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kernels.invoke(
            backend_shape(input.shape),
            QuantizedDenseMLPBuffers {
                hidden_state: input.hidden_state,
                next_hidden_state: input.next_hidden_state,
            },
            QuantizedDenseMLPScratch {
                gate_up_proj: input.scratch.gate_up_proj,
                activation: input.scratch.activation,
            },
            input.weights,
        )));
        input.next_hidden_state
    }
}

fn backend_shape(shape: DenseMLPReplayShape) -> QuantizedDenseMLPShape {
    QuantizedDenseMLPShape {
        num_tokens: shape.num_tokens,
    }
}

fn backend_config(core: &DenseMLPCore, config: DenseMLPMetalConfig) -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: core.hidden_dim.try_into().expect("dense MLP hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("dense MLP intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.dtype,
    }
}
