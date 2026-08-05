use inference_backend_metal::components::QuantizedDenseMLP;
use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPReplayTopology;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::DenseMLPReplayShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratchBindings;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMLPMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub io_dtype: Dtype,
}

impl DenseMLPMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 dense MLP model boundary is not supported"),
            dtype => panic!("unsupported dense MLP model boundary dtype {dtype:?}"),
        }
    }
}

pub struct DenseMLP {
    compute: QuantizedDenseMLP,
}

#[derive(Clone, Copy)]
pub struct DenseMLPReplayInput<'a> {
    pub shape: DenseMLPReplayShape,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: DenseMLPScratchBindings<'a>,
    pub weights: QuantizedDenseMLPWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct DenseMLPBucketedReplayInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens_key: ReplayParameterKey,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: DenseMLPScratchBindings<'a>,
    pub weights: QuantizedDenseMLPWeights<'a>,
}

impl DenseMLP {
    pub fn new(device: &Device, core: DenseMLPCore, config: DenseMLPMetalConfig) -> Self {
        core.validate();
        config.validate();
        Self {
            compute: QuantizedDenseMLP::new(device, backend_config(&core, config)),
        }
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> QuantizedDenseMLPReplayTopology {
        self.compute.topology(num_total_tokens)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.compute.topology_boundaries()
    }

    pub fn record_bucketed<'a, R>(&'a self, recorder: &mut R, input: DenseMLPBucketedReplayInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke_bucketed(
            input.num_total_tokens,
            input.num_active_tokens_key,
            QuantizedDenseMLPBuffers {
                hidden_state: input.hidden_state,
                next_hidden_state: input.next_hidden_state,
            },
            QuantizedDenseMLPScratch {
                gate_up: input.scratch.gate_up,
                swiglu: input.scratch.swiglu,
            },
            input.weights,
        )));
        input.next_hidden_state
    }
}

impl ReplayLayer for DenseMLP {
    type Input<'a> = DenseMLPReplayInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        input.shape.validate();
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke(
            backend_shape(input.shape),
            QuantizedDenseMLPBuffers {
                hidden_state: input.hidden_state,
                next_hidden_state: input.next_hidden_state,
            },
            QuantizedDenseMLPScratch {
                gate_up: input.scratch.gate_up,
                swiglu: input.scratch.swiglu,
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

pub(super) fn backend_config(core: &DenseMLPCore, config: DenseMLPMetalConfig) -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: core.hidden_dim.try_into().expect("dense MLP hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("dense MLP intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.io_dtype,
    }
}
