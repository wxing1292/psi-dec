use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::num_target_hidden_states;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

pub struct Qwen3GatherUnembed {
    gather: Gather,
    unembed: Unembed,
    hidden_dim: u32,
}

#[derive(Clone, Copy)]
pub struct Qwen3GatherUnembedArgs<'a> {
    pub num_rows: u32,
    pub hidden_input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Qwen3GatherUnembed {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: UnembedConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let unembed = Unembed::load(device, store, config, bindings)?;
        Ok(Self {
            gather: Gather::new(device),
            unembed,
            hidden_dim: config.hidden_dim,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen3GatherUnembedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.gather.record(
            recorder,
            args.num_rows,
            self.hidden_dim,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        <Unembed as ReplayLayer>::record(
            &self.unembed,
            recorder,
            UnembedInput {
                num_rows: args.num_rows,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3GatherUnembedReplayKey {
    num_target_hidden_states: u32,
}

impl Qwen3GatherUnembedReplayKey {
    pub fn from_microbatch(microbatch: &Qwen3Microbatch) -> Self {
        let num_target_hidden_states = num_target_hidden_states(microbatch)
            .try_into()
            .expect("qwen3 target hidden-state count must fit u32");
        assert!(
            num_target_hidden_states > 0,
            "qwen3 GatherUnembed replay requires target hidden states"
        );
        Self {
            num_target_hidden_states,
        }
    }

    pub fn num_target_hidden_states(&self) -> u32 {
        self.num_target_hidden_states
    }
}

impl ReplayComponent for Qwen3GatherUnembed {
    type Key = Qwen3GatherUnembedReplayKey;
    type Input<'a> = Qwen3GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        assert!(input.num_rows > 0, "qwen3 GatherUnembed requires target hidden states");
        Qwen3GatherUnembedReplayKey {
            num_target_hidden_states: input.num_rows,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen3GatherUnembed::record(self, recorder, *input);
    }
}
