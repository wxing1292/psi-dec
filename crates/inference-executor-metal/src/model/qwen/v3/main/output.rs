use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::num_main_output_rows;

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
    unembed: Rc<Unembed>,
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
        let unembed = Rc::new(Unembed::load(device, store, config, bindings)?);
        Ok(Self::new(device, config.hidden_dim, unembed))
    }

    pub fn new(device: &Device, hidden_dim: u32, unembed: Rc<Unembed>) -> Self {
        Self {
            gather: Gather::new(device),
            unembed,
            hidden_dim,
        }
    }

    pub fn unembed(&self) -> Rc<Unembed> {
        Rc::clone(&self.unembed)
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
    num_main_output_rows: u32,
}

impl Qwen3GatherUnembedReplayKey {
    pub fn from_microbatch(microbatch: &Qwen3Microbatch) -> Self {
        let num_main_output_rows = num_main_output_rows(microbatch)
            .try_into()
            .expect("qwen3 Main output row count must fit u32");
        assert!(
            num_main_output_rows > 0,
            "qwen3 GatherUnembed replay requires Main output rows"
        );
        Self { num_main_output_rows }
    }

    pub fn num_main_output_rows(&self) -> u32 {
        self.num_main_output_rows
    }
}

impl ReplayComponent for Qwen3GatherUnembed {
    type Key = Qwen3GatherUnembedReplayKey;
    type Input<'a> = Qwen3GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        assert!(input.num_rows > 0, "qwen3 GatherUnembed requires Main output rows");
        Qwen3GatherUnembedReplayKey {
            num_main_output_rows: input.num_rows,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen3GatherUnembed::record(self, recorder, *input);
    }
}
use std::rc::Rc;
