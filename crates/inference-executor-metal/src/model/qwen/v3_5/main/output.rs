use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::num_main_output_rows;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

pub struct Qwen35GatherUnembed {
    gather: Gather,
    unembed: Unembed,
}

#[derive(Clone, Copy)]
pub struct Qwen35GatherUnembedArgs<'a> {
    pub num_rows: u32,
    pub hidden_input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Qwen35GatherUnembed {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: UnembedConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let unembed = Unembed::load(device, store, config, bindings)?;
        Ok(Self {
            gather: Gather::new(device, config.hidden_dim),
            unembed,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35GatherUnembedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.gather.record(
            recorder,
            args.num_rows,
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
pub struct Qwen35GatherUnembedReplayKey {
    num_main_output_rows: u32,
}

impl Qwen35GatherUnembedReplayKey {
    pub fn from_microbatch(microbatch: &Qwen35Microbatch) -> Self {
        let num_main_output_rows = num_main_output_rows(microbatch)
            .try_into()
            .expect("qwen3.5 Main output row count must fit u32");
        assert!(
            num_main_output_rows > 0,
            "qwen3.5 GatherUnembed replay requires Main output rows"
        );
        Self { num_main_output_rows }
    }

    pub fn num_main_output_rows(&self) -> u32 {
        self.num_main_output_rows
    }
}

impl ReplayComponent for Qwen35GatherUnembed {
    type Key = Qwen35GatherUnembedReplayKey;
    type Input<'a> = Qwen35GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        assert!(input.num_rows > 0, "qwen3.5 GatherUnembed requires Main output rows");
        Qwen35GatherUnembedReplayKey {
            num_main_output_rows: input.num_rows,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35GatherUnembed::record(self, recorder, *input);
    }
}
