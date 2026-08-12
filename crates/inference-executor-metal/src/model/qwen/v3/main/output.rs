use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::num_main_output_rows;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

pub struct Qwen3GatherUnembed {
    gather: Gather,
    unembed: Option<Rc<Unembed>>,
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
    pub fn new(device: &Device, hidden_dim: u32, unembed: Rc<Unembed>) -> Self {
        Self {
            gather: Gather::new(device, hidden_dim),
            unembed: Some(unembed),
        }
    }

    pub fn unembed(&self) -> Rc<Unembed> {
        Rc::clone(
            self.unembed
                .as_ref()
                .expect("qwen3 GatherUnembed weights must be loaded before use"),
        )
    }

    pub fn load_weights(&mut self, unembed: Rc<Unembed>) {
        assert!(self.unembed.is_none(), "qwen3 GatherUnembed weights are already loaded");
        self.unembed = Some(unembed);
    }

    pub fn unload_weights(&mut self) -> Rc<Unembed> {
        self.unembed.take().expect("qwen3 GatherUnembed weights are not loaded")
    }

    fn loaded_unembed(&self) -> &Unembed {
        self.unembed
            .as_deref()
            .expect("qwen3 GatherUnembed weights must be loaded before execution")
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen3GatherUnembedArgs<'a>) -> &'a Buffer
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
            self.loaded_unembed(),
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
