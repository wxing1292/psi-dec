use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

const QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.gather_unembed.num_active_rows");

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

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_rows: u32,
        num_active_rows: ReplayU32,
        args: Qwen3GatherUnembedArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        if let ReplayU32::Fixed(value) = num_active_rows {
            assert_eq!(value, args.num_rows);
            assert_eq!(value, num_total_rows);
        } else {
            assert_eq!(args.num_rows, num_total_rows);
        }
        self.gather.record(
            recorder,
            num_total_rows,
            num_active_rows,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        <Unembed as ReplayLayer>::record(
            self.loaded_unembed(),
            recorder,
            UnembedInput {
                num_total_rows,
                num_active_rows,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }

    pub fn prepare_replay(&self, num_active_rows: u32) -> (Qwen3GatherUnembedReplayKey, ReplayArguments) {
        assert!(
            num_active_rows > 0,
            "qwen3 GatherUnembed replay requires Main output rows"
        );
        let key = Qwen3GatherUnembedReplayKey {
            num_total_rows: num_active_rows,
            unembed_topology: self.loaded_unembed().replay_topology(num_active_rows),
        };
        let arguments = ReplayArguments::new().with_u32(QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows);
        (key, arguments)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3GatherUnembedReplayKey {
    num_total_rows: u32,
    unembed_topology: affine_quantized::KernelKind,
}

impl ReplayComponent for Qwen3GatherUnembed {
    type Key = Qwen3GatherUnembedReplayKey;
    type Input<'a> = Qwen3GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.prepare_replay(input.num_rows).0
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen3GatherUnembed::record(
            self,
            recorder,
            key.num_total_rows,
            ReplayU32::Parameter(QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS),
            *input,
        );
    }
}
