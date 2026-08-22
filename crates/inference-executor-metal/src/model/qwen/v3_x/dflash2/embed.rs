use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayU32;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

pub struct Qwen3xDFlash2Embed {
    embed: Option<Rc<Embed>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDFlash2EmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDFlash2EmbedReplayKey {
    num_tokens: u32,
}

impl Qwen3xDFlash2Embed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed: Some(embed) }
    }

    pub fn load_weights(&mut self, embed: Rc<Embed>) {
        assert!(self.embed.is_none(), "Qwen3.x DFlash2 embed weights are already loaded");
        self.embed = Some(embed);
    }

    pub fn unload_weights(&mut self) -> Rc<Embed> {
        self.embed.take().expect("Qwen3.x DFlash2 embed weights are not loaded")
    }

    fn embed(&self) -> &Embed {
        self.embed
            .as_deref()
            .expect("Qwen3.x DFlash2 embed weights must be loaded before execution")
    }
}

impl ReplayComponent for Qwen3xDFlash2Embed {
    type Key = Qwen3xDFlash2EmbedReplayKey;
    type Input<'a> = Qwen3xDFlash2EmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDFlash2EmbedReplayKey {
            num_tokens: input.num_tokens,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let _ = <Embed as ReplayLayer>::record(
            self.embed(),
            recorder,
            EmbedInput {
                num_total_tokens: input.num_tokens,
                num_active_tokens: ReplayU32::Fixed(input.num_tokens),
                token_ids: input.token_ids,
                output_hidden: input.hidden_output,
            },
        );
    }
}
