use std::rc::Rc;

use inference_backend_metal::metal::Buffer;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

pub struct Qwen3xDSparkEmbed {
    embed: Rc<Embed>,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkEmbedReplayKey {
    num_tokens: u32,
}

impl Qwen3xDSparkEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed }
    }
}

impl ReplayComponent for Qwen3xDSparkEmbed {
    type Key = Qwen3xDSparkEmbedReplayKey;
    type Input<'a> = Qwen3xDSparkEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDSparkEmbedReplayKey {
            num_tokens: input.num_tokens,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let _ = <Embed as ReplayLayer>::record(
            &self.embed,
            recorder,
            EmbedInput {
                num_tokens: input.num_tokens,
                token_ids: input.token_ids,
                output_hidden: input.hidden_output,
            },
        );
    }
}
