use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

pub struct Qwen3MainEmbed {
    embed: Rc<Embed>,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

impl Qwen3MainEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen3MainEmbedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        <Embed as ReplayLayer>::record(
            &self.embed,
            recorder,
            EmbedInput {
                num_tokens: args.num_tokens,
                token_ids: args.token_ids,
                output_hidden: args.hidden_output,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3MainEmbedReplayKey {
    num_tokens: u32,
}

impl Qwen3MainEmbedReplayKey {
    pub fn new(num_tokens: u32) -> Self {
        Self { num_tokens }
    }
}

impl ReplayComponent for Qwen3MainEmbed {
    type Key = Qwen3MainEmbedReplayKey;
    type Input<'a> = Qwen3MainEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3MainEmbedReplayKey::new(input.num_tokens)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen3MainEmbed::record(self, recorder, *input);
    }
}
