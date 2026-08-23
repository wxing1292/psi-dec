use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

const DSPARK_EMBED_NUM_ACTIVE_TOKENS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.dspark.embed.num_active_tokens");

pub struct Qwen3xDSparkEmbed {
    embed: Option<Rc<Embed>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkEmbedReplayKey {
    num_total_tokens: u32,
}

impl Qwen3xDSparkEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed: Some(embed) }
    }

    pub fn load_weights(&mut self, embed: Rc<Embed>) {
        assert!(self.embed.is_none(), "Qwen3.x DSpark embed weights are already loaded");
        self.embed = Some(embed);
    }

    pub fn unload_weights(&mut self) -> Rc<Embed> {
        self.embed.take().expect("Qwen3.x DSpark embed weights are not loaded")
    }

    fn embed(&self) -> &Embed {
        self.embed
            .as_deref()
            .expect("Qwen3.x DSpark embed weights must be loaded before execution")
    }

    pub fn prepare_replay(&self, num_active_tokens: u32) -> (Qwen3xDSparkEmbedReplayKey, ReplayArguments) {
        assert!(num_active_tokens > 0, "Qwen3.x DSpark Embed requires active tokens");
        let key = Qwen3xDSparkEmbedReplayKey {
            num_total_tokens: num_active_tokens,
        };
        let arguments = ReplayArguments::new().with_u32(DSPARK_EMBED_NUM_ACTIVE_TOKENS, num_active_tokens);
        (key, arguments)
    }
}

impl ReplayComponent for Qwen3xDSparkEmbed {
    type Key = Qwen3xDSparkEmbedReplayKey;
    type Input<'a> = Qwen3xDSparkEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDSparkEmbedReplayKey {
            num_total_tokens: input.num_tokens,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let _ = <Embed as ReplayLayer>::record(
            self.embed(),
            recorder,
            EmbedInput {
                num_total_tokens: input.num_tokens,
                num_active_tokens: ReplayU32::Parameter(DSPARK_EMBED_NUM_ACTIVE_TOKENS),
                token_ids: input.token_ids,
                output_hidden: input.hidden_output,
            },
        );
    }
}
