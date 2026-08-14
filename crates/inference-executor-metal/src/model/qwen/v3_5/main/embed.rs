use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedBucketedInput;
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

const QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.main_embed.num_active_tokens");

pub struct Qwen35MainEmbed {
    embed: Option<Rc<Embed>>,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

impl Qwen35MainEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        let max_tokens = embed.max_tokens();
        Self {
            embed: Some(embed),
            replay_bucket_policy: ReplayBucketPolicy::new(max_tokens),
        }
    }

    pub fn load_weights(&mut self, embed: Rc<Embed>) {
        assert!(self.embed.is_none(), "qwen3.5 Main embed weights are already loaded");
        self.embed = Some(embed);
    }

    pub fn unload_weights(&mut self) -> Rc<Embed> {
        self.embed.take().expect("qwen3.5 Main embed weights are not loaded")
    }

    fn embed(&self) -> &Embed {
        self.embed
            .as_deref()
            .expect("qwen3.5 Main embed weights must be loaded before execution")
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainEmbedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        <Embed as ReplayLayer>::record(
            self.embed(),
            recorder,
            EmbedInput {
                num_tokens: args.num_tokens,
                token_ids: args.token_ids,
                output_hidden: args.hidden_output,
            },
        )
    }

    pub fn prepare_replay(&self, num_active_tokens: u32) -> (Qwen35MainEmbedReplayKey, ReplayArguments) {
        let key = self.replay_key_for_active_tokens(num_active_tokens);
        let arguments = ReplayArguments::new().with_u32(QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS, num_active_tokens);
        (key, arguments)
    }

    fn replay_key_for_active_tokens(&self, num_active_tokens: u32) -> Qwen35MainEmbedReplayKey {
        Qwen35MainEmbedReplayKey::new(self.replay_bucket_policy.capacity(num_active_tokens))
    }

    fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        args: Qwen35MainEmbedArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.embed().record_bucketed(
            recorder,
            EmbedBucketedInput {
                num_total_tokens,
                num_active_tokens_key: QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS,
                token_ids: args.token_ids,
                output_hidden: args.hidden_output,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainEmbedReplayKey {
    num_total_tokens: u32,
}

impl Qwen35MainEmbedReplayKey {
    /// Creates a key for an already-selected replay token capacity.
    pub fn new(num_total_tokens: u32) -> Self {
        Self { num_total_tokens }
    }
}

impl ReplayComponent for Qwen35MainEmbed {
    type Key = Qwen35MainEmbedReplayKey;
    type Input<'a> = Qwen35MainEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for_active_tokens(input.num_tokens)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        self.record_bucketed(recorder, key.num_total_tokens, *input);
    }
}
