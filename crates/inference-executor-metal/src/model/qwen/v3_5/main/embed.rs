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
use crate::model::embedding::EmbedInput;
use crate::replay::ReplayComponent;

const QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.main_embed.num_active_tokens");

pub struct Qwen35MainEmbed {
    embed: Rc<Embed>,
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
            embed,
            replay_bucket_policy: ReplayBucketPolicy::new(max_tokens),
        }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainEmbedArgs<'a>) -> &'a Buffer
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
        self.embed.record_bucketed(
            recorder,
            num_total_tokens,
            QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS,
            args.token_ids,
            args.hidden_output,
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::checkpoint::QuantizedTensorBindings;
    use inference_executor_core::checkpoint::SafeTensorIndex;
    use safetensors::Dtype as SafeTensorDtype;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS;
    use super::Qwen35MainEmbed;
    use super::Qwen35MainEmbedArgs;
    use crate::checkpoint::SafeTensorStore;
    use crate::def::replay_op::MetalReplayRuntime;
    use crate::model::embedding::Embed;
    use crate::model::embedding::EmbedConfig;
    use crate::replay::Replay;
    use crate::replay::ReplayComponent;

    const MAX_TOKENS: u32 = 6;
    const VOCAB_SIZE: u32 = 32;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;

    struct OwnedTensor {
        dtype: SafeTensorDtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> SafeTensorDtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    struct TempModelDir(PathBuf);

    impl TempModelDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "psi-qwen35-main-embed-test-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn stage_policy_uses_embedding_capacity_and_prepares_one_active_argument() {
        let device = Device::system_default();
        let embed = Rc::new(test_embed(&device));
        assert_eq!(embed.max_tokens(), MAX_TOKENS);
        let component = Qwen35MainEmbed::new(embed);
        let token_ids = Buffer::new_zeroed_elements(&device, MAX_TOKENS, Dtype::Int32);
        let hidden_output =
            Buffer::new_zeroed_elements(&device, MAX_TOKENS as usize * HIDDEN_DIM as usize, Dtype::Bfloat16);
        let input = |num_tokens| {
            Qwen35MainEmbedArgs {
                num_tokens,
                token_ids: &token_ids,
                hidden_output: &hidden_output,
            }
        };

        let (three_key, three_arguments) = component.prepare_replay(3);
        let (four_key, four_arguments) = component.prepare_replay(4);
        let (five_key, five_arguments) = component.prepare_replay(5);

        assert_eq!(three_key.num_total_tokens, 4);
        assert_eq!(three_key, four_key);
        assert_eq!(five_key.num_total_tokens, 6);
        assert_ne!(four_key, five_key);
        assert_eq!(component.replay_key(&input(3)), three_key);
        assert_eq!(
            three_arguments,
            ReplayArguments::new().with_u32(QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS, 3)
        );
        assert_eq!(
            four_arguments,
            ReplayArguments::new().with_u32(QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS, 4)
        );
        assert_eq!(
            five_arguments,
            ReplayArguments::new().with_u32(QWEN35_MAIN_EMBED_NUM_ACTIVE_TOKENS, 5)
        );
        assert_panics(|| {
            component.prepare_replay(0);
        });
        assert_panics(|| {
            component.prepare_replay(MAX_TOKENS + 1);
        });
    }

    #[test]
    fn replay_component_reuses_bucket_program_and_preserves_exact_recording() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let component = Qwen35MainEmbed::new(Rc::new(test_embed(&device)));
        let token_ids = Buffer::new_zeroed_elements(&device, MAX_TOKENS, Dtype::Int32);
        let hidden_output =
            Buffer::new_zeroed_elements(&device, MAX_TOKENS as usize * HIDDEN_DIM as usize, Dtype::Bfloat16);
        let input = |num_tokens| {
            Qwen35MainEmbedArgs {
                num_tokens,
                token_ids: &token_ids,
                hidden_output: &hidden_output,
            }
        };
        let (_, active_three_arguments) = component.prepare_replay(3);
        let mut exact_recorder = runtime.create_recorder();
        component.record(&mut exact_recorder, input(3));
        assert_eq!(exact_recorder.build().stats().parameter_count, 0);
        let mut replay = Replay::new("qwen3.5 MainEmbed test", component);

        let (three_key, three_cache_hit) = replay.record(&runtime, &input(3));
        assert!(!three_cache_hit);
        assert_eq!(replay.replay(&three_key).stats().parameter_count, 1);
        stream
            .submit_replay_with_arguments(replay.replay(&three_key), &active_three_arguments)
            .wait();

        let (four_key, four_cache_hit) = replay.record(&runtime, &input(4));
        assert!(four_cache_hit);
        assert_eq!(four_key, three_key);

        let (five_key, five_cache_hit) = replay.record(&runtime, &input(5));
        assert!(!five_cache_hit);
        assert_ne!(five_key, three_key);
        assert_eq!(replay.replay(&five_key).stats().parameter_count, 1);
    }

    fn test_embed(device: &Device) -> Embed {
        const FILE_NAME: &str = "model.safetensors";
        const WEIGHT: &str = "embed.weight";
        const SCALES: &str = "embed.scales";
        const BIASES: &str = "embed.biases";
        let model_dir = TempModelDir::new();
        let weight_bytes = VOCAB_SIZE as usize * HIDDEN_DIM as usize;
        let affine_elements = VOCAB_SIZE as usize * HIDDEN_DIM as usize / GROUP_SIZE as usize;
        let tensors = HashMap::from([
            (
                WEIGHT.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::U32,
                    shape: vec![weight_bytes / size_of::<u32>()],
                    data: vec![0; weight_bytes],
                },
            ),
            (
                SCALES.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![affine_elements],
                    data: vec![0; affine_elements * size_of::<u16>()],
                },
            ),
            (
                BIASES.to_string(),
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![affine_elements],
                    data: vec![0; affine_elements * size_of::<u16>()],
                },
            ),
        ]);
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &model_dir.0.join(FILE_NAME),
        )
        .unwrap();
        let index = SafeTensorIndex::new(HashMap::from([
            (WEIGHT.to_string(), FILE_NAME.to_string()),
            (SCALES.to_string(), FILE_NAME.to_string()),
            (BIASES.to_string(), FILE_NAME.to_string()),
        ]))
        .unwrap();
        let mut store = SafeTensorStore::new(&model_dir.0, index);
        let mut embed = Embed::new(
            device,
            EmbedConfig {
                max_tokens: MAX_TOKENS,
                vocab_size: VOCAB_SIZE,
                hidden_dim: HIDDEN_DIM,
                group_size: GROUP_SIZE,
                bits: 8,
                scale_bias_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
            },
        );
        embed
            .load_weights(
                device,
                &mut store,
                QuantizedTensorBindings {
                    weight: WEIGHT.to_string(),
                    scales: SCALES.to_string(),
                    biases: BIASES.to_string(),
                },
            )
            .unwrap();
        embed
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
