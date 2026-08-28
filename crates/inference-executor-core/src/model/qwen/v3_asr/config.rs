use std::fs::File;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::def::ModelExecutorError;
use crate::model::qwen::v3::Qwen3TextConfig;
use crate::model::qwen::v3_x::QuantizationConfig;

#[derive(Clone, Debug)]
pub struct Qwen3ASRModelConfig {
    pub audio: Qwen3ASRAudioConfig,
    pub text: Qwen3TextConfig,
    pub quantization: QuantizationConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub support_languages: Vec<String>,
    pub generation: Qwen3ASRGenerationConfig,
    pub preprocessor: Qwen3ASRPreprocessorConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRAudioConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub d_model: usize,
    pub max_source_positions: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub conv_chunksize: usize,
    pub downsample_hidden_size: usize,
    pub output_dim: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRGenerationConfig {
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ASRPreprocessorConfig {
    pub chunk_length_seconds: usize,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_samples: usize,
    pub max_frames: usize,
    pub dither: f32,
}

#[derive(Debug, Deserialize)]
struct CheckpointConfig {
    architectures: Vec<String>,
    model_type: String,
    quantization: Value,
    quantization_config: Value,
    support_languages: Vec<String>,
    thinker_config: ThinkerConfig,
}

#[derive(Debug, Deserialize)]
struct ThinkerConfig {
    architectures: Vec<String>,
    model_type: String,
    audio_config: AudioCheckpointConfig,
    text_config: TextCheckpointConfig,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_token_id: u32,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct AudioCheckpointConfig {
    model_type: String,
    activation_function: String,
    activation_dropout: f32,
    attention_dropout: f32,
    dropout: f32,
    scale_embedding: bool,
    num_mel_bins: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    d_model: usize,
    max_source_positions: usize,
    n_window: usize,
    n_window_infer: usize,
    conv_chunksize: usize,
    downsample_hidden_size: usize,
    output_dim: usize,
}

#[derive(Debug, Deserialize)]
struct TextCheckpointConfig {
    model_type: String,
    attention_bias: bool,
    attention_dropout: f32,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    rope_scaling: RopeScaling,
    tie_word_embeddings: bool,
    use_cache: bool,
}

#[derive(Debug, Deserialize)]
struct RopeScaling {
    interleaved: bool,
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    rope_type: String,
    #[serde(rename = "type")]
    type_name: String,
}

#[derive(Debug, Deserialize)]
struct GenerationCheckpointConfig {
    eos_token_id: Vec<u32>,
    pad_token_id: u32,
    do_sample: bool,
}

#[derive(Debug, Deserialize)]
struct PreprocessorCheckpointConfig {
    chunk_length: usize,
    dither: f32,
    feature_extractor_type: String,
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    n_samples: usize,
    nb_max_frames: usize,
    padding_side: String,
    padding_value: f32,
    processor_class: String,
    return_attention_mask: bool,
}

pub fn init_qwen3_asr_config(model_dir: impl AsRef<Path>) -> Result<Qwen3ASRModelConfig, ModelExecutorError> {
    let model_dir = model_dir.as_ref();
    let checkpoint = load_json(model_dir.join("config.json"), "model")?;
    let generation = load_json(model_dir.join("generation_config.json"), "generation")?;
    let preprocessor = load_json(model_dir.join("preprocessor_config.json"), "preprocessor")?;
    normalize(checkpoint, generation, preprocessor)
}

pub const fn audio_output_rows(num_frames: usize) -> usize {
    let complete_chunks = num_frames / 100;
    let tail_frames = num_frames % 100;
    complete_chunks * 13 + tail_frames.div_ceil(8)
}

fn load_json<T>(path: impl AsRef<Path>, kind: &str) -> Result<T, ModelExecutorError>
where
    T: for<'de> Deserialize<'de>,
{
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| {
        ModelExecutorError::custom(format!("unable to open Qwen3-ASR {kind} config {path:?}: {error}"))
    })?;
    serde_json::from_reader(file).map_err(|error| {
        ModelExecutorError::custom(format!("unable to parse Qwen3-ASR {kind} config {path:?}: {error}"))
    })
}

fn normalize(
    checkpoint: CheckpointConfig,
    generation: GenerationCheckpointConfig,
    preprocessor: PreprocessorCheckpointConfig,
) -> Result<Qwen3ASRModelConfig, ModelExecutorError> {
    validate_identity(&checkpoint.model_type, &checkpoint.architectures, "top-level")?;
    validate_identity(
        &checkpoint.thinker_config.model_type,
        &checkpoint.thinker_config.architectures,
        "thinker",
    )?;
    if !checkpoint.thinker_config.dtype.eq_ignore_ascii_case("bfloat16") {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3-ASR thinker dtype {:?}; expected bfloat16",
            checkpoint.thinker_config.dtype
        )));
    }

    let mut quantization = parse_quantization(checkpoint.quantization)?;
    let quantization_config = parse_quantization(checkpoint.quantization_config)?;
    if quantization != quantization_config {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR quantization and quantization_config must match",
        ));
    }
    quantization.normalize_tensor_overrides();
    if quantization.group_size != 64
        || quantization.bits != 8
        || quantization.mode.as_deref() != Some("affine")
        || !quantization.tensor_overrides.is_empty()
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR requires affine 8-bit text weights with group_size=64 and no overrides",
        ));
    }

    let thinker = checkpoint.thinker_config;
    let audio = normalize_audio(thinker.audio_config)?;
    let text = normalize_text(thinker.text_config)?;
    if audio.output_dim != text.hidden_size {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3-ASR audio output_dim={} must equal text hidden_size={}",
            audio.output_dim, text.hidden_size
        )));
    }
    for (name, token_id) in [
        ("audio_start_token_id", thinker.audio_start_token_id),
        ("audio_end_token_id", thinker.audio_end_token_id),
        ("audio_token_id", thinker.audio_token_id),
    ] {
        if token_id as usize >= text.vocab_size {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3-ASR {name}={token_id} must be below vocab_size={}",
                text.vocab_size
            )));
        }
    }
    if checkpoint.support_languages.is_empty()
        || checkpoint
            .support_languages
            .iter()
            .any(|language| language.trim().is_empty())
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR support_languages must contain nonempty names",
        ));
    }

    let generation = normalize_generation(generation, text.vocab_size)?;
    let preprocessor = normalize_preprocessor(preprocessor, &audio)?;
    Ok(Qwen3ASRModelConfig {
        audio,
        text,
        quantization,
        audio_start_token_id: thinker.audio_start_token_id,
        audio_end_token_id: thinker.audio_end_token_id,
        audio_token_id: thinker.audio_token_id,
        support_languages: checkpoint.support_languages,
        generation,
        preprocessor,
    })
}

fn validate_identity(model_type: &str, architectures: &[String], owner: &str) -> Result<(), ModelExecutorError> {
    if model_type != "qwen3_asr" || architectures != ["Qwen3ASRForConditionalGeneration"] {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3-ASR {owner} identity: model_type={model_type:?}, architectures={architectures:?}"
        )));
    }
    Ok(())
}

fn parse_quantization(value: Value) -> Result<QuantizationConfig, ModelExecutorError> {
    serde_json::from_value(value)
        .map_err(|error| ModelExecutorError::custom(format!("unable to parse Qwen3-ASR quantization: {error}")))
}

fn normalize_audio(raw: AudioCheckpointConfig) -> Result<Qwen3ASRAudioConfig, ModelExecutorError> {
    if raw.model_type != "qwen3_asr_audio_encoder"
        || raw.activation_function != "gelu"
        || raw.activation_dropout != 0.0
        || raw.attention_dropout != 0.0
        || raw.dropout != 0.0
        || raw.scale_embedding
    {
        return Err(ModelExecutorError::custom(
            "unsupported Qwen3-ASR audio encoder semantics",
        ));
    }
    let config = Qwen3ASRAudioConfig {
        num_mel_bins: raw.num_mel_bins,
        encoder_layers: raw.encoder_layers,
        encoder_attention_heads: raw.encoder_attention_heads,
        encoder_ffn_dim: raw.encoder_ffn_dim,
        d_model: raw.d_model,
        max_source_positions: raw.max_source_positions,
        n_window: raw.n_window,
        n_window_infer: raw.n_window_infer,
        conv_chunksize: raw.conv_chunksize,
        downsample_hidden_size: raw.downsample_hidden_size,
        output_dim: raw.output_dim,
    };
    let target = Qwen3ASRAudioConfig {
        num_mel_bins: 128,
        encoder_layers: 24,
        encoder_attention_heads: 16,
        encoder_ffn_dim: 4096,
        d_model: 1024,
        max_source_positions: 1500,
        n_window: 50,
        n_window_infer: 800,
        conv_chunksize: 500,
        downsample_hidden_size: 480,
        output_dim: 2048,
    };
    if config != target {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3-ASR audio geometry: {config:?}"
        )));
    }
    Ok(config)
}

fn normalize_text(raw: TextCheckpointConfig) -> Result<Qwen3TextConfig, ModelExecutorError> {
    if raw.model_type != "qwen3"
        || raw.attention_bias
        || raw.attention_dropout != 0.0
        || raw.hidden_act != "silu"
        || !raw.tie_word_embeddings
        || !raw.use_cache
    {
        return Err(ModelExecutorError::custom(
            "unsupported Qwen3-ASR text decoder semantics",
        ));
    }
    if !raw.rope_scaling.interleaved
        || !raw.rope_scaling.mrope_interleaved
        || raw.rope_scaling.mrope_section != [24, 20, 20]
        || raw.rope_scaling.rope_type != "default"
        || raw.rope_scaling.type_name != "default"
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR requires default interleaved M-RoPE with mrope_section=[24,20,20]",
        ));
    }
    if raw.hidden_size == 0
        || raw.intermediate_size == 0
        || raw.num_hidden_layers == 0
        || raw.num_attention_heads == 0
        || raw.num_key_value_heads == 0
        || raw.head_dim == 0
        || raw.vocab_size == 0
        || raw.max_position_embeddings == 0
        || !raw.rms_norm_eps.is_finite()
        || raw.rms_norm_eps <= 0.0
        || !raw.rope_theta.is_finite()
        || raw.rope_theta <= 0.0
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR text dimensions and scalar parameters must be positive and finite",
        ));
    }
    if raw.num_attention_heads.checked_mul(raw.head_dim) != Some(raw.hidden_size)
        || !raw.num_attention_heads.is_multiple_of(raw.num_key_value_heads)
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR text attention geometry is inconsistent",
        ));
    }
    Ok(Qwen3TextConfig {
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        rms_norm_eps: raw.rms_norm_eps,
        vocab_size: raw.vocab_size,
        max_position_embeddings: raw.max_position_embeddings,
        rope_theta: raw.rope_theta,
    })
}

fn normalize_generation(
    raw: GenerationCheckpointConfig,
    vocab_size: usize,
) -> Result<Qwen3ASRGenerationConfig, ModelExecutorError> {
    if raw.do_sample || raw.eos_token_id.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR generation must use greedy decoding and at least one EOS token",
        ));
    }
    if raw
        .eos_token_id
        .iter()
        .chain(std::iter::once(&raw.pad_token_id))
        .any(|&token_id| token_id as usize >= vocab_size)
    {
        return Err(ModelExecutorError::custom(
            "Qwen3-ASR generation token IDs must be below vocab_size",
        ));
    }
    Ok(Qwen3ASRGenerationConfig {
        eos_token_ids: raw.eos_token_id,
        pad_token_id: raw.pad_token_id,
    })
}

fn normalize_preprocessor(
    raw: PreprocessorCheckpointConfig,
    audio: &Qwen3ASRAudioConfig,
) -> Result<Qwen3ASRPreprocessorConfig, ModelExecutorError> {
    if raw.feature_extractor_type != "WhisperFeatureExtractor"
        || raw.processor_class != "Qwen3ASRProcessor"
        || raw.padding_side != "right"
        || raw.padding_value != 0.0
        || !raw.return_attention_mask
        || raw.dither != 0.0
        || raw.feature_size != audio.num_mel_bins
        || raw.chunk_length != 30
        || raw.hop_length != 160
        || raw.n_fft != 400
        || raw.n_samples != 480_000
        || raw.nb_max_frames != 3_000
    {
        return Err(ModelExecutorError::custom(
            "unsupported Qwen3-ASR Whisper preprocessor contract",
        ));
    }
    Ok(Qwen3ASRPreprocessorConfig {
        chunk_length_seconds: raw.chunk_length,
        feature_size: raw.feature_size,
        hop_length: raw.hop_length,
        n_fft: raw.n_fft,
        n_samples: raw.n_samples,
        max_frames: raw.nb_max_frames,
        dither: raw.dither,
    })
}
