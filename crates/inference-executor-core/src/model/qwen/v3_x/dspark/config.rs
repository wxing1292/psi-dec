use std::num::NonZeroUsize;
use std::path::Path;

use serde::Deserialize;

use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::QuantizationConfig;
use crate::model::qwen::v3_x::RopeParameters;

#[derive(Clone, Debug)]
pub struct Qwen3xDSparkConfig {
    pub block_size: usize,
    pub mask_token_id: usize,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub markov_rank: usize,
    pub num_anchors: usize,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3xDSparkMainConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl Qwen3xDSparkConfig {
    pub fn resolve_num_spec_tokens(&self, requested: Option<NonZeroUsize>) -> Result<NonZeroUsize, ModelExecutorError> {
        let num_spec_tokens = requested.unwrap_or_else(|| {
            NonZeroUsize::new(self.block_size).expect("validated Qwen3 DSpark checkpoint block_size must be positive")
        });
        if num_spec_tokens.get() > self.block_size {
            return Err(ModelExecutorError::custom(format!(
                "requested num_spec_tokens={num_spec_tokens} exceeds Qwen3 DSpark checkpoint block_size={}",
                self.block_size
            )));
        }
        Ok(num_spec_tokens)
    }

    pub fn validate_main(&self, main: Qwen3xDSparkMainConfig) -> Result<(), ModelExecutorError> {
        let mismatches = [
            (self.hidden_size != main.hidden_size)
                .then(|| format!("hidden_size dspark={} main={}", self.hidden_size, main.hidden_size)),
            (self.num_target_layers != main.num_hidden_layers).then(|| {
                format!(
                    "num_target_layers dspark={} main={}",
                    self.num_target_layers, main.num_hidden_layers
                )
            }),
            (self.vocab_size != main.vocab_size)
                .then(|| format!("vocab_size dspark={} main={}", self.vocab_size, main.vocab_size)),
            (self.max_position_embeddings != main.max_position_embeddings).then(|| {
                format!(
                    "max_position_embeddings dspark={} main={}",
                    self.max_position_embeddings, main.max_position_embeddings
                )
            }),
            (self.rope_theta != main.rope_theta)
                .then(|| format!("rope_theta dspark={} main={}", self.rope_theta, main.rope_theta)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if !mismatches.is_empty() {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3 DSpark config is incompatible with the Main model: {}",
                mismatches.join(", ")
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct Qwen3xDSparkCheckpointConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attention_dropout: f32,
    block_size: usize,
    #[serde(default)]
    confidence_head_with_markov: bool,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    enable_confidence_head: bool,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    markov_head_type: String,
    markov_rank: usize,
    mask_token_id: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_anchors: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    #[serde(default)]
    quantization: Option<QuantizationConfig>,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    torch_dtype: Option<String>,
    target_layer_ids: Vec<usize>,
    #[serde(default = "default_true")]
    use_cache: bool,
    #[serde(default)]
    use_sliding_window: bool,
    vocab_size: usize,
}

pub fn init_qwen3x_dspark_config(model_dir: impl AsRef<Path>) -> Result<Qwen3xDSparkConfig, ModelExecutorError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&config_path).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to open Qwen3 DSpark config file {config_path:?}, error: {error:?}"
        ))
    })?;
    let checkpoint = serde_json::from_reader::<_, Qwen3xDSparkCheckpointConfig>(file).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to parse Qwen3 DSpark config file {config_path:?}, error: {error:?}"
        ))
    })?;
    normalize_qwen3x_dspark_config(checkpoint).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to normalize Qwen3 DSpark config from {config_path:?}, error: {error}"
        ))
    })
}

fn normalize_qwen3x_dspark_config(
    mut checkpoint: Qwen3xDSparkCheckpointConfig,
) -> Result<Qwen3xDSparkConfig, ModelExecutorError> {
    validate_checkpoint_semantics(&checkpoint)?;
    for dtype in [checkpoint.dtype.as_deref(), checkpoint.torch_dtype.as_deref()]
        .into_iter()
        .flatten()
    {
        validate_dtype(dtype)?;
    }
    if let Some(quantization) = &mut checkpoint.quantization {
        validate_quantization(quantization)?;
        quantization.normalize_tensor_overrides();
    }
    let rope_theta = checkpoint
        .rope_parameters
        .rope_theta
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark rope_parameters.rope_theta must be present"))?;
    Ok(Qwen3xDSparkConfig {
        block_size: checkpoint.block_size,
        mask_token_id: checkpoint.mask_token_id,
        target_layer_ids: checkpoint.target_layer_ids,
        num_target_layers: checkpoint.num_target_layers,
        hidden_size: checkpoint.hidden_size,
        intermediate_size: checkpoint.intermediate_size,
        num_hidden_layers: checkpoint.num_hidden_layers,
        num_attention_heads: checkpoint.num_attention_heads,
        num_key_value_heads: checkpoint.num_key_value_heads,
        head_dim: checkpoint.head_dim,
        rms_norm_eps: checkpoint.rms_norm_eps,
        rope_theta,
        max_position_embeddings: checkpoint.max_position_embeddings,
        vocab_size: checkpoint.vocab_size,
        markov_rank: checkpoint.markov_rank,
        num_anchors: checkpoint.num_anchors,
        quantization: checkpoint.quantization,
    })
}

fn validate_checkpoint_semantics(config: &Qwen3xDSparkCheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.model_type != "qwen3" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark model_type {:?}; expected \"qwen3\"",
            config.model_type
        )));
    }
    if let Some(architecture) = config
        .architectures
        .iter()
        .find(|architecture| architecture.as_str() != "Qwen3DSparkModel")
    {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark architecture {architecture:?}; expected \"Qwen3DSparkModel\""
        )));
    }
    if config.attention_bias {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark attention_bias=true is unsupported",
        ));
    }
    if !config.attention_dropout.is_finite() || config.attention_dropout != 0.0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark attention_dropout must be 0, got {}",
            config.attention_dropout
        )));
    }
    if config.hidden_act != "silu" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark hidden_act {:?}; expected \"silu\"",
            config.hidden_act
        )));
    }
    if config.tie_word_embeddings {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark tie_word_embeddings=true is unsupported",
        ));
    }
    if !config.use_cache {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark use_cache=false is unsupported",
        ));
    }
    if config.use_sliding_window || config.sliding_window.is_some() {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark sliding-window attention is unsupported",
        ));
    }
    if !config.layer_types.is_empty()
        && (config.layer_types.len() != config.num_hidden_layers
            || config
                .layer_types
                .iter()
                .any(|layer_type| layer_type != "full_attention"))
    {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark layer_types must contain only full_attention layers",
        ));
    }
    validate_positive_dimensions(config)?;
    validate_attention_dimensions(config)?;
    validate_main_layer_selection(config)?;
    validate_rope(config)?;
    if config.mask_token_id >= config.vocab_size {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark mask_token_id={} must be below vocab_size={}",
            config.mask_token_id, config.vocab_size
        )));
    }
    if config.markov_head_type != "vanilla" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark markov_head_type {:?}; expected \"vanilla\"",
            config.markov_head_type
        )));
    }
    if !config.enable_confidence_head {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark requires enable_confidence_head=true",
        ));
    }
    if !config.confidence_head_with_markov {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark requires confidence_head_with_markov=true",
        ));
    }
    Ok(())
}

fn validate_positive_dimensions(config: &Qwen3xDSparkCheckpointConfig) -> Result<(), ModelExecutorError> {
    let dimensions = [
        ("block_size", config.block_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("num_target_layers", config.num_target_layers),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("vocab_size", config.vocab_size),
        ("markov_rank", config.markov_rank),
        ("num_anchors", config.num_anchors),
    ];
    if let Some((name, _)) = dimensions.into_iter().find(|&(_, value)| value == 0) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark {name} must be positive"
        )));
    }
    if config.num_target_layers <= 1 {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark num_target_layers must be greater than 1",
        ));
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark rms_norm_eps must be finite and positive, got {}",
            config.rms_norm_eps
        )));
    }
    Ok(())
}

fn validate_attention_dimensions(config: &Qwen3xDSparkCheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.head_dim < 2 || !config.head_dim.is_multiple_of(2) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark head_dim must be positive and even, got {}",
            config.head_dim
        )));
    }
    let _q_dim = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark query dimension must fit usize"))?;
    let _kv_dim = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark key/value dimension must fit usize"))?;
    if !config.num_attention_heads.is_multiple_of(config.num_key_value_heads) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark num_attention_heads={} must be divisible by num_key_value_heads={}",
            config.num_attention_heads, config.num_key_value_heads
        )));
    }
    Ok(())
}

fn validate_main_layer_selection(config: &Qwen3xDSparkCheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.target_layer_ids.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark target_layer_ids must not be empty",
        ));
    }
    for pair in config.target_layer_ids.windows(2) {
        if pair[0] >= pair[1] {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3 DSpark target_layer_ids must be strictly increasing: {:?}",
                config.target_layer_ids
            )));
        }
    }
    let unsupported_final_layer_id = config.num_target_layers - 1;
    if let Some(&layer_id) = config
        .target_layer_ids
        .iter()
        .find(|&&layer_id| layer_id >= unsupported_final_layer_id)
    {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark target_layer_ids entry {layer_id} must be below unsupported final layer \
             {unsupported_final_layer_id} for num_target_layers={}",
            config.num_target_layers,
        )));
    }
    Ok(())
}

fn validate_rope(config: &Qwen3xDSparkCheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.rope_parameters.rope_type.as_deref() != Some("default") {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark rope_type {:?}; expected \"default\"",
            config.rope_parameters.rope_type
        )));
    }
    let rope_theta = config
        .rope_parameters
        .rope_theta
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark rope_parameters.rope_theta must be present"))?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark rope_theta must be finite and positive, got {rope_theta}"
        )));
    }
    if config
        .rope_parameters
        .partial_rotary_factor
        .is_some_and(|factor| !factor.is_finite() || factor != 1.0)
        || config.rope_parameters.factor.is_some()
        || config.rope_parameters.original_max_position_embeddings.is_some()
    {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark scaled or partial RoPE is unsupported",
        ));
    }
    Ok(())
}

fn validate_dtype(dtype: &str) -> Result<(), ModelExecutorError> {
    match dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(()),
        _ => {
            Err(ModelExecutorError::custom(format!(
                "unsupported Qwen3 DSpark dtype {dtype:?}; expected bfloat16"
            )))
        },
    }
}

fn validate_quantization(config: &QuantizationConfig) -> Result<(), ModelExecutorError> {
    if config.group_size == 0 {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark quantization group_size must be positive",
        ));
    }
    if !matches!(config.bits, 2 | 3 | 4 | 6 | 8) {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 DSpark quantization bits={}; expected 2, 3, 4, 6, or 8",
            config.bits
        )));
    }
    for (tensor_name, tensor) in &config.tensor_overrides {
        if tensor.group_size == Some(0) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3 DSpark quantization override {tensor_name:?} group_size must be positive"
            )));
        }
        if tensor.bits.is_some_and(|bits| !matches!(bits, 2 | 3 | 4 | 6 | 8)) {
            return Err(ModelExecutorError::custom(format!(
                "unsupported Qwen3 DSpark quantization override {tensor_name:?} bits={:?}",
                tensor.bits
            )));
        }
    }
    Ok(())
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
