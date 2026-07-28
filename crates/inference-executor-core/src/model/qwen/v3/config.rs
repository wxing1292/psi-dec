use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::QuantizationConfig;

#[derive(Clone, Debug)]
pub struct Qwen3ModelConfig {
    pub text_config: Qwen3TextConfig,
    pub quantization: Option<QuantizationConfig>,
    eos_token_ids: Vec<u32>,
}

impl Qwen3ModelConfig {
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Deserialize)]
struct Qwen3CheckpointConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    decoder_sparse_step: usize,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    full_attention_interval: usize,
    #[serde(default)]
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_num_key_heads: usize,
    #[serde(default)]
    linear_key_head_dim: usize,
    #[serde(default)]
    linear_value_head_dim: usize,
    #[serde(default)]
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default)]
    quantization: Option<Value>,
    #[serde(default)]
    quantization_config: Option<Value>,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_parameters: Option<Value>,
    #[serde(default)]
    rope_scaling: Option<Value>,
    rope_theta: f32,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default = "default_true")]
    use_cache: bool,
    #[serde(default)]
    use_sliding_window: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIdOrIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(token_id) => vec![token_id],
            Self::Many(token_ids) => token_ids,
        }
    }
}

pub fn init_qwen3_model_config(model_dir: impl AsRef<Path>) -> Result<Qwen3ModelConfig, ModelExecutorError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&config_path).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to open Qwen3 model config file {config_path:?}, err: {err:?}"
        ))
    })?;
    let envelope = serde_json::from_reader::<_, Value>(file).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to parse Qwen3 model config file {config_path:?}, err: {err:?}"
        ))
    })?;
    parse_qwen3_config(envelope).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to normalize Qwen3 model config from {config_path:?}, err: {err}"
        ))
    })
}

fn parse_qwen3_config(envelope: Value) -> Result<Qwen3ModelConfig, ModelExecutorError> {
    let checkpoint = serde_json::from_value::<Qwen3CheckpointConfig>(envelope).map_err(|err| {
        ModelExecutorError::custom(format!("unable to parse flat Qwen3 checkpoint config, err: {err:?}"))
    })?;
    validate_checkpoint_semantics(&checkpoint)?;

    for dtype in [checkpoint.dtype.as_deref(), checkpoint.torch_dtype.as_deref()]
        .into_iter()
        .flatten()
    {
        validate_dtype(dtype)?;
    }
    let quantization = normalize_quantization(checkpoint.quantization, checkpoint.quantization_config)?;
    let eos_token_ids = normalize_eos_token_ids(checkpoint.eos_token_id, checkpoint.vocab_size)?;
    Ok(Qwen3ModelConfig {
        text_config: Qwen3TextConfig {
            hidden_size: checkpoint.hidden_size,
            intermediate_size: checkpoint.intermediate_size,
            num_hidden_layers: checkpoint.num_hidden_layers,
            num_attention_heads: checkpoint.num_attention_heads,
            num_key_value_heads: checkpoint.num_key_value_heads,
            head_dim: checkpoint.head_dim,
            rms_norm_eps: checkpoint.rms_norm_eps,
            vocab_size: checkpoint.vocab_size,
            max_position_embeddings: checkpoint.max_position_embeddings,
            rope_theta: checkpoint.rope_theta,
        },
        quantization,
        eos_token_ids,
    })
}

fn validate_checkpoint_semantics(config: &Qwen3CheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.model_type != "qwen3" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 model_type {:?}; expected \"qwen3\"",
            config.model_type
        )));
    }
    if let Some(architecture) = config
        .architectures
        .iter()
        .find(|architecture| architecture.as_str() != "Qwen3ForCausalLM")
    {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 architecture {architecture:?}; expected \"Qwen3ForCausalLM\""
        )));
    }
    if config.attention_bias {
        return Err(ModelExecutorError::custom("Qwen3 attention_bias=true is unsupported"));
    }
    if config.tie_word_embeddings {
        return Err(ModelExecutorError::custom(
            "Qwen3 tie_word_embeddings=true is unsupported",
        ));
    }
    if config.hidden_act != "silu" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 hidden_act {:?}; expected \"silu\"",
            config.hidden_act
        )));
    }
    if !config.use_cache {
        return Err(ModelExecutorError::custom("Qwen3 use_cache=false is unsupported"));
    }
    if config.rope_scaling.is_some() {
        return Err(ModelExecutorError::custom("Qwen3 rope_scaling is unsupported"));
    }
    if config.rope_parameters.is_some() {
        return Err(ModelExecutorError::custom("Qwen3 rope_parameters is unsupported"));
    }
    if config
        .partial_rotary_factor
        .is_some_and(|factor| !factor.is_finite() || factor != 1.0)
    {
        return Err(ModelExecutorError::custom(
            "Qwen3 requires partial_rotary_factor=1 when the field is present",
        ));
    }
    if config.use_sliding_window || config.sliding_window.is_some() {
        return Err(ModelExecutorError::custom(
            "Qwen3 sliding-window attention is unsupported",
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
            "Qwen3 layer_types must contain only full_attention layers",
        ));
    }
    if !matches!(config.full_attention_interval, 0 | 1) {
        return Err(ModelExecutorError::custom(
            "Qwen3 full_attention_interval must be 1 when present",
        ));
    }
    if [
        config.linear_num_value_heads,
        config.linear_num_key_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
    ]
    .into_iter()
    .any(|value| value != 0)
    {
        return Err(ModelExecutorError::custom("Qwen3 GDN configuration is unsupported"));
    }
    if [
        config.decoder_sparse_step,
        config.num_experts,
        config.num_experts_per_tok,
        config.shared_expert_intermediate_size,
        config.moe_intermediate_size,
    ]
    .into_iter()
    .any(|value| value != 0)
    {
        return Err(ModelExecutorError::custom("Qwen3 MoE configuration is unsupported"));
    }
    if config.mtp_num_hidden_layers != 0 || config.mtp_use_dedicated_embeddings {
        return Err(ModelExecutorError::custom("Qwen3 MTP configuration is unsupported"));
    }
    if config.hidden_size == 0 {
        return Err(ModelExecutorError::custom("Qwen3 hidden_size must be positive"));
    }
    if config.intermediate_size == 0 {
        return Err(ModelExecutorError::custom("Qwen3 intermediate_size must be positive"));
    }
    if config.num_hidden_layers == 0 {
        return Err(ModelExecutorError::custom("Qwen3 num_hidden_layers must be positive"));
    }
    if config.num_attention_heads == 0 {
        return Err(ModelExecutorError::custom("Qwen3 num_attention_heads must be positive"));
    }
    if config.num_key_value_heads == 0 {
        return Err(ModelExecutorError::custom("Qwen3 num_key_value_heads must be positive"));
    }
    if config.head_dim < 2 || !config.head_dim.is_multiple_of(2) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 head_dim must be positive and even, got {}",
            config.head_dim
        )));
    }
    let q_dim = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 query dimension must fit usize"))?;
    let _kv_dim = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 key/value dimension must fit usize"))?;
    if q_dim != config.hidden_size {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 num_attention_heads={} * head_dim={} must equal hidden_size={}",
            config.num_attention_heads, config.head_dim, config.hidden_size
        )));
    }
    if !config.num_attention_heads.is_multiple_of(config.num_key_value_heads) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 num_attention_heads={} must be divisible by num_key_value_heads={}",
            config.num_attention_heads, config.num_key_value_heads
        )));
    }
    if config.vocab_size == 0 {
        return Err(ModelExecutorError::custom("Qwen3 vocab_size must be positive"));
    }
    if config.max_position_embeddings == 0 {
        return Err(ModelExecutorError::custom(
            "Qwen3 max_position_embeddings must be positive",
        ));
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 rms_norm_eps must be finite and positive, got {}",
            config.rms_norm_eps
        )));
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 rope_theta must be finite and positive, got {}",
            config.rope_theta
        )));
    }
    Ok(())
}

fn validate_dtype(dtype: &str) -> Result<(), ModelExecutorError> {
    match dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(()),
        _ => {
            Err(ModelExecutorError::custom(format!(
                "unsupported Qwen3 dtype {dtype:?}; expected bfloat16"
            )))
        },
    }
}

fn normalize_quantization(
    quantization: Option<Value>,
    quantization_config: Option<Value>,
) -> Result<Option<QuantizationConfig>, ModelExecutorError> {
    let quantization = quantization
        .map(|value| parse_quantization("quantization", value))
        .transpose()?;
    let quantization_config = quantization_config
        .map(|value| parse_quantization("quantization_config", value))
        .transpose()?;
    match (quantization, quantization_config) {
        (None, None) => Ok(None),
        (Some(config), None) | (None, Some(config)) => Ok(Some(config)),
        (Some(config), Some(nested)) => {
            let config_value =
                serde_json::to_value(&config).expect("Qwen3 normalized quantization must serialize to JSON");
            let nested_value =
                serde_json::to_value(&nested).expect("Qwen3 normalized quantization must serialize to JSON");
            if config_value != nested_value {
                return Err(ModelExecutorError::custom(
                    "Qwen3 quantization and quantization_config must describe the same layout",
                ));
            }
            Ok(Some(config))
        },
    }
}

fn parse_quantization(name: &str, value: Value) -> Result<QuantizationConfig, ModelExecutorError> {
    let mut config = serde_json::from_value::<QuantizationConfig>(value)
        .map_err(|err| ModelExecutorError::custom(format!("unable to parse Qwen3 {name}, err: {err:?}")))?;
    validate_quantization_value(name, config.group_size, config.bits, config.mode.as_deref())?;
    for (tensor_name, tensor) in &config.tensor_overrides {
        validate_quantization_value(
            &format!("{name}.{tensor_name}"),
            tensor.group_size.unwrap_or(config.group_size),
            tensor.bits.unwrap_or(config.bits),
            tensor.mode.as_deref().or(config.mode.as_deref()),
        )?;
    }
    config.normalize_tensor_overrides();
    Ok(config)
}

fn validate_quantization_value(
    name: &str,
    group_size: usize,
    bits: usize,
    mode: Option<&str>,
) -> Result<(), ModelExecutorError> {
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 {name} group_size={group_size}; expected 32, 64, or 128"
        )));
    }
    if !matches!(bits, 2 | 3 | 4 | 6 | 8) {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 {name} bits={bits}; expected 2, 3, 4, 6, or 8"
        )));
    }
    if mode.is_some_and(|mode| !mode.eq_ignore_ascii_case("affine")) {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3 {name} mode={mode:?}; expected affine"
        )));
    }
    Ok(())
}

fn normalize_eos_token_ids(
    eos_token_id: Option<TokenIdOrIds>,
    vocab_size: usize,
) -> Result<Vec<u32>, ModelExecutorError> {
    let Some(eos_token_id) = eos_token_id else {
        return Ok(Vec::new());
    };
    let eos_token_ids = eos_token_id.into_vec();
    if eos_token_ids.is_empty() {
        return Err(ModelExecutorError::custom("Qwen3 eos_token_id list must not be empty"));
    }
    for (index, &token_id) in eos_token_ids.iter().enumerate() {
        if token_id as u64 >= vocab_size as u64 {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3 eos_token_id={token_id} must be below vocab_size={vocab_size}"
            )));
        }
        if eos_token_ids[..index].contains(&token_id) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3 eos_token_id contains duplicate token {token_id}"
            )));
        }
    }
    Ok(eos_token_ids)
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
