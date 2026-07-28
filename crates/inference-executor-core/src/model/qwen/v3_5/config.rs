use std::path::Path;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde_json::Value;

use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::QuantizationConfig;
use crate::model::qwen::v3_x::RopeParameters;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen35ModelConfig {
    pub model_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub text_config: Qwen35TextConfig,
    #[serde(default, deserialize_with = "deserialize_quantization_config")]
    pub quantization: Option<QuantizationConfig>,
}

fn deserialize_quantization_config<'de, D>(deserializer: D) -> Result<Option<QuantizationConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut quantization = Option::<QuantizationConfig>::deserialize(deserializer)?;
    if let Some(config) = &mut quantization {
        config.normalize_tensor_overrides();
    }
    Ok(quantization)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen35TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default)]
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub full_attention_interval: usize,
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    #[serde(default)]
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub dtype: Option<String>,

    #[serde(skip)]
    pub scale: f32,
    #[serde(skip)]
    pub rope_dim: usize,
}

fn default_norm_topk_prob() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerType {
    GDN,
    FullAttention,
}

impl Qwen35ModelConfig {
    pub fn normalize(&mut self) -> Result<(), ModelExecutorError> {
        if let Some(quantization) = &mut self.quantization {
            quantization.normalize_tensor_overrides();
        }
        normalize_text_config(&mut self.text_config)
    }

    pub fn layer_type_at(&self, layer_index: usize) -> Result<LayerType, ModelExecutorError> {
        layer_type_at(&self.text_config, layer_index)
    }

    pub fn layer_uses_moe(&self, layer_index: usize) -> bool {
        layer_uses_moe(&self.text_config, layer_index)
    }
}

pub fn init_qwen35_model_config(model_dir: impl AsRef<Path>) -> Result<Qwen35ModelConfig, ModelExecutorError> {
    let model_config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&model_config_path).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to open qwen3.5 model config file {:?}, err: {err:?}",
            model_config_path
        ))
    })?;
    let envelope = serde_json::from_reader::<_, Value>(file).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to parse qwen3.5 model config file {:?}, err: {err:?}",
            model_config_path
        ))
    })?;
    parse_qwen35_config(envelope).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to normalize qwen3.5 model config from {:?}, err: {err}",
            model_config_path
        ))
    })
}

fn parse_qwen35_config(envelope: Value) -> Result<Qwen35ModelConfig, ModelExecutorError> {
    let mut model_config = serde_json::from_value::<Qwen35ModelConfig>(envelope.clone()).map_err(|err| {
        ModelExecutorError::custom(format!("unable to parse qwen3.5 model config envelope, err: {err:?}"))
    })?;
    let nested_quantization = parse_nested_quantization_config(&envelope)?;
    match (&model_config.quantization, nested_quantization) {
        (Some(quantization), Some(nested_quantization)) if quantization != &nested_quantization => {
            return Err(ModelExecutorError::custom(
                "qwen3.5 quantization and quantization_config must describe the same layout",
            ));
        },
        (None, nested_quantization) => model_config.quantization = nested_quantization,
        _ => {},
    }
    model_config.normalize()?;
    Ok(model_config)
}

fn parse_nested_quantization_config(envelope: &Value) -> Result<Option<QuantizationConfig>, ModelExecutorError> {
    let Some(value) = envelope.get("quantization_config").cloned() else {
        return Ok(None);
    };
    let mut config = serde_json::from_value::<QuantizationConfig>(value).map_err(|err| {
        ModelExecutorError::custom(format!("unable to parse qwen3.5 quantization_config, err: {err:?}"))
    })?;
    config.normalize_tensor_overrides();
    Ok(Some(config))
}

fn layer_type_at(config: &Qwen35TextConfig, layer_index: usize) -> Result<LayerType, ModelExecutorError> {
    if layer_index >= config.num_hidden_layers {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 layer_index={layer_index} is outside num_hidden_layers={}",
            config.num_hidden_layers
        )));
    }
    if let Some(layer_type) = config.layer_types.get(layer_index) {
        return match layer_type.as_str() {
            "gated_delta_net" | "linear_attention" => Ok(LayerType::GDN),
            "full_attention" => Ok(LayerType::FullAttention),
            other => {
                Err(ModelExecutorError::custom(format!(
                    "unknown qwen3.5 layer type {other:?} at layer {layer_index}"
                )))
            },
        };
    }

    if config.full_attention_interval > 0 && (layer_index + 1).is_multiple_of(config.full_attention_interval) {
        Ok(LayerType::FullAttention)
    } else {
        Ok(LayerType::GDN)
    }
}

fn layer_uses_moe(config: &Qwen35TextConfig, layer_index: usize) -> bool {
    if config.num_experts == 0 {
        return false;
    }
    if config.decoder_sparse_step <= 1 {
        return true;
    }
    (layer_index + 1).is_multiple_of(config.decoder_sparse_step)
}

fn normalize_text_config(config: &mut Qwen35TextConfig) -> Result<(), ModelExecutorError> {
    if config.hidden_size == 0 {
        return Err(ModelExecutorError::custom("qwen3.5 hidden_size must be positive"));
    }
    if config.num_hidden_layers == 0 {
        return Err(ModelExecutorError::custom("qwen3.5 num_hidden_layers must be positive"));
    }
    if config.num_attention_heads == 0 {
        return Err(ModelExecutorError::custom(
            "qwen3.5 num_attention_heads must be positive",
        ));
    }
    if config.hidden_act.is_empty() {
        config.hidden_act = "silu".to_string();
    }
    if config.num_key_value_heads == 0 {
        config.num_key_value_heads = config.num_attention_heads;
    }
    if config.head_dim == 0 {
        if !config.hidden_size.is_multiple_of(config.num_attention_heads) {
            return Err(ModelExecutorError::custom(format!(
                "qwen3.5 hidden_size={} must be divisible by num_attention_heads={}",
                config.hidden_size, config.num_attention_heads
            )));
        }
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    if config.rms_norm_eps == 0.0 {
        config.rms_norm_eps = 1e-6;
    }
    if config.linear_conv_kernel_dim == 0 {
        config.linear_conv_kernel_dim = 4;
    }
    if config.rope_theta == 0.0 {
        config.rope_theta = config
            .rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_theta)
            .unwrap_or(100_000.0);
    }
    if config.partial_rotary_factor == 0.0 {
        config.partial_rotary_factor = config
            .rope_parameters
            .as_ref()
            .and_then(|rope| rope.partial_rotary_factor)
            .unwrap_or(0.25);
    }
    if config.full_attention_interval == 0 {
        config.full_attention_interval = config
            .layer_types
            .iter()
            .position(|layer_type| layer_type.to_ascii_lowercase().contains("full"))
            .map(|index| index + 1)
            .unwrap_or(4);
    }
    if config.full_attention_interval > config.num_hidden_layers {
        config.full_attention_interval = config.num_hidden_layers;
    }
    config.scale = (config.head_dim as f32).sqrt().recip();
    config.rope_dim = ((config.head_dim as f32) * config.partial_rotary_factor)
        .round()
        .clamp(2.0, config.head_dim as f32) as usize;
    if !config.rope_dim.is_multiple_of(2) {
        config.rope_dim -= 1;
    }
    Ok(())
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
