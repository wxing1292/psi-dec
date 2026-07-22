use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde_json::Value;

use crate::def::ModelExecutorError;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(flatten, default)]
    pub tensor_overrides: HashMap<String, TensorQuantizationOverride>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TensorQuantizationOverride {
    #[serde(default)]
    pub group_size: Option<usize>,
    #[serde(default)]
    pub bits: Option<usize>,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedQuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: Option<String>,
}

impl TensorQuantizationOverride {
    fn resolve_with_defaults(&self, defaults: &QuantizationConfig) -> ResolvedQuantizationConfig {
        ResolvedQuantizationConfig {
            group_size: self.group_size.unwrap_or(defaults.group_size),
            bits: self.bits.unwrap_or(defaults.bits),
            mode: self.mode.clone().or_else(|| defaults.mode.clone()),
        }
    }
}

impl QuantizationConfig {
    pub fn resolve_for_tensor(&self, tensor_name: &str) -> ResolvedQuantizationConfig {
        let tensor_base = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
        let internal_name = normalize_qwen_name(tensor_name);
        let internal_base = normalize_qwen_name(tensor_base);
        self.tensor_overrides
            .get(tensor_name)
            .or_else(|| self.tensor_overrides.get(tensor_base))
            .or_else(|| self.tensor_overrides.get(&internal_name))
            .or_else(|| self.tensor_overrides.get(&internal_base))
            .map(|tensor_override| tensor_override.resolve_with_defaults(self))
            .unwrap_or_else(|| {
                ResolvedQuantizationConfig {
                    group_size: self.group_size,
                    bits: self.bits,
                    mode: self.mode.clone(),
                }
            })
    }

    pub fn normalize_tensor_overrides(&mut self) {
        if self.tensor_overrides.is_empty() {
            return;
        }

        let explicit_overrides = std::mem::take(&mut self.tensor_overrides);
        for (name, tensor_override) in &explicit_overrides {
            self.tensor_overrides.insert(name.clone(), tensor_override.clone());
        }
        for (name, tensor_override) in explicit_overrides {
            for alias in quant_override_aliases(&name) {
                self.tensor_overrides
                    .entry(alias)
                    .or_insert_with(|| tensor_override.clone());
            }
        }
    }
}

fn quant_override_aliases(name: &str) -> [String; 3] {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    [base.to_string(), normalize_qwen_name(name), normalize_qwen_name(base)]
}

pub fn normalize_qwen_name(name: &str) -> String {
    let mut normalized = name;
    for prefix in [
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ] {
        if let Some(stripped) = normalized.strip_prefix(prefix) {
            normalized = stripped;
            break;
        }
    }
    if let Some(suffix) = normalized.strip_prefix("lm_head") {
        return format!("unembed{suffix}");
    }
    normalized.to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen35ModelConfig {
    pub model_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub text_config: TextConfig,
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
pub struct TextConfig {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorPathLayout {
    pub container_prefix: &'static str,
    pub model_prefix: &'static str,
}

impl TensorPathLayout {
    pub fn model_path(&self, suffix: &str) -> String {
        format!("{}{}{}", self.container_prefix, self.model_prefix, suffix)
    }

    pub fn container_path(&self, suffix: &str) -> String {
        format!("{}{}", self.container_prefix, suffix)
    }
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

pub fn init_model_config(model_dir: impl AsRef<Path>) -> Result<Qwen35ModelConfig, ModelExecutorError> {
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
    let mut model_config = serde_json::from_value::<Qwen35ModelConfig>(envelope.clone()).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to parse qwen3.5 model config envelope from {:?}, err: {err:?}",
            model_config_path
        ))
    })?;
    model_config.quantization = model_config
        .quantization
        .or_else(|| parse_nested_quantization_config(&envelope));
    model_config.normalize()?;
    Ok(model_config)
}

pub fn parse_nested_quantization_config(envelope: &Value) -> Option<QuantizationConfig> {
    envelope
        .get("quantization_config")
        .cloned()
        .and_then(|value| serde_json::from_value::<QuantizationConfig>(value).ok())
        .map(|mut config| {
            config.normalize_tensor_overrides();
            config
        })
}

pub fn resolve_tensor_path_layout_from_names<'a>(tensor_names: impl IntoIterator<Item = &'a str>) -> TensorPathLayout {
    let names = tensor_names.into_iter().collect::<Vec<_>>();
    for layout in tensor_path_layout_candidates() {
        if names
            .iter()
            .any(|name| *name == layout.model_path("embed_tokens.weight"))
        {
            return layout;
        }
    }
    default_tensor_path_layout()
}

pub fn default_tensor_path_layout() -> TensorPathLayout {
    TensorPathLayout {
        container_prefix: "",
        model_prefix: "model.",
    }
}

pub fn tensor_path_layout_candidates() -> [TensorPathLayout; 5] {
    [
        TensorPathLayout {
            container_prefix: "",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "",
        },
        TensorPathLayout {
            container_prefix: "model.language_model.",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "model.language_model.",
            model_prefix: "",
        },
    ]
}

pub fn layer_type_at(config: &TextConfig, layer_index: usize) -> Result<LayerType, ModelExecutorError> {
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

pub fn layer_uses_moe(config: &TextConfig, layer_index: usize) -> bool {
    if config.num_experts == 0 {
        return false;
    }
    if config.decoder_sparse_step <= 1 {
        return true;
    }
    (layer_index + 1).is_multiple_of(config.decoder_sparse_step)
}

pub fn normalize_text_config(config: &mut TextConfig) -> Result<(), ModelExecutorError> {
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
