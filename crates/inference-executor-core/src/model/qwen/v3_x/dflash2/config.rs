use std::num::NonZeroUsize;
use std::path::Path;

use serde::Deserialize;
use serde_json::Map;
use serde_json::Value;

use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::QuantizationConfig;
use crate::model::qwen::v3_x::RopeParameters;

#[derive(Clone, Debug)]
pub struct Qwen3xDFlash2Config {
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub mask_token_id: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
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
    pub sliding_window: usize,
    pub vocab_size: usize,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3xDFlash2MainConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl Qwen3xDFlash2Config {
    pub fn num_spec_tokens(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.block_size - 1)
            .expect("validated Qwen3x DFlash2 block_size must contain an anchor and proposal token")
    }

    pub fn validate_main(&self, main: Qwen3xDFlash2MainConfig) -> Result<(), ModelExecutorError> {
        let mismatches = [
            (self.hidden_size != main.hidden_size)
                .then(|| format!("hidden_size dflash2={} main={}", self.hidden_size, main.hidden_size)),
            (self.num_target_layers != main.num_hidden_layers).then(|| {
                format!(
                    "num_target_layers dflash2={} main={}",
                    self.num_target_layers, main.num_hidden_layers
                )
            }),
            (self.vocab_size != main.vocab_size)
                .then(|| format!("vocab_size dflash2={} main={}", self.vocab_size, main.vocab_size)),
            (self.max_position_embeddings != main.max_position_embeddings).then(|| {
                format!(
                    "max_position_embeddings dflash2={} main={}",
                    self.max_position_embeddings, main.max_position_embeddings
                )
            }),
            (self.rope_theta != main.rope_theta)
                .then(|| format!("rope_theta dflash2={} main={}", self.rope_theta, main.rope_theta)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if !mismatches.is_empty() {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 config is incompatible with the Main model: {}",
                mismatches.join(", ")
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct CheckpointConfig {
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attention_dropout: f32,
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    #[serde(default)]
    dtype: Option<String>,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    #[serde(default)]
    is_causal: bool,
    layer_types: Vec<String>,
    mask_token_id: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    max_window_layers: Option<usize>,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    #[serde(default)]
    quantization: Option<QuantizationConfig>,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    selector_rank: usize,
    selector_top_k: usize,
    sliding_window: usize,
    target_layer_ids: Vec<usize>,
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

struct CheckpointConfigAdapter {
    architecture: &'static str,
    adapt: fn(&mut Map<String, Value>) -> Result<(), ModelExecutorError>,
}

const CANONICAL_CHECKPOINT_ARCHITECTURE: &str = "Qwen3DFlash2Model";
const CHECKPOINT_CONFIG_ADAPTERS: &[CheckpointConfigAdapter] = &[
    CheckpointConfigAdapter {
        architecture: CANONICAL_CHECKPOINT_ARCHITECTURE,
        adapt: adapt_canonical_checkpoint_config,
    },
    CheckpointConfigAdapter {
        architecture: "DFlash2DraftModel",
        adapt: adapt_dflash2_draft_checkpoint_config,
    },
];
const DFLASH2_DRAFT_CANONICAL_FIELDS: &[&str] = &[
    "block_size",
    "conv_group_size",
    "conv_kernel_size",
    "mask_token_id",
    "selector_rank",
    "selector_top_k",
    "target_layer_ids",
];

pub fn init_qwen3x_dflash2_config(model_dir: impl AsRef<Path>) -> Result<Qwen3xDFlash2Config, ModelExecutorError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&config_path).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to open Qwen3x DFlash2 config file {config_path:?}, error: {error:?}"
        ))
    })?;
    let value = serde_json::from_reader::<_, Value>(file).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to parse Qwen3x DFlash2 config file {config_path:?}, error: {error:?}"
        ))
    })?;
    let value = adapt_checkpoint_config(value).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to adapt Qwen3x DFlash2 config file {config_path:?}, error: {error}"
        ))
    })?;
    let checkpoint = serde_json::from_value::<CheckpointConfig>(value).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to parse canonical Qwen3x DFlash2 config from {config_path:?}, error: {error:?}"
        ))
    })?;
    normalize(checkpoint).map_err(|error| {
        ModelExecutorError::custom(format!(
            "unable to normalize Qwen3x DFlash2 config from {config_path:?}, error: {error}"
        ))
    })
}

fn adapt_checkpoint_config(mut value: Value) -> Result<Value, ModelExecutorError> {
    let config = value
        .as_object_mut()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 config must be a JSON object"))?;
    let architecture = checkpoint_architecture(config)?
        .unwrap_or(CANONICAL_CHECKPOINT_ARCHITECTURE)
        .to_owned();
    let adapter = CHECKPOINT_CONFIG_ADAPTERS
        .iter()
        .find(|adapter| adapter.architecture == architecture)
        .ok_or_else(|| {
            ModelExecutorError::custom(format!(
                "unsupported Qwen3x DFlash2 checkpoint architecture {architecture:?}; no config adapter is registered"
            ))
        })?;
    (adapter.adapt)(config)?;
    config.remove("architectures");
    Ok(value)
}

fn checkpoint_architecture(config: &Map<String, Value>) -> Result<Option<&str>, ModelExecutorError> {
    let Some(architectures) = config.get("architectures") else {
        return Ok(None);
    };
    let architectures = architectures
        .as_array()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 architectures must be a JSON array"))?;
    match architectures.as_slice() {
        [] => Ok(None),
        [architecture] => {
            architecture
                .as_str()
                .map(Some)
                .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 architecture must be a string"))
        },
        _ => {
            Err(ModelExecutorError::custom(
                "Qwen3x DFlash2 architectures must contain at most one entry",
            ))
        },
    }
}

fn adapt_canonical_checkpoint_config(config: &mut Map<String, Value>) -> Result<(), ModelExecutorError> {
    if config.contains_key("dflash_config") {
        return Err(ModelExecutorError::custom(
            "canonical Qwen3x DFlash2 checkpoint config must use flat fields",
        ));
    }
    Ok(())
}

fn adapt_dflash2_draft_checkpoint_config(config: &mut Map<String, Value>) -> Result<(), ModelExecutorError> {
    let nested = config
        .remove("dflash_config")
        .ok_or_else(|| ModelExecutorError::custom("DFlash2DraftModel config requires dflash_config"))?;
    let nested = nested
        .as_object()
        .ok_or_else(|| ModelExecutorError::custom("DFlash2DraftModel dflash_config must be a JSON object"))?;
    let unexpected = nested
        .keys()
        .filter(|name| !DFLASH2_DRAFT_CANONICAL_FIELDS.contains(&name.as_str()))
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(ModelExecutorError::custom(format!(
            "DFlash2DraftModel dflash_config contains unsupported fields {unexpected:?}"
        )));
    }
    for &name in DFLASH2_DRAFT_CANONICAL_FIELDS {
        let nested_value = nested.get(name).ok_or_else(|| {
            ModelExecutorError::custom(format!("DFlash2DraftModel dflash_config.{name} must be present"))
        })?;
        if let Some(flat_value) = config.get(name) {
            if flat_value != nested_value {
                return Err(ModelExecutorError::custom(format!(
                    "DFlash2DraftModel {name} and dflash_config.{name} must match"
                )));
            }
        } else {
            config.insert(name.to_owned(), nested_value.clone());
        }
    }
    Ok(())
}

fn normalize(mut config: CheckpointConfig) -> Result<Qwen3xDFlash2Config, ModelExecutorError> {
    validate_semantics(&config)?;
    for dtype in [config.dtype.as_deref(), config.torch_dtype.as_deref()]
        .into_iter()
        .flatten()
    {
        validate_dtype(dtype)?;
    }
    if let Some(quantization) = &mut config.quantization {
        validate_quantization(quantization)?;
        quantization.normalize_tensor_overrides();
    }
    let rope_theta = normalize_rope(&config)?;
    Ok(Qwen3xDFlash2Config {
        block_size: config.block_size,
        conv_group_size: config.conv_group_size,
        conv_kernel_size: config.conv_kernel_size,
        mask_token_id: config.mask_token_id,
        selector_rank: config.selector_rank,
        selector_top_k: config.selector_top_k,
        target_layer_ids: config.target_layer_ids,
        num_target_layers: config.num_target_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta,
        max_position_embeddings: config.max_position_embeddings,
        sliding_window: config.sliding_window,
        vocab_size: config.vocab_size,
        quantization: config.quantization,
    })
}

fn validate_semantics(config: &CheckpointConfig) -> Result<(), ModelExecutorError> {
    if config.model_type != "qwen3" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3x DFlash2 model_type {:?}; expected \"qwen3\"",
            config.model_type
        )));
    }
    if config.attention_bias || config.attention_dropout != 0.0 || !config.attention_dropout.is_finite() {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 requires attention_bias=false and attention_dropout=0",
        ));
    }
    if config.is_causal {
        return Err(ModelExecutorError::custom("Qwen3x DFlash2 requires is_causal=false"));
    }
    if config.hidden_act != "silu" {
        return Err(ModelExecutorError::custom(format!(
            "unsupported Qwen3x DFlash2 hidden_act {:?}; expected \"silu\"",
            config.hidden_act
        )));
    }
    if config.tie_word_embeddings || !config.use_cache || !config.use_sliding_window {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 requires tie_word_embeddings=false, use_cache=true, and use_sliding_window=true",
        ));
    }
    let dimensions = [
        ("block_size", config.block_size),
        ("conv_group_size", config.conv_group_size),
        ("conv_kernel_size", config.conv_kernel_size),
        ("selector_rank", config.selector_rank),
        ("selector_top_k", config.selector_top_k),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("num_target_layers", config.num_target_layers),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("sliding_window", config.sliding_window),
        ("vocab_size", config.vocab_size),
    ];
    if let Some((name, _)) = dimensions.into_iter().find(|&(_, value)| value == 0) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 {name} must be positive"
        )));
    }
    if config.block_size < 2 {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 block_size must contain an anchor and at least one proposal token",
        ));
    }
    if config.sliding_window <= config.block_size {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 sliding_window={} must exceed block_size={} so every query row has persistent history",
            config.sliding_window, config.block_size
        )));
    }
    if config.num_target_layers < 2 {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 num_target_layers must contain a captured layer and the unsupported final Main layer",
        ));
    }
    if !config.hidden_size.is_multiple_of(config.conv_group_size) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 conv_group_size={} must divide hidden_size={}",
            config.conv_group_size, config.hidden_size
        )));
    }
    let num_conv_groups = config.hidden_size / config.conv_group_size;
    let _conv_projection_dim = num_conv_groups
        .checked_mul(config.conv_kernel_size)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 convolution projection width must fit usize"))?;
    if config.head_dim < 2 || !config.head_dim.is_multiple_of(2) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 head_dim must be positive and even",
        ));
    }
    let _q_dim = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 query dimension must fit usize"))?;
    let _kv_dim = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 key/value dimension must fit usize"))?;
    if !config.num_attention_heads.is_multiple_of(config.num_key_value_heads) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 num_attention_heads must be divisible by num_key_value_heads",
        ));
    }
    if config.layer_types.len() != config.num_hidden_layers
        || config
            .layer_types
            .iter()
            .any(|layer_type| layer_type != "sliding_attention")
    {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 layer_types must contain one sliding_attention entry per layer",
        ));
    }
    if config
        .max_window_layers
        .is_some_and(|layers| layers != config.num_hidden_layers)
    {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 max_window_layers must equal num_hidden_layers when present",
        ));
    }
    if config.target_layer_ids.len() != config.num_hidden_layers {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 target_layer_ids length must equal num_hidden_layers",
        ));
    }
    if config.target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 target_layer_ids must be strictly increasing",
        ));
    }
    let final_main_layer = config.num_target_layers - 1;
    if config.target_layer_ids.iter().any(|&layer| layer >= final_main_layer) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 target_layer_ids must be below unsupported final Main layer {final_main_layer}"
        )));
    }
    if config.mask_token_id >= config.vocab_size || config.selector_top_k > config.vocab_size {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 mask_token_id and selector_top_k must fit the vocabulary",
        ));
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 rms_norm_eps must be finite and positive",
        ));
    }
    Ok(())
}

fn normalize_rope(config: &CheckpointConfig) -> Result<f32, ModelExecutorError> {
    let rope_theta = config
        .rope_parameters
        .rope_theta
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 rope_parameters.rope_theta must be present"))?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 rope_theta must be finite and positive",
        ));
    }
    if config.rope_parameters.rope_type.as_deref() != Some("default")
        || config.rope_parameters.partial_rotary_factor.unwrap_or(1.0) != 1.0
        || config.rope_parameters.factor.is_some()
        || config.rope_parameters.original_max_position_embeddings.is_some()
        || config.rope_parameters.attention_factor.is_some()
        || config.rope_parameters.beta_fast.is_some()
        || config.rope_parameters.beta_slow.is_some()
        || config.rope_parameters.mscale.is_some()
        || config.rope_parameters.mscale_all_dim.is_some()
        || config.rope_parameters.truncate.is_some()
    {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 requires full-dimension default RoPE without scaling parameters",
        ));
    }
    Ok(rope_theta)
}

fn validate_dtype(dtype: &str) -> Result<(), ModelExecutorError> {
    match dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(()),
        _ => {
            Err(ModelExecutorError::custom(format!(
                "unsupported Qwen3x DFlash2 model IO dtype {dtype:?}; expected bfloat16"
            )))
        },
    }
}

fn validate_quantization(config: &QuantizationConfig) -> Result<(), ModelExecutorError> {
    if !matches!(config.group_size, 32 | 64 | 128) || !matches!(config.bits, 2 | 3 | 4 | 6 | 8) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 affine quantization requires group_size 32, 64, or 128 and bits 2, 3, 4, 6, or 8",
        ));
    }
    if !matches!(config.mode.as_deref(), None | Some("affine")) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 quantization mode must be affine",
        ));
    }
    for (name, tensor) in &config.tensor_overrides {
        if tensor
            .group_size
            .is_some_and(|group_size| !matches!(group_size, 32 | 64 | 128))
            || tensor.bits.is_some_and(|bits| !matches!(bits, 2 | 3 | 4 | 6 | 8))
            || tensor.mode.as_deref().is_some_and(|mode| mode != "affine")
        {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 quantization override {name:?} is not a supported affine layout"
            )));
        }
    }
    Ok(())
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_adapter_maps_the_published_dflash2_contract() {
        let value = serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]
            },
            "attention_bias": false,
            "attention_dropout": 0.0,
            "is_causal": false,
            "dtype": "bfloat16",
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "max_position_embeddings": 262144,
            "max_window_layers": 5,
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 64,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 10000000.0, "rope_type": "default" },
            "sliding_window": 2048,
            "tie_word_embeddings": false,
            "use_cache": true,
            "use_sliding_window": true,
            "vocab_size": 248320
        });

        let config = normalize(serde_json::from_value(adapt_checkpoint_config(value).unwrap()).unwrap()).unwrap();

        assert_eq!(config.block_size, 8);
        assert_eq!(config.conv_group_size, 16);
        assert_eq!(config.selector_top_k, 16);
        assert_eq!(config.target_layer_ids, [5, 19, 33, 47, 61]);
        assert_eq!(config.sliding_window, 2048);
        assert_eq!(config.num_spec_tokens().get(), 7);
    }

    #[test]
    fn test_main_validation_reports_all_mismatches() {
        let config = Qwen3xDFlash2Config {
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            mask_token_id: 63,
            selector_rank: 32,
            selector_top_k: 16,
            target_layer_ids: vec![1],
            num_target_layers: 4,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
            sliding_window: 64,
            vocab_size: 64,
            quantization: None,
        };

        let error = config
            .validate_main(Qwen3xDFlash2MainConfig {
                hidden_size: 32,
                num_hidden_layers: 5,
                vocab_size: 65,
                max_position_embeddings: 256,
                rope_theta: 1_000.0,
            })
            .unwrap_err();

        let message = error.to_string();
        for field in [
            "hidden_size",
            "num_target_layers",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
        ] {
            assert!(message.contains(field));
        }
    }
}
