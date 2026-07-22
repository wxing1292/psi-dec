use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;
use serde::Serialize;

use super::QuantizationConfig;
use super::TextConfig;
use crate::def::ModelExecutorError;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DSparkConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub block_size: usize,
    pub dflash_config: DSparkDFlashConfig,
    #[serde(default)]
    pub dtype: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_target_layers: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DSparkDFlashConfig {
    pub causal_head: bool,
    pub causal: bool,
    pub mask_token_id: usize,
    #[serde(rename = "target_layer_ids")]
    pub target_residual_layer_indices: Vec<usize>,
}

impl DSparkConfig {
    pub fn normalize_and_validate(&mut self) -> Result<(), ModelExecutorError> {
        if let Some(quantization) = &mut self.quantization {
            quantization.normalize_tensor_overrides();
        }
        if self.model_type != "qwen3" {
            return Err(ModelExecutorError::custom(format!(
                "unsupported DSpark model_type {:?}; expected qwen3",
                self.model_type
            )));
        }
        if self.block_size < 2 {
            return Err(ModelExecutorError::custom("DSpark block_size must be at least 2"));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_target_layers == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.markov_rank == 0
        {
            return Err(ModelExecutorError::custom(format!(
                "invalid zero DSpark dimension: hidden={} intermediate={} layers={} q_heads={} kv_heads={} \
                 target_layers={} head_dim={} vocab={} max_positions={} markov_rank={}",
                self.hidden_size,
                self.intermediate_size,
                self.num_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.num_target_layers,
                self.head_dim,
                self.vocab_size,
                self.max_position_embeddings,
                self.markov_rank
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ModelExecutorError::custom(format!(
                "DSpark rms_norm_eps={} must be finite and positive",
                self.rms_norm_eps
            )));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(ModelExecutorError::custom(format!(
                "DSpark rope_theta={} must be finite and positive",
                self.rope_theta
            )));
        }
        if self.dflash_config.causal || self.dflash_config.causal_head {
            return Err(ModelExecutorError::custom(
                "causal DSpark block/head checkpoints are not supported",
            ));
        }
        if self.dflash_config.mask_token_id >= self.vocab_size {
            return Err(ModelExecutorError::custom(format!(
                "DSpark mask_token_id={} must be below vocab_size={}",
                self.dflash_config.mask_token_id, self.vocab_size
            )));
        }
        if self.dflash_config.target_residual_layer_indices.is_empty() {
            return Err(ModelExecutorError::custom(
                "DSpark target_residual_layer_indices (dflash_config.target_layer_ids) must not be empty",
            ));
        }
        let unique_target_layers = self
            .dflash_config
            .target_residual_layer_indices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        if unique_target_layers.len() != self.dflash_config.target_residual_layer_indices.len() {
            return Err(ModelExecutorError::custom(format!(
                "DSpark target_residual_layer_indices (dflash_config.target_layer_ids) must be unique: {:?}",
                self.dflash_config.target_residual_layer_indices
            )));
        }
        if !self.layer_types.is_empty()
            && (self.layer_types.len() != self.num_layers
                || self.layer_types.iter().any(|layer_type| layer_type != "full_attention"))
        {
            return Err(ModelExecutorError::custom(format!(
                "DSpark layer_types must contain one full_attention entry per draft layer: {:?}",
                self.layer_types
            )));
        }
        if self.markov_head_type != "vanilla" {
            return Err(ModelExecutorError::custom(format!(
                "unsupported DSpark markov_head_type {:?}; expected vanilla",
                self.markov_head_type
            )));
        }
        Ok(())
    }

    pub fn validate_target(&self, target: &TextConfig) -> Result<(), ModelExecutorError> {
        let target_kv_elements = target
            .num_key_value_heads
            .checked_mul(target.head_dim)
            .ok_or_else(|| ModelExecutorError::custom("target GQA K/V token size must fit usize"))?;
        let dspark_kv_elements = self
            .num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| ModelExecutorError::custom("DSpark context K/V token size must fit usize"))?;
        let dimensions_match = self.hidden_size == target.hidden_size
            && self.vocab_size == target.vocab_size
            && self.num_target_layers == target.num_hidden_layers
            && target.max_position_embeddings >= self.max_position_embeddings
            && target_kv_elements == dspark_kv_elements;
        if !dimensions_match {
            return Err(ModelExecutorError::custom(format!(
                "DSpark/target dimensions must match: target hidden={} vocab={} layers={} max_positions={} \
                 kv_elements={} DSpark hidden={} vocab={} target_layers={} max_positions={} kv_elements={}",
                target.hidden_size,
                target.vocab_size,
                target.num_hidden_layers,
                target.max_position_embeddings,
                target_kv_elements,
                self.hidden_size,
                self.vocab_size,
                self.num_target_layers,
                self.max_position_embeddings,
                dspark_kv_elements
            )));
        }
        if let Some(&layer_id) = self
            .dflash_config
            .target_residual_layer_indices
            .iter()
            .find(|&&layer_id| layer_id >= target.num_hidden_layers)
        {
            return Err(ModelExecutorError::custom(format!(
                "DSpark target layer {layer_id} is outside target num_hidden_layers={}",
                target.num_hidden_layers
            )));
        }
        Ok(())
    }
}

pub fn init_dspark_config(model_dir: impl AsRef<Path>) -> Result<DSparkConfig, ModelExecutorError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(&config_path).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to open DSpark config file {config_path:?}, err: {err:?}"
        ))
    })?;
    let mut config = serde_json::from_reader::<_, DSparkConfig>(file).map_err(|err| {
        ModelExecutorError::custom(format!(
            "unable to parse DSpark config file {config_path:?}, err: {err:?}"
        ))
    })?;
    config.normalize_and_validate()?;
    Ok(config)
}

#[cfg(test)]
#[path = "dspark_config_tests.rs"]
mod tests;
