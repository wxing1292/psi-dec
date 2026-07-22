use std::path::Path;

use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::config::SamplingConfig;
use serde::Deserialize;
use serde::Serialize;

use crate::def::ModelExecutorError;

const DEFAULT_SAMPLER_SEED: u32 = 42;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct SamplerConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_seed")]
    pub seed: u32,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct HFGenerationConfig {
    #[serde(flatten)]
    sampler: SamplerConfig,
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    One(u32),
    Many(Vec<u32>),
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_k: default_top_k(),
            top_p: default_top_p(),
            seed: default_seed(),
        }
    }
}

impl SamplerConfig {
    pub fn from_runtime(config: &SamplingConfig, seed: u32) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            seed,
        }
    }

    pub fn load(model_dir: impl AsRef<Path>) -> Result<SamplerConfig, ModelExecutorError> {
        Ok(HFGenerationConfig::load(model_dir)?.sampler)
    }

    pub fn validate(&self) -> Result<(), ModelExecutorError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ModelExecutorError::custom(format!(
                "sampler temperature must be finite and non-negative, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(ModelExecutorError::custom(format!(
                "sampler top_p must be finite and in [0, 1], got {}",
                self.top_p
            )));
        }
        Ok(())
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1 || self.top_p == 0.0
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }
}

impl HFGenerationConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ModelExecutorError> {
        let generation_config_path = model_dir.as_ref().join("generation_config.json");
        match std::fs::File::open(&generation_config_path) {
            Ok(file) => {
                let config = serde_json::from_reader::<_, Self>(file).map_err(|err| {
                    ModelExecutorError::custom(format!(
                        "unable to parse sampling config file {:?}, err: {err:?}",
                        generation_config_path
                    ))
                })?;
                config.sampler.validate()?;
                Ok(config)
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => {
                Err(ModelExecutorError::custom(format!(
                    "unable to open generation config file {:?}, err: {err}",
                    generation_config_path
                )))
            },
        }
    }

    pub fn sampler(&self) -> &SamplerConfig {
        &self.sampler
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        match &self.eos_token_id {
            None => &[],
            Some(TokenIdOrIds::One(token_id)) => std::slice::from_ref(token_id),
            Some(TokenIdOrIds::Many(token_ids)) => token_ids,
        }
    }
}

fn default_temperature() -> f32 {
    DEFAULT_SAMPLING_TEMPERATURE
}

fn default_top_k() -> usize {
    DEFAULT_SAMPLING_TOP_K
}

fn default_top_p() -> f32 {
    DEFAULT_SAMPLING_TOP_P
}

fn default_seed() -> u32 {
    DEFAULT_SAMPLER_SEED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let single = serde_json::from_str::<HFGenerationConfig>(r#"{"eos_token_id": 7}"#).unwrap();
        let multiple = serde_json::from_str::<HFGenerationConfig>(r#"{"eos_token_id": [8, 9]}"#).unwrap();

        assert_eq!(single.eos_token_ids(), &[7]);
        assert_eq!(multiple.eos_token_ids(), &[8, 9]);
        assert_eq!(single.sampler(), &SamplerConfig::default());
    }
}
