use std::collections::HashSet;

use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use crate::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use crate::model::qwen::v3_x::weight_layout::dense_mlp_bindings;
use crate::model::qwen::v3_x::weight_layout::push_quantized_tensor_names;
use crate::model::qwen::v3_x::weight_layout::quantized;
use crate::model::qwen::v3_x::weight_layout::quantized_path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkWeightBindings {
    pub embed: Option<QuantizedTensorBindings>,
    pub main_feature: Qwen3xDSparkMainFeatureWeightBindings,
    pub layers: Vec<Qwen3xDSparkLayerWeightBindings>,
    pub final_norm_weight: String,
    pub unembed: Option<QuantizedTensorBindings>,
    pub markov: Qwen3xDSparkMarkovWeightBindings,
    pub confidence: Option<Qwen3xDSparkConfidenceWeightBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkMainFeatureWeightBindings {
    pub fc: QuantizedTensorBindings,
    pub hidden_norm_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkLayerWeightBindings {
    pub input_norm_weight: String,
    pub post_attention_norm_weight: String,
    pub gqa: Qwen3xGQAWeightBindings,
    pub mlp: Qwen3xDenseMLPWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkMarkovWeightBindings {
    pub w1: QuantizedTensorBindings,
    pub w2: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkConfidenceWeightBindings {
    pub weight: String,
    pub bias: String,
}

impl Qwen3xDSparkWeightBindings {
    pub fn from_config(config: &Qwen3xDSparkConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let layer_prefix = format!("layers.{layer_index}");
            let attention_prefix = format!("{layer_prefix}.self_attn");
            layers.push(Qwen3xDSparkLayerWeightBindings {
                input_norm_weight: format!("{layer_prefix}.input_layernorm.weight"),
                post_attention_norm_weight: format!("{layer_prefix}.post_attention_layernorm.weight"),
                gqa: Qwen3xGQAWeightBindings {
                    q: quantized(&attention_prefix, "q_proj"),
                    k: quantized(&attention_prefix, "k_proj"),
                    v: quantized(&attention_prefix, "v_proj"),
                    q_norm_weight: format!("{attention_prefix}.q_norm.weight"),
                    k_norm_weight: format!("{attention_prefix}.k_norm.weight"),
                    output: quantized(&attention_prefix, "o_proj"),
                },
                mlp: dense_mlp_bindings(&format!("{layer_prefix}.mlp")),
            });
        }
        Self {
            embed: None,
            main_feature: Qwen3xDSparkMainFeatureWeightBindings {
                fc: quantized_path("fc".to_string()),
                hidden_norm_weight: "hidden_norm.weight".to_string(),
            },
            layers,
            final_norm_weight: "norm.weight".to_string(),
            unembed: None,
            markov: Qwen3xDSparkMarkovWeightBindings {
                w1: quantized_path("markov_head.markov_w1".to_string()),
                w2: quantized_path("markov_head.markov_w2".to_string()),
            },
            confidence: None,
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        if let Some(embed) = &self.embed {
            push_quantized_tensor_names(embed, &mut names);
        }
        push_quantized_tensor_names(&self.main_feature.fc, &mut names);
        names.push(self.main_feature.hidden_norm_weight.as_str());
        for layer in &self.layers {
            layer.push_tensor_names(&mut names);
        }
        names.push(self.final_norm_weight.as_str());
        if let Some(unembed) = &self.unembed {
            push_quantized_tensor_names(unembed, &mut names);
        }
        push_quantized_tensor_names(&self.markov.w1, &mut names);
        push_quantized_tensor_names(&self.markov.w2, &mut names);
        if let Some(confidence) = &self.confidence {
            names.extend([confidence.weight.as_str(), confidence.bias.as_str()]);
        }
        names
    }

    pub fn source_tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        if let Some(embed) = &self.embed {
            names.push(embed.weight.as_str());
        }
        names.extend([
            self.main_feature.fc.weight.as_str(),
            self.main_feature.hidden_norm_weight.as_str(),
        ]);
        for layer in &self.layers {
            layer.push_source_tensor_names(&mut names);
        }
        names.push(self.final_norm_weight.as_str());
        if let Some(unembed) = &self.unembed {
            names.push(unembed.weight.as_str());
        }
        names.extend([self.markov.w1.weight.as_str(), self.markov.w2.weight.as_str()]);
        if let Some(confidence) = &self.confidence {
            names.extend([confidence.weight.as_str(), confidence.bias.as_str()]);
        }
        names
    }
}

impl Qwen3xDSparkLayerWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.input_norm_weight.as_str(),
            self.post_attention_norm_weight.as_str(),
        ]);
        self.gqa.push_tensor_names(names);
        self.mlp.push_tensor_names(names);
    }

    fn push_source_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.input_norm_weight.as_str(),
            self.post_attention_norm_weight.as_str(),
            self.gqa.q.weight.as_str(),
            self.gqa.k.weight.as_str(),
            self.gqa.v.weight.as_str(),
            self.gqa.q_norm_weight.as_str(),
            self.gqa.k_norm_weight.as_str(),
            self.gqa.output.weight.as_str(),
            self.mlp.gate.weight.as_str(),
            self.mlp.up.weight.as_str(),
            self.mlp.down.weight.as_str(),
        ]);
    }
}

pub fn resolve_qwen3x_dspark_weight_bindings<'a>(
    config: &Qwen3xDSparkConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3xDSparkWeightBindings, ModelExecutorError> {
    let actual = tensor_names.into_iter().collect::<HashSet<_>>();
    if actual.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }

    let mut bindings = Qwen3xDSparkWeightBindings::from_config(config);
    bindings.embed = resolve_optional_quantized_group(&actual, "embed_tokens")?;
    bindings.unembed = resolve_optional_quantized_group(&actual, "lm_head")?;

    let expected_names = bindings.tensor_names();
    let expected = expected_names.iter().copied().collect::<HashSet<_>>();
    let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let mut unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    unexpected.sort_unstable();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark checkpoint must match the exact affine tensor layout; missing={missing:?}, \
             unexpected={unexpected:?}"
        )));
    }
    Ok(bindings)
}

pub fn resolve_qwen3x_dspark_source_weight_bindings<'a>(
    config: &Qwen3xDSparkConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3xDSparkWeightBindings, ModelExecutorError> {
    let actual = tensor_names.into_iter().collect::<HashSet<_>>();
    if actual.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark source checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }
    let mut bindings = Qwen3xDSparkWeightBindings::from_config(config);
    bindings.embed = actual
        .contains("embed_tokens.weight")
        .then(|| quantized_path("embed_tokens".to_string()));
    bindings.unembed = actual
        .contains("lm_head.weight")
        .then(|| quantized_path("lm_head".to_string()));
    let has_confidence_weight = actual.contains("confidence_head.proj.weight");
    let has_confidence_bias = actual.contains("confidence_head.proj.bias");
    if has_confidence_weight != has_confidence_bias {
        return Err(ModelExecutorError::custom(
            "Qwen3 DSpark source confidence head must contain both weight and bias",
        ));
    }
    bindings.confidence = has_confidence_weight.then(|| {
        Qwen3xDSparkConfidenceWeightBindings {
            weight: "confidence_head.proj.weight".to_string(),
            bias: "confidence_head.proj.bias".to_string(),
        }
    });

    let expected_names = bindings.source_tensor_names();
    let expected = expected_names.iter().copied().collect::<HashSet<_>>();
    let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let mut unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    unexpected.sort_unstable();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3 DSpark source checkpoint must match the official tensor layout; missing={missing:?}, \
             unexpected={unexpected:?}"
        )));
    }
    Ok(bindings)
}

fn resolve_optional_quantized_group(
    actual: &HashSet<&str>,
    prefix: &str,
) -> Result<Option<QuantizedTensorBindings>, ModelExecutorError> {
    let bindings = quantized_path(prefix.to_string());
    let names = [
        bindings.weight.as_str(),
        bindings.scales.as_str(),
        bindings.biases.as_str(),
    ];
    let count = names.iter().filter(|name| actual.contains(**name)).count();
    match count {
        0 => Ok(None),
        3 => Ok(Some(bindings)),
        _ => {
            Err(ModelExecutorError::custom(format!(
                "Qwen3 DSpark optional affine tensor group {prefix:?} must be absent or complete"
            )))
        },
    }
}

#[cfg(test)]
#[path = "weight_layout_tests.rs"]
mod tests;
