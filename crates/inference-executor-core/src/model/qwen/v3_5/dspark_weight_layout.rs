use std::collections::HashSet;

use super::DSparkConfig;
use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkWeightBindings {
    pub target_projection: Qwen35DSparkTargetWeightBindings,
    pub layers: Vec<Qwen35DSparkLayerWeightBindings>,
    pub final_norm_weight: String,
    pub markov: Qwen35DSparkMarkovWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkTargetWeightBindings {
    pub fc: QuantizedTensorBindings,
    pub hidden_norm_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkLayerWeightBindings {
    pub input_norm_weight: String,
    pub post_attention_norm_weight: String,
    pub attention: Qwen35DSparkAttentionWeightBindings,
    pub mlp: Qwen35DSparkMLPWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkAttentionWeightBindings {
    pub q: QuantizedTensorBindings,
    pub k: QuantizedTensorBindings,
    pub v: QuantizedTensorBindings,
    pub q_norm_weight: String,
    pub k_norm_weight: String,
    pub output: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkMLPWeightBindings {
    pub gate: QuantizedTensorBindings,
    pub up: QuantizedTensorBindings,
    pub down: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkMarkovWeightBindings {
    pub w1: QuantizedTensorBindings,
    pub w2: QuantizedTensorBindings,
}

impl Qwen35DSparkWeightBindings {
    pub fn from_config(config: &DSparkConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_index in 0..config.num_layers {
            let prefix = format!("layers.{layer_index}");
            let attention_prefix = format!("{prefix}.self_attn");
            let mlp_prefix = format!("{prefix}.mlp");
            layers.push(Qwen35DSparkLayerWeightBindings {
                input_norm_weight: format!("{prefix}.input_layernorm.weight"),
                post_attention_norm_weight: format!("{prefix}.post_attention_layernorm.weight"),
                attention: Qwen35DSparkAttentionWeightBindings {
                    q: quantized(&attention_prefix, "q_proj"),
                    k: quantized(&attention_prefix, "k_proj"),
                    v: quantized(&attention_prefix, "v_proj"),
                    q_norm_weight: format!("{attention_prefix}.q_norm.weight"),
                    k_norm_weight: format!("{attention_prefix}.k_norm.weight"),
                    output: quantized(&attention_prefix, "o_proj"),
                },
                mlp: Qwen35DSparkMLPWeightBindings {
                    gate: quantized(&mlp_prefix, "gate_proj"),
                    up: quantized(&mlp_prefix, "up_proj"),
                    down: quantized(&mlp_prefix, "down_proj"),
                },
            });
        }
        Self {
            target_projection: Qwen35DSparkTargetWeightBindings {
                fc: quantized_path("fc"),
                hidden_norm_weight: "hidden_norm.weight".to_string(),
            },
            layers,
            final_norm_weight: "norm.weight".to_string(),
            markov: Qwen35DSparkMarkovWeightBindings {
                w1: quantized_path("markov_head.markov_w1"),
                w2: quantized_path("markov_head.markov_w2"),
            },
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        push_quantized_names(&self.target_projection.fc, &mut names);
        names.push(&self.target_projection.hidden_norm_weight);
        for layer in &self.layers {
            layer.push_tensor_names(&mut names);
        }
        names.push(&self.final_norm_weight);
        push_quantized_names(&self.markov.w1, &mut names);
        push_quantized_names(&self.markov.w2, &mut names);
        names
    }

    pub fn source_tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        names.push(self.target_projection.fc.weight.as_str());
        names.push(&self.target_projection.hidden_norm_weight);
        for layer in &self.layers {
            layer.push_source_tensor_names(&mut names);
        }
        names.push(&self.final_norm_weight);
        names.push(self.markov.w1.weight.as_str());
        names.push(self.markov.w2.weight.as_str());
        names
    }
}

impl Qwen35DSparkLayerWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.input_norm_weight.as_str(),
            self.post_attention_norm_weight.as_str(),
        ]);
        self.attention.push_tensor_names(names);
        self.mlp.push_tensor_names(names);
    }

    fn push_source_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.input_norm_weight.as_str(),
            self.post_attention_norm_weight.as_str(),
        ]);
        self.attention.push_source_tensor_names(names);
        self.mlp.push_source_tensor_names(names);
    }
}

impl Qwen35DSparkAttentionWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_names(&self.q, names);
        push_quantized_names(&self.k, names);
        push_quantized_names(&self.v, names);
        names.extend([self.q_norm_weight.as_str(), self.k_norm_weight.as_str()]);
        push_quantized_names(&self.output, names);
    }

    fn push_source_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.q.weight.as_str(),
            self.k.weight.as_str(),
            self.v.weight.as_str(),
            self.q_norm_weight.as_str(),
            self.k_norm_weight.as_str(),
            self.output.weight.as_str(),
        ]);
    }
}

impl Qwen35DSparkMLPWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_names(&self.gate, names);
        push_quantized_names(&self.up, names);
        push_quantized_names(&self.down, names);
    }

    fn push_source_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.gate.weight.as_str(),
            self.up.weight.as_str(),
            self.down.weight.as_str(),
        ]);
    }
}

pub fn resolve_qwen35_dspark_weight_bindings<'a>(
    config: &DSparkConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen35DSparkWeightBindings, ModelExecutorError> {
    let bindings = Qwen35DSparkWeightBindings::from_config(config);
    let actual = tensor_names.into_iter().collect::<HashSet<_>>();
    let expected_names = bindings.tensor_names();
    let expected = expected_names.iter().copied().collect::<HashSet<_>>();
    let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let mut unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    unexpected.sort_unstable();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModelExecutorError::custom(format!(
            "DSpark checkpoint must match the exact Hikari affine tensor layout; missing={missing:?}, \
             unexpected={unexpected:?}"
        )));
    }
    Ok(bindings)
}

fn quantized(prefix: &str, relative_name: &str) -> QuantizedTensorBindings {
    quantized_path(&format!("{prefix}.{relative_name}"))
}

fn quantized_path(prefix: &str) -> QuantizedTensorBindings {
    QuantizedTensorBindings {
        weight: format!("{prefix}.weight"),
        scales: format!("{prefix}.scales"),
        biases: format!("{prefix}.biases"),
    }
}

fn push_quantized_names<'a>(bindings: &'a QuantizedTensorBindings, names: &mut Vec<&'a str>) {
    names.extend([
        bindings.weight.as_str(),
        bindings.scales.as_str(),
        bindings.biases.as_str(),
    ]);
}

#[cfg(test)]
#[path = "dspark_weight_layout_tests.rs"]
mod tests;
