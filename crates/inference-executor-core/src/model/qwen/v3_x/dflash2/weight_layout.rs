use std::collections::HashSet;

use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use crate::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use crate::model::qwen::v3_x::weight_layout::dense_mlp_bindings;
use crate::model::qwen::v3_x::weight_layout::push_quantized_tensor_names;
use crate::model::qwen::v3_x::weight_layout::quantized;
use crate::model::qwen::v3_x::weight_layout::quantized_path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2WeightBindings {
    pub main_feature: Qwen3xDFlash2MainFeatureWeightBindings,
    pub layers: Vec<Qwen3xDFlash2LayerWeightBindings>,
    pub final_norm_weight: String,
    pub selector: Qwen3xDFlash2SelectorWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2MainFeatureWeightBindings {
    pub fc: QuantizedTensorBindings,
    pub hidden_norm_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2LayerWeightBindings {
    pub input_norm_weight: String,
    pub attention_conv: Qwen3xDFlash2ConvWeightBindings,
    pub gqa: Qwen3xGQAWeightBindings,
    pub post_attention_norm_weight: String,
    pub mlp_conv: Qwen3xDFlash2ConvWeightBindings,
    pub mlp: Qwen3xDenseMLPWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2ConvWeightBindings {
    pub base_kernel: String,
    pub kernel_projection: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2SelectorWeightBindings {
    pub hidden_projection: QuantizedTensorBindings,
    pub predecessor_codebook: QuantizedTensorBindings,
    pub successor_codebook: QuantizedTensorBindings,
}

impl Qwen3xDFlash2WeightBindings {
    pub fn from_config(config: &Qwen3xDFlash2Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_index}");
            let attention_prefix = format!("{prefix}.self_attn");
            layers.push(Qwen3xDFlash2LayerWeightBindings {
                input_norm_weight: format!("{prefix}.input_layernorm.weight"),
                attention_conv: conv_bindings(&format!("{prefix}.attention_conv")),
                gqa: Qwen3xGQAWeightBindings {
                    q: quantized(&attention_prefix, "q_proj"),
                    k: quantized(&attention_prefix, "k_proj"),
                    v: quantized(&attention_prefix, "v_proj"),
                    q_norm_weight: format!("{attention_prefix}.q_norm.weight"),
                    k_norm_weight: format!("{attention_prefix}.k_norm.weight"),
                    output: quantized(&attention_prefix, "o_proj"),
                },
                post_attention_norm_weight: format!("{prefix}.post_attention_layernorm.weight"),
                mlp_conv: conv_bindings(&format!("{prefix}.mlp_conv")),
                mlp: dense_mlp_bindings(&format!("{prefix}.mlp")),
            });
        }
        Self {
            main_feature: Qwen3xDFlash2MainFeatureWeightBindings {
                fc: quantized_path("fc".to_string()),
                hidden_norm_weight: "hidden_norm.weight".to_string(),
            },
            layers,
            final_norm_weight: "norm.weight".to_string(),
            selector: Qwen3xDFlash2SelectorWeightBindings {
                hidden_projection: quantized_path("candidate_selector.hidden_projection".to_string()),
                predecessor_codebook: quantized_path("candidate_selector.predecessor_codebook".to_string()),
                successor_codebook: quantized_path("candidate_selector.successor_codebook".to_string()),
            },
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        self.main_feature.push_tensor_names(&mut names);
        for layer in &self.layers {
            layer.push_tensor_names(&mut names);
        }
        names.push(&self.final_norm_weight);
        self.selector.push_tensor_names(&mut names);
        names
    }

    pub fn source_tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        names.extend([
            self.main_feature.fc.weight.as_str(),
            self.main_feature.hidden_norm_weight.as_str(),
        ]);
        for layer in &self.layers {
            layer.push_source_tensor_names(&mut names);
        }
        names.push(&self.final_norm_weight);
        names.extend([
            self.selector.hidden_projection.weight.as_str(),
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        ]);
        names
    }
}

impl Qwen3xDFlash2MainFeatureWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.fc, names);
        names.push(&self.hidden_norm_weight);
    }
}

impl Qwen3xDFlash2LayerWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.push(&self.input_norm_weight);
        self.attention_conv.push_tensor_names(names);
        self.gqa.push_tensor_names(names);
        names.push(&self.post_attention_norm_weight);
        self.mlp_conv.push_tensor_names(names);
        self.mlp.push_tensor_names(names);
    }

    fn push_source_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([
            self.input_norm_weight.as_str(),
            self.attention_conv.base_kernel.as_str(),
            self.attention_conv.kernel_projection.weight.as_str(),
            self.gqa.q.weight.as_str(),
            self.gqa.k.weight.as_str(),
            self.gqa.v.weight.as_str(),
            self.gqa.q_norm_weight.as_str(),
            self.gqa.k_norm_weight.as_str(),
            self.gqa.output.weight.as_str(),
            self.post_attention_norm_weight.as_str(),
            self.mlp_conv.base_kernel.as_str(),
            self.mlp_conv.kernel_projection.weight.as_str(),
            self.mlp.gate.weight.as_str(),
            self.mlp.up.weight.as_str(),
            self.mlp.down.weight.as_str(),
        ]);
    }
}

impl Qwen3xDFlash2ConvWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.push(&self.base_kernel);
        push_quantized_tensor_names(&self.kernel_projection, names);
    }
}

impl Qwen3xDFlash2SelectorWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.hidden_projection, names);
        push_quantized_tensor_names(&self.predecessor_codebook, names);
        push_quantized_tensor_names(&self.successor_codebook, names);
    }
}

pub fn resolve_qwen3x_dflash2_weight_bindings<'a>(
    config: &Qwen3xDFlash2Config,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3xDFlash2WeightBindings, ModelExecutorError> {
    resolve_exact_layout(config, tensor_names, false)
}

pub fn resolve_qwen3x_dflash2_source_weight_bindings<'a>(
    config: &Qwen3xDFlash2Config,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3xDFlash2WeightBindings, ModelExecutorError> {
    resolve_exact_layout(config, tensor_names, true)
}

fn resolve_exact_layout<'a>(
    config: &Qwen3xDFlash2Config,
    tensor_names: impl IntoIterator<Item = &'a str>,
    source: bool,
) -> Result<Qwen3xDFlash2WeightBindings, ModelExecutorError> {
    let actual = tensor_names.into_iter().collect::<HashSet<_>>();
    if actual.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }
    let bindings = Qwen3xDFlash2WeightBindings::from_config(config);
    let expected_names = if source {
        bindings.source_tensor_names()
    } else {
        bindings.tensor_names()
    };
    let expected = expected_names.iter().copied().collect::<HashSet<_>>();
    assert_eq!(
        expected.len(),
        expected_names.len(),
        "Qwen3x DFlash2 binding tree must not contain duplicate tensor names"
    );
    let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let mut unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    unexpected.sort_unstable();
    if !missing.is_empty() || !unexpected.is_empty() {
        let kind = if source { "source" } else { "affine" };
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 {kind} checkpoint must match the exact tensor layout; missing={missing:?}, \
             unexpected={unexpected:?}"
        )));
    }
    Ok(bindings)
}

fn conv_bindings(prefix: &str) -> Qwen3xDFlash2ConvWeightBindings {
    Qwen3xDFlash2ConvWeightBindings {
        base_kernel: format!("{prefix}.base_kernel"),
        kernel_projection: quantized(prefix, "kernel_projection"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_tree_matches_published_semantic_groups() {
        let config = fixture_config();
        let bindings = Qwen3xDFlash2WeightBindings::from_config(&config);

        assert_eq!(bindings.layers.len(), 2);
        assert_eq!(
            bindings.layers[1].attention_conv.base_kernel,
            "layers.1.attention_conv.base_kernel"
        );
        assert_eq!(
            bindings.layers[1].mlp_conv.kernel_projection.weight,
            "layers.1.mlp_conv.kernel_projection.weight"
        );
        assert_eq!(
            bindings.selector.predecessor_codebook.weight,
            "candidate_selector.predecessor_codebook.weight"
        );
        assert_eq!(bindings.source_tensor_names().len(), 36);
        assert_eq!(bindings.tensor_names().len(), 80);
        resolve_qwen3x_dflash2_source_weight_bindings(&config, bindings.source_tensor_names()).unwrap();
        resolve_qwen3x_dflash2_weight_bindings(&config, bindings.tensor_names()).unwrap();
    }

    fn fixture_config() -> Qwen3xDFlash2Config {
        Qwen3xDFlash2Config {
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            mask_token_id: 63,
            selector_rank: 32,
            selector_top_k: 16,
            target_layer_ids: vec![1, 3],
            num_target_layers: 5,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
            sliding_window: 64,
            vocab_size: 64,
            quantization: None,
        }
    }
}
