use std::collections::HashSet;

use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3::Qwen3ModelConfig;
use crate::model::qwen::v3_x::TensorPathLayout;
use crate::model::qwen::v3_x::tensor_path_layout_candidates;
use crate::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use crate::model::qwen::v3_x::weight_layout::dense_mlp_bindings;
use crate::model::qwen::v3_x::weight_layout::push_quantized_tensor_names;
use crate::model::qwen::v3_x::weight_layout::quantized;
use crate::model::qwen::v3_x::weight_layout::quantized_path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ModelWeightBindings {
    pub embed: QuantizedTensorBindings,
    pub main: Qwen3MainWeightBindings,
    pub unembed: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3MainWeightBindings {
    pub final_norm_weight: String,
    pub layers: Vec<Qwen3LayerWeightBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3LayerWeightBindings {
    pub input_norm_weight: String,
    pub post_attention_norm_weight: String,
    pub gqa: Qwen3xGQAWeightBindings,
    pub mlp: Qwen3xDenseMLPWeightBindings,
}

#[derive(Clone, Copy)]
struct Qwen3WeightLayout {
    tensor: TensorPathLayout,
}

impl Qwen3ModelWeightBindings {
    fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        push_quantized_tensor_names(&self.embed, &mut names);
        names.push(&self.main.final_norm_weight);
        push_quantized_tensor_names(&self.unembed, &mut names);
        for layer in &self.main.layers {
            layer.push_tensor_names(&mut names);
        }
        names
    }
}

impl Qwen3LayerWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.push(&self.input_norm_weight);
        names.push(&self.post_attention_norm_weight);
        self.gqa.push_tensor_names(names);
        self.mlp.push_tensor_names(names);
    }
}

impl Qwen3WeightLayout {
    fn bind(self, model_config: &Qwen3ModelConfig) -> Qwen3ModelWeightBindings {
        let mut layers = Vec::with_capacity(model_config.text_config.num_hidden_layers);
        for model_layer_index in 0..model_config.text_config.num_hidden_layers {
            let layer_prefix = self.tensor.model_path(&format!("layers.{model_layer_index}"));
            let attention_prefix = format!("{layer_prefix}.self_attn");
            let mlp_prefix = format!("{layer_prefix}.mlp");
            layers.push(Qwen3LayerWeightBindings {
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
                mlp: dense_mlp_bindings(&mlp_prefix),
            });
        }
        Qwen3ModelWeightBindings {
            embed: quantized_path(self.tensor.model_path("embed_tokens")),
            main: Qwen3MainWeightBindings {
                final_norm_weight: self.tensor.model_path("norm.weight"),
                layers,
            },
            unembed: quantized_path(self.tensor.container_path("lm_head")),
        }
    }

    fn label(self) -> String {
        format!(
            "container_prefix={:?}, model_prefix={:?}",
            self.tensor.container_prefix, self.tensor.model_prefix
        )
    }
}

pub fn resolve_qwen3_model_weight_bindings<'a>(
    model_config: &Qwen3ModelConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3ModelWeightBindings, ModelExecutorError> {
    let tensor_names = tensor_names.into_iter().collect::<HashSet<_>>();
    if tensor_names.is_empty() {
        return Err(ModelExecutorError::custom(
            "Qwen3 checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }

    let mut matches = Vec::new();
    let mut missing_by_layout = Vec::new();
    for tensor in tensor_path_layout_candidates() {
        let layout = Qwen3WeightLayout { tensor };
        let bindings = layout.bind(model_config);
        let missing = bindings
            .tensor_names()
            .into_iter()
            .find(|name| !tensor_names.contains(name))
            .map(str::to_string);
        if let Some(missing) = missing {
            missing_by_layout.push(format!("{} missing {missing:?}", layout.label()));
        } else {
            matches.push((layout.label(), bindings));
        }
    }

    match matches.len() {
        1 => Ok(matches.pop().expect("Qwen3 layout match count checked").1),
        0 => {
            Err(ModelExecutorError::custom(format!(
                "Qwen3 checkpoint tensor manifest does not match a supported exact weight layout: {}",
                missing_by_layout.join("; ")
            )))
        },
        _ => {
            Err(ModelExecutorError::custom(format!(
                "Qwen3 checkpoint tensor manifest matches multiple weight layouts: {:?}",
                matches.into_iter().map(|(label, _)| label).collect::<Vec<_>>()
            )))
        },
    }
}

#[cfg(test)]
#[path = "weight_layout_tests.rs"]
mod tests;
