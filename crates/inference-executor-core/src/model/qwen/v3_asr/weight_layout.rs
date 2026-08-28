use std::collections::HashSet;

use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3::weight_layout::Qwen3LayerWeightBindings;
use crate::model::qwen::v3::weight_layout::Qwen3MainWeightBindings;
use crate::model::qwen::v3_asr::Qwen3ASRModelConfig;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use crate::model::qwen::v3_x::weight_layout::dense_mlp_bindings;
use crate::model::qwen::v3_x::weight_layout::push_quantized_tensor_names;
use crate::model::qwen::v3_x::weight_layout::quantized;
use crate::model::qwen::v3_x::weight_layout::quantized_path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRWeightBindings {
    pub audio: Qwen3ASRAudioWeightBindings,
    pub embed: QuantizedTensorBindings,
    pub text: Qwen3MainWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRAudioWeightBindings {
    pub conv: [Qwen3ASRAffineWeightBindings; 3],
    pub conv_out_weight: String,
    pub layers: Vec<Qwen3ASRAudioLayerWeightBindings>,
    pub ln_post: Qwen3ASRNormWeightBindings,
    pub proj1: Qwen3ASRAffineWeightBindings,
    pub proj2: Qwen3ASRAffineWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRAudioLayerWeightBindings {
    pub self_attention_norm: Qwen3ASRNormWeightBindings,
    pub q: Qwen3ASRAffineWeightBindings,
    pub k: Qwen3ASRAffineWeightBindings,
    pub v: Qwen3ASRAffineWeightBindings,
    pub output: Qwen3ASRAffineWeightBindings,
    pub final_norm: Qwen3ASRNormWeightBindings,
    pub fc1: Qwen3ASRAffineWeightBindings,
    pub fc2: Qwen3ASRAffineWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRAffineWeightBindings {
    pub weight: String,
    pub bias: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3ASRNormWeightBindings {
    pub weight: String,
    pub bias: String,
}

impl Qwen3ASRWeightBindings {
    fn tensor_names(&self) -> Vec<&str> {
        let mut names = self.audio.tensor_names();
        push_quantized_tensor_names(&self.embed, &mut names);
        names.push(&self.text.final_norm_weight);
        for layer in &self.text.layers {
            names.extend([
                layer.input_norm_weight.as_str(),
                layer.post_attention_norm_weight.as_str(),
            ]);
            layer.gqa.push_tensor_names(&mut names);
            layer.mlp.push_tensor_names(&mut names);
        }
        names
    }
}

impl Qwen3ASRAudioWeightBindings {
    fn tensor_names(&self) -> Vec<&str> {
        let mut names = vec![];
        for conv in &self.conv {
            conv.push_tensor_names(&mut names);
        }
        names.push(&self.conv_out_weight);
        for layer in &self.layers {
            layer.push_tensor_names(&mut names);
        }
        self.ln_post.push_tensor_names(&mut names);
        self.proj1.push_tensor_names(&mut names);
        self.proj2.push_tensor_names(&mut names);
        names
    }
}

impl Qwen3ASRAudioLayerWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.self_attention_norm.push_tensor_names(names);
        self.q.push_tensor_names(names);
        self.k.push_tensor_names(names);
        self.v.push_tensor_names(names);
        self.output.push_tensor_names(names);
        self.final_norm.push_tensor_names(names);
        self.fc1.push_tensor_names(names);
        self.fc2.push_tensor_names(names);
    }
}

impl Qwen3ASRAffineWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([self.weight.as_str(), self.bias.as_str()]);
    }
}

impl Qwen3ASRNormWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([self.weight.as_str(), self.bias.as_str()]);
    }
}

pub fn resolve_qwen3_asr_weight_bindings<'a>(
    config: &Qwen3ASRModelConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen3ASRWeightBindings, ModelExecutorError> {
    let bindings = bind(config.audio.encoder_layers, config.text.num_hidden_layers);
    let tensor_names = tensor_names.into_iter().collect::<HashSet<_>>();
    if let Some(missing) = bindings
        .tensor_names()
        .into_iter()
        .find(|name| !tensor_names.contains(name))
    {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3-ASR checkpoint is missing required tensor {missing:?}"
        )));
    }
    Ok(bindings)
}

fn bind(num_audio_layers: usize, num_text_layers: usize) -> Qwen3ASRWeightBindings {
    let conv = [1, 2, 3].map(|index| affine(&format!("audio_tower.conv2d{index}")));
    let mut audio_layers = Vec::with_capacity(num_audio_layers);
    for index in 0..num_audio_layers {
        let prefix = format!("audio_tower.layers.{index}");
        audio_layers.push(Qwen3ASRAudioLayerWeightBindings {
            self_attention_norm: norm(&format!("{prefix}.self_attn_layer_norm")),
            q: affine(&format!("{prefix}.self_attn.q_proj")),
            k: affine(&format!("{prefix}.self_attn.k_proj")),
            v: affine(&format!("{prefix}.self_attn.v_proj")),
            output: affine(&format!("{prefix}.self_attn.out_proj")),
            final_norm: norm(&format!("{prefix}.final_layer_norm")),
            fc1: affine(&format!("{prefix}.fc1")),
            fc2: affine(&format!("{prefix}.fc2")),
        });
    }
    Qwen3ASRWeightBindings {
        audio: Qwen3ASRAudioWeightBindings {
            conv,
            conv_out_weight: "audio_tower.conv_out.weight".to_string(),
            layers: audio_layers,
            ln_post: norm("audio_tower.ln_post"),
            proj1: affine("audio_tower.proj1"),
            proj2: affine("audio_tower.proj2"),
        },
        embed: quantized_path("model.embed_tokens".to_string()),
        text: text_bindings("model", num_text_layers),
    }
}

fn text_bindings(model_prefix: &str, num_layers: usize) -> Qwen3MainWeightBindings {
    let mut layers = Vec::with_capacity(num_layers);
    for index in 0..num_layers {
        let layer_prefix = format!("{model_prefix}.layers.{index}");
        let attention_prefix = format!("{layer_prefix}.self_attn");
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
            mlp: dense_mlp_bindings(&format!("{layer_prefix}.mlp")),
        });
    }
    Qwen3MainWeightBindings {
        final_norm_weight: format!("{model_prefix}.norm.weight"),
        layers,
    }
}

fn affine(prefix: &str) -> Qwen3ASRAffineWeightBindings {
    Qwen3ASRAffineWeightBindings {
        weight: format!("{prefix}.weight"),
        bias: format!("{prefix}.bias"),
    }
}

fn norm(prefix: &str) -> Qwen3ASRNormWeightBindings {
    Qwen3ASRNormWeightBindings {
        weight: format!("{prefix}.weight"),
        bias: format!("{prefix}.bias"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_bindings_use_tied_text_embeddings() {
        let bindings = bind(2, 2);
        let names = bindings.tensor_names();
        assert!(names.contains(&"audio_tower.layers.1.self_attn.q_proj.weight"));
        assert!(names.contains(&"model.layers.1.mlp.down_proj.weight"));
        assert!(names.contains(&"model.embed_tokens.weight"));
        assert!(!names.iter().any(|name| name.starts_with("lm_head")));
    }
}
