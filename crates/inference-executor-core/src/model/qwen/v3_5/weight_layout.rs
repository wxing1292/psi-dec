use std::collections::HashSet;

use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3_5::LayerType;
use crate::model::qwen::v3_5::Qwen35ModelConfig;
use crate::model::qwen::v3_5::TensorPathLayout;
use crate::model::qwen::v3_5::default_tensor_path_layout;
use crate::model::qwen::v3_5::tensor_path_layout_candidates;

const QWEN35_GDN_COMPONENT_NAMES: [&str; 2] = ["gated_delta_net", "linear_attn"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35ModelWeightBindings {
    pub embed: QuantizedTensorBindings,
    pub main: Qwen35MainWeightBindings,
    pub unembed: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MainWeightBindings {
    pub final_norm_weight: String,
    pub layers: Vec<Qwen35LayerWeightBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MTPWeightBindings {
    pub embed: Qwen35MTPEmbedWeightBindings,
    pub body: Qwen35LayerWeightBindings,
    pub final_norm_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MTPEmbedWeightBindings {
    pub prev_hidden_norm_weight: String,
    pub token_hidden_norm_weight: String,
    pub projection: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35LayerWeightBindings {
    pub input_norm_weight: String,
    pub post_attention_norm_weight: String,
    pub attention: Qwen35AttentionWeightBindings,
    pub mlp: Qwen35MLPWeightBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen35AttentionWeightBindings {
    GDN(Qwen35GDNWeightBindings),
    GQA(Qwen35GQAWeightBindings),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen35MLPWeightBindings {
    Dense(Box<Qwen35DenseMLPWeightBindings>),
    MoE(Box<Qwen35MoEWeightBindings>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GDNWeightBindings {
    pub qkv: QuantizedTensorBindings,
    pub a: QuantizedTensorBindings,
    pub b: QuantizedTensorBindings,
    pub z: QuantizedTensorBindings,
    pub conv_weight: String,
    pub norm_weight: String,
    pub a_log: String,
    pub dt_bias: String,
    pub output: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GQAWeightBindings {
    pub q: QuantizedTensorBindings,
    pub k: QuantizedTensorBindings,
    pub v: QuantizedTensorBindings,
    pub q_norm_weight: String,
    pub k_norm_weight: String,
    pub output: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35DenseMLPWeightBindings {
    pub gate: QuantizedTensorBindings,
    pub up: QuantizedTensorBindings,
    pub down: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoEWeightBindings {
    pub router: QuantizedTensorBindings,
    pub experts: Qwen35SparseExpertWeightBindings,
    pub shared_expert_gate: Option<QuantizedTensorBindings>,
    pub shared_expert: Option<Qwen35DenseMLPWeightBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35SparseExpertWeightBindings {
    pub gate: QuantizedTensorBindings,
    pub up: QuantizedTensorBindings,
    pub down: QuantizedTensorBindings,
}

#[derive(Clone, Copy)]
struct Qwen35WeightLayout {
    tensor: TensorPathLayout,
    gdn_component_name: &'static str,
}

impl QuantizedTensorBindings {
    fn from_prefix(prefix: String) -> Self {
        Self {
            weight: format!("{prefix}.weight"),
            scales: format!("{prefix}.scales"),
            biases: format!("{prefix}.biases"),
        }
    }

    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([self.weight.as_str(), self.scales.as_str(), self.biases.as_str()]);
    }
}

impl Qwen35ModelWeightBindings {
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        self.embed.push_tensor_names(&mut names);
        names.push(&self.main.final_norm_weight);
        self.unembed.push_tensor_names(&mut names);
        for layer in &self.main.layers {
            layer.push_tensor_names(&mut names);
        }
        names
    }
}

impl Qwen35MTPWeightBindings {
    pub fn from_config(model_config: &Qwen35ModelConfig, num_modules: usize) -> Result<Self, ModelExecutorError> {
        assert_eq!(num_modules, 1, "qwen3.5 supports exactly one MTP module");
        assert!(
            num_modules <= model_config.text_config.num_hidden_layers,
            "qwen3.5 MTP binding count exceeds configured layers"
        );
        let uses_moe = model_config.layer_uses_moe(0);
        let has_shared_expert = model_config.text_config.shared_expert_intermediate_size > 0;
        Ok(Self {
            embed: Qwen35MTPEmbedWeightBindings {
                prev_hidden_norm_weight: "pre_fc_norm_hidden.weight".to_string(),
                token_hidden_norm_weight: "pre_fc_norm_embedding.weight".to_string(),
                projection: quantized_path("fc".to_string()),
            },
            body: qwen35_gqa_layer_weight_bindings("layers.0", uses_moe, has_shared_expert),
            final_norm_weight: "norm.weight".to_string(),
        })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        names.extend([
            self.embed.prev_hidden_norm_weight.as_str(),
            self.embed.token_hidden_norm_weight.as_str(),
        ]);
        self.embed.projection.push_tensor_names(&mut names);
        names.push(&self.final_norm_weight);
        self.body.push_tensor_names(&mut names);
        names
    }
}

impl Qwen35LayerWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.push(&self.input_norm_weight);
        names.push(&self.post_attention_norm_weight);
        self.attention.push_tensor_names(names);
        self.mlp.push_tensor_names(names);
    }
}

impl Qwen35AttentionWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        match self {
            Self::GDN(bindings) => bindings.push_tensor_names(names),
            Self::GQA(bindings) => bindings.push_tensor_names(names),
        }
    }
}

impl Qwen35MLPWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        match self {
            Self::Dense(bindings) => bindings.push_tensor_names(names),
            Self::MoE(bindings) => bindings.push_tensor_names(names),
        }
    }
}

impl Qwen35GDNWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.qkv.push_tensor_names(names);
        self.a.push_tensor_names(names);
        self.b.push_tensor_names(names);
        self.z.push_tensor_names(names);
        names.extend([
            self.conv_weight.as_str(),
            self.norm_weight.as_str(),
            self.a_log.as_str(),
            self.dt_bias.as_str(),
        ]);
        self.output.push_tensor_names(names);
    }
}

impl Qwen35GQAWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.q.push_tensor_names(names);
        self.k.push_tensor_names(names);
        self.v.push_tensor_names(names);
        names.extend([self.q_norm_weight.as_str(), self.k_norm_weight.as_str()]);
        self.output.push_tensor_names(names);
    }
}

impl Qwen35DenseMLPWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.gate.push_tensor_names(names);
        self.up.push_tensor_names(names);
        self.down.push_tensor_names(names);
    }
}

impl Qwen35MoEWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.router.push_tensor_names(names);
        self.experts.push_tensor_names(names);
        if let Some(bindings) = &self.shared_expert_gate {
            bindings.push_tensor_names(names);
        }
        if let Some(bindings) = &self.shared_expert {
            bindings.push_tensor_names(names);
        }
    }
}

impl Qwen35SparseExpertWeightBindings {
    fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        self.gate.push_tensor_names(names);
        self.up.push_tensor_names(names);
        self.down.push_tensor_names(names);
    }
}

impl Qwen35WeightLayout {
    fn bind(self, model_config: &Qwen35ModelConfig) -> Result<Qwen35ModelWeightBindings, ModelExecutorError> {
        let text = &model_config.text_config;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for model_layer_index in 0..text.num_hidden_layers {
            let prefix = self.tensor.model_path(&format!("layers.{model_layer_index}"));
            let attention = match model_config.layer_type_at(model_layer_index)? {
                LayerType::GDN => {
                    let prefix = format!("{prefix}.{}", self.gdn_component_name);
                    Qwen35AttentionWeightBindings::GDN(Qwen35GDNWeightBindings {
                        qkv: quantized(&prefix, "in_proj_qkv"),
                        a: quantized(&prefix, "in_proj_a"),
                        b: quantized(&prefix, "in_proj_b"),
                        z: quantized(&prefix, "in_proj_z"),
                        conv_weight: format!("{prefix}.conv1d.weight"),
                        norm_weight: format!("{prefix}.norm.weight"),
                        a_log: format!("{prefix}.A_log"),
                        dt_bias: format!("{prefix}.dt_bias"),
                        output: quantized(&prefix, "out_proj"),
                    })
                },
                LayerType::FullAttention => {
                    let prefix = format!("{prefix}.self_attn");
                    Qwen35AttentionWeightBindings::GQA(Qwen35GQAWeightBindings {
                        q: quantized(&prefix, "q_proj"),
                        k: quantized(&prefix, "k_proj"),
                        v: quantized(&prefix, "v_proj"),
                        q_norm_weight: format!("{prefix}.q_norm.weight"),
                        k_norm_weight: format!("{prefix}.k_norm.weight"),
                        output: quantized(&prefix, "o_proj"),
                    })
                },
            };
            let mlp_prefix = format!("{prefix}.mlp");
            let mlp = if model_config.layer_uses_moe(model_layer_index) {
                let has_shared_expert = text.shared_expert_intermediate_size > 0;
                Qwen35MLPWeightBindings::MoE(Box::new(Qwen35MoEWeightBindings {
                    router: quantized_path(format!("{mlp_prefix}.gate")),
                    experts: Qwen35SparseExpertWeightBindings {
                        gate: quantized(&mlp_prefix, "switch_mlp.gate_proj"),
                        up: quantized(&mlp_prefix, "switch_mlp.up_proj"),
                        down: quantized(&mlp_prefix, "switch_mlp.down_proj"),
                    },
                    shared_expert_gate: has_shared_expert
                        .then(|| quantized_path(format!("{mlp_prefix}.shared_expert_gate"))),
                    shared_expert: has_shared_expert
                        .then(|| dense_mlp_bindings(&format!("{mlp_prefix}.shared_expert"))),
                }))
            } else {
                Qwen35MLPWeightBindings::Dense(Box::new(dense_mlp_bindings(&mlp_prefix)))
            };
            layers.push(Qwen35LayerWeightBindings {
                input_norm_weight: format!("{prefix}.input_layernorm.weight"),
                post_attention_norm_weight: format!("{prefix}.post_attention_layernorm.weight"),
                attention,
                mlp,
            });
        }
        Ok(Qwen35ModelWeightBindings {
            embed: quantized_path(self.tensor.model_path("embed_tokens")),
            main: Qwen35MainWeightBindings {
                final_norm_weight: self.tensor.model_path("norm.weight"),
                layers,
            },
            unembed: quantized_path(self.tensor.container_path("lm_head")),
        })
    }

    fn label(self) -> String {
        format!(
            "container_prefix={:?}, model_prefix={:?}, gdn_component={:?}",
            self.tensor.container_prefix, self.tensor.model_prefix, self.gdn_component_name
        )
    }
}

pub fn default_qwen35_model_weight_bindings(
    model_config: &Qwen35ModelConfig,
) -> Result<Qwen35ModelWeightBindings, ModelExecutorError> {
    Qwen35WeightLayout {
        tensor: default_tensor_path_layout(),
        gdn_component_name: QWEN35_GDN_COMPONENT_NAMES[0],
    }
    .bind(model_config)
}

pub fn qwen35_gqa_layer_weight_bindings(
    layer_prefix: &str,
    uses_moe: bool,
    has_shared_expert: bool,
) -> Qwen35LayerWeightBindings {
    let attention_prefix = format!("{layer_prefix}.self_attn");
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let mlp = if uses_moe {
        Qwen35MLPWeightBindings::MoE(Box::new(Qwen35MoEWeightBindings {
            router: quantized_path(format!("{mlp_prefix}.gate")),
            experts: Qwen35SparseExpertWeightBindings {
                gate: quantized(&mlp_prefix, "switch_mlp.gate_proj"),
                up: quantized(&mlp_prefix, "switch_mlp.up_proj"),
                down: quantized(&mlp_prefix, "switch_mlp.down_proj"),
            },
            shared_expert_gate: has_shared_expert.then(|| quantized_path(format!("{mlp_prefix}.shared_expert_gate"))),
            shared_expert: has_shared_expert.then(|| dense_mlp_bindings(&format!("{mlp_prefix}.shared_expert"))),
        }))
    } else {
        Qwen35MLPWeightBindings::Dense(Box::new(dense_mlp_bindings(&mlp_prefix)))
    };
    Qwen35LayerWeightBindings {
        input_norm_weight: format!("{layer_prefix}.input_layernorm.weight"),
        post_attention_norm_weight: format!("{layer_prefix}.post_attention_layernorm.weight"),
        attention: Qwen35AttentionWeightBindings::GQA(Qwen35GQAWeightBindings {
            q: quantized(&attention_prefix, "q_proj"),
            k: quantized(&attention_prefix, "k_proj"),
            v: quantized(&attention_prefix, "v_proj"),
            q_norm_weight: format!("{attention_prefix}.q_norm.weight"),
            k_norm_weight: format!("{attention_prefix}.k_norm.weight"),
            output: quantized(&attention_prefix, "o_proj"),
        }),
        mlp,
    }
}

pub fn resolve_qwen35_model_weight_bindings<'a>(
    model_config: &Qwen35ModelConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen35ModelWeightBindings, ModelExecutorError> {
    let tensor_names = tensor_names.into_iter().collect::<HashSet<_>>();
    if tensor_names.is_empty() {
        return Err(ModelExecutorError::custom(
            "qwen3.5 checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }
    let has_gdn = (0..model_config.text_config.num_hidden_layers)
        .map(|model_layer_index| model_config.layer_type_at(model_layer_index))
        .collect::<Result<Vec<_>, _>>()?
        .contains(&LayerType::GDN);
    let gdn_component_names = if has_gdn {
        &QWEN35_GDN_COMPONENT_NAMES[..]
    } else {
        &QWEN35_GDN_COMPONENT_NAMES[..1]
    };

    let mut matches = Vec::new();
    let mut missing_by_layout = Vec::new();
    for tensor in tensor_path_layout_candidates() {
        for &gdn_component_name in gdn_component_names {
            let layout = Qwen35WeightLayout {
                tensor,
                gdn_component_name,
            };
            let bindings = layout.bind(model_config)?;
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
    }

    match matches.len() {
        1 => Ok(matches.pop().expect("qwen3.5 layout match count checked").1),
        0 => {
            Err(ModelExecutorError::custom(format!(
                "qwen3.5 checkpoint tensor manifest does not match a supported exact weight layout: {}",
                missing_by_layout.join("; ")
            )))
        },
        _ => {
            Err(ModelExecutorError::custom(format!(
                "qwen3.5 checkpoint tensor manifest matches multiple weight layouts: {:?}",
                matches.into_iter().map(|(label, _)| label).collect::<Vec<_>>()
            )))
        },
    }
}

pub fn resolve_qwen35_mtp_weight_bindings<'a>(
    model_config: &Qwen35ModelConfig,
    num_modules: usize,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen35MTPWeightBindings, ModelExecutorError> {
    let tensor_names = tensor_names.into_iter().collect::<HashSet<_>>();
    if tensor_names.is_empty() {
        return Err(ModelExecutorError::custom(
            "qwen3.5 MTP checkpoint layout resolution requires a nonempty tensor manifest",
        ));
    }
    let bindings = Qwen35MTPWeightBindings::from_config(model_config, num_modules)?;
    let mut missing = bindings
        .tensor_names()
        .into_iter()
        .filter(|name| !tensor_names.contains(name))
        .collect::<Vec<_>>();
    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 MTP checkpoint tensor manifest does not match the canonical exact weight layout; \
             missing={missing:?}"
        )));
    }
    Ok(bindings)
}

fn quantized(prefix: &str, relative_name: &str) -> QuantizedTensorBindings {
    quantized_path(format!("{prefix}.{relative_name}"))
}

fn quantized_path(prefix: String) -> QuantizedTensorBindings {
    QuantizedTensorBindings::from_prefix(prefix)
}

fn dense_mlp_bindings(prefix: &str) -> Qwen35DenseMLPWeightBindings {
    Qwen35DenseMLPWeightBindings {
        gate: quantized(prefix, "gate_proj"),
        up: quantized(prefix, "up_proj"),
        down: quantized(prefix, "down_proj"),
    }
}

#[cfg(test)]
#[path = "weight_layout_tests.rs"]
mod tests;
