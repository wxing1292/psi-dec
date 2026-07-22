use serde_json::json;

use super::QWEN35_GDN_COMPONENT_NAMES;
use super::Qwen35AttentionWeightBindings;
use super::Qwen35MLPWeightBindings;
use super::Qwen35WeightLayout;
use super::resolve_qwen35_model_weight_bindings;
use super::resolve_qwen35_mtp_weight_bindings;
use crate::model::qwen::v3_5::Qwen35ModelConfig;
use crate::model::qwen::v3_5::TensorPathLayout;

fn model_config(num_experts: usize) -> Qwen35ModelConfig {
    let intermediate_size = if num_experts == 0 { 16 } else { 0 };
    let moe_intermediate_size = if num_experts == 0 { 0 } else { 8 };
    let shared_expert_intermediate_size = if num_experts == 0 { 0 } else { 16 };
    let mut config = serde_json::from_value::<Qwen35ModelConfig>(json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 8,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "decoder_sparse_step": 1,
            "num_experts": num_experts,
            "num_experts_per_tok": if num_experts == 0 { 0 } else { 2 },
            "shared_expert_intermediate_size": shared_expert_intermediate_size,
            "moe_intermediate_size": moe_intermediate_size
        }
    }))
    .unwrap();
    config.normalize().unwrap();
    config
}

#[test]
fn resolves_complete_linear_attn_dense_layout() {
    let config = model_config(0);
    let layout = Qwen35WeightLayout {
        tensor: TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        gdn_component_name: "linear_attn",
    };
    let expected = layout.bind(&config).unwrap();
    let names = expected
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let actual = resolve_qwen35_model_weight_bindings(&config, names.iter().map(String::as_str)).unwrap();

    assert_eq!(actual, expected);
    assert!(matches!(
        actual.main.layers[0].attention,
        Qwen35AttentionWeightBindings::GDN(_)
    ));
    assert!(matches!(
        actual.main.layers[3].attention,
        Qwen35AttentionWeightBindings::GQA(_)
    ));
    assert!(matches!(actual.main.layers[0].mlp, Qwen35MLPWeightBindings::Dense(_)));
    assert_eq!(
        actual.main.layers[0].input_norm_weight,
        "language_model.model.layers.0.input_layernorm.weight"
    );
}

#[test]
fn resolves_complete_moe_layout() {
    let config = model_config(4);
    let layout = Qwen35WeightLayout {
        tensor: TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        gdn_component_name: "linear_attn",
    };
    let expected = layout.bind(&config).unwrap();
    let names = expected
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let actual = resolve_qwen35_model_weight_bindings(&config, names.iter().map(String::as_str)).unwrap();

    let Qwen35MLPWeightBindings::MoE(mlp) = &actual.main.layers[0].mlp else {
        panic!("qwen3.5 MoE config must bind MoE weights");
    };
    assert_eq!(
        mlp.experts.gate.weight,
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    );
    assert!(mlp.shared_expert_gate.is_some());
    assert!(mlp.shared_expert.is_some());
}

#[test]
fn rejects_missing_exact_tensor() {
    let config = model_config(0);
    let layout = Qwen35WeightLayout {
        tensor: TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        gdn_component_name: "linear_attn",
    };
    let expected = layout.bind(&config).unwrap();
    let missing_name = expected.main.layers[3].input_norm_weight.clone();
    let names = expected
        .tensor_names()
        .into_iter()
        .filter(|&name| name != missing_name)
        .map(str::to_string)
        .collect::<Vec<_>>();

    let err = resolve_qwen35_model_weight_bindings(&config, names.iter().map(String::as_str)).unwrap_err();

    assert!(
        err.to_string()
            .contains("does not match a supported exact weight layout")
    );
    assert!(err.to_string().contains(&missing_name));
}

#[test]
fn rejects_mixed_gdn_component_names() {
    let config = model_config(0);
    let layout = Qwen35WeightLayout {
        tensor: TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        gdn_component_name: "linear_attn",
    };
    let expected = layout.bind(&config).unwrap();
    let names = expected
        .tensor_names()
        .into_iter()
        .map(|name| {
            if name.contains("layers.1.linear_attn") {
                name.replace("linear_attn", QWEN35_GDN_COMPONENT_NAMES[0])
            } else {
                name.to_string()
            }
        })
        .collect::<Vec<_>>();

    let err = resolve_qwen35_model_weight_bindings(&config, names.iter().map(String::as_str)).unwrap_err();

    assert!(
        err.to_string()
            .contains("does not match a supported exact weight layout")
    );
}

#[test]
fn resolves_single_mtp_module_from_complete_names() {
    let config = model_config(0);
    let expected = super::Qwen35MTPWeightBindings::from_config(&config, 1).unwrap();
    let names = expected
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let actual = resolve_qwen35_mtp_weight_bindings(&config, 1, names.iter().map(String::as_str)).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.embed.projection.weight, "fc.weight");
    assert_eq!(actual.body.input_norm_weight, "layers.0.input_layernorm.weight");
}

#[test]
fn rejects_missing_mtp_body_tensor() {
    let config = model_config(0);
    let expected = super::Qwen35MTPWeightBindings::from_config(&config, 1).unwrap();
    let missing = expected.body.input_norm_weight.clone();
    let names = expected
        .tensor_names()
        .into_iter()
        .filter(|&name| name != missing)
        .map(str::to_string)
        .collect::<Vec<_>>();

    let err = resolve_qwen35_mtp_weight_bindings(&config, 1, names.iter().map(String::as_str)).unwrap_err();

    assert!(err.to_string().contains(&missing));
}

#[test]
#[should_panic(expected = "qwen3.5 supports exactly one MTP module")]
fn rejects_multiple_mtp_modules() {
    let _ = super::Qwen35MTPWeightBindings::from_config(&model_config(0), 2);
}
