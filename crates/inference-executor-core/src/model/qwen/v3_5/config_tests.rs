use super::*;

fn config_json() -> Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.layers.0.mlp.gate.weight": {
                "group_size": 128,
                "bits": 3
            }
        },
        "quantization_config": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.layers.0.mlp.gate.weight": {
                "group_size": 128,
                "bits": 3
            }
        },
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 0.0,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "layer_types": ["linear_attention", "gated_delta_net", "linear_attention", "full_attention"],
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 0,
            "decoder_sparse_step": 2,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "shared_expert_intermediate_size": 768,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }
    })
}

#[test]
fn test_parses_and_normalizes_nested_text_config() {
    let model_config = parse_qwen35_config(config_json()).unwrap();

    assert_eq!(model_config.text_config.hidden_act, "silu");
    assert_eq!(model_config.text_config.rms_norm_eps, 1e-6);
    assert_eq!(model_config.text_config.linear_conv_kernel_dim, 4);
    assert_eq!(model_config.text_config.full_attention_interval, 4);
    assert_eq!(model_config.text_config.rope_theta, 10_000_000.0);
    assert_eq!(model_config.text_config.rope_dim, 64);
}

#[test]
fn test_resolves_layer_types_and_moe_schedule() {
    let mut model_config = serde_json::from_value::<Qwen35ModelConfig>(config_json()).unwrap();
    model_config.normalize().unwrap();

    assert_eq!(model_config.layer_type_at(0).unwrap(), LayerType::GDN);
    assert_eq!(model_config.layer_type_at(3).unwrap(), LayerType::FullAttention);
    assert!(!model_config.layer_uses_moe(0));
    assert!(model_config.layer_uses_moe(1));
}

#[test]
fn test_rejects_malformed_nested_quantization() {
    let mut envelope = config_json();
    envelope["quantization"] = serde_json::json!({
        "group_size": 64,
        "bits": 4,
        "mode": "affine"
    });
    envelope["quantization_config"]["bits"] = serde_json::json!("four");

    let error = parse_qwen35_config(envelope).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("unable to parse qwen3.5 quantization_config")
    );
}

#[test]
fn test_rejects_conflicting_quantization_fields() {
    let mut envelope = config_json();
    envelope["quantization"] = serde_json::json!({
        "group_size": 64,
        "bits": 8,
        "mode": "affine"
    });

    let error = parse_qwen35_config(envelope).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("quantization and quantization_config must describe the same layout")
    );
}
