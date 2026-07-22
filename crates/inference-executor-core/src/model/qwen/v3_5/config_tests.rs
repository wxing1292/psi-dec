use super::*;

fn config_json() -> Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
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
    let envelope = config_json();
    let mut model_config = serde_json::from_value::<Qwen35ModelConfig>(envelope.clone()).unwrap();
    model_config.quantization = model_config
        .quantization
        .or_else(|| parse_nested_quantization_config(&envelope));
    model_config.normalize().unwrap();

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
fn test_resolves_tensor_path_layout() {
    let layout = resolve_tensor_path_layout_from_names([
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.gated_delta_net.in_proj_qkv.weight",
    ]);

    assert_eq!(
        layout,
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        }
    );
    assert_eq!(
        layout.model_path("layers.0.self_attn.q_proj.weight"),
        "language_model.model.layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn test_resolves_quantization_override_aliases() {
    let mut model_config = serde_json::from_value::<Qwen35ModelConfig>(config_json()).unwrap();
    model_config.quantization = parse_nested_quantization_config(&config_json());
    model_config.normalize().unwrap();
    let quantization = model_config.quantization.unwrap();

    let direct = quantization.resolve_for_tensor("model.layers.0.mlp.gate.weight");
    let normalized = quantization.resolve_for_tensor("layers.0.mlp.gate.weight");
    let fallback = quantization.resolve_for_tensor("model.layers.0.mlp.up_proj.weight");

    assert_eq!(direct.group_size, 128);
    assert_eq!(normalized.bits, 3);
    assert_eq!(fallback.group_size, 64);
    assert_eq!(fallback.bits, 4);
}

#[test]
fn test_normalizes_common_qwen_container_prefixes() {
    assert_eq!(
        normalize_qwen_name("model.layers.0.mlp.gate_proj.weight"),
        "layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(
        normalize_qwen_name("language_model.model.layers.1.self_attn.q_proj.weight"),
        "layers.1.self_attn.q_proj.weight"
    );
    assert_eq!(
        normalize_qwen_name("model.language_model.layers.2.mlp.up_proj.weight"),
        "layers.2.mlp.up_proj.weight"
    );
    assert_eq!(normalize_qwen_name("lm_head.weight"), "unembed.weight");
}
