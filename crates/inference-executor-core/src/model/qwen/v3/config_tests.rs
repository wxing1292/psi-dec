use super::*;

#[test]
fn test_rejects_unsupported_rope_scaling() {
    let mut value = valid_config();
    value["rope_scaling"] = serde_json::json!({
        "rope_type": "yarn",
        "factor": 4.0
    });

    let error = parse_qwen3_config(value).unwrap_err();

    assert!(error.to_string().contains("rope_scaling is unsupported"));
}

#[test]
fn test_rejects_malformed_quantization() {
    let mut value = valid_config();
    value["quantization"]["bits"] = serde_json::json!("four");

    let error = parse_qwen3_config(value).unwrap_err();

    assert!(error.to_string().contains("unable to parse Qwen3 quantization"));
}

#[test]
fn test_rejects_conflicting_quantization_fields() {
    let mut value = valid_config();
    value["quantization_config"]["bits"] = serde_json::json!(3);

    let error = parse_qwen3_config(value).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("quantization and quantization_config must describe the same layout")
    );
}

#[test]
fn test_rejects_unsupported_checkpoint_constants() {
    for (field, value, expected) in [
        ("model_type", serde_json::json!("qwen2"), "unsupported Qwen3 model_type"),
        (
            "attention_bias",
            serde_json::json!(true),
            "attention_bias=true is unsupported",
        ),
        (
            "tie_word_embeddings",
            serde_json::json!(true),
            "tie_word_embeddings=true is unsupported",
        ),
        ("hidden_act", serde_json::json!("gelu"), "unsupported Qwen3 hidden_act"),
        ("use_cache", serde_json::json!(false), "use_cache=false is unsupported"),
        ("torch_dtype", serde_json::json!("float16"), "unsupported Qwen3 dtype"),
    ] {
        let mut config = valid_config();
        config[field] = value;

        let error = parse_qwen3_config(config).unwrap_err();

        assert!(error.to_string().contains(expected), "{field}: {error}");
    }
}

#[test]
fn test_rejects_non_positive_model_geometry() {
    for (field, value, expected) in [
        ("hidden_size", serde_json::json!(0), "hidden_size must be positive"),
        (
            "num_hidden_layers",
            serde_json::json!(0),
            "num_hidden_layers must be positive",
        ),
        (
            "num_attention_heads",
            serde_json::json!(0),
            "num_attention_heads must be positive",
        ),
        (
            "num_key_value_heads",
            serde_json::json!(0),
            "num_key_value_heads must be positive",
        ),
        ("head_dim", serde_json::json!(0), "head_dim must be positive and even"),
        ("head_dim", serde_json::json!(1), "head_dim must be positive and even"),
        ("head_dim", serde_json::json!(3), "head_dim must be positive and even"),
    ] {
        let mut config = valid_config();
        config[field] = value;

        let error = parse_qwen3_config(config).unwrap_err();

        assert!(error.to_string().contains(expected), "{field}: {error}");
    }
}

#[test]
fn test_rejects_inconsistent_attention_geometry() {
    let mut query_width = valid_config();
    query_width["num_attention_heads"] = serde_json::json!(4);
    assert!(
        parse_qwen3_config(query_width)
            .unwrap_err()
            .to_string()
            .contains("num_attention_heads=4 * head_dim=16 must equal hidden_size=128")
    );

    let mut grouped_heads = valid_config();
    grouped_heads["num_key_value_heads"] = serde_json::json!(3);
    assert!(
        parse_qwen3_config(grouped_heads)
            .unwrap_err()
            .to_string()
            .contains("num_attention_heads=8 must be divisible by num_key_value_heads=3")
    );
}

#[test]
fn test_rejects_attention_dimension_overflow() {
    let mut query = valid_config();
    query["num_attention_heads"] = serde_json::json!(usize::MAX);
    query["head_dim"] = serde_json::json!(2);
    assert!(
        parse_qwen3_config(query)
            .unwrap_err()
            .to_string()
            .contains("query dimension must fit usize")
    );

    let mut key_value = valid_config();
    key_value["num_key_value_heads"] = serde_json::json!(usize::MAX);
    key_value["head_dim"] = serde_json::json!(2);
    assert!(
        parse_qwen3_config(key_value)
            .unwrap_err()
            .to_string()
            .contains("key/value dimension must fit usize")
    );
}

#[test]
fn test_rejects_non_dense_or_mixed_attention_config() {
    let mut moe = valid_config();
    moe["num_experts"] = serde_json::json!(8);
    assert!(
        parse_qwen3_config(moe)
            .unwrap_err()
            .to_string()
            .contains("MoE configuration is unsupported")
    );

    let mut mixed_attention = valid_config();
    mixed_attention["layer_types"] =
        serde_json::json!(["full_attention", "linear_attention", "full_attention", "full_attention"]);
    assert!(
        parse_qwen3_config(mixed_attention)
            .unwrap_err()
            .to_string()
            .contains("layer_types must contain only full_attention")
    );
}

#[test]
fn test_rejects_invalid_eos_tokens() {
    let mut duplicate = valid_config();
    duplicate["eos_token_id"] = serde_json::json!([100, 100]);
    assert!(
        parse_qwen3_config(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate token")
    );

    let mut outside_vocab = valid_config();
    outside_vocab["eos_token_id"] = serde_json::json!(1024);
    assert!(
        parse_qwen3_config(outside_vocab)
            .unwrap_err()
            .to_string()
            .contains("must be below vocab_size")
    );
}

fn valid_config() -> Value {
    serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "eos_token_id": 101,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 128,
        "intermediate_size": 256,
        "max_position_embeddings": 4096,
        "model_type": "qwen3",
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "quantization_config": {
            "group_size": 64,
            "bits": 4
        },
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1_000_000.0,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 1024
    })
}
