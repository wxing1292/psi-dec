use super::Qwen35DSparkWeightBindings;
use super::resolve_qwen35_dspark_weight_bindings;
use crate::model::qwen::v3_5::DSparkConfig;

fn config() -> DSparkConfig {
    let mut config = serde_json::from_value::<DSparkConfig>(serde_json::json!({
        "model_type": "qwen3",
        "block_size": 5,
        "dflash_config": {
            "causal_head": false,
            "causal": false,
            "mask_token_id": 15,
            "target_layer_ids": [0, 3, 7]
        },
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_target_layers": 8,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 32,
        "vocab_size": 16,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "layer_types": ["full_attention", "full_attention"]
    }))
    .unwrap();
    config.normalize_and_validate().unwrap();
    config
}

#[test]
fn builds_exact_hikari_names_from_draft_layer_count() {
    let bindings = Qwen35DSparkWeightBindings::from_config(&config());

    assert_eq!(bindings.layers.len(), 2);
    assert_eq!(bindings.target_projection.fc.weight, "fc.weight");
    assert_eq!(
        bindings.layers[1].attention.q.weight,
        "layers.1.self_attn.q_proj.weight"
    );
    assert_eq!(bindings.layers[1].mlp.down.biases, "layers.1.mlp.down_proj.biases");
    assert_eq!(bindings.markov.w2.scales, "markov_head.markov_w2.scales");
    assert_eq!(bindings.source_tensor_names().len(), 27);
    assert_eq!(bindings.tensor_names().len(), 61);
}

#[test]
fn resolves_only_the_complete_affine_manifest() {
    let config = config();
    let expected = Qwen35DSparkWeightBindings::from_config(&config);
    let names = expected
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let actual = resolve_qwen35_dspark_weight_bindings(&config, names.iter().map(String::as_str)).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn rejects_missing_or_unexpected_affine_tensors() {
    let config = config();
    let expected = Qwen35DSparkWeightBindings::from_config(&config);
    let mut names = expected
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    names.retain(|name| name != "layers.1.self_attn.q_proj.scales");
    names.push("layers.1.self_attn.unknown.weight".to_string());

    let err = resolve_qwen35_dspark_weight_bindings(&config, names.iter().map(String::as_str)).unwrap_err();

    assert!(err.to_string().contains("layers.1.self_attn.q_proj.scales"));
    assert!(err.to_string().contains("layers.1.self_attn.unknown.weight"));
}
