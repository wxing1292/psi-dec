use super::DSparkConfig;

fn dspark_config() -> DSparkConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3",
        "block_size": 11,
        "dflash_config": {
            "causal_head": false,
            "causal": false,
            "mask_token_id": 248070,
            "target_layer_ids": [1, 16, 31, 46, 61]
        },
        "dtype": "bfloat16",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_target_layers": 64,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "max_position_embeddings": 8192,
        "vocab_size": 248320,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"]
    }))
    .unwrap()
}

#[test]
fn test_linked_dspark_config() {
    let mut config = dspark_config();
    config.normalize_and_validate().unwrap();

    assert_eq!(config.block_size, 11);
    assert_eq!(config.num_layers, 5);
    assert_eq!(config.dflash_config.mask_token_id, 248070);
    assert_eq!(config.dflash_config.target_residual_layer_indices, [1, 16, 31, 46, 61]);
    assert_eq!(config.markov_rank, 256);
}

#[test]
fn test_rejects_duplicate_target_layers() {
    let mut config = dspark_config();
    config.dflash_config.target_residual_layer_indices = vec![1, 16, 16, 46, 61];

    let err = config.normalize_and_validate().unwrap_err();

    assert!(err.to_string().contains("target_residual_layer_indices"));
    assert!(err.to_string().contains("must be unique"));
}

#[test]
fn test_selected_residual_count_is_independent_from_draft_layer_count() {
    let mut config = dspark_config();
    config.dflash_config.target_residual_layer_indices = vec![1, 31, 61];

    config.normalize_and_validate().unwrap();

    assert_eq!(config.num_layers, 5);
    assert_eq!(config.dflash_config.target_residual_layer_indices.len(), 3);
}

#[test]
fn test_rejects_empty_target_residual_layers() {
    let mut config = dspark_config();
    config.dflash_config.target_residual_layer_indices.clear();

    let err = config.normalize_and_validate().unwrap_err();

    assert!(err.to_string().contains("target_residual_layer_indices"));
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn test_serializes_upstream_field_names() {
    let config = dspark_config();

    let value = serde_json::to_value(config).unwrap();

    assert_eq!(value["num_hidden_layers"], 5);
    assert_eq!(
        value["dflash_config"]["target_layer_ids"],
        serde_json::json!([1, 16, 31, 46, 61])
    );
    assert!(value.get("num_layers").is_none());
    assert!(value["dflash_config"].get("target_residual_layer_indices").is_none());
}

#[test]
fn test_rejects_unsupported_markov_head() {
    let mut config = dspark_config();
    config.markov_head_type = "causal".to_string();

    let err = config.normalize_and_validate().unwrap_err();

    assert!(err.to_string().contains("markov_head_type"));
}
