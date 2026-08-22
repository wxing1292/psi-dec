use serde_json::Value;

use super::Qwen3xDSparkCheckpointConfig;
use super::Qwen3xDSparkMainConfig;
use super::Qwen3xDSparkRopeScaling;
use super::adapt_checkpoint_config;
use super::normalize_qwen3x_dspark_config;

fn canonical_config() -> Value {
    serde_json::json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "block_size": 7,
        "confidence_head_with_markov": true,
        "dtype": "bfloat16",
        "enable_confidence_head": true,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": [
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ],
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 151669,
        "max_position_embeddings": 40960,
        "model_type": "qwen3",
        "num_attention_heads": 40,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 40,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "rope_type": "default"
        },
        "sliding_window": null,
        "target_layer_ids": [1, 10, 19, 28, 37],
        "tie_word_embeddings": false,
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    })
}

fn parse(value: Value) -> Result<super::Qwen3xDSparkConfig, crate::def::ModelExecutorError> {
    let value = adapt_checkpoint_config(value)?;
    let checkpoint = serde_json::from_value::<Qwen3xDSparkCheckpointConfig>(value).map_err(|error| {
        crate::def::ModelExecutorError::custom(format!("unable to parse canonical test config: {error}"))
    })?;
    normalize_qwen3x_dspark_config(checkpoint)
}

#[test]
fn test_parses_canonical_config() {
    let config = parse(canonical_config()).unwrap();

    assert_eq!(config.block_size, 7);
    assert_eq!(config.target_layer_ids, [1, 10, 19, 28, 37]);
    assert_eq!(config.num_hidden_layers, 5);
    assert_eq!(config.num_attention_heads, 40);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.markov_rank, 256);
    assert_eq!(config.quantization.unwrap().bits, 4);
}

#[test]
fn test_requires_registered_checkpoint_adapter() {
    let mut value = canonical_config();
    value["architectures"] = serde_json::json!(["UnknownDSparkModel"]);

    let error = parse(value).unwrap_err();

    assert!(error.to_string().contains("no config adapter is registered"));
}

#[test]
fn test_adapts_qwen38_draft_config() {
    let mut value = canonical_config();
    value["architectures"] = serde_json::json!(["DSparkDraftModel"]);
    value["max_position_embeddings"] = serde_json::json!(262144);
    value["num_target_layers"] = serde_json::json!(64);
    value["vocab_size"] = serde_json::json!(248320);
    value.as_object_mut().unwrap().remove("mask_token_id");
    value.as_object_mut().unwrap().remove("target_layer_ids");
    value["dflash_config"] = serde_json::json!({
        "attention_mode": "gqa",
        "confidence_head_with_markov": true,
        "enable_confidence_head": true,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 248077,
        "projector_type": "dspark",
        "target_layer_ids": [4, 16, 28, 40, 52]
    });
    value["rope_parameters"] = serde_json::json!({
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 8192,
        "rope_theta": 10000000.0,
        "rope_type": "yarn"
    });

    let config = parse(value).unwrap();

    assert_eq!(config.mask_token_id, 248077);
    assert_eq!(config.target_layer_ids, [4, 16, 28, 40, 52]);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert_eq!(
        config.rope_scaling,
        Qwen3xDSparkRopeScaling::Yarn {
            factor: 32.0,
            attention_factor: 1.0 + 0.1 * 32.0_f32.ln(),
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 8192,
            truncate: true,
        }
    );
}

#[test]
fn test_requires_markov_confidence_head() {
    let mut disabled = canonical_config();
    disabled["enable_confidence_head"] = serde_json::json!(false);
    assert!(
        parse(disabled)
            .unwrap_err()
            .to_string()
            .contains("enable_confidence_head=true")
    );

    let mut without_markov = canonical_config();
    without_markov["confidence_head_with_markov"] = serde_json::json!(false);
    assert!(
        parse(without_markov)
            .unwrap_err()
            .to_string()
            .contains("confidence_head_with_markov=true")
    );
}

#[test]
fn test_rejects_nonincreasing_target_layer_ids() {
    let mut value = canonical_config();
    value["target_layer_ids"] = serde_json::json!([1, 10, 10, 28, 37]);

    let error = parse(value).unwrap_err();

    assert!(error.to_string().contains("strictly increasing"));
}

#[test]
fn test_rejects_target_layer_id_outside_main_model() {
    let mut value = canonical_config();
    value["target_layer_ids"] = serde_json::json!([1, 10, 19, 28, 40]);

    let error = parse(value).unwrap_err();

    assert!(error.to_string().contains("must be below unsupported final layer 39"));
}

#[test]
fn test_rejects_final_main_decoder_layer() {
    let mut value = canonical_config();
    value["target_layer_ids"] = serde_json::json!([1, 10, 19, 28, 39]);

    let error = parse(value).unwrap_err();

    assert!(error.to_string().contains("must be below unsupported final layer 39"));
}

#[test]
fn test_rejects_gated_or_non_full_attention_config() {
    let mut gated = canonical_config();
    gated["attention_bias"] = serde_json::json!(true);
    assert!(parse(gated).unwrap_err().to_string().contains("attention_bias"));

    let mut sliding = canonical_config();
    sliding["layer_types"][2] = serde_json::json!("sliding_attention");
    assert!(parse(sliding).unwrap_err().to_string().contains("layer_types"));
}

#[test]
fn test_rejects_invalid_attention_geometry() {
    let mut value = canonical_config();
    value["num_key_value_heads"] = serde_json::json!(6);

    let error = parse(value).unwrap_err();

    assert!(error.to_string().contains("must be divisible"));
}

#[test]
fn test_accepts_query_width_distinct_from_hidden_width() {
    let mut value = canonical_config();
    value["num_attention_heads"] = serde_json::json!(32);

    let config = parse(value).unwrap();

    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.num_attention_heads * config.head_dim, 4096);
}

#[test]
fn test_validates_compatible_main_contract() {
    let config = parse(canonical_config()).unwrap();

    config
        .validate_main(Qwen3xDSparkMainConfig {
            hidden_size: 5120,
            num_hidden_layers: 40,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rope_theta: 1_000_000.0,
        })
        .unwrap();
}

#[test]
fn test_reports_all_main_contract_mismatches() {
    let config = parse(canonical_config()).unwrap();

    let error = config
        .validate_main(Qwen3xDSparkMainConfig {
            hidden_size: 4096,
            num_hidden_layers: 32,
            vocab_size: 128000,
            max_position_embeddings: 32768,
            rope_theta: 10_000.0,
        })
        .unwrap_err()
        .to_string();

    for field in [
        "hidden_size",
        "num_target_layers",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
    ] {
        assert!(error.contains(field), "missing mismatch for {field}: {error}");
    }
}
