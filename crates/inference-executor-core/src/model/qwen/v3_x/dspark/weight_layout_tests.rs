use super::Qwen3xDSparkWeightBindings;
use super::resolve_qwen3x_dspark_source_weight_bindings;
use super::resolve_qwen3x_dspark_weight_bindings;
use crate::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;

fn config() -> Qwen3xDSparkConfig {
    Qwen3xDSparkConfig {
        block_size: 5,
        mask_token_id: 15,
        target_layer_ids: vec![0, 3, 7],
        num_target_layers: 8,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        max_position_embeddings: 32,
        vocab_size: 16,
        markov_rank: 4,
        num_anchors: 8,
        enable_confidence_head: true,
        confidence_head_with_markov: true,
        quantization: None,
    }
}

fn required_names() -> Vec<String> {
    Qwen3xDSparkWeightBindings::from_config(&config())
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

#[test]
fn test_builds_qwen3x_component_bindings() {
    let bindings = Qwen3xDSparkWeightBindings::from_config(&config());

    assert_eq!(bindings.layers.len(), 2);
    assert_eq!(bindings.main_feature.fc.weight, "fc.weight");
    assert_eq!(bindings.layers[1].gqa.q.weight, "layers.1.self_attn.q_proj.weight");
    assert_eq!(bindings.layers[1].mlp.down.biases, "layers.1.mlp.down_proj.biases");
    assert_eq!(bindings.markov.w2.scales, "markov_head.markov_w2.scales");
    assert!(bindings.embed.is_none());
    assert!(bindings.unembed.is_none());
    assert_eq!(
        bindings.confidence.as_ref().unwrap().weight,
        "confidence_head.proj.weight"
    );
}

#[test]
fn test_resolves_required_affine_manifest() {
    let names = required_names();

    let actual = resolve_qwen3x_dspark_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap();

    assert_eq!(actual, Qwen3xDSparkWeightBindings::from_config(&config()));
}

#[test]
fn test_resolves_complete_optional_weight_groups() {
    let mut names = required_names();
    names.extend([
        "embed_tokens.weight".to_string(),
        "embed_tokens.scales".to_string(),
        "embed_tokens.biases".to_string(),
        "lm_head.weight".to_string(),
        "lm_head.scales".to_string(),
        "lm_head.biases".to_string(),
    ]);

    let actual = resolve_qwen3x_dspark_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap();

    assert!(actual.embed.is_some());
    assert!(actual.unembed.is_some());
    assert!(actual.confidence.is_some());
}

#[test]
fn test_resolves_official_source_confidence_head() {
    let names = Qwen3xDSparkWeightBindings::from_config(&config())
        .source_tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let actual = resolve_qwen3x_dspark_source_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap();

    assert_eq!(
        actual.confidence.as_ref().unwrap().weight,
        "confidence_head.proj.weight"
    );
    assert_eq!(actual.confidence.as_ref().unwrap().bias, "confidence_head.proj.bias");
}

#[test]
fn test_rejects_missing_confidence_weights_when_enabled() {
    let mut names = required_names();
    names.retain(|name| !name.starts_with("confidence_head."));

    let error = resolve_qwen3x_dspark_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap_err();

    assert!(error.to_string().contains("confidence_head.proj.weight"));
    assert!(error.to_string().contains("official BF16 DSpark checkpoint"));
    assert!(error.to_string().contains("qwen3_dspark_quantize"));
}

#[test]
fn test_rejects_confidence_weights_when_disabled() {
    let mut config = config();
    config.enable_confidence_head = false;
    config.confidence_head_with_markov = false;
    let mut names = Qwen3xDSparkWeightBindings::from_config(&config)
        .tensor_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    names.extend([
        "confidence_head.proj.weight".to_string(),
        "confidence_head.proj.bias".to_string(),
    ]);

    let error = resolve_qwen3x_dspark_weight_bindings(&config, names.iter().map(String::as_str)).unwrap_err();

    assert!(error.to_string().contains("confidence_head.proj.weight"));
}

#[test]
fn test_rejects_partial_optional_groups() {
    let mut names = required_names();
    names.push("embed_tokens.weight".to_string());

    let error = resolve_qwen3x_dspark_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap_err();

    assert!(error.to_string().contains("must be absent or complete"));
}

#[test]
fn test_rejects_missing_or_unexpected_affine_tensors() {
    let mut names = required_names();
    names.retain(|name| name != "layers.1.self_attn.q_proj.scales");
    names.push("layers.1.self_attn.unknown.weight".to_string());

    let error = resolve_qwen3x_dspark_weight_bindings(&config(), names.iter().map(String::as_str)).unwrap_err();

    assert!(error.to_string().contains("layers.1.self_attn.q_proj.scales"));
    assert!(error.to_string().contains("layers.1.self_attn.unknown.weight"));
}
