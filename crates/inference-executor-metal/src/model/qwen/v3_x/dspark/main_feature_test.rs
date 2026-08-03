use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;

use super::Qwen3xDSparkMainFeatureLayout;
use super::Qwen3xDSparkMainResidualBindings;

#[test]
fn test_layout_preserves_selected_decoder_output_order() {
    let config = config();
    let layout = Qwen3xDSparkMainFeatureLayout::new(&config, 128);
    let bindings = Qwen3xDSparkMainResidualBindings::new(&config.target_layer_ids);

    assert_eq!(layout.selected_hidden_dim, 96);
    assert_eq!(layout.capture_columns(0), 0..32);
    assert_eq!(layout.capture_columns(2), 64..96);
    assert_eq!(layout.main_residual_elements(), 12_288);
    assert_eq!(bindings.get(1), Some(0));
    assert_eq!(bindings.get(4), Some(1));
    assert_eq!(bindings.get(6), Some(2));
    assert_eq!(bindings.get(2), None);
}

fn config() -> Qwen3xDSparkConfig {
    Qwen3xDSparkConfig {
        block_size: 7,
        mask_token_id: 1,
        target_layer_ids: vec![1, 4, 6],
        num_target_layers: 8,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 8,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        max_position_embeddings: 32,
        vocab_size: 64,
        markov_rank: 8,
        num_anchors: 8,
        quantization: None,
    }
}
