use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::mlp::dense::DenseMLPCore;

use super::Qwen3xDSparkMainFeatureLayout;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkLayerPlan;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkMainResidualPlan;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkPlan;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkQuantizedEmbeddingPlan;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkQuantizedLinearPlan;

#[test]
fn test_layout_preserves_selected_decoder_output_order() {
    let layout = Qwen3xDSparkMainFeatureLayout::new(&new_plan(), 128);

    assert_eq!(layout.selected_hidden_dim, 96);
    assert_eq!(layout.capture_columns(0), 0..32);
    assert_eq!(layout.capture_columns(2), 64..96);
    assert_eq!(layout.main_residual_elements(), 12_288);
}

fn new_plan() -> Qwen3xDSparkPlan {
    let attention = UngatedGQACore::new(0, 32, 8, 4, 1, 8.0_f32.sqrt().recip());
    Qwen3xDSparkPlan {
        block_size: 7,
        mask_token_id: 1,
        main_residuals: [1, 4, 6]
            .into_iter()
            .enumerate()
            .map(|(residual_slice_index, model_layer_index)| {
                Qwen3xDSparkMainResidualPlan {
                    model_layer_index,
                    residual_slice_index,
                }
            })
            .collect(),
        embedding: Qwen3xDSparkQuantizedEmbeddingPlan {
            num_embeddings: 64,
            embedding_dim: 32,
            group_size: 32,
            bits: 4,
        },
        fc: Qwen3xDSparkQuantizedLinearPlan {
            input_dim: 96,
            output_dim: 32,
            group_size: 32,
            bits: 4,
        },
        hidden_norm_eps: 1e-6,
        layers: vec![Qwen3xDSparkLayerPlan {
            dspark_layer_index: 0,
            input_norm_eps: 1e-6,
            post_attention_norm_eps: 1e-6,
            attention_core: UngatedDSparkGQACore::new(attention, 7),
            attention_metal: GQAMetalConfig {
                group_size: 32,
                bits: 4,
                page_bytes: 32 * 1024,
                rope_dim: 8,
                norm_eps: 1e-6,
                rope_theta: 10_000.0,
                rope_scale: 1.0,
                io_dtype: Dtype::Bfloat16,
            },
            mlp_core: DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: 32,
                intermediate_dim: 64,
            },
            mlp_metal: DenseMLPMetalConfig {
                group_size: 32,
                bits: 4,
                io_dtype: Dtype::Bfloat16,
            },
        }],
        norm_eps: 1e-6,
        unembed: Qwen3xDSparkQuantizedLinearPlan {
            input_dim: 32,
            output_dim: 64,
            group_size: 32,
            bits: 4,
        },
        markov_w1: Qwen3xDSparkQuantizedEmbeddingPlan {
            num_embeddings: 64,
            embedding_dim: 8,
            group_size: 32,
            bits: 4,
        },
        markov_w2: Qwen3xDSparkQuantizedLinearPlan {
            input_dim: 8,
            output_dim: 64,
            group_size: 8,
            bits: 4,
        },
    }
}
