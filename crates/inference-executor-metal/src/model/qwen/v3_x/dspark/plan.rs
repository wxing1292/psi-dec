use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::ResolvedQuantizationConfig;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::model::qwen::v3_x::weight::to_u32;

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3xDSparkPlan {
    pub block_size: usize,
    pub mask_token_id: usize,
    pub main_residuals: Vec<Qwen3xDSparkMainResidualPlan>,
    pub embedding: Qwen3xDSparkQuantizedEmbeddingPlan,
    pub fc: Qwen3xDSparkQuantizedLinearPlan,
    pub hidden_norm_eps: f32,
    pub layers: Vec<Qwen3xDSparkLayerPlan>,
    pub norm_eps: f32,
    pub unembed: Qwen3xDSparkQuantizedLinearPlan,
    pub markov_w1: Qwen3xDSparkQuantizedEmbeddingPlan,
    pub markov_w2: Qwen3xDSparkQuantizedLinearPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkMainResidualPlan {
    pub model_layer_index: usize,
    pub residual_slice_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3xDSparkLayerPlan {
    pub dspark_layer_index: usize,
    pub input_norm_eps: f32,
    pub post_attention_norm_eps: f32,
    pub attention_core: UngatedDSparkGQACore,
    pub attention_metal: GQAMetalConfig,
    pub mlp_core: DenseMLPCore,
    pub mlp_metal: DenseMLPMetalConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkQuantizedLinearPlan {
    pub input_dim: usize,
    pub output_dim: usize,
    pub group_size: u32,
    pub bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDSparkQuantizedEmbeddingPlan {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub group_size: u32,
    pub bits: u32,
}

pub fn build_qwen3x_dspark_plan(
    config: &Qwen3xDSparkConfig,
    page_bytes: usize,
) -> Result<Qwen3xDSparkPlan, ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark Metal executor requires quantization config"))?;
    let main_residuals = config
        .target_layer_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(residual_slice_index, model_layer_index)| {
            Qwen3xDSparkMainResidualPlan {
                model_layer_index,
                residual_slice_index,
            }
        })
        .collect::<Vec<_>>();

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for dspark_layer_index in 0..config.num_hidden_layers {
        let layer_prefix = format!("layers.{dspark_layer_index}");
        let attention_quantization = uniform_quantization(
            quantization,
            &[
                format!("{layer_prefix}.self_attn.q_proj.weight"),
                format!("{layer_prefix}.self_attn.k_proj.weight"),
                format!("{layer_prefix}.self_attn.v_proj.weight"),
                format!("{layer_prefix}.self_attn.o_proj.weight"),
            ],
            "Qwen3 DSpark fused attention",
        )?;
        let attention = UngatedGQACore::new(
            dspark_layer_index,
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            (config.head_dim as f32).sqrt().recip(),
        );
        let attention_core = UngatedDSparkGQACore::new(attention, config.block_size);
        let attention_metal = GQAMetalConfig {
            group_size: to_u32("Qwen3 DSpark attention group_size", attention_quantization.group_size)?,
            bits: to_u32("Qwen3 DSpark attention bits", attention_quantization.bits)?,
            page_bytes: to_u32("Qwen3 DSpark GQA page_bytes", page_bytes)?,
            rope_dim: to_u32("Qwen3 DSpark rope_dim", config.head_dim)?,
            norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            rope_scale: 1.0,
            io_dtype: Dtype::Bfloat16,
        };
        attention_metal.validate();
        assert!(
            attention_metal.num_ungated_tokens_per_page(&attention_core.attention) > 0,
            "Qwen3 DSpark attention geometry must fit one cache page"
        );

        let mlp_quantization = uniform_quantization(
            quantization,
            &[
                format!("{layer_prefix}.mlp.gate_proj.weight"),
                format!("{layer_prefix}.mlp.up_proj.weight"),
                format!("{layer_prefix}.mlp.down_proj.weight"),
            ],
            "Qwen3 DSpark fused MLP",
        )?;
        let mlp_core = DenseMLPCore {
            model_layer_index: dspark_layer_index,
            hidden_dim: config.hidden_size,
            intermediate_dim: config.intermediate_size,
        };
        mlp_core.validate();
        let mlp_metal = DenseMLPMetalConfig {
            group_size: to_u32("Qwen3 DSpark MLP group_size", mlp_quantization.group_size)?,
            bits: to_u32("Qwen3 DSpark MLP bits", mlp_quantization.bits)?,
            io_dtype: Dtype::Bfloat16,
        };
        mlp_metal.validate();
        layers.push(Qwen3xDSparkLayerPlan {
            dspark_layer_index,
            input_norm_eps: config.rms_norm_eps,
            post_attention_norm_eps: config.rms_norm_eps,
            attention_core,
            attention_metal,
            mlp_core,
            mlp_metal,
        });
    }

    let embedding_quantization = quantization.resolve_for_tensor("embed_tokens.weight");
    let fc_quantization = quantization.resolve_for_tensor("fc.weight");
    let unembed_quantization = quantization.resolve_for_tensor("lm_head.weight");
    let markov_w1_quantization = quantization.resolve_for_tensor("markov_head.markov_w1.weight");
    let markov_w2_quantization = quantization.resolve_for_tensor("markov_head.markov_w2.weight");
    Ok(Qwen3xDSparkPlan {
        block_size: config.block_size,
        mask_token_id: config.mask_token_id,
        main_residuals,
        embedding: embedding_plan(config.vocab_size, config.hidden_size, embedding_quantization)?,
        fc: linear_plan(
            config
                .hidden_size
                .checked_mul(config.target_layer_ids.len())
                .ok_or_else(|| ModelExecutorError::custom("Qwen3 DSpark Main feature width must fit usize"))?,
            config.hidden_size,
            fc_quantization,
        )?,
        hidden_norm_eps: config.rms_norm_eps,
        layers,
        norm_eps: config.rms_norm_eps,
        unembed: linear_plan(config.hidden_size, config.vocab_size, unembed_quantization)?,
        markov_w1: embedding_plan(config.vocab_size, config.markov_rank, markov_w1_quantization)?,
        markov_w2: linear_plan(config.markov_rank, config.vocab_size, markov_w2_quantization)?,
    })
}

fn uniform_quantization(
    config: &inference_executor_core::model::qwen::v3_x::QuantizationConfig,
    tensor_names: &[String],
    component: &str,
) -> Result<ResolvedQuantizationConfig, ModelExecutorError> {
    let first_name = tensor_names
        .first()
        .expect("uniform quantization requires tensor names");
    let first = config.resolve_for_tensor(first_name);
    if let Some((tensor_name, resolved)) = tensor_names.iter().skip(1).find_map(|tensor_name| {
        let resolved = config.resolve_for_tensor(tensor_name);
        (resolved != first).then_some((tensor_name, resolved))
    }) {
        return Err(ModelExecutorError::custom(format!(
            "{component} requires one affine layout, but {first_name:?} uses group_size={} bits={} and \
             {tensor_name:?} uses group_size={} bits={}",
            first.group_size, first.bits, resolved.group_size, resolved.bits
        )));
    }
    Ok(first)
}

fn linear_plan(
    input_dim: usize,
    output_dim: usize,
    quantization: ResolvedQuantizationConfig,
) -> Result<Qwen3xDSparkQuantizedLinearPlan, ModelExecutorError> {
    Ok(Qwen3xDSparkQuantizedLinearPlan {
        input_dim,
        output_dim,
        group_size: to_u32("Qwen3 DSpark linear group_size", quantization.group_size)?,
        bits: to_u32("Qwen3 DSpark linear bits", quantization.bits)?,
    })
}

fn embedding_plan(
    num_embeddings: usize,
    embedding_dim: usize,
    quantization: ResolvedQuantizationConfig,
) -> Result<Qwen3xDSparkQuantizedEmbeddingPlan, ModelExecutorError> {
    Ok(Qwen3xDSparkQuantizedEmbeddingPlan {
        num_embeddings,
        embedding_dim,
        group_size: to_u32("Qwen3 DSpark embedding group_size", quantization.group_size)?,
        bits: to_u32("Qwen3 DSpark embedding bits", quantization.bits)?,
    })
}

#[cfg(test)]
mod tests {
    use inference_executor_core::model::qwen::v3_x::QuantizationConfig;

    use super::*;

    fn config() -> Qwen3xDSparkConfig {
        Qwen3xDSparkConfig {
            block_size: 7,
            mask_token_id: 15,
            target_layer_ids: vec![1, 4],
            num_target_layers: 8,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            vocab_size: 64,
            markov_rank: 8,
            num_anchors: 8,
            enable_confidence_head: true,
            confidence_head_with_markov: true,
            quantization: Some(QuantizationConfig {
                group_size: 32,
                bits: 4,
                mode: None,
                tensor_overrides: Default::default(),
            }),
        }
    }

    #[test]
    fn test_plan_uses_official_block_and_selected_main_outputs() {
        let plan = build_qwen3x_dspark_plan(&config(), 32 * 1024).unwrap();

        assert_eq!(plan.block_size, 7);
        assert_eq!(plan.main_residuals.len(), 2);
        assert_eq!(plan.main_residuals[1].model_layer_index, 4);
        assert_eq!(plan.fc.input_dim, 64);
        assert_eq!(plan.fc.output_dim, 32);
        assert_eq!(plan.layers.len(), 2);
        assert_eq!(plan.layers[0].attention_core.block_size, 7);
        assert_eq!(plan.markov_w1.embedding_dim, 8);
    }

    #[test]
    fn test_plan_rejects_mixed_fused_attention_quantization() {
        let mut config = config();
        config.quantization.as_mut().unwrap().tensor_overrides.insert(
            "layers.0.self_attn.k_proj.weight".to_string(),
            inference_executor_core::model::qwen::v3_x::TensorQuantizationOverride {
                group_size: Some(32),
                bits: Some(8),
                mode: None,
            },
        );

        let error = build_qwen3x_dspark_plan(&config, 32 * 1024).unwrap_err();

        assert!(error.to_string().contains("requires one affine layout"));
    }
}
