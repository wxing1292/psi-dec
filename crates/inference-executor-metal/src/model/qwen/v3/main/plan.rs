use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::model::qwen::v3_x::weight::to_u32;

pub fn qwen3_gqa_core_and_metal(
    model_layer_index: usize,
    config: &Qwen3ModelConfig,
) -> Result<(UngatedGQACore, GQAMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let core = UngatedGQACore::new(
        model_layer_index,
        text.hidden_size,
        text.head_dim,
        text.num_attention_heads,
        text.num_key_value_heads,
        (text.head_dim as f32).sqrt().recip(),
    );
    core.validate();
    let (group_size, bits) = quantization(config)?;
    let metal = GQAMetalConfig {
        group_size,
        bits,
        page_bytes: to_u32("Qwen3 GQA page_bytes", QWEN3_PAGE_SIZE_BYTES)?,
        single_q_token_kv_token_tile_size: 128,
        single_q_token_num_threads_per_threadblock: 128,
        single_q_token_max_q_head_tile_size: 5,
        tiled_q_token_tile_size: 8,
        tiled_kv_token_tile_size: 16,
        rope_dim: to_u32("Qwen3 GQA rope_dim", text.head_dim)?,
        norm_eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        rope_scale: 1.0,
        dtype: Dtype::Bfloat16,
    };
    metal.validate();
    assert!(metal.num_ungated_tokens_per_page(&core) > 0);
    Ok((core, metal))
}

pub fn qwen3_dense_mlp_core_and_metal(
    model_layer_index: usize,
    config: &Qwen3ModelConfig,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let core = DenseMLPCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.intermediate_size,
    };
    core.validate();
    let (group_size, bits) = quantization(config)?;
    let metal = DenseMLPMetalConfig {
        group_size,
        bits,
        dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

fn quantization(config: &Qwen3ModelConfig) -> Result<(u32, u32), ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3 Metal executor requires quantization config"))?;
    Ok((
        to_u32("Qwen3 quantization group_size", quantization.group_size)?,
        to_u32("Qwen3 quantization bits", quantization.bits)?,
    ))
}
