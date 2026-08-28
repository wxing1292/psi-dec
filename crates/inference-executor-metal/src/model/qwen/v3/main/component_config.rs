//! Derives Metal component configuration from a Qwen3 checkpoint configuration.

use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3::Qwen3TextConfig;
use inference_executor_core::model::qwen::v3_x::QuantizationConfig;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::model::qwen::v3_x::weight::to_u32;

#[derive(Clone, Copy)]
pub struct Qwen3MainConfig<'a> {
    pub text: &'a Qwen3TextConfig,
    pub quantization: &'a QuantizationConfig,
    pub page_size_bytes: usize,
}

pub fn derive_qwen3_gqa_configs(
    model_layer_index: usize,
    config: Qwen3MainConfig<'_>,
) -> Result<(UngatedGQACore, GQAMetalConfig), ModelExecutorError> {
    let text = config.text;
    let core = UngatedGQACore::new(
        model_layer_index,
        text.hidden_size,
        text.head_dim,
        text.num_attention_heads,
        text.num_key_value_heads,
        (text.head_dim as f32).sqrt().recip(),
    );
    core.validate();
    let (group_size, bits) = quantization(config.quantization)?;
    let metal = GQAMetalConfig {
        group_size,
        bits,
        page_bytes: to_u32("Qwen3 GQA page_bytes", config.page_size_bytes)?,
        rope_dim: to_u32("Qwen3 GQA rope_dim", text.head_dim)?,
        norm_eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        rope_scaling: RopeScaling::Default,
        io_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    assert!(metal.num_ungated_tokens_per_page(&core) > 0);
    Ok((core, metal))
}

pub fn derive_qwen3_dense_mlp_configs(
    model_layer_index: usize,
    config: Qwen3MainConfig<'_>,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    let text = config.text;
    let core = DenseMLPCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.intermediate_size,
    };
    core.validate();
    let (group_size, bits) = quantization(config.quantization)?;
    let affine = QuantizedAffineLayout {
        group_size,
        bits,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let metal = DenseMLPMetalConfig {
        gate_up: affine,
        down: affine,
        io_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

fn quantization(quantization: &QuantizationConfig) -> Result<(u32, u32), ModelExecutorError> {
    Ok((
        to_u32("Qwen3 quantization group_size", quantization.group_size)?,
        to_u32("Qwen3 quantization bits", quantization.bits)?,
    ))
}
