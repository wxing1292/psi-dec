use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35TextConfig;
use inference_executor_core::model::qwen::v3_x::QuantizationConfig;

use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::moe::backend::GatedMoEMetalConfig;
use crate::model::qwen::v3_x::weight::to_u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MetalDefaults {
    pub group_size: u32,
    pub bits: u32,
    pub hidden_dtype: Dtype,
}

impl Default for Qwen35MetalDefaults {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            hidden_dtype: Dtype::Bfloat16,
        }
    }
}

impl Qwen35MetalDefaults {
    pub fn from_quantization(quantization: Option<&QuantizationConfig>) -> Result<Self, ModelExecutorError> {
        let mut defaults = Self::default();
        if let Some(quantization) = quantization {
            defaults.group_size = to_u32("Qwen3.5 quantization group_size", quantization.group_size)?;
            defaults.bits = to_u32("Qwen3.5 quantization bits", quantization.bits)?;
        }
        Ok(defaults)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35LayerCounts {
    pub gqa: usize,
    pub gdn: usize,
    pub has_dense_mlp: bool,
    pub has_moe: bool,
}

pub fn qwen35_layer_counts(config: &Qwen35ModelConfig) -> Result<Qwen35LayerCounts, ModelExecutorError> {
    let mut counts = Qwen35LayerCounts {
        gqa: 0,
        gdn: 0,
        has_dense_mlp: false,
        has_moe: false,
    };
    for model_layer_index in 0..config.text_config.num_hidden_layers {
        match config.layer_type_at(model_layer_index)? {
            LayerType::GDN => counts.gdn += 1,
            LayerType::FullAttention => counts.gqa += 1,
        }
        if config.layer_uses_moe(model_layer_index) {
            counts.has_moe = true;
        } else {
            counts.has_dense_mlp = true;
        }
    }
    Ok(counts)
}

pub fn qwen35_gdn_core_and_metal(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GDNCore, GDNMetalConfig), ModelExecutorError> {
    let core = GDNCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        num_qk_heads: text.linear_num_key_heads,
        qk_head_dim: text.linear_key_head_dim,
        num_v_heads: text.linear_num_value_heads,
        v_head_dim: text.linear_value_head_dim,
        conv_kernel_size: text.linear_conv_kernel_dim,
        q_scale: (text.linear_key_head_dim as f32).sqrt().recip(),
    };
    core.validate();
    let metal = GDNMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        norm_eps: text.rms_norm_eps,
        input_dtype: defaults.hidden_dtype,
        output_dtype: defaults.hidden_dtype,
        qkvabz_scale_bias_dtype: Dtype::Bfloat16,
        output_scale_bias_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn qwen35_gqa_core_and_metal(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GQACore, GQAMetalConfig), ModelExecutorError> {
    let core = GQACore::new(
        model_layer_index,
        text.hidden_size,
        text.head_dim,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.scale,
    );
    core.validate();
    let metal = GQAMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        page_bytes: to_u32("Qwen3.5 GQA page_bytes", QWEN35_PAGE_SIZE_BYTES)?,
        rope_dim: to_u32("Qwen3.5 GQA rope_dim", text.rope_dim)?,
        norm_eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        rope_scale: 1.0,
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    assert!(metal.num_tokens_per_page(&core) > 0);
    Ok((core, metal))
}

pub fn qwen35_dense_mlp_core_and_metal(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    if text.intermediate_size == 0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3.5 layer {model_layer_index} uses dense MLP but intermediate_size is zero"
        )));
    }
    let core = DenseMLPCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.intermediate_size,
    };
    core.validate();
    let metal = DenseMLPMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn qwen35_moe_core_and_metal(
    layer_prefix: &str,
    model_layer_index: usize,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GatedMoECore, GatedMoEMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let core = GatedMoECore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.moe_intermediate_size,
        shared_experts_intermediate_dim: (text.shared_expert_intermediate_size > 0)
            .then_some(text.shared_expert_intermediate_size),
        num_experts: text.num_experts,
        num_experts_per_token: text.num_experts_per_tok,
        norm_topk_prob: text.norm_topk_prob,
    };
    core.validate();
    let metal = GatedMoEMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        router_bits: quant_bits_for(config, &format!("{layer_prefix}.mlp.gate.weight"), defaults.bits)?,
        shared_expert_gate_bits: quant_bits_for(
            config,
            &format!("{layer_prefix}.mlp.shared_expert_gate.weight"),
            defaults.bits,
        )?,
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn validate_qwen35_mtp_config(
    main_model_config: &Qwen35ModelConfig,
    mtp_model_config: &Qwen35ModelConfig,
) -> Result<(), ModelExecutorError> {
    let main = &main_model_config.text_config;
    let mtp = &mtp_model_config.text_config;
    if main.hidden_size != mtp.hidden_size
        || main.num_attention_heads != mtp.num_attention_heads
        || main.num_key_value_heads != mtp.num_key_value_heads
        || main.head_dim != mtp.head_dim
        || main.num_experts != mtp.num_experts
    {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 MTP config must match main model dimensions: main hidden={} q_heads={} kv_heads={} head_dim={} \
             experts={} mtp hidden={} q_heads={} kv_heads={} head_dim={} experts={}",
            main.hidden_size,
            main.num_attention_heads,
            main.num_key_value_heads,
            main.head_dim,
            main.num_experts,
            mtp.hidden_size,
            mtp.num_attention_heads,
            mtp.num_key_value_heads,
            mtp.head_dim,
            mtp.num_experts
        )));
    }
    assert_eq!(
        mtp.mtp_num_hidden_layers, 1,
        "qwen3.5 MTP checkpoint must contain exactly one physical body layer"
    );
    if mtp.mtp_use_dedicated_embeddings {
        return Err(ModelExecutorError::custom(
            "qwen3.5 MTP checkpoint must share the Main token embedding",
        ));
    }
    Ok(())
}

fn quant_bits_for(config: &Qwen35ModelConfig, tensor_name: &str, default_bits: u32) -> Result<u32, ModelExecutorError> {
    let bits = config
        .quantization
        .as_ref()
        .map(|quantization| quantization.resolve_for_tensor(tensor_name).bits)
        .unwrap_or(default_bits as usize);
    to_u32("Qwen3.5 quantization bits", bits)
}
