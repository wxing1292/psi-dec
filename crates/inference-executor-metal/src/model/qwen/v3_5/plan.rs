use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::mlp::moe::MoEExecutionPolicyConfig;
use inference_executor_core::model::qwen::v3_5::DSparkConfig;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::QuantizationConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::TextConfig;

use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::moe::backend::GatedMoEMetalConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MetalDefaults {
    pub group_size: u32,
    pub bits: u32,
    pub hidden_dtype: Dtype,
    pub gdn_recurrent_v_tile_size: u32,
    pub context_parallel_kv_token_tile_size: u32,
    pub context_parallel_num_threads_per_threadblock: u32,
    pub context_parallel_max_q_head_tile_size: u32,
    pub q_token_tile_size: u32,
    pub tiled_kv_token_tile_size: u32,
    pub moe_execution_policy: MoEExecutionPolicyConfig,
}

impl Default for Qwen35MetalDefaults {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            hidden_dtype: Dtype::Bfloat16,
            gdn_recurrent_v_tile_size: 8,
            context_parallel_kv_token_tile_size: 256,
            context_parallel_num_threads_per_threadblock: 256,
            context_parallel_max_q_head_tile_size: 8,
            // Tiled GQA uses Tq_tile=8 and Tkv_tile=16. Hq_tile is selected
            // dynamically from Q-token-tile utilization: 27B uses 3/6 and 35B
            // uses 4/8 Q heads per fixed KV head.
            q_token_tile_size: 8,
            tiled_kv_token_tile_size: 16,
            moe_execution_policy: MoEExecutionPolicyConfig::default(),
        }
    }
}

impl Qwen35MetalDefaults {
    pub fn from_quantization(quantization: Option<&QuantizationConfig>) -> Result<Self, ModelExecutorError> {
        let mut defaults = Self::default();
        if let Some(quantization) = quantization {
            defaults.group_size = to_u32("qwen3.5 quantization group_size", quantization.group_size)?;
            defaults.bits = to_u32("qwen3.5 quantization bits", quantization.bits)?;
        }
        Ok(defaults)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35DSparkPlan {
    pub block_size: usize,
    pub mask_token_id: usize,
    pub target_residuals: Vec<Qwen35DSparkTargetResidualPlan>,
    pub fc: Qwen35QuantizedLinearPlan,
    pub hidden_norm_eps: f32,
    pub layers: Vec<Qwen35DSparkLayerPlan>,
    pub norm_eps: f32,
    pub markov_w1: Qwen35QuantizedEmbeddingPlan,
    pub markov_w2: Qwen35QuantizedLinearPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkTargetResidualPlan {
    pub model_layer_index: usize,
    pub residual_slice_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35DSparkLayerPlan {
    pub dspark_layer_index: usize,
    pub input_layernorm_eps: f32,
    pub post_attention_layernorm_eps: f32,
    pub attention_core: GQACore,
    pub attention_metal: GQAMetalConfig,
    pub mlp_core: DenseMLPCore,
    pub mlp_metal: DenseMLPMetalConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35QuantizedLinearPlan {
    pub input_dim: usize,
    pub output_dim: usize,
    pub group_size: u32,
    pub bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35QuantizedEmbeddingPlan {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub group_size: u32,
    pub bits: u32,
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
    for layer_index in 0..config.text_config.num_hidden_layers {
        match config.layer_type_at(layer_index)? {
            LayerType::GDN => counts.gdn += 1,
            LayerType::FullAttention => counts.gqa += 1,
        }
        if config.layer_uses_moe(layer_index) {
            counts.has_moe = true;
        } else {
            counts.has_dense_mlp = true;
        }
    }
    Ok(counts)
}

pub fn build_qwen35_dspark_plan(
    target_model_config: &Qwen35ModelConfig,
    mut dspark_config: DSparkConfig,
) -> Result<Qwen35DSparkPlan, ModelExecutorError> {
    dspark_config.normalize_and_validate()?;
    dspark_config.validate_target(&target_model_config.text_config)?;
    let metal_defaults = Qwen35MetalDefaults::from_quantization(dspark_config.quantization.as_ref())?;

    let target_residuals = dspark_config
        .dflash_config
        .target_residual_layer_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(residual_slice_index, model_layer_index)| {
            assert!(
                model_layer_index < target_model_config.text_config.num_hidden_layers,
                "Qwen3.5 DSpark target residual layer index must be in the target model"
            );
            Qwen35DSparkTargetResidualPlan {
                model_layer_index,
                residual_slice_index,
            }
        })
        .collect::<Vec<_>>();

    let mut layers = Vec::with_capacity(dspark_config.num_layers);
    for dspark_layer_index in 0..dspark_config.num_layers {
        let attention_core = GQACore::new(
            dspark_layer_index,
            dspark_config.hidden_size,
            dspark_config.head_dim,
            dspark_config.num_attention_heads,
            dspark_config.num_key_value_heads,
            (dspark_config.head_dim as f32).sqrt().recip(),
        );
        attention_core.validate();
        let attention_metal = GQAMetalConfig {
            group_size: metal_defaults.group_size,
            bits: metal_defaults.bits,
            page_bytes: to_u32("DSpark GQA page_bytes", QWEN35_PAGE_SIZE_BYTES)?,
            context_parallel_kv_token_tile_size: metal_defaults.context_parallel_kv_token_tile_size,
            context_parallel_num_threads_per_threadblock: metal_defaults.context_parallel_num_threads_per_threadblock,
            context_parallel_max_q_head_tile_size: metal_defaults.context_parallel_max_q_head_tile_size,
            q_token_tile_size: metal_defaults.q_token_tile_size,
            tiled_kv_token_tile_size: metal_defaults.tiled_kv_token_tile_size,
            rope_dim: to_u32("DSpark GQA rope_dim", dspark_config.head_dim)?,
            norm_eps: dspark_config.rms_norm_eps,
            rope_theta: dspark_config.rope_theta,
            rope_scale: 1.0,
            dtype: metal_defaults.hidden_dtype,
        };
        attention_metal.validate();
        assert!(attention_metal.num_tokens_per_page(&attention_core) > 0);

        let mlp_core = DenseMLPCore {
            model_layer_index: dspark_layer_index,
            hidden_dim: dspark_config.hidden_size,
            intermediate_dim: dspark_config.intermediate_size,
        };
        mlp_core.validate();
        let mlp_metal = DenseMLPMetalConfig {
            group_size: metal_defaults.group_size,
            bits: metal_defaults.bits,
            dtype: metal_defaults.hidden_dtype,
        };
        mlp_metal.validate();
        layers.push(Qwen35DSparkLayerPlan {
            dspark_layer_index,
            input_layernorm_eps: dspark_config.rms_norm_eps,
            post_attention_layernorm_eps: dspark_config.rms_norm_eps,
            attention_core,
            attention_metal,
            mlp_core,
            mlp_metal,
        });
    }

    let fc_quantization = dspark_quantization_for(&dspark_config, "fc.weight", metal_defaults)?;
    let markov_w1_quantization =
        dspark_quantization_for(&dspark_config, "markov_head.markov_w1.weight", metal_defaults)?;
    let markov_w2_quantization =
        dspark_quantization_for(&dspark_config, "markov_head.markov_w2.weight", metal_defaults)?;
    let selected_hidden_dim = dspark_config
        .hidden_size
        .checked_mul(target_residuals.len())
        .ok_or_else(|| ModelExecutorError::custom("DSpark selected hidden dimension must fit usize"))?;
    Ok(Qwen35DSparkPlan {
        block_size: dspark_config.block_size,
        mask_token_id: dspark_config.dflash_config.mask_token_id,
        target_residuals,
        fc: Qwen35QuantizedLinearPlan {
            input_dim: selected_hidden_dim,
            output_dim: dspark_config.hidden_size,
            group_size: fc_quantization.0,
            bits: fc_quantization.1,
        },
        hidden_norm_eps: dspark_config.rms_norm_eps,
        layers,
        norm_eps: dspark_config.rms_norm_eps,
        markov_w1: Qwen35QuantizedEmbeddingPlan {
            num_embeddings: dspark_config.vocab_size,
            embedding_dim: dspark_config.markov_rank,
            group_size: markov_w1_quantization.0,
            bits: markov_w1_quantization.1,
        },
        markov_w2: Qwen35QuantizedLinearPlan {
            input_dim: dspark_config.markov_rank,
            output_dim: dspark_config.vocab_size,
            group_size: markov_w2_quantization.0,
            bits: markov_w2_quantization.1,
        },
    })
}

fn dspark_quantization_for(
    config: &DSparkConfig,
    tensor_name: &str,
    defaults: Qwen35MetalDefaults,
) -> Result<(u32, u32), ModelExecutorError> {
    let Some(quantization) = &config.quantization else {
        return Ok((defaults.group_size, defaults.bits));
    };
    let resolved = quantization.resolve_for_tensor(tensor_name);
    Ok((
        to_u32("DSpark quantization group_size", resolved.group_size)?,
        to_u32("DSpark quantization bits", resolved.bits)?,
    ))
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
    if mtp.mtp_num_hidden_layers != 1 {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 MTP checkpoint must contain exactly one body layer, got {}",
            mtp.mtp_num_hidden_layers
        )));
    }
    if mtp.mtp_use_dedicated_embeddings {
        return Err(ModelExecutorError::custom(
            "qwen3.5 MTP checkpoint must share the Main token embedding",
        ));
    }
    Ok(())
}

pub fn qwen35_gdn_core_and_metal(
    model_layer_index: usize,
    text: &TextConfig,
    metal_defaults: Qwen35MetalDefaults,
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
        group_size: metal_defaults.group_size,
        bits: metal_defaults.bits,
        recurrent_v_tile_size: metal_defaults.gdn_recurrent_v_tile_size,
        norm_eps: text.rms_norm_eps,
        input_dtype: Dtype::Float32,
        qkvabz_affine_dtype: Dtype::Float32,
        output_affine_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn qwen35_gqa_core_and_metal(
    model_layer_index: usize,
    text: &TextConfig,
    metal_defaults: Qwen35MetalDefaults,
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
        group_size: metal_defaults.group_size,
        bits: metal_defaults.bits,
        page_bytes: to_u32("qwen3.5 GQA page_bytes", QWEN35_PAGE_SIZE_BYTES)?,
        context_parallel_kv_token_tile_size: metal_defaults.context_parallel_kv_token_tile_size,
        context_parallel_num_threads_per_threadblock: metal_defaults.context_parallel_num_threads_per_threadblock,
        context_parallel_max_q_head_tile_size: metal_defaults.context_parallel_max_q_head_tile_size,
        q_token_tile_size: metal_defaults.q_token_tile_size,
        tiled_kv_token_tile_size: metal_defaults.tiled_kv_token_tile_size,
        rope_dim: to_u32("qwen3.5 GQA rope_dim", text.rope_dim)?,
        norm_eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        rope_scale: 1.0,
        dtype: metal_defaults.hidden_dtype,
    };
    metal.validate();
    assert!(metal.num_tokens_per_page(&core) > 0);
    Ok((core, metal))
}

pub fn qwen35_dense_mlp_core_and_metal(
    model_layer_index: usize,
    text: &TextConfig,
    metal_defaults: Qwen35MetalDefaults,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    if text.intermediate_size == 0 {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 layer {model_layer_index} uses dense MLP but intermediate_size is zero"
        )));
    }
    let core = DenseMLPCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.intermediate_size,
    };
    core.validate();
    let metal = DenseMLPMetalConfig {
        group_size: metal_defaults.group_size,
        bits: metal_defaults.bits,
        dtype: metal_defaults.hidden_dtype,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn qwen35_moe_core_and_metal(
    layer_prefix: &str,
    model_layer_index: usize,
    model_config: &Qwen35ModelConfig,
    metal_defaults: Qwen35MetalDefaults,
) -> Result<(GatedMoECore, GatedMoEMetalConfig), ModelExecutorError> {
    let text = &model_config.text_config;
    let core = GatedMoECore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.moe_intermediate_size,
        common_expert_intermediate_dim: (text.shared_expert_intermediate_size > 0)
            .then_some(text.shared_expert_intermediate_size),
        num_experts: text.num_experts,
        num_experts_per_token: text.num_experts_per_tok,
        norm_topk_prob: text.norm_topk_prob,
    };
    core.validate();
    let metal = GatedMoEMetalConfig {
        group_size: metal_defaults.group_size,
        bits: metal_defaults.bits,
        router_bits: quant_bits_for(
            model_config,
            &format!("{layer_prefix}.mlp.gate.weight"),
            metal_defaults.bits,
        )?,
        common_gate_bits: quant_bits_for(
            model_config,
            &format!("{layer_prefix}.mlp.shared_expert_gate.weight"),
            metal_defaults.bits,
        )?,
        dtype: metal_defaults.hidden_dtype,
        execution_policy: metal_defaults.moe_execution_policy,
    };
    metal.validate();
    Ok((core, metal))
}

fn quant_bits_for(
    model_config: &Qwen35ModelConfig,
    tensor_name: &str,
    default_bits: u32,
) -> Result<u32, ModelExecutorError> {
    let bits = model_config
        .quantization
        .as_ref()
        .map(|quantization| quantization.resolve_for_tensor(tensor_name).bits)
        .unwrap_or(default_bits as usize);
    to_u32("qwen3.5 quantization bits", bits)
}

fn to_u32(name: &str, value: usize) -> Result<u32, ModelExecutorError> {
    value
        .try_into()
        .map_err(|_| ModelExecutorError::custom(format!("{name}={value} must fit u32")))
}
