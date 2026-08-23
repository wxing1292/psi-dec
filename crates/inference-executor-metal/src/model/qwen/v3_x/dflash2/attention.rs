use std::rc::Rc;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BlockSpecGQACore;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::block_spec::backend::BlockSpecGQA;
use crate::attn::block_spec::backend::BlockSpecGQAInput;
use crate::attn::block_spec::backend::BlockSpecGQAMetalConfig;
use crate::attn::block_spec::context::BlockSpecGQAContextAppender;
use crate::attn::block_spec::context::BlockSpecGQAContextInput;
use crate::attn::block_spec::context::BlockSpecGQAContextScratch;
use crate::attn::block_spec::metadata::BlockSpecGQAMetadataBuffers;
use crate::attn::block_spec::scratch::BlockSpecScratch;
use crate::attn::block_spec::state::BlockSpecGQAState;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::layer::Qwen3xBlockSpecGQAWeightBuffers;
use crate::model::qwen::v3_x::weight::to_u32;

pub struct Qwen3xDFlash2Attention {
    dflash2_layer_index: u32,
    core: BlockSpecGQACore,
    metal: BlockSpecGQAMetalConfig,
    weights: Option<Qwen3xBlockSpecGQAWeightBuffers>,
    backend: BlockSpecGQA,
    context_appender: BlockSpecGQAContextAppender,
    block_scratch: Option<Rc<BlockSpecScratch>>,
    context_scratch: Option<Rc<BlockSpecGQAContextScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
}

impl Qwen3xDFlash2Attention {
    pub fn new(
        device: &Device,
        core: BlockSpecGQACore,
        metal: BlockSpecGQAMetalConfig,
        dflash2_layer_index: usize,
        state: &BlockSpecGQAState,
    ) -> Self {
        Self {
            dflash2_layer_index: dflash2_layer_index
                .try_into()
                .expect("Qwen3 DFlash2 layer index must fit u32"),
            core: core.clone(),
            metal,
            weights: None,
            backend: state.new_gqa(device, core.clone(), metal),
            context_appender: BlockSpecGQAContextAppender::new(device, core, metal),
            block_scratch: Some(state.block_scratch()),
            context_scratch: Some(state.context_scratch()),
            request_page_table: Some(state.request_page_table()),
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xGQAWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3.x DFlash2 attention weights are already loaded"
        );
        self.weights = Some(Qwen3xBlockSpecGQAWeightBuffers::load(
            device,
            store,
            &bindings,
            &self.core.attention,
            self.metal,
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(
            self.weights.is_some(),
            "Qwen3.x DFlash2 attention weights are not loaded"
        );
        self.weights.take();
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.block_scratch.is_some() && self.context_scratch.is_some() && self.request_page_table.is_some(),
            "Qwen3.x DFlash2 attention state is not loaded"
        );
        self.request_page_table.take();
        self.context_scratch.take();
        self.block_scratch.take();
    }

    pub fn load_state(&mut self, state: &BlockSpecGQAState) {
        assert!(
            self.block_scratch.is_none() && self.context_scratch.is_none() && self.request_page_table.is_none(),
            "Qwen3.x DFlash2 attention state is already loaded"
        );
        self.block_scratch = Some(state.block_scratch());
        self.context_scratch = Some(state.context_scratch());
        self.request_page_table = Some(state.request_page_table());
    }

    fn weights(&self) -> &Qwen3xBlockSpecGQAWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3.x DFlash2 attention weights must be loaded before execution")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_prefill<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: inference_backend_metal::metal::ReplayU32,
        main_feature: &'a Buffer,
        req_slots: &'a Buffer,
        flat_token_indices: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.context_appender.record(
            recorder,
            BlockSpecGQAContextInput {
                num_total_tokens,
                num_active_tokens,
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: self.dflash2_layer_index,
                main_feature,
                req_slots,
                flat_token_indices,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.context_scratch().bindings(),
            },
        );
    }

    pub fn record_block<'a, R>(
        &'a self,
        recorder: &mut R,
        num_active_tokens: inference_backend_metal::metal::ReplayU32,
        metadata: &'a BlockSpecGQAMetadataBuffers,
        hidden_input: &'a Buffer,
        hidden_output: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <BlockSpecGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            BlockSpecGQAInput {
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: self.dflash2_layer_index,
                metadata,
                hidden_state: hidden_input,
                next_hidden_state: hidden_output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.block_scratch().bindings(),
                num_active_tokens,
            },
        );
    }

    fn block_scratch(&self) -> &BlockSpecScratch {
        self.block_scratch
            .as_deref()
            .expect("Qwen3.x DFlash2 block scratch must be loaded before execution")
    }

    fn context_scratch(&self) -> &BlockSpecGQAContextScratch {
        self.context_scratch
            .as_deref()
            .expect("Qwen3.x DFlash2 context scratch must be loaded before execution")
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("Qwen3.x DFlash2 request page-table state must be loaded before execution")
    }
}

pub fn derive_qwen3x_dflash2_gqa_configs(
    config: &Qwen3xDFlash2Config,
    num_spec_tokens: usize,
    dflash2_layer_index: usize,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
    scale_bias_dtype: Dtype,
) -> Result<(BlockSpecGQACore, BlockSpecGQAMetalConfig), ModelExecutorError> {
    let core = qwen3x_dflash2_gqa_core(config, num_spec_tokens, dflash2_layer_index);
    let metal = qwen3x_dflash2_gqa_metal_config(config, bindings, page_bytes, scale_bias_dtype)?;
    Ok((core, metal))
}

pub fn qwen3x_dflash2_gqa_core(
    config: &Qwen3xDFlash2Config,
    num_spec_tokens: usize,
    dflash2_layer_index: usize,
) -> BlockSpecGQACore {
    assert!(
        num_spec_tokens > 0,
        "Qwen3x DFlash2 attention requires speculative tokens"
    );
    assert_eq!(
        num_spec_tokens,
        config.num_spec_tokens().get(),
        "Qwen3x DFlash2 proposal count must match the checkpoint"
    );
    let num_query_rows = config.block_size;
    assert!(
        dflash2_layer_index < config.num_hidden_layers,
        "Qwen3x DFlash2 attention layer index must be within the model"
    );
    let attention = UngatedGQACore::new(
        dflash2_layer_index,
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        (config.head_dim as f32).sqrt().recip(),
    );
    attention.validate();
    BlockSpecGQACore::new(attention, num_query_rows)
}

pub fn qwen3x_dflash2_gqa_sdpa_config(
    config: &Qwen3xDFlash2Config,
    page_bytes: usize,
) -> Result<backend_sdpa::Config, ModelExecutorError> {
    let num_q_heads = to_u32("Qwen3x DFlash2 GQA Q-head count", config.num_attention_heads)?;
    let num_kv_heads = to_u32("Qwen3x DFlash2 GQA KV-head count", config.num_key_value_heads)?;
    let head_dim = to_u32("Qwen3x DFlash2 GQA head_dim", config.head_dim)?;
    let page_bytes = to_u32("Qwen3x DFlash2 GQA page_bytes", page_bytes)?;
    let kv_bytes_per_token = num_kv_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 GQA KV bytes per token must fit u32"))?;
    if !page_bytes.is_multiple_of(kv_bytes_per_token) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DFlash2 GQA page bytes must contain whole KV tokens",
        ));
    }
    let sdpa_config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads,
        num_kv_heads,
        head_dim,
        tokens_per_page: page_bytes / kv_bytes_per_token,
    };
    sdpa_config.validate();
    Ok(sdpa_config)
}

fn qwen3x_dflash2_gqa_metal_config(
    config: &Qwen3xDFlash2Config,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
    scale_bias_dtype: Dtype,
) -> Result<BlockSpecGQAMetalConfig, ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 Metal executor requires quantization config"))?;
    let affine_layout = |tensor_name: &str| -> Result<QuantizedAffineLayout, ModelExecutorError> {
        let resolved = quantization.resolve_for_tensor(tensor_name);
        if !matches!(resolved.mode.as_deref(), None | Some("affine")) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 GQA tensor {tensor_name:?} requires affine quantization, got mode={:?}",
                resolved.mode
            )));
        }
        Ok(QuantizedAffineLayout {
            group_size: to_u32("Qwen3x DFlash2 GQA group_size", resolved.group_size)?,
            bits: to_u32("Qwen3x DFlash2 GQA bits", resolved.bits)?,
            scale_bias_dtype,
        })
    };
    let metal = BlockSpecGQAMetalConfig {
        q: affine_layout(&bindings.q.weight)?,
        k: affine_layout(&bindings.k.weight)?,
        v: affine_layout(&bindings.v.weight)?,
        output: affine_layout(&bindings.output.weight)?,
        page_bytes: to_u32("Qwen3x DFlash2 GQA page_bytes", page_bytes)?,
        rope_dim: to_u32("Qwen3x DFlash2 GQA rope_dim", config.head_dim)?,
        norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        rope_scaling: RopeScaling::Default,
        io_dtype: Dtype::Bfloat16,
        norm_weight_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok(metal)
}

#[cfg(test)]
mod tests {
    use inference_executor_core::model::qwen::v3_x::QuantizationConfig;
    use inference_executor_core::model::qwen::v3_x::TensorQuantizationOverride;
    use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2WeightBindings;

    use super::*;

    #[test]
    fn test_gqa_uses_layer_geometry_and_checkpoint_bindings() {
        let config = config();
        let bindings = Qwen3xDFlash2WeightBindings::from_config(&config);

        let (core, metal) =
            derive_qwen3x_dflash2_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024, Dtype::Bfloat16)
                .unwrap();

        assert_eq!(core.block_size, 8);
        assert_eq!(core.attention.model_layer_index, 1);
        assert_eq!(core.attention.hidden_dim, 32);
        assert_eq!(core.attention.num_q_heads, 4);
        assert_eq!(core.attention.num_kv_heads, 1);
        assert_eq!(metal.q.group_size, 32);
        assert_eq!(metal.q.bits, 4);
    }

    #[test]
    fn test_gqa_accepts_a_layer_specific_affine_layout() {
        let mut config = config();
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            config.quantization.as_mut().unwrap().tensor_overrides.insert(
                format!("layers.1.self_attn.{projection}.weight"),
                TensorQuantizationOverride {
                    group_size: Some(32),
                    bits: Some(8),
                    mode: None,
                },
            );
        }
        let bindings = Qwen3xDFlash2WeightBindings::from_config(&config);

        let (_, layer_0_metal) =
            derive_qwen3x_dflash2_gqa_configs(&config, 7, 0, &bindings.layers[0].gqa, 32 * 1024, Dtype::Bfloat16)
                .unwrap();
        let (_, layer_1_metal) =
            derive_qwen3x_dflash2_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024, Dtype::Bfloat16)
                .unwrap();

        assert_eq!(layer_0_metal.q.bits, 4);
        assert_eq!(layer_1_metal.q.bits, 8);
    }

    #[test]
    fn test_gqa_sdpa_config_contains_only_shared_workload_facts() {
        let mut config = config();
        config.quantization.as_mut().unwrap().tensor_overrides.insert(
            "layers.1.self_attn.q_proj.weight".to_string(),
            TensorQuantizationOverride {
                group_size: Some(32),
                bits: Some(8),
                mode: None,
            },
        );

        let sdpa = qwen3x_dflash2_gqa_sdpa_config(&config, 32 * 1024).unwrap();

        assert_eq!(sdpa.io_dtype, Dtype::Bfloat16);
        assert_eq!(sdpa.num_q_heads, 4);
        assert_eq!(sdpa.num_kv_heads, 1);
        assert_eq!(sdpa.head_dim, 8);
        assert_eq!(sdpa.tokens_per_page, 1024);
    }

    #[test]
    fn test_gqa_preserves_mixed_projection_layouts_within_one_layer() {
        let mut config = config();
        config.quantization.as_mut().unwrap().tensor_overrides.insert(
            "layers.1.self_attn.k_proj.weight".to_string(),
            TensorQuantizationOverride {
                group_size: Some(32),
                bits: Some(8),
                mode: None,
            },
        );
        let bindings = Qwen3xDFlash2WeightBindings::from_config(&config);

        let (_, metal) =
            derive_qwen3x_dflash2_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024, Dtype::Float32)
                .unwrap();

        assert_eq!(metal.q.bits, 4);
        assert_eq!(metal.k.bits, 8);
        assert_eq!(metal.v.bits, 4);
        assert_eq!(metal.output.bits, 4);
        assert_eq!(metal.q.scale_bias_dtype, Dtype::Float32);
        assert_eq!(metal.k.scale_bias_dtype, Dtype::Float32);
        assert_eq!(metal.v.scale_bias_dtype, Dtype::Float32);
        assert_eq!(metal.output.scale_bias_dtype, Dtype::Float32);
    }

    fn config() -> Qwen3xDFlash2Config {
        Qwen3xDFlash2Config {
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            mask_token_id: 15,
            selector_rank: 8,
            selector_top_k: 16,
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
            sliding_window: 16,
            vocab_size: 64,
            quantization: Some(QuantizationConfig {
                group_size: 32,
                bits: 4,
                mode: None,
                tensor_overrides: Default::default(),
            }),
        }
    }
}
