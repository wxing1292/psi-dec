use std::rc::Rc;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BiDiBlockGQACore;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkRopeScaling;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::bidi_block_gqa::backend::BiDiBlockGQA;
use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAInput;
use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAMetalConfig;
use crate::attn::bidi_block_gqa::kv_cache_write::BiDiBlockGQAKVCacheWriteInput;
use crate::attn::bidi_block_gqa::kv_cache_write::BiDiBlockGQAKVCacheWriteScratch;
use crate::attn::bidi_block_gqa::kv_cache_write::BiDiBlockGQAKVCacheWriter;
use crate::attn::bidi_block_gqa::metadata::BiDiBlockGQAMetadataBuffers;
use crate::attn::bidi_block_gqa::scratch::BiDiBlockGQAScratch;
use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::layer::Qwen3xBiDiBlockGQAWeightBuffers;
use crate::model::qwen::v3_x::weight::to_u32;

pub struct Qwen3xDSparkAttention {
    dspark_layer_index: u32,
    core: BiDiBlockGQACore,
    metal: BiDiBlockGQAMetalConfig,
    weights: Option<Qwen3xBiDiBlockGQAWeightBuffers>,
    backend: BiDiBlockGQA,
    kv_cache_writer: BiDiBlockGQAKVCacheWriter,
    bidi_block_scratch: Option<Rc<BiDiBlockGQAScratch>>,
    kv_cache_write_scratch: Option<Rc<BiDiBlockGQAKVCacheWriteScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
}

impl Qwen3xDSparkAttention {
    pub fn new(
        device: &Device,
        core: BiDiBlockGQACore,
        metal: BiDiBlockGQAMetalConfig,
        dspark_layer_index: usize,
        state: &BiDiBlockGQAState,
    ) -> Self {
        Self {
            dspark_layer_index: dspark_layer_index
                .try_into()
                .expect("Qwen3 DSpark layer index must fit u32"),
            core: core.clone(),
            metal,
            weights: None,
            backend: state.new_gqa(device, core.clone(), metal),
            kv_cache_writer: BiDiBlockGQAKVCacheWriter::new(device, core, metal),
            bidi_block_scratch: Some(state.bidi_block_scratch()),
            kv_cache_write_scratch: Some(state.kv_cache_write_scratch()),
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
            "Qwen3.x DSpark attention weights are already loaded"
        );
        self.weights = Some(Qwen3xBiDiBlockGQAWeightBuffers::load(
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
            "Qwen3.x DSpark attention weights are not loaded"
        );
        self.weights.take();
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.bidi_block_scratch.is_some()
                && self.kv_cache_write_scratch.is_some()
                && self.request_page_table.is_some(),
            "Qwen3.x DSpark attention state is not loaded"
        );
        self.request_page_table.take();
        self.kv_cache_write_scratch.take();
        self.bidi_block_scratch.take();
    }

    pub fn load_state(&mut self, state: &BiDiBlockGQAState) {
        assert!(
            self.bidi_block_scratch.is_none()
                && self.kv_cache_write_scratch.is_none()
                && self.request_page_table.is_none(),
            "Qwen3.x DSpark attention state is already loaded"
        );
        self.bidi_block_scratch = Some(state.bidi_block_scratch());
        self.kv_cache_write_scratch = Some(state.kv_cache_write_scratch());
        self.request_page_table = Some(state.request_page_table());
    }

    fn weights(&self) -> &Qwen3xBiDiBlockGQAWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3.x DSpark attention weights must be loaded before execution")
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
        self.kv_cache_writer.record(
            recorder,
            BiDiBlockGQAKVCacheWriteInput {
                num_total_tokens,
                num_active_tokens,
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: self.dspark_layer_index,
                main_feature,
                req_slots,
                flat_token_indices,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.kv_cache_write_scratch().bindings(),
            },
        );
    }

    pub fn record_bidi_block<'a, R>(
        &'a self,
        recorder: &mut R,
        num_active_tokens: inference_backend_metal::metal::ReplayU32,
        metadata: &'a BiDiBlockGQAMetadataBuffers,
        hidden_input: &'a Buffer,
        hidden_output: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <BiDiBlockGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            BiDiBlockGQAInput {
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: self.dspark_layer_index,
                metadata,
                hidden_state: hidden_input,
                next_hidden_state: hidden_output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.bidi_block_scratch().bindings(),
                num_active_tokens,
            },
        );
    }

    fn bidi_block_scratch(&self) -> &BiDiBlockGQAScratch {
        self.bidi_block_scratch
            .as_deref()
            .expect("Qwen3.x DSpark block scratch must be loaded before execution")
    }

    fn kv_cache_write_scratch(&self) -> &BiDiBlockGQAKVCacheWriteScratch {
        self.kv_cache_write_scratch
            .as_deref()
            .expect("Qwen3.x DSpark context scratch must be loaded before execution")
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("Qwen3.x DSpark request page-table state must be loaded before execution")
    }
}

pub fn derive_qwen3x_dspark_gqa_configs(
    config: &Qwen3xDSparkConfig,
    num_spec_tokens: usize,
    dspark_layer_index: usize,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
) -> Result<(BiDiBlockGQACore, BiDiBlockGQAMetalConfig), ModelExecutorError> {
    let core = qwen3x_dspark_gqa_core(config, num_spec_tokens, dspark_layer_index);
    let metal = qwen3x_dspark_gqa_metal_config(config, bindings, page_bytes)?;
    Ok((core, metal))
}

pub fn qwen3x_dspark_gqa_core(
    config: &Qwen3xDSparkConfig,
    num_spec_tokens: usize,
    dspark_layer_index: usize,
) -> BiDiBlockGQACore {
    assert_eq!(
        num_spec_tokens,
        config.num_spec_tokens().get(),
        "Qwen3x DSpark proposal count must match the checkpoint"
    );
    assert!(
        dspark_layer_index < config.num_hidden_layers,
        "Qwen3x DSpark attention layer index must be within the model"
    );
    let attention = UngatedGQACore::new(
        dspark_layer_index,
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        (config.head_dim as f32).sqrt().recip(),
    );
    attention.validate();
    BiDiBlockGQACore::new(attention, num_spec_tokens)
}

pub fn qwen3x_dspark_gqa_sdpa_config(
    config: &Qwen3xDSparkConfig,
    page_bytes: usize,
) -> Result<backend_sdpa::Config, ModelExecutorError> {
    let num_q_heads = to_u32("Qwen3x DSpark GQA Q-head count", config.num_attention_heads)?;
    let num_kv_heads = to_u32("Qwen3x DSpark GQA KV-head count", config.num_key_value_heads)?;
    let head_dim = to_u32("Qwen3x DSpark GQA head_dim", config.head_dim)?;
    let page_bytes = to_u32("Qwen3x DSpark GQA page_bytes", page_bytes)?;
    let kv_bytes_per_token = num_kv_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark GQA KV bytes per token must fit u32"))?;
    if !page_bytes.is_multiple_of(kv_bytes_per_token) {
        return Err(ModelExecutorError::custom(
            "Qwen3x DSpark GQA page bytes must contain whole KV tokens",
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

fn qwen3x_dspark_gqa_metal_config(
    config: &Qwen3xDSparkConfig,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
) -> Result<BiDiBlockGQAMetalConfig, ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Metal executor requires quantization config"))?;
    let affine_layout = |tensor_name: &str| -> Result<QuantizedAffineLayout, ModelExecutorError> {
        let resolved = quantization.resolve_for_tensor(tensor_name);
        if !matches!(resolved.mode.as_deref(), None | Some("affine")) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DSpark GQA tensor {tensor_name:?} requires affine quantization, got mode={:?}",
                resolved.mode
            )));
        }
        Ok(QuantizedAffineLayout {
            group_size: to_u32("Qwen3x DSpark GQA group_size", resolved.group_size)?,
            bits: to_u32("Qwen3x DSpark GQA bits", resolved.bits)?,
            scale_bias_dtype: Dtype::Bfloat16,
        })
    };
    let rope_scaling = match config.rope_scaling {
        Qwen3xDSparkRopeScaling::Default => RopeScaling::Default,
        Qwen3xDSparkRopeScaling::Yarn {
            factor,
            attention_factor,
            beta_fast,
            beta_slow,
            original_max_position_embeddings,
            truncate,
        } => {
            RopeScaling::Yarn {
                factor,
                attention_factor,
                beta_fast,
                beta_slow,
                original_max_position_embeddings: to_u32(
                    "Qwen3x DSpark Yarn original_max_position_embeddings",
                    original_max_position_embeddings,
                )?,
                truncate,
            }
        },
    };
    let metal = BiDiBlockGQAMetalConfig {
        q: affine_layout(&bindings.q.weight)?,
        k: affine_layout(&bindings.k.weight)?,
        v: affine_layout(&bindings.v.weight)?,
        output: affine_layout(&bindings.output.weight)?,
        page_bytes: to_u32("Qwen3x DSpark GQA page_bytes", page_bytes)?,
        rope_dim: to_u32("Qwen3x DSpark GQA rope_dim", config.head_dim)?,
        norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        rope_scaling,
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
    use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;

    use super::*;

    #[test]
    fn test_gqa_uses_layer_geometry_and_checkpoint_bindings() {
        let config = config();
        let bindings = Qwen3xDSparkWeightBindings::from_config(&config);

        let (core, metal) =
            derive_qwen3x_dspark_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap();

        assert_eq!(core.block_size, 7);
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
        let bindings = Qwen3xDSparkWeightBindings::from_config(&config);

        let (_, layer_0_metal) =
            derive_qwen3x_dspark_gqa_configs(&config, 7, 0, &bindings.layers[0].gqa, 32 * 1024).unwrap();
        let (_, layer_1_metal) =
            derive_qwen3x_dspark_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap();

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

        let sdpa = qwen3x_dspark_gqa_sdpa_config(&config, 32 * 1024).unwrap();

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
        let bindings = Qwen3xDSparkWeightBindings::from_config(&config);

        let (_, metal) = derive_qwen3x_dspark_gqa_configs(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap();

        assert_eq!(metal.q.bits, 4);
        assert_eq!(metal.k.bits, 8);
        assert_eq!(metal.v.bits, 4);
        assert_eq!(metal.output.bits, 4);
    }

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
            rope_scaling: Qwen3xDSparkRopeScaling::Default,
            max_position_embeddings: 32,
            vocab_size: 64,
            markov_rank: 8,
            quantization: Some(QuantizationConfig {
                group_size: 32,
                bits: 4,
                mode: None,
                tensor_overrides: Default::default(),
            }),
        }
    }
}
