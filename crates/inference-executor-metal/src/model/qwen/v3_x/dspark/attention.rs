use std::rc::Rc;

use inference_backend_metal::components::GQAComputeConfig;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::dspark::backend::UngatedDSparkGQA;
use crate::attn::dspark::backend::UngatedDSparkGQAInput;
use crate::attn::dspark::context::DSparkGQAContextScratch;
use crate::attn::dspark::context::UngatedDSparkGQAContextAppender;
use crate::attn::dspark::context::UngatedDSparkGQAContextInput;
use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratch;
use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::layer::Qwen3xUngatedGQAWeightBuffers;
use crate::model::qwen::v3_x::weight::resolve_uniform_quantization;
use crate::model::qwen::v3_x::weight::to_u32;

pub struct Qwen3xDSparkAttention {
    dspark_layer_index: u32,
    weights: Option<Qwen3xUngatedGQAWeightBuffers>,
    backend: UngatedDSparkGQA,
    context_appender: UngatedDSparkGQAContextAppender,
    block_scratch: Option<Rc<DSparkBlockScratch>>,
    context_scratch: Option<Rc<DSparkGQAContextScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
}

impl Qwen3xDSparkAttention {
    pub fn new(
        device: &Device,
        core: &UngatedDSparkGQACore,
        metal: GQAMetalConfig,
        dspark_layer_index: usize,
        state: &UngatedDSparkGQAState,
    ) -> Self {
        Self {
            dspark_layer_index: dspark_layer_index
                .try_into()
                .expect("Qwen3 DSpark layer index must fit u32"),
            weights: None,
            backend: UngatedDSparkGQA::new(device, core.clone(), metal),
            context_appender: UngatedDSparkGQAContextAppender::new(device, core.clone(), metal),
            block_scratch: Some(state.block_scratch()),
            context_scratch: Some(state.context_scratch()),
            request_page_table: Some(state.request_page_table()),
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        core: &UngatedDSparkGQACore,
        metal: GQAMetalConfig,
        bindings: Qwen3xGQAWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3.x DSpark attention weights are already loaded"
        );
        self.weights = Some(Qwen3xUngatedGQAWeightBuffers::load(
            device,
            store,
            &bindings,
            &core.attention,
            metal,
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
            self.block_scratch.is_some() && self.context_scratch.is_some() && self.request_page_table.is_some(),
            "Qwen3.x DSpark attention state is not loaded"
        );
        self.request_page_table.take();
        self.context_scratch.take();
        self.block_scratch.take();
    }

    pub fn load_state(&mut self, state: &UngatedDSparkGQAState) {
        assert!(
            self.block_scratch.is_none() && self.context_scratch.is_none() && self.request_page_table.is_none(),
            "Qwen3.x DSpark attention state is already loaded"
        );
        self.block_scratch = Some(state.block_scratch());
        self.context_scratch = Some(state.context_scratch());
        self.request_page_table = Some(state.request_page_table());
    }

    fn weights(&self) -> &Qwen3xUngatedGQAWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3.x DSpark attention weights must be loaded before execution")
    }

    pub fn record_context<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        main_feature: &'a Buffer,
        req_slots: &'a Buffer,
        flat_token_indices: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.context_appender.record(
            recorder,
            UngatedDSparkGQAContextInput {
                num_tokens,
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
                scratch: self.context_scratch().bindings(),
            },
        );
    }

    pub fn record_block<'a, R>(
        &'a self,
        recorder: &mut R,
        metadata: &'a DSparkGQAMetadataBuffers,
        hidden_input: &'a Buffer,
        hidden_output: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <UngatedDSparkGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            UngatedDSparkGQAInput {
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
                scratch: self.block_scratch().bindings(),
            },
        );
    }

    fn block_scratch(&self) -> &DSparkBlockScratch {
        self.block_scratch
            .as_deref()
            .expect("Qwen3.x DSpark block scratch must be loaded before execution")
    }

    fn context_scratch(&self) -> &DSparkGQAContextScratch {
        self.context_scratch
            .as_deref()
            .expect("Qwen3.x DSpark context scratch must be loaded before execution")
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("Qwen3.x DSpark request page-table state must be loaded before execution")
    }
}

pub fn qwen3x_dspark_gqa_core_and_metal(
    config: &Qwen3xDSparkConfig,
    num_spec_tokens: usize,
    dspark_layer_index: usize,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
) -> Result<(UngatedDSparkGQACore, GQAMetalConfig), ModelExecutorError> {
    let core = qwen3x_dspark_gqa_core(config, num_spec_tokens, dspark_layer_index);
    let metal = qwen3x_dspark_gqa_metal_config(config, bindings, page_bytes)?;
    assert!(
        metal.num_ungated_tokens_per_page(&core.attention) > 0,
        "Qwen3x DSpark GQA geometry must fit one cache page"
    );
    Ok((core, metal))
}

pub fn qwen3x_dspark_gqa_core(
    config: &Qwen3xDSparkConfig,
    num_spec_tokens: usize,
    dspark_layer_index: usize,
) -> UngatedDSparkGQACore {
    assert!(
        num_spec_tokens > 0,
        "Qwen3x DSpark attention requires speculative tokens"
    );
    assert!(
        num_spec_tokens <= config.block_size,
        "Qwen3x DSpark attention proposal length must not exceed the checkpoint block_size"
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
    UngatedDSparkGQACore::new(attention, num_spec_tokens)
}

pub fn qwen3x_dspark_gqa_compute_config(
    config: &Qwen3xDSparkConfig,
    page_bytes: usize,
) -> Result<GQAComputeConfig, ModelExecutorError> {
    let compute_config = GQAComputeConfig {
        io_dtype: Dtype::Bfloat16,
        page_bytes: to_u32("Qwen3x DSpark GQA page_bytes", page_bytes)?,
        num_q_heads: to_u32("Qwen3x DSpark GQA Q-head count", config.num_attention_heads)?,
        num_kv_heads: to_u32("Qwen3x DSpark GQA KV-head count", config.num_key_value_heads)?,
        head_dim: to_u32("Qwen3x DSpark GQA head_dim", config.head_dim)?,
    };
    compute_config.validate();
    Ok(compute_config)
}

fn qwen3x_dspark_gqa_metal_config(
    config: &Qwen3xDSparkConfig,
    bindings: &Qwen3xGQAWeightBindings,
    page_bytes: usize,
) -> Result<GQAMetalConfig, ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Metal executor requires quantization config"))?;
    let resolved = resolve_uniform_quantization(
        quantization,
        &[
            bindings.q.weight.as_str(),
            bindings.k.weight.as_str(),
            bindings.v.weight.as_str(),
            bindings.output.weight.as_str(),
        ],
        "Qwen3x DSpark GQA",
    )?;
    let metal = GQAMetalConfig {
        group_size: to_u32("Qwen3x DSpark GQA group_size", resolved.group_size)?,
        bits: to_u32("Qwen3x DSpark GQA bits", resolved.bits)?,
        page_bytes: to_u32("Qwen3x DSpark GQA page_bytes", page_bytes)?,
        rope_dim: to_u32("Qwen3x DSpark GQA rope_dim", config.head_dim)?,
        norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        rope_scale: 1.0,
        io_dtype: Dtype::Bfloat16,
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
            qwen3x_dspark_gqa_core_and_metal(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap();

        assert_eq!(core.block_size, 7);
        assert_eq!(core.attention.model_layer_index, 1);
        assert_eq!(core.attention.hidden_dim, 32);
        assert_eq!(core.attention.num_q_heads, 4);
        assert_eq!(core.attention.num_kv_heads, 1);
        assert_eq!(metal.group_size, 32);
        assert_eq!(metal.bits, 4);
    }

    #[test]
    fn test_gqa_uses_configured_spec_tokens_below_checkpoint_limit() {
        let config = config();
        let core = qwen3x_dspark_gqa_core(&config, 3, 0);

        assert_eq!(config.block_size, 7);
        assert_eq!(core.block_size, 3);
    }

    #[test]
    #[should_panic(expected = "must not exceed the checkpoint block_size")]
    fn test_gqa_rejects_configured_spec_tokens_above_checkpoint_limit() {
        let config = config();
        let _ = qwen3x_dspark_gqa_core(&config, 8, 0);
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
            qwen3x_dspark_gqa_core_and_metal(&config, 7, 0, &bindings.layers[0].gqa, 32 * 1024).unwrap();
        let (_, layer_1_metal) =
            qwen3x_dspark_gqa_core_and_metal(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap();

        assert_eq!(layer_0_metal.bits, 4);
        assert_eq!(layer_1_metal.bits, 8);
    }

    #[test]
    fn test_gqa_compute_config_contains_only_shared_workload_facts() {
        let mut config = config();
        config.quantization.as_mut().unwrap().tensor_overrides.insert(
            "layers.1.self_attn.q_proj.weight".to_string(),
            TensorQuantizationOverride {
                group_size: Some(32),
                bits: Some(8),
                mode: None,
            },
        );

        let compute = qwen3x_dspark_gqa_compute_config(&config, 32 * 1024).unwrap();

        assert_eq!(compute.io_dtype, Dtype::Bfloat16);
        assert_eq!(compute.page_bytes, 32 * 1024);
        assert_eq!(compute.num_q_heads, 4);
        assert_eq!(compute.num_kv_heads, 1);
        assert_eq!(compute.head_dim, 8);
        assert_eq!(compute.num_tokens_per_page(), 1024);
    }

    #[test]
    fn test_gqa_rejects_mixed_projection_layouts_within_one_layer() {
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

        let error = qwen3x_dspark_gqa_core_and_metal(&config, 7, 1, &bindings.layers[1].gqa, 32 * 1024).unwrap_err();

        assert!(error.to_string().contains("GQA requires one affine layout"));
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
            max_position_embeddings: 32,
            vocab_size: 64,
            markov_rank: 8,
            num_anchors: 8,
            quantization: Some(QuantizationConfig {
                group_size: 32,
                bits: 4,
                mode: None,
                tensor_overrides: Default::default(),
            }),
        }
    }
}
