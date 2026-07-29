use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::ungated_backend::UngatedGQA;
use crate::attn::gqa::ungated_backend::UngatedGQAInput;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::weight::affine_config;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::quant_weight;
use crate::model::qwen::v3_x::weight::typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3MainGQA {
    model_layer_index: usize,
    weights: Qwen3MainGQAWeights,
    backend: Rc<UngatedGQA>,
    scratch: Rc<UngatedGQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
}

pub struct Qwen3MainGQAState {
    backend: Rc<UngatedGQA>,
    scratch: Rc<UngatedGQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
    metadata: GQAMetadataBuffers,
    num_cache_pages: usize,
    cache_lane: usize,
}

impl Qwen3MainGQA {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
        bindings: Qwen3xGQAWeightBindings,
        state: &Qwen3MainGQAState,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            model_layer_index: core.model_layer_index,
            weights: Qwen3MainGQAWeights::load(device, store, &bindings, core, metal)?,
            backend: Rc::clone(&state.backend),
            scratch: Rc::clone(&state.scratch),
            request_page_table: Rc::clone(&state.request_page_table),
        })
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        pages: &'a Buffer,
        metadata: &'a GQAMetadataBuffers,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <UngatedGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            UngatedGQAInput {
                page_table_layout: self.request_page_table.layout(),
                gqa_layer_index: self
                    .model_layer_index
                    .try_into()
                    .expect("qwen3 Main GQA layer index must fit u32"),
                batch_metadata: metadata,
                hidden_state: input,
                next_hidden_state: output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table.page_ids_buffer(),
                },
                weights: self.weights.as_borrowed(),
                scratch: self.scratch.bindings(),
            },
        );
    }
}

impl Qwen3MainGQAState {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        core: UngatedGQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
        cache_lane: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3 Main GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3 Main cache page IDs must fit u32"
        );
        page_table_layout.validate();
        Self {
            backend: Rc::new(UngatedGQA::new(device, core.clone(), metal)),
            scratch: Rc::new(UngatedGQAScratch::new(device, &core, metal, max_tokens)),
            request_page_table: Rc::new(GQARequestPageTable::new(device, page_table_layout)),
            metadata: GQAMetadataBuffers::new(device, max_tokens),
            num_cache_pages,
            cache_lane,
        }
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.backend.num_tokens_per_page() as usize
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        &self.metadata
    }

    pub fn prepare_pages(&self, core_batch: &BatchDeviceRequest) {
        self.request_page_table
            .prepare(core_batch, self.cache_lane, self.num_cache_pages);
    }

    pub fn prepare_metadata(&self, req_slots: &[u32], token_indices: &[u32], cu_tokens: &[u32]) -> GQAReplayShape {
        self.backend
            .prepare(&self.metadata, req_slots, token_indices, cu_tokens)
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table.reset_req_slots(req_slots);
    }
}

struct Qwen3MainGQAWeights {
    qkv_weight: Buffer,
    qkv_scales: Buffer,
    qkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen3MainGQAWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xGQAWeightBindings,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let q_weight = quant_weight(store, &bindings.q.weight)?;
        let k_weight = quant_weight(store, &bindings.k.weight)?;
        let v_weight = quant_weight(store, &bindings.v.weight)?;
        let q_scales = typed_tensor(store, &bindings.q.scales, safetensors::Dtype::BF16)?.into_data();
        let k_scales = typed_tensor(store, &bindings.k.scales, safetensors::Dtype::BF16)?.into_data();
        let v_scales = typed_tensor(store, &bindings.v.scales, safetensors::Dtype::BF16)?.into_data();
        let q_biases = typed_tensor(store, &bindings.q.biases, safetensors::Dtype::BF16)?.into_data();
        let k_biases = typed_tensor(store, &bindings.k.biases, safetensors::Dtype::BF16)?.into_data();
        let v_biases = typed_tensor(store, &bindings.v.biases, safetensors::Dtype::BF16)?.into_data();
        let qkv_weight = concat_bytes(&[&q_weight, &k_weight, &v_weight]);
        let qkv_scales = concat_bytes(&[&q_scales, &k_scales, &v_scales]);
        let qkv_biases = concat_bytes(&[&q_biases, &k_biases, &v_biases]);
        let qkv_config = affine_config(
            core.qkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        validate_len("Qwen3 GQA qkv weight", qkv_weight.len(), qkv_config.weight_bytes())?;
        validate_len(
            "Qwen3 GQA qkv scales",
            qkv_scales.len(),
            qkv_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3 GQA qkv biases",
            qkv_biases.len(),
            qkv_config.scale_or_bias_bytes(),
        )?;
        let output_config = affine_config(
            core.hidden_dim,
            core.q_dim(),
            metal.group_size,
            metal.bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        let output_weight = quant_weight(store, &bindings.output.weight)?;
        let output_scales = typed_tensor(store, &bindings.output.scales, safetensors::Dtype::BF16)?.into_data();
        let output_biases = typed_tensor(store, &bindings.output.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "Qwen3 GQA output weight",
            output_weight.len(),
            output_config.weight_bytes(),
        )?;
        validate_len(
            "Qwen3 GQA output scales",
            output_scales.len(),
            output_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3 GQA output biases",
            output_biases.len(),
            output_config.scale_or_bias_bytes(),
        )?;

        Ok(Self {
            qkv_weight: Buffer::from_slice(device, &qkv_weight),
            qkv_scales: Buffer::from_slice(device, &qkv_scales),
            qkv_biases: Buffer::from_slice(device, &qkv_biases),
            q_norm_weight: load_qwen3x_norm_weight(device, store, &bindings.q_norm_weight, &[core.head_dim])?,
            k_norm_weight: load_qwen3x_norm_weight(device, store, &bindings.k_norm_weight, &[core.head_dim])?,
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        })
    }

    fn as_borrowed(&self) -> UngatedGQAWeights<'_> {
        UngatedGQAWeights {
            qkv_weight: &self.qkv_weight,
            qkv_scales: &self.qkv_scales,
            qkv_biases: &self.qkv_biases,
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}
