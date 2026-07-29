use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::backend::GQAInput;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::backend::GQAWeights;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::weight::affine_config;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::quant_weight;
use crate::model::qwen::v3_x::weight::typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xGQA {
    compact_gqa_layer_index: usize,
    weights: Qwen3xGQAWeights,
    backend: Rc<GQA>,
    scratch: Rc<GQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
}

impl Qwen3xGQA {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        core: &GQACore,
        metal: GQAMetalConfig,
        compact_gqa_layer_index: usize,
        bindings: Qwen3xGQAWeightBindings,
        backend: Rc<GQA>,
        scratch: Rc<GQAScratch>,
        request_page_table: Rc<GQARequestPageTable>,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            compact_gqa_layer_index,
            weights: Qwen3xGQAWeights::load(device, store, &bindings, core, metal)?,
            backend,
            scratch,
            request_page_table,
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
        let _ = <GQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            GQAInput {
                page_table_layout: self.request_page_table.layout(),
                gqa_layer_index: self
                    .compact_gqa_layer_index
                    .try_into()
                    .expect("qwen3.x compact GQA layer index must fit u32"),
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

    pub fn num_tokens_per_page(&self) -> usize {
        self.backend.num_tokens_per_page() as usize
    }
}

struct Qwen3xGQAWeights {
    qgkv_weight: Buffer,
    qgkv_scales: Buffer,
    qgkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen3xGQAWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xGQAWeightBindings,
        core: &GQACore,
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
        let qgkv_weight = concat_bytes(&[&q_weight, &k_weight, &v_weight]);
        let qgkv_scales = concat_bytes(&[&q_scales, &k_scales, &v_scales]);
        let qgkv_biases = concat_bytes(&[&q_biases, &k_biases, &v_biases]);

        let qgkv_config = affine_config(
            core.qgkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        validate_len("GQA qgkv weight", qgkv_weight.len(), qgkv_config.weight_bytes())?;
        validate_len("GQA qgkv scales", qgkv_scales.len(), qgkv_config.scale_or_bias_bytes())?;
        validate_len("GQA qgkv biases", qgkv_biases.len(), qgkv_config.scale_or_bias_bytes())?;
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
        validate_len("GQA output weight", output_weight.len(), output_config.weight_bytes())?;
        validate_len(
            "GQA output scales",
            output_scales.len(),
            output_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "GQA output biases",
            output_biases.len(),
            output_config.scale_or_bias_bytes(),
        )?;

        Ok(Self {
            qgkv_weight: Buffer::from_slice(device, &qgkv_weight),
            qgkv_scales: Buffer::from_slice(device, &qgkv_scales),
            qgkv_biases: Buffer::from_slice(device, &qgkv_biases),
            q_norm_weight: load_qwen3x_norm_weight(device, store, &bindings.q_norm_weight, &[core.head_dim])?,
            k_norm_weight: load_qwen3x_norm_weight(device, store, &bindings.k_norm_weight, &[core.head_dim])?,
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        })
    }

    fn as_borrowed(&self) -> GQAWeights<'_> {
        GQAWeights {
            qgkv_weight: &self.qgkv_weight,
            qgkv_scales: &self.qgkv_scales,
            qgkv_biases: &self.qgkv_biases,
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}
