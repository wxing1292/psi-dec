use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35GQAWeightBindings;

use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::backend::GQAInput;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAWeights;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;
use crate::model::qwen::v3_5::weight::affine_shape;
use crate::model::qwen::v3_5::weight::concat_bytes;
use crate::model::qwen::v3_5::weight::quant_weight;
use crate::model::qwen::v3_5::weight::qwen_next_norm_f32_buffer;
use crate::model::qwen::v3_5::weight::typed_tensor;
use crate::model::qwen::v3_5::weight::validate_len;

pub struct Qwen35GQA {
    layer_index: usize,
    weights: Qwen35GQAWeights,
    backend: Rc<GQA>,
    scratch: Rc<GQAScratch>,
    request_page_table: Rc<GQARequestPageTable>,
}

impl Qwen35GQA {
    pub fn num_tokens_per_page(&self) -> usize {
        self.backend.num_tokens_per_page() as usize
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        layer_index: usize,
        bindings: Qwen35GQAWeightBindings,
        backend: Rc<GQA>,
        scratch: Rc<GQAScratch>,
        request_page_table: Rc<GQARequestPageTable>,
    ) -> Result<Self, ModelExecutorError> {
        let (core, metal) = qwen35_gqa_core_and_metal(model_layer_index, &config.text_config, defaults)?;
        Ok(Self {
            layer_index,
            weights: Qwen35GQAWeights::load(device, store, &bindings, &core, metal, config.quantization.is_some())?,
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
                    .layer_index
                    .try_into()
                    .expect("qwen3.5 compact GQA layer index must fit u32"),
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

struct Qwen35GQAWeights {
    qgkv_weight: Buffer,
    qgkv_scales: Buffer,
    qgkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen35GQAWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35GQAWeightBindings,
        core: &GQACore,
        metal: crate::attn::gqa::backend::GQAMetalConfig,
        norms_store_actual_scale: bool,
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

        let qgkv_shape = affine_shape(
            core.qgkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        validate_len("GQA qgkv weight", qgkv_weight.len(), qgkv_shape.weight_bytes())?;
        validate_len("GQA qgkv scales", qgkv_scales.len(), qgkv_shape.affine_param_bytes())?;
        validate_len("GQA qgkv biases", qgkv_biases.len(), qgkv_shape.affine_param_bytes())?;
        let output_shape = affine_shape(
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
        validate_len("GQA output weight", output_weight.len(), output_shape.weight_bytes())?;
        validate_len(
            "GQA output scales",
            output_scales.len(),
            output_shape.affine_param_bytes(),
        )?;
        validate_len(
            "GQA output biases",
            output_biases.len(),
            output_shape.affine_param_bytes(),
        )?;

        Ok(Self {
            qgkv_weight: Buffer::from_slice(device, &qgkv_weight),
            qgkv_scales: Buffer::from_slice(device, &qgkv_scales),
            qgkv_biases: Buffer::from_slice(device, &qgkv_biases),
            q_norm_weight: qwen_next_norm_f32_buffer(
                device,
                store,
                &bindings.q_norm_weight,
                &[core.head_dim],
                norms_store_actual_scale,
            )?,
            k_norm_weight: qwen_next_norm_f32_buffer(
                device,
                store,
                &bindings.k_norm_weight,
                &[core.head_dim],
                norms_store_actual_scale,
            )?,
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
