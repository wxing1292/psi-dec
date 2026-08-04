use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GQACore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::TensorMap;
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
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::remove_typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xGQA {
    gqa_layer_index: ReplayU32,
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
        gqa_layer_index: ReplayU32,
        bindings: Qwen3xGQAWeightBindings,
        backend: Rc<GQA>,
        scratch: Rc<GQAScratch>,
        request_page_table: Rc<GQARequestPageTable>,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            gqa_layer_index,
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
                gqa_layer_index: self.gqa_layer_index,
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
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "GQA must consume its tensor map");
        Ok(weights)
    }

    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xGQAWeightBindings,
        core: &GQACore,
        metal: GQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let q_weight = remove_quant_weight(tensors, &bindings.q.weight)?;
        let k_weight = remove_quant_weight(tensors, &bindings.k.weight)?;
        let v_weight = remove_quant_weight(tensors, &bindings.v.weight)?;
        let q_scales = remove_typed_tensor(tensors, &bindings.q.scales, safetensors::Dtype::BF16)?.into_data();
        let k_scales = remove_typed_tensor(tensors, &bindings.k.scales, safetensors::Dtype::BF16)?.into_data();
        let v_scales = remove_typed_tensor(tensors, &bindings.v.scales, safetensors::Dtype::BF16)?.into_data();
        let q_biases = remove_typed_tensor(tensors, &bindings.q.biases, safetensors::Dtype::BF16)?.into_data();
        let k_biases = remove_typed_tensor(tensors, &bindings.k.biases, safetensors::Dtype::BF16)?.into_data();
        let v_biases = remove_typed_tensor(tensors, &bindings.v.biases, safetensors::Dtype::BF16)?.into_data();
        let qgkv_weight = concat_bytes(&[&q_weight, &k_weight, &v_weight]);
        let qgkv_scales = concat_bytes(&[&q_scales, &k_scales, &v_scales]);
        let qgkv_biases = concat_bytes(&[&q_biases, &k_biases, &v_biases]);

        let qgkv_config = affine_config(
            core.qgkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.io_dtype,
            metal.io_dtype,
            metal.io_dtype,
        );
        validate_len("GQA qgkv weight", qgkv_weight.len(), qgkv_config.weight_bytes())?;
        validate_len("GQA qgkv scales", qgkv_scales.len(), qgkv_config.scale_or_bias_bytes())?;
        validate_len("GQA qgkv biases", qgkv_biases.len(), qgkv_config.scale_or_bias_bytes())?;
        let output_config = affine_config(
            core.hidden_dim,
            core.q_dim(),
            metal.group_size,
            metal.bits,
            metal.io_dtype,
            metal.io_dtype,
            metal.io_dtype,
        );
        let output_weight = remove_quant_weight(tensors, &bindings.output.weight)?;
        let output_scales =
            remove_typed_tensor(tensors, &bindings.output.scales, safetensors::Dtype::BF16)?.into_data();
        let output_biases =
            remove_typed_tensor(tensors, &bindings.output.biases, safetensors::Dtype::BF16)?.into_data();
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
            q_norm_weight: remove_qwen3x_norm_weight(device, tensors, &bindings.q_norm_weight, &[core.head_dim])?,
            k_norm_weight: remove_qwen3x_norm_weight(device, tensors, &bindings.k_norm_weight, &[core.head_dim])?,
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
