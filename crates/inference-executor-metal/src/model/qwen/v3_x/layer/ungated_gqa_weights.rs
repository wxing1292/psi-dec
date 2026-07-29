use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::checkpoint::SafeTensorStore;
use crate::model::qwen::v3_x::weight::affine_shape;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::quant_weight;
use crate::model::qwen::v3_x::weight::typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xUngatedGQAWeightBuffers {
    qkv_weight: Buffer,
    qkv_scales: Buffer,
    qkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen3xUngatedGQAWeightBuffers {
    pub fn load(
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
        let qkv_shape = affine_shape(
            core.qkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        validate_len("ungated GQA qkv weight", qkv_weight.len(), qkv_shape.weight_bytes())?;
        validate_len(
            "ungated GQA qkv scales",
            qkv_scales.len(),
            qkv_shape.affine_param_bytes(),
        )?;
        validate_len(
            "ungated GQA qkv biases",
            qkv_biases.len(),
            qkv_shape.affine_param_bytes(),
        )?;

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
        validate_len(
            "ungated GQA output weight",
            output_weight.len(),
            output_shape.weight_bytes(),
        )?;
        validate_len(
            "ungated GQA output scales",
            output_scales.len(),
            output_shape.affine_param_bytes(),
        )?;
        validate_len(
            "ungated GQA output biases",
            output_biases.len(),
            output_shape.affine_param_bytes(),
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

    pub fn as_borrowed(&self) -> UngatedGQAWeights<'_> {
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
