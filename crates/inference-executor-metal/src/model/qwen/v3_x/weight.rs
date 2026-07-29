use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::checkpoint::TensorBytes;
use crate::mlp::moe::backend::GatedMoEMetalConfig;
pub fn typed_tensor(
    store: &mut SafeTensorStore,
    name: &str,
    dtype: safetensors::Dtype,
) -> Result<TensorBytes, ModelExecutorError> {
    store.tensor_bytes(name, dtype)
}

pub fn quant_weight(store: &mut SafeTensorStore, name: &str) -> Result<Vec<u8>, ModelExecutorError> {
    Ok(typed_tensor(store, name, safetensors::Dtype::U32)?.into_data())
}

pub fn load_qwen3x_norm_weight(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, ModelExecutorError> {
    let data = typed_tensor(store, name, safetensors::Dtype::BF16)?;
    validate_shape(name, data.shape(), expected_shape)?;
    Ok(Buffer::from_slice(device, data.data()))
}

pub fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

pub fn affine_config(
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
    input_dtype: Dtype,
    output_dtype: Dtype,
    scale_bias_dtype: Dtype,
) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("affine n must fit i32"),
        k: k.try_into().expect("affine k must fit i32"),
        group_size: group_size.try_into().expect("affine group_size must fit i32"),
        bits: bits.try_into().expect("affine bits must fit i32"),
        input_dtype,
        output_dtype,
        scale_bias_dtype,
    }
}

pub fn sparse_affine_layout(
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    metal: GatedMoEMetalConfig,
) -> SparseAffineLayout {
    SparseAffineLayout {
        experts,
        output_dim,
        input_dim,
        group_size: metal.group_size as usize,
        bits: metal.bits as usize,
        scale_bias_dtype: metal.dtype,
    }
}

pub struct SparseAffineLayout {
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    group_size: usize,
    bits: usize,
    scale_bias_dtype: Dtype,
}

impl SparseAffineLayout {
    pub fn weight_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim * self.bits / 32) * std::mem::size_of::<u32>()
    }

    pub fn scale_or_bias_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim / self.group_size) * self.scale_bias_dtype.item_size()
    }
}

pub fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

pub fn validate_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} shape mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

pub fn to_u32(name: &str, value: usize) -> Result<u32, ModelExecutorError> {
    value
        .try_into()
        .map_err(|_| ModelExecutorError::custom(format!("{name}={value} must fit u32")))
}
