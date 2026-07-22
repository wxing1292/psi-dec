use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::checkpoint::TensorBytes;
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

pub fn load_qwen35_norm_weight(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
    stores_actual_scale: bool,
) -> Result<Buffer, ModelExecutorError> {
    let data = typed_tensor(store, name, safetensors::Dtype::BF16)?;
    validate_shape(name, data.shape(), expected_shape)?;
    Ok(Buffer::from_slice(
        device,
        &qwen_next_bf16_weight(data.data(), stores_actual_scale),
    ))
}

pub fn qwen_next_norm_f32_buffer(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
    stores_actual_scale: bool,
) -> Result<Buffer, ModelExecutorError> {
    let data = typed_tensor(store, name, safetensors::Dtype::BF16)?;
    validate_shape(name, data.shape(), expected_shape)?;
    Ok(Buffer::from_slice(
        device,
        &qwen_next_f32_weight(data.data(), stores_actual_scale),
    ))
}

pub fn bf16_tensor_as_f32(store: &mut SafeTensorStore, name: &str) -> Result<Vec<f32>, ModelExecutorError> {
    let data = typed_tensor(store, name, safetensors::Dtype::BF16)?;
    Ok(bf16_bytes_to_f32(data.data()))
}

pub fn qwen_next_bf16_weight(bytes: &[u8], stores_actual_scale: bool) -> Vec<u16> {
    bf16_bytes_to_f32(bytes)
        .into_iter()
        .map(|value| {
            let scale = if stores_actual_scale { value } else { value + 1.0 };
            bf16::from_f32(scale).to_bits()
        })
        .collect()
}

pub fn qwen_next_f32_weight(bytes: &[u8], stores_actual_scale: bool) -> Vec<f32> {
    bf16_bytes_to_f32(bytes)
        .into_iter()
        .map(|value| if stores_actual_scale { value } else { value + 1.0 })
        .collect()
}

pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

pub fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

pub fn concat_f32(parts: &[&[f32]]) -> Vec<f32> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

pub fn affine_shape(
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
    input_dtype: Dtype,
    output_dtype: Dtype,
    affine_dtype: Dtype,
) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: 1,
        n: n.try_into().expect("affine n must fit i32"),
        k: k.try_into().expect("affine k must fit i32"),
        group_size: group_size.try_into().expect("affine group_size must fit i32"),
        bits: bits.try_into().expect("affine bits must fit i32"),
        input_dtype,
        output_dtype,
        affine_dtype,
    }
}

pub fn sparse_affine_layout(
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    metal: crate::mlp::moe::backend::GatedMoEMetalConfig,
) -> SparseAffineLayout {
    SparseAffineLayout {
        experts,
        output_dim,
        input_dim,
        group_size: metal.group_size as usize,
        bits: metal.bits as usize,
        affine_dtype: metal.dtype,
    }
}

pub struct SparseAffineLayout {
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    group_size: usize,
    bits: usize,
    affine_dtype: Dtype,
}

impl SparseAffineLayout {
    pub fn weight_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim * self.bits / 32) * std::mem::size_of::<u32>()
    }

    pub fn affine_param_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim / self.group_size) * self.affine_dtype.item_size()
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
