use half::bf16;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkAttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkLayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkMLPWeightBindings;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkMarkovWeightBindings;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkTargetWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::checkpoint::TensorBytes;
use crate::model::qwen::v3_5::dspark::attention::Qwen35DSparkQKVLayout;
use crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan;
use crate::model::qwen::v3_5::plan::Qwen35DSparkPlan;

pub struct Qwen35DSparkTargetWeights {
    pub fc_weight: Buffer,
    pub fc_scales: Buffer,
    pub fc_biases: Buffer,
    pub hidden_norm_weight: Buffer,
}

pub struct Qwen35DSparkFinalWeights {
    pub norm_weight: Buffer,
}

impl Qwen35DSparkFinalWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        norm_weight_name: &str,
        hidden_dim: usize,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            norm_weight: actual_scale_norm(device, store, norm_weight_name, hidden_dim)?,
        })
    }
}

impl Qwen35DSparkTargetWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkPlan,
        bindings: &Qwen35DSparkTargetWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let fc = quantized_matrix(
            store,
            &bindings.fc,
            plan.fc.output_dim,
            plan.fc.input_dim,
            plan.fc.group_size,
            plan.fc.bits,
        )?;
        let hidden_norm = tensor(
            store,
            &bindings.hidden_norm_weight,
            safetensors::Dtype::BF16,
            &[plan.fc.output_dim],
        )?;
        Ok(Self {
            fc_weight: Buffer::from_slice(device, &fc.weight),
            fc_scales: Buffer::from_slice(device, &fc.scales),
            fc_biases: Buffer::from_slice(device, &fc.biases),
            // DSpark/Qwen3 RMSNorm checkpoints store the actual multiplicative
            // scale, unlike Qwen3.5 target Qwen2RMSNorm weights.
            hidden_norm_weight: Buffer::from_slice(device, hidden_norm.data()),
        })
    }
}

pub struct Qwen35DSparkLayerWeights {
    pub input_norm_weight: Buffer,
    pub post_attention_norm_weight: Buffer,
    pub attention: Qwen35DSparkAttentionWeights,
    pub mlp: Qwen35DSparkMLPWeights,
}

pub struct Qwen35DSparkMarkovWeights {
    pub w1_weight: Buffer,
    pub w1_scales: Buffer,
    pub w1_biases: Buffer,
    pub w2_weight: Buffer,
    pub w2_scales: Buffer,
    pub w2_biases: Buffer,
}

pub struct Qwen35DSparkAttentionWeights {
    /// Row-major affine storage ordered as Q, K, then V. Context append
    /// projects K/V via row offsets into this same immutable allocation.
    pub qkv_weight: Buffer,
    pub qkv_scales: Buffer,
    pub qkv_biases: Buffer,
    pub q_norm_weight: Buffer,
    pub k_norm_weight: Buffer,
    pub output_weight: Buffer,
    pub output_scales: Buffer,
    pub output_biases: Buffer,
}

pub struct Qwen35DSparkMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl Qwen35DSparkLayerWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkLayerPlan,
        bindings: &Qwen35DSparkLayerWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        plan.attention_core.validate();
        plan.attention_metal.validate();
        plan.mlp_core.validate();
        plan.mlp_metal.validate();
        let hidden_dim = plan.attention_core.hidden_dim;
        assert_eq!(hidden_dim, plan.mlp_core.hidden_dim);
        Ok(Self {
            input_norm_weight: actual_scale_norm(device, store, &bindings.input_norm_weight, hidden_dim)?,
            post_attention_norm_weight: actual_scale_norm(
                device,
                store,
                &bindings.post_attention_norm_weight,
                hidden_dim,
            )?,
            attention: Qwen35DSparkAttentionWeights::load(device, store, plan, &bindings.attention)?,
            mlp: Qwen35DSparkMLPWeights::load(device, store, plan, &bindings.mlp)?,
        })
    }
}

impl Qwen35DSparkMarkovWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkPlan,
        bindings: &Qwen35DSparkMarkovWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        assert_eq!(plan.markov_w1.num_embeddings, plan.markov_w2.output_dim);
        assert_eq!(plan.markov_w1.embedding_dim, plan.markov_w2.input_dim);
        let w1 = quantized_matrix(
            store,
            &bindings.w1,
            plan.markov_w1.num_embeddings,
            plan.markov_w1.embedding_dim,
            plan.markov_w1.group_size,
            plan.markov_w1.bits,
        )?;
        let w2 = quantized_matrix(
            store,
            &bindings.w2,
            plan.markov_w2.output_dim,
            plan.markov_w2.input_dim,
            plan.markov_w2.group_size,
            plan.markov_w2.bits,
        )?;
        Ok(Self {
            w1_weight: Buffer::from_slice(device, &w1.weight),
            w1_scales: Buffer::from_slice(device, &w1.scales),
            w1_biases: Buffer::from_slice(device, &w1.biases),
            w2_weight: Buffer::from_slice(device, &w2.weight),
            w2_scales: Buffer::from_slice(device, &w2.scales),
            w2_biases: Buffer::from_slice(device, &w2.biases),
        })
    }
}

impl Qwen35DSparkAttentionWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkLayerPlan,
        bindings: &Qwen35DSparkAttentionWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let core = &plan.attention_core;
        let metal = plan.attention_metal;
        assert_eq!(metal.dtype, Dtype::Bfloat16);
        let q = quantized_matrix(
            store,
            &bindings.q,
            core.q_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
        )?;
        let k = quantized_matrix(
            store,
            &bindings.k,
            core.k_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
        )?;
        let v = quantized_matrix(
            store,
            &bindings.v,
            core.v_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
        )?;
        let output = quantized_matrix(
            store,
            &bindings.output,
            core.hidden_dim,
            core.q_dim(),
            metal.group_size,
            metal.bits,
        )?;
        let qkv_weight = concat_bytes(&[&q.weight, &k.weight, &v.weight]);
        let qkv_scales = concat_bytes(&[&q.scales, &k.scales, &v.scales]);
        let qkv_biases = concat_bytes(&[&q.biases, &k.biases, &v.biases]);
        let qkv_shape = Qwen35DSparkQKVLayout::from_plan(plan).qkv_shape(1);
        validate_len("DSpark QKV weight", qkv_weight.len(), qkv_shape.weight_bytes())?;
        validate_len("DSpark QKV scales", qkv_scales.len(), qkv_shape.affine_param_bytes())?;
        validate_len("DSpark QKV biases", qkv_biases.len(), qkv_shape.affine_param_bytes())?;
        Ok(Self {
            qkv_weight: Buffer::from_slice(device, &qkv_weight),
            qkv_scales: Buffer::from_slice(device, &qkv_scales),
            qkv_biases: Buffer::from_slice(device, &qkv_biases),
            q_norm_weight: actual_scale_norm_f32(device, store, &bindings.q_norm_weight, core.head_dim)?,
            k_norm_weight: actual_scale_norm_f32(device, store, &bindings.k_norm_weight, core.head_dim)?,
            output_weight: Buffer::from_slice(device, &output.weight),
            output_scales: Buffer::from_slice(device, &output.scales),
            output_biases: Buffer::from_slice(device, &output.biases),
        })
    }
}

impl Qwen35DSparkMLPWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkLayerPlan,
        bindings: &Qwen35DSparkMLPWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let core = &plan.mlp_core;
        let metal = plan.mlp_metal;
        assert_eq!(metal.dtype, Dtype::Bfloat16);
        let gate = quantized_matrix(
            store,
            &bindings.gate,
            core.intermediate_dim,
            core.hidden_dim,
            metal.group_size,
            metal.bits,
        )?;
        let up = quantized_matrix(
            store,
            &bindings.up,
            core.intermediate_dim,
            core.hidden_dim,
            metal.group_size,
            metal.bits,
        )?;
        let down = quantized_matrix(
            store,
            &bindings.down,
            core.hidden_dim,
            core.intermediate_dim,
            metal.group_size,
            metal.bits,
        )?;
        let config = QuantizedDenseMLPConfig {
            hidden_dim: to_u32("DSpark MLP hidden_dim", core.hidden_dim)?,
            intermediate_dim: to_u32("DSpark MLP intermediate_dim", core.intermediate_dim)?,
            group_size: metal.group_size,
            bits: metal.bits,
            dtype: metal.dtype,
        };
        let shape = QuantizedDenseMLPShape { num_tokens: 1 };
        let gate_up_shape = config.gate_up_shape(shape);
        let down_shape = config.down_shape(shape);
        let gate_up_weight = concat_bytes(&[&gate.weight, &up.weight]);
        let gate_up_scales = concat_bytes(&[&gate.scales, &up.scales]);
        let gate_up_biases = concat_bytes(&[&gate.biases, &up.biases]);
        validate_len(
            "DSpark MLP gate/up weight",
            gate_up_weight.len(),
            gate_up_shape.weight_bytes(),
        )?;
        validate_len(
            "DSpark MLP gate/up scales",
            gate_up_scales.len(),
            gate_up_shape.affine_param_bytes(),
        )?;
        validate_len(
            "DSpark MLP gate/up biases",
            gate_up_biases.len(),
            gate_up_shape.affine_param_bytes(),
        )?;
        validate_len("DSpark MLP down weight", down.weight.len(), down_shape.weight_bytes())?;
        validate_len(
            "DSpark MLP down scales",
            down.scales.len(),
            down_shape.affine_param_bytes(),
        )?;
        validate_len(
            "DSpark MLP down biases",
            down.biases.len(),
            down_shape.affine_param_bytes(),
        )?;
        Ok(Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight),
            gate_up_scales: Buffer::from_slice(device, &gate_up_scales),
            gate_up_biases: Buffer::from_slice(device, &gate_up_biases),
            down_weight: Buffer::from_slice(device, &down.weight),
            down_scales: Buffer::from_slice(device, &down.scales),
            down_biases: Buffer::from_slice(device, &down.biases),
        })
    }

    pub fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct QuantizedMatrixBytes {
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct QuantizedMatrixLayout {
    shape: AffineQuantizedMatmulShape,
    packed_u32_columns: usize,
    affine_columns: usize,
}

fn quantized_matrix(
    store: &mut SafeTensorStore,
    bindings: &QuantizedTensorBindings,
    output_dim: usize,
    input_dim: usize,
    group_size: u32,
    bits: u32,
) -> Result<QuantizedMatrixBytes, ModelExecutorError> {
    let layout = quantized_matrix_layout(output_dim, input_dim, group_size, bits)?;
    let weight = tensor(
        store,
        &bindings.weight,
        safetensors::Dtype::U32,
        &[output_dim, layout.packed_u32_columns],
    )?
    .into_data();
    let scales = tensor(
        store,
        &bindings.scales,
        safetensors::Dtype::BF16,
        &[output_dim, layout.affine_columns],
    )?
    .into_data();
    let biases = tensor(
        store,
        &bindings.biases,
        safetensors::Dtype::BF16,
        &[output_dim, layout.affine_columns],
    )?
    .into_data();
    validate_len("DSpark affine weight", weight.len(), layout.shape.weight_bytes())?;
    validate_len("DSpark affine scales", scales.len(), layout.shape.affine_param_bytes())?;
    validate_len("DSpark affine biases", biases.len(), layout.shape.affine_param_bytes())?;
    Ok(QuantizedMatrixBytes { weight, scales, biases })
}

fn quantized_matrix_layout(
    output_dim: usize,
    input_dim: usize,
    group_size: u32,
    bits: u32,
) -> Result<QuantizedMatrixLayout, ModelExecutorError> {
    let shape = AffineQuantizedMatmulShape::same_dtype(
        1,
        to_i32("DSpark affine output_dim", output_dim)?,
        to_i32("DSpark affine input_dim", input_dim)?,
        group_size
            .try_into()
            .map_err(|_| ModelExecutorError::custom("DSpark affine group_size must fit i32"))?,
        bits.try_into()
            .map_err(|_| ModelExecutorError::custom("DSpark affine bits must fit i32"))?,
        Dtype::Bfloat16,
    );
    shape.validate();
    let packed_input_bits = input_dim
        .checked_mul(bits as usize)
        .ok_or_else(|| ModelExecutorError::custom("DSpark affine packed input dimension must fit usize"))?;
    if !packed_input_bits.is_multiple_of(32) {
        return Err(ModelExecutorError::custom(format!(
            "DSpark affine packed input bits={packed_input_bits} must be divisible by 32"
        )));
    }
    Ok(QuantizedMatrixLayout {
        shape,
        packed_u32_columns: packed_input_bits / 32,
        affine_columns: input_dim / group_size as usize,
    })
}

fn actual_scale_norm(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    dimension: usize,
) -> Result<Buffer, ModelExecutorError> {
    let norm = tensor(store, name, safetensors::Dtype::BF16, &[dimension])?;
    Ok(Buffer::from_slice(device, norm.data()))
}

fn actual_scale_norm_f32(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    dimension: usize,
) -> Result<Buffer, ModelExecutorError> {
    let norm = tensor(store, name, safetensors::Dtype::BF16, &[dimension])?;
    let values = actual_scale_bf16_bytes_to_f32(norm.data());
    assert_eq!(values.len(), dimension, "DSpark norm length must match its dimension");
    Ok(Buffer::from_slice(device, &values))
}

fn actual_scale_bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0, "DSpark BF16 norm bytes must contain full elements");
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut output = Vec::with_capacity(len);
    for part in parts {
        output.extend_from_slice(part);
    }
    output
}

fn to_i32(name: &str, value: usize) -> Result<i32, ModelExecutorError> {
    value
        .try_into()
        .map_err(|_| ModelExecutorError::custom(format!("{name} must fit i32")))
}

fn to_u32(name: &str, value: usize) -> Result<u32, ModelExecutorError> {
    value
        .try_into()
        .map_err(|_| ModelExecutorError::custom(format!("{name} must fit u32")))
}

fn tensor(
    store: &mut SafeTensorStore,
    name: &str,
    dtype: safetensors::Dtype,
    expected_shape: &[usize],
) -> Result<TensorBytes, ModelExecutorError> {
    let tensor = store.tensor_bytes(name, dtype)?;
    if tensor.shape() != expected_shape {
        return Err(ModelExecutorError::custom(format!(
            "unexpected shape for tensor {name:?}: expected {expected_shape:?}, got {:?}",
            tensor.shape()
        )));
    }
    Ok(tensor)
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "unexpected {name} byte length: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::actual_scale_bf16_bytes_to_f32;
    use super::concat_bytes;
    use super::quantized_matrix_layout;

    #[test]
    fn quantized_layout_matches_converter_shapes() {
        let body = quantized_matrix_layout(5120, 5120, 64, 4).unwrap();
        assert_eq!(body.packed_u32_columns, 640);
        assert_eq!(body.affine_columns, 80);

        let selected_fc = quantized_matrix_layout(5120, 25_600, 64, 3).unwrap();
        assert_eq!(selected_fc.packed_u32_columns, 2400);
        assert_eq!(selected_fc.affine_columns, 400);

        let markov_w1 = quantized_matrix_layout(248_320, 1024, 64, 4).unwrap();
        assert_eq!(markov_w1.packed_u32_columns, 128);
        assert_eq!(markov_w1.affine_columns, 16);

        let markov_w2 = quantized_matrix_layout(248_320, 1024, 64, 8).unwrap();
        assert_eq!(markov_w2.packed_u32_columns, 256);
        assert_eq!(markov_w2.affine_columns, 16);
    }

    #[test]
    fn qkv_byte_concatenation_preserves_projection_row_order() {
        assert_eq!(concat_bytes(&[&[1, 2], &[3], &[4, 5]]), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn qk_norm_weights_expand_from_bf16_to_f32_without_qwen_next_offset() {
        let values = [bf16::from_f32(0.5).to_bits(), bf16::from_f32(1.25).to_bits()];
        let bytes = values.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(actual_scale_bf16_bytes_to_f32(&bytes), [0.5, 1.25]);
    }
}
