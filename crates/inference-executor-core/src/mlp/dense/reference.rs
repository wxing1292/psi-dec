//! CPU reference implementation for dense gated MLP tests.

use crate::mlp::dense::DenseMLPCore;
use crate::reference::dense_linear_reference;
use crate::reference::silu_reference;

#[derive(Clone, Debug)]
pub struct DenseMLPReferenceWeights<'a> {
    pub gate_weight: &'a [f32],
    pub gate_bias: Option<&'a [f32]>,
    pub up_weight: &'a [f32],
    pub up_bias: Option<&'a [f32]>,
    pub down_weight: &'a [f32],
    pub down_bias: Option<&'a [f32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedAffineReferenceShape {
    pub num_rows: usize,
    pub output_dim: usize,
    pub input_dim: usize,
    pub group_size: usize,
    pub bits: usize,
}

#[derive(Clone, Debug)]
pub struct QuantizedDenseMLPReferenceWeights<'a> {
    pub gate_up_weight: &'a [u8],
    pub gate_up_scales: &'a [f32],
    pub gate_up_biases: &'a [f32],
    pub down_weight: &'a [u8],
    pub down_scales: &'a [f32],
    pub down_biases: &'a [f32],
}

pub fn dense_mlp_reference(
    core: &DenseMLPCore,
    input: &[f32],
    num_tokens: usize,
    weights: DenseMLPReferenceWeights<'_>,
) -> Vec<f32> {
    core.validate();
    assert_eq!(input.len(), num_tokens * core.hidden_dim);
    let gate = dense_linear_reference(
        input,
        weights.gate_weight,
        weights.gate_bias,
        num_tokens,
        core.hidden_dim,
        core.intermediate_dim,
    );
    let up = dense_linear_reference(
        input,
        weights.up_weight,
        weights.up_bias,
        num_tokens,
        core.hidden_dim,
        core.intermediate_dim,
    );
    let activation = gate
        .iter()
        .zip(up.iter())
        .map(|(&gate, &up)| silu_reference(gate) * up)
        .collect::<Vec<_>>();
    dense_linear_reference(
        &activation,
        weights.down_weight,
        weights.down_bias,
        num_tokens,
        core.intermediate_dim,
        core.hidden_dim,
    )
}

pub fn quantized_affine_reference(
    shape: QuantizedAffineReferenceShape,
    input: &[f32],
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    shape.validate();
    assert_eq!(input.len(), shape.num_rows * shape.input_dim);
    assert_eq!(weight.len(), shape.weight_bytes());
    assert_eq!(scales.len(), shape.affine_param_len());
    assert_eq!(biases.len(), shape.affine_param_len());

    let groups = shape.input_dim / shape.group_size;
    let mut output = vec![0.0_f32; shape.num_rows * shape.output_dim];
    for row in 0..shape.num_rows {
        let input_row = &input[row * shape.input_dim..(row + 1) * shape.input_dim];
        for output_col in 0..shape.output_dim {
            let mut value = 0.0_f32;
            for group in 0..groups {
                let group_start = group * shape.group_size;
                let group_end = group_start + shape.group_size;
                let mut dot = 0.0_f32;
                let mut input_sum = 0.0_f32;
                for (input_col, &input_value) in input_row.iter().enumerate().take(group_end).skip(group_start) {
                    input_sum += input_value;
                    dot += input_value * quantized_weight_value(shape, weight, output_col, input_col) as f32;
                }
                let affine_index = output_col * groups + group;
                value += scales[affine_index] * dot + biases[affine_index] * input_sum;
            }
            output[row * shape.output_dim + output_col] = value;
        }
    }
    output
}

pub fn quantized_dense_mlp_reference(
    core: &DenseMLPCore,
    input: &[f32],
    num_tokens: usize,
    group_size: usize,
    bits: usize,
    weights: QuantizedDenseMLPReferenceWeights<'_>,
) -> Vec<f32> {
    core.validate();
    assert_eq!(input.len(), num_tokens * core.hidden_dim);

    let gate_up = quantized_affine_reference(
        QuantizedAffineReferenceShape {
            num_rows: num_tokens,
            output_dim: core.intermediate_dim * 2,
            input_dim: core.hidden_dim,
            group_size,
            bits,
        },
        input,
        weights.gate_up_weight,
        weights.gate_up_scales,
        weights.gate_up_biases,
    );
    let mut activation = vec![0.0_f32; num_tokens * core.intermediate_dim];
    for row in 0..num_tokens {
        let gate_up_row = &gate_up[row * core.intermediate_dim * 2..(row + 1) * core.intermediate_dim * 2];
        for col in 0..core.intermediate_dim {
            let gate = gate_up_row[col];
            let up = gate_up_row[core.intermediate_dim + col];
            activation[row * core.intermediate_dim + col] = silu_reference(gate) * up;
        }
    }
    quantized_affine_reference(
        QuantizedAffineReferenceShape {
            num_rows: num_tokens,
            output_dim: core.hidden_dim,
            input_dim: core.intermediate_dim,
            group_size,
            bits,
        },
        &activation,
        weights.down_weight,
        weights.down_scales,
        weights.down_biases,
    )
}

impl QuantizedAffineReferenceShape {
    pub fn validate(self) {
        assert!(self.num_rows > 0);
        assert!(self.output_dim > 0);
        assert!(self.input_dim > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.input_dim % self.group_size, 0);
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        let pack_factor = quantized_pack_factor(self.bits);
        let bytes_per_pack = quantized_bytes_per_pack(self.bits);
        self.output_dim * self.input_dim * bytes_per_pack / pack_factor
    }

    pub fn affine_param_len(self) -> usize {
        self.validate();
        self.output_dim * (self.input_dim / self.group_size)
    }
}

fn quantized_weight_value(
    shape: QuantizedAffineReferenceShape,
    weight: &[u8],
    output_col: usize,
    input_col: usize,
) -> u32 {
    let pack_factor = quantized_pack_factor(shape.bits);
    let bytes_per_pack = quantized_bytes_per_pack(shape.bits);
    let row_bytes = shape.input_dim * bytes_per_pack / pack_factor;
    let pack_index = input_col / pack_factor;
    let value_index = input_col % pack_factor;
    let byte_index = output_col * row_bytes + pack_index * bytes_per_pack;
    let mut packed = 0_u32;
    for byte_offset in 0..bytes_per_pack {
        packed |= u32::from(weight[byte_index + byte_offset]) << (8 * byte_offset);
    }
    let mask = (1_u32 << shape.bits) - 1;
    (packed >> (value_index * shape.bits)) & mask
}

fn quantized_pack_factor(bits: usize) -> usize {
    match bits {
        3 => 8,
        6 => 4,
        2 | 4 | 8 => 8 / bits,
        _ => panic!("unsupported quantized affine bits {bits}"),
    }
}

fn quantized_bytes_per_pack(bits: usize) -> usize {
    match bits {
        3 | 6 => 3,
        2 | 4 | 8 => 1,
        _ => panic!("unsupported quantized affine bits {bits}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed() {
        let core = DenseMLPCore {
            model_layer_index: 0,
            hidden_dim: 2,
            intermediate_dim: 2,
        };
        let output = dense_mlp_reference(
            &core,
            &[1.0, -2.0],
            1,
            DenseMLPReferenceWeights {
                gate_weight: &[1.0, 0.0, 0.0, 1.0],
                gate_bias: None,
                up_weight: &[2.0, 0.0, 0.0, 3.0],
                up_bias: None,
                down_weight: &[1.0, 0.5, -1.0, 2.0],
                down_bias: Some(&[0.25, -0.5]),
            },
        );

        let hidden0 = silu_reference(1.0) * 2.0;
        let hidden1 = silu_reference(-2.0) * -6.0;
        assert!((output[0] - (hidden0 + 0.5 * hidden1 + 0.25)).abs() < 1.0e-6);
        assert!((output[1] - (-hidden0 + 2.0 * hidden1 - 0.5)).abs() < 1.0e-6);
    }

    #[test]
    fn test_quantized() {
        let shape = QuantizedAffineReferenceShape {
            num_rows: 1,
            output_dim: 1,
            input_dim: 32,
            group_size: 32,
            bits: 4,
        };
        let mut weight = vec![0_u8; shape.weight_bytes()];
        for input_col in 0..shape.input_dim {
            let value = (input_col % 16) as u8;
            let byte_index = input_col / 2;
            if input_col % 2 == 0 {
                weight[byte_index] |= value;
            } else {
                weight[byte_index] |= value << 4;
            }
        }
        let input = vec![1.0_f32; shape.input_dim];
        let output = quantized_affine_reference(shape, &input, &weight, &[0.5], &[0.25]);
        let expected_dot = (0..shape.input_dim).map(|index| (index % 16) as f32).sum::<f32>();
        assert_eq!(output, vec![0.5 * expected_dot + 0.25 * shape.input_dim as f32]);
    }
}
