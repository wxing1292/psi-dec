//! Small CPU reference helpers shared by component-specific test oracles.

pub fn sigmoid_reference(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn silu_reference(value: f32) -> f32 {
    value * sigmoid_reference(value)
}

pub fn softplus_reference(value: f32) -> f32 {
    if value > 20.0 { value } else { (1.0 + value.exp()).ln() }
}

pub fn softmax_reference(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max_value, value| max_value.max(value));
    let weights = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    assert!(
        total > 0.0 && total.is_finite(),
        "softmax reference requires finite positive total"
    );
    weights.into_iter().map(|weight| weight / total).collect()
}

pub fn rms_norm_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    num_rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), num_rows * hidden_dim);
    assert_eq!(weight.len(), hidden_dim);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), hidden_dim);
    }
    let mut output = vec![0.0; input.len()];
    for row_index in 0..num_rows {
        let row_offset = row_index * hidden_dim;
        let sum_squares = input[row_offset..row_offset + hidden_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inverse_rms = (sum_squares / hidden_dim as f32 + eps).sqrt().recip();
        for hidden_index in 0..hidden_dim {
            output[row_offset + hidden_index] = input[row_offset + hidden_index] * inverse_rms * weight[hidden_index]
                + bias.map(|bias| bias[hidden_index]).unwrap_or(0.0);
        }
    }
    output
}

pub fn dense_linear_reference(
    input: &[f32],  // [num_rows, input_dim] row major
    weight: &[f32], // [output_dim, input_dim] row major
    bias: Option<&[f32]>,
    num_rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), num_rows * input_dim);
    assert_eq!(weight.len(), output_dim * input_dim);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_dim);
    }

    let mut output = vec![0.0; num_rows * output_dim];

    for row_index in 0..num_rows {
        for output_index in 0..output_dim {
            let mut value = bias.map(|bias| bias[output_index]).unwrap_or(0.0);
            for input_index in 0..input_dim {
                value += input[row_index * input_dim + input_index] * weight[output_index * input_dim + input_index];
            }
            output[row_index * output_dim + output_index] = value;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_activations() {
        assert_eq!(sigmoid_reference(0.0), 0.5);
        assert_eq!(silu_reference(0.0), 0.0);
        assert!((softplus_reference(0.0) - 2.0_f32.ln()).abs() < 1.0e-6);
    }

    #[test]
    fn test_softmax_reference_normalizes_logits() {
        let output = softmax_reference(&[0.0, 3.0_f32.ln()]);

        assert!((output[0] - 0.25).abs() < 1.0e-6);
        assert!((output[1] - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn test_rms_norm_reference_normalizes_rows() {
        let output = rms_norm_reference(&[3.0, 4.0], &[1.0, 2.0], None, 1, 2, 0.0);

        assert!((output[0] - 0.84852815).abs() < 1.0e-6);
        assert!((output[1] - 2.2627418).abs() < 1.0e-6);
    }

    #[test]
    fn test_dense_linear_reference_uses_row_major_weight() {
        let output = dense_linear_reference(&[1.0, 2.0], &[3.0, 4.0, -1.0, 0.5], Some(&[0.25, -0.5]), 1, 2, 2);

        assert_eq!(output, vec![11.25, -0.5]);
    }
}
