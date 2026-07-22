//! Small CPU reference helpers shared by component-specific test oracles.

pub fn silu_reference(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn sigmoid_reference(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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
    rows: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), dim);
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let row_offset = row * dim;
        let sum_sq = input[row_offset..row_offset + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inv_rms = (sum_sq / dim as f32 + eps).sqrt().recip();
        for dim_index in 0..dim {
            output[row_offset + dim_index] = input[row_offset + dim_index] * inv_rms * weight[dim_index]
                + bias.map(|bias| bias[dim_index]).unwrap_or(0.0);
        }
    }
    output
}

pub fn dense_linear_reference(
    input: &[f32],  // [m, k] row major
    weight: &[f32], // [n, k] row major
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), m * k);
    assert_eq!(weight.len(), n * k);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n);
    }

    let mut output = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = bias.map(|b| b[j]).unwrap_or(0.0);
            for p in 0..k {
                acc += input[i * k + p] * weight[j * k + p];
            }
            output[i * n + j] = acc;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

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
