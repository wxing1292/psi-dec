//! CPU reference implementation for MoE routing and combine tests.

use half::bf16;

use crate::mlp::dense::reference::QuantizedAffineReferenceShape;
use crate::mlp::dense::reference::quantized_affine_reference;
use crate::reference::sigmoid_reference;
use crate::reference::silu_reference;
use crate::reference::softmax_reference;

#[derive(Clone, Debug, PartialEq)]
pub struct MoERoutingReference {
    pub expert_indices: Vec<u32>,
    pub expert_probs: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPReferenceWeights<'a> {
    pub gate_weight: &'a [u8],
    pub gate_scales: &'a [f32],
    pub gate_biases: &'a [f32],
    pub up_weight: &'a [u8],
    pub up_scales: &'a [f32],
    pub up_biases: &'a [f32],
    pub down_weight: &'a [u8],
    pub down_scales: &'a [f32],
    pub down_biases: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPTokenMajorReferenceInput<'a> {
    pub input: &'a [f32],
    pub token_indices: &'a [u32],
    pub expert_indices: &'a [u32],
    pub route_indices: &'a [u32],
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub group_size: usize,
    pub bits: usize,
    pub num_experts: usize,
    pub weights: QuantizedSparseMLPReferenceWeights<'a>,
}

pub fn moe_routing_reference(
    router_logits: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_logits.len(), num_tokens * num_experts);
    let mut router_probs = Vec::with_capacity(router_logits.len());
    for token in 0..num_tokens {
        router_probs.extend(softmax_reference(
            &router_logits[token * num_experts..(token + 1) * num_experts],
        ));
    }
    moe_routing_from_probs_reference(
        &router_probs,
        num_tokens,
        num_experts,
        num_experts_per_token,
        norm_topk_prob,
    )
}

pub fn moe_routing_from_probs_reference(
    router_probs: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_probs.len(), num_tokens * num_experts);
    assert!(num_experts_per_token > 0);
    assert!(num_experts_per_token <= num_experts);

    let mut expert_indices = Vec::with_capacity(num_tokens * num_experts_per_token);
    let mut expert_probs = Vec::with_capacity(num_tokens * num_experts_per_token);
    for token in 0..num_tokens {
        let mut ranked = router_probs[token * num_experts..(token + 1) * num_experts]
            .iter()
            .enumerate()
            .map(|(expert, &prob)| (expert as u32, prob))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap().then_with(|| left.0.cmp(&right.0)));
        ranked.truncate(num_experts_per_token);
        let selected_total = ranked.iter().map(|(_, prob)| *prob).sum::<f32>();
        for (expert, prob) in ranked {
            expert_indices.push(expert);
            expert_probs.push(if norm_topk_prob { prob / selected_total } else { prob });
        }
    }
    MoERoutingReference {
        expert_indices,
        expert_probs,
    }
}

pub fn moe_routing_from_bf16_probs_reference(
    router_probs: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_probs.len(), num_tokens * num_experts);
    assert!(num_experts_per_token > 0);
    assert!(num_experts_per_token <= num_experts);
    assert!(num_experts <= 256);
    assert!(num_experts_per_token <= 16);

    let router_probs = router_probs
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let mut expert_indices = Vec::with_capacity(num_tokens * num_experts_per_token);
    let mut expert_probs = Vec::with_capacity(num_tokens * num_experts_per_token);
    for token in 0..num_tokens {
        let mut selected = Vec::<(u32, f32)>::with_capacity(num_experts_per_token);
        for _ in 0..num_experts_per_token {
            let mut best = (u32::MAX, -1.0f32);
            for expert in 0..num_experts {
                if selected
                    .iter()
                    .any(|(selected_expert, _)| *selected_expert == expert as u32)
                {
                    continue;
                }
                let prob = router_probs[token * num_experts + expert];
                if prob > best.1 || (prob == best.1 && (expert as u32) < best.0) {
                    best = (expert as u32, prob);
                }
            }
            selected.push(best);
        }

        let mut topk_sum = 0.0f32;
        for (_, prob) in &selected {
            topk_sum = bf16::from_f32(topk_sum + *prob).to_f32();
        }
        for (expert, mut prob) in selected {
            if norm_topk_prob && num_experts_per_token > 1 && topk_sum > 0.0 {
                prob = bf16::from_f32(prob / topk_sum).to_f32();
            }
            expert_indices.push(expert);
            expert_probs.push(bf16::from_f32(prob).to_f32());
        }
    }
    MoERoutingReference {
        expert_indices,
        expert_probs,
    }
}

pub fn quantized_sparse_mlp_token_major_reference(
    reference_input: QuantizedSparseMLPTokenMajorReferenceInput<'_>,
) -> Vec<f32> {
    assert_eq!(
        reference_input.token_indices.len(),
        reference_input.expert_indices.len()
    );
    assert_eq!(reference_input.token_indices.len(), reference_input.route_indices.len());
    assert_eq!(reference_input.input.len() % reference_input.hidden_dim, 0);
    let hidden_dim = reference_input.hidden_dim;
    let intermediate_dim = reference_input.intermediate_dim;
    let group_size = reference_input.group_size;
    let bits = reference_input.bits;
    let num_experts = reference_input.num_experts;
    let weights = reference_input.weights;
    let num_input_vectors = reference_input.input.len() / hidden_dim;
    let num_routes = reference_input.token_indices.len();
    let gate_up_weight_bytes = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: intermediate_dim,
        input_dim: hidden_dim,
        group_size,
        bits,
    }
    .weight_bytes();
    let gate_up_affine_len = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: intermediate_dim,
        input_dim: hidden_dim,
        group_size,
        bits,
    }
    .affine_param_len();
    let down_weight_bytes = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: hidden_dim,
        input_dim: intermediate_dim,
        group_size,
        bits,
    }
    .weight_bytes();
    let down_affine_len = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: hidden_dim,
        input_dim: intermediate_dim,
        group_size,
        bits,
    }
    .affine_param_len();
    assert_eq!(weights.gate_weight.len(), num_experts * gate_up_weight_bytes);
    assert_eq!(weights.up_weight.len(), num_experts * gate_up_weight_bytes);
    assert_eq!(weights.down_weight.len(), num_experts * down_weight_bytes);
    assert_eq!(weights.gate_scales.len(), num_experts * gate_up_affine_len);
    assert_eq!(weights.gate_biases.len(), num_experts * gate_up_affine_len);
    assert_eq!(weights.up_scales.len(), num_experts * gate_up_affine_len);
    assert_eq!(weights.up_biases.len(), num_experts * gate_up_affine_len);
    assert_eq!(weights.down_scales.len(), num_experts * down_affine_len);
    assert_eq!(weights.down_biases.len(), num_experts * down_affine_len);

    let mut activation_by_route = vec![0.0_f32; num_routes * intermediate_dim];
    for route in 0..num_routes {
        let token = reference_input.token_indices[route] as usize;
        let expert = reference_input.expert_indices[route] as usize;
        assert!(token < num_input_vectors);
        assert!(expert < num_experts);

        let input_row = &reference_input.input[token * hidden_dim..(token + 1) * hidden_dim];
        let gate = quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: 1,
                output_dim: intermediate_dim,
                input_dim: hidden_dim,
                group_size,
                bits,
            },
            input_row,
            expert_slice(weights.gate_weight, expert, gate_up_weight_bytes),
            expert_slice(weights.gate_scales, expert, gate_up_affine_len),
            expert_slice(weights.gate_biases, expert, gate_up_affine_len),
        );
        let up = quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: 1,
                output_dim: intermediate_dim,
                input_dim: hidden_dim,
                group_size,
                bits,
            },
            input_row,
            expert_slice(weights.up_weight, expert, gate_up_weight_bytes),
            expert_slice(weights.up_scales, expert, gate_up_affine_len),
            expert_slice(weights.up_biases, expert, gate_up_affine_len),
        );
        let activation = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate, &up)| silu_reference(gate) * up)
            .collect::<Vec<_>>();
        activation_by_route[route * intermediate_dim..(route + 1) * intermediate_dim].copy_from_slice(&activation);
    }

    let mut output = vec![0.0_f32; num_routes * hidden_dim];
    for route in 0..num_routes {
        let activation_route = reference_input.route_indices[route] as usize;
        let expert = reference_input.expert_indices[route] as usize;
        assert!(activation_route < num_routes);
        assert!(expert < num_experts);
        let route_output = quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: 1,
                output_dim: hidden_dim,
                input_dim: intermediate_dim,
                group_size,
                bits,
            },
            &activation_by_route[activation_route * intermediate_dim..(activation_route + 1) * intermediate_dim],
            expert_slice(weights.down_weight, expert, down_weight_bytes),
            expert_slice(weights.down_scales, expert, down_affine_len),
            expert_slice(weights.down_biases, expert, down_affine_len),
        );
        output[route * hidden_dim..(route + 1) * hidden_dim].copy_from_slice(&route_output);
    }
    output
}

fn expert_slice<T>(values: &[T], expert: usize, stride: usize) -> &[T] {
    &values[expert * stride..(expert + 1) * stride]
}

pub fn moe_combine_without_common_reference(
    routed_hidden: &[f32],
    routed_probs: &[f32],
    num_tokens: usize,
    num_experts_per_token: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(routed_hidden.len(), num_tokens * num_experts_per_token * hidden_dim);
    assert_eq!(routed_probs.len(), num_tokens * num_experts_per_token);
    let mut output = vec![0.0; num_tokens * hidden_dim];
    for token in 0..num_tokens {
        for route in 0..num_experts_per_token {
            let route_index = token * num_experts_per_token + route;
            let route_weight = routed_probs[route_index];
            for dim in 0..hidden_dim {
                output[token * hidden_dim + dim] += route_weight * routed_hidden[route_index * hidden_dim + dim];
            }
        }
    }
    output
}

pub fn moe_combine_with_common_reference(
    routed_output: &[f32],
    common_hidden: &[f32],
    common_gate_logits: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(routed_output.len(), num_tokens * hidden_dim);
    assert_eq!(common_hidden.len(), num_tokens * hidden_dim);
    assert_eq!(common_gate_logits.len(), num_tokens);
    let mut output = routed_output.to_vec();
    for token in 0..num_tokens {
        let gate = sigmoid_reference(common_gate_logits[token]);
        for dim in 0..hidden_dim {
            output[token * hidden_dim + dim] += gate * common_hidden[token * hidden_dim + dim];
        }
    }
    output
}

pub fn moe_combine_without_common_bf16_reference(
    routed_hidden: &[f32],
    routed_probs: &[f32],
    num_tokens: usize,
    num_experts_per_token: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    assert_eq!(routed_hidden.len(), num_tokens * num_experts_per_token * hidden_dim);
    assert_eq!(routed_probs.len(), num_tokens * num_experts_per_token);
    let mut output = Vec::with_capacity(num_tokens * hidden_dim);
    for token in 0..num_tokens {
        for dim in 0..hidden_dim {
            let mut acc = 0.0f32;
            for slot in 0..num_experts_per_token {
                let route = token * num_experts_per_token + slot;
                let route_weight = bf16::from_f32(routed_probs[route]).to_f32();
                let hidden = bf16::from_f32(routed_hidden[route * hidden_dim + dim]).to_f32();
                let weighted = bf16::from_f32(route_weight * hidden).to_f32();
                acc = bf16::from_f32(acc + weighted).to_f32();
            }
            output.push(bf16::from_f32(acc).to_bits());
        }
    }
    output
}

pub fn moe_combine_with_common_bf16_reference(
    routed_output: &[u16],
    common_hidden: &[f32],
    common_gate_logits: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    assert_eq!(routed_output.len(), num_tokens * hidden_dim);
    assert_eq!(common_hidden.len(), num_tokens * hidden_dim);
    assert_eq!(common_gate_logits.len(), num_tokens);
    let mut output = Vec::with_capacity(num_tokens * hidden_dim);
    for (token, &common_gate_logit) in common_gate_logits.iter().enumerate().take(num_tokens) {
        let gate_logit = bf16::from_f32(common_gate_logit).to_f32();
        let gate = sigmoid_reference(gate_logit);
        for dim in 0..hidden_dim {
            let gid = token * hidden_dim + dim;
            let routed = bf16::from_bits(routed_output[gid]).to_f32();
            let common = bf16::from_f32(common_hidden[gid]).to_f32();
            output.push(bf16::from_f32(routed + gate * common).to_bits());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing() {
        let routed = moe_routing_reference(&[0.25, 2.0, -1.0, 1.0, 3.0, 3.0, 0.5, -2.0], 2, 4, 2, true);

        assert_eq!(routed.expert_indices, vec![1, 3, 0, 1]);
        assert!((routed.expert_probs[0] + routed.expert_probs[1] - 1.0).abs() < 1.0e-6);
        assert!((routed.expert_probs[2] + routed.expert_probs[3] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_combine() {
        let routed = moe_combine_without_common_reference(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.75], 1, 2, 2);
        assert_eq!(routed, vec![2.5, 3.5]);

        let output = moe_combine_with_common_reference(&routed, &[10.0, -2.0], &[0.0], 1, 2);
        assert_eq!(output, vec![7.5, 2.5]);
    }
}
