//! CPU reference implementation for Gated DeltaNet primitive tests.

use crate::attn::gdn::GDNCore;
use crate::reference::silu_reference;
use crate::reference::softplus_reference;

#[derive(Clone, Debug, PartialEq)]
pub struct GDNShortConvReference {
    pub conv_qkv: Vec<f32>,
    pub next_conv_state: Vec<f32>,
}

pub fn gdn_short_conv_reference(
    core: &GDNCore,
    cu_tokens: &[u32],
    source_conv_state: &[f32],
    projected_qkv: &[f32],
    conv_weight: &[f32],
) -> GDNShortConvReference {
    core.validate();
    let num_reqs = cu_tokens.len() - 1;
    let num_tokens = *cu_tokens.last().unwrap() as usize;
    let conv_state_len = core.conv_state_len();
    assert_eq!(source_conv_state.len(), num_reqs * core.qkv_dim() * conv_state_len);
    assert_eq!(projected_qkv.len(), num_tokens * core.qkv_dim());
    assert_eq!(conv_weight.len(), core.qkv_dim() * core.conv_kernel_size);

    let mut conv_qkv = vec![0.0; num_tokens * core.qkv_dim()];
    let mut next_conv_state = vec![0.0; num_reqs * core.qkv_dim() * conv_state_len];
    for req_index in 0..num_reqs {
        let flat_token_begin = cu_tokens[req_index] as usize;
        let flat_token_end = cu_tokens[req_index + 1] as usize;
        let num_req_tokens = flat_token_end - flat_token_begin;
        for qkv_channel_index in 0..core.qkv_dim() {
            for token_index_in_req in 0..num_req_tokens {
                let mut acc = 0.0;
                for kernel_index in 0..core.conv_kernel_size {
                    let sequence_index =
                        token_index_in_req as isize + kernel_index as isize - core.conv_state_len() as isize;
                    let x = if sequence_index < 0 {
                        let state_index = (sequence_index + core.conv_state_len() as isize) as usize;
                        source_conv_state
                            [(req_index * core.qkv_dim() + qkv_channel_index) * conv_state_len + state_index]
                    } else {
                        projected_qkv
                            [((flat_token_begin + sequence_index as usize) * core.qkv_dim()) + qkv_channel_index]
                    };
                    acc += x * conv_weight[qkv_channel_index * core.conv_kernel_size + kernel_index];
                }
                conv_qkv[((flat_token_begin + token_index_in_req) * core.qkv_dim()) + qkv_channel_index] =
                    silu_reference(acc);
            }

            for state_index in 0..conv_state_len {
                let sequence_index = num_req_tokens as isize + state_index as isize - conv_state_len as isize;
                let x = if sequence_index < 0 {
                    let source_state_index = state_index + num_req_tokens;
                    source_conv_state
                        [(req_index * core.qkv_dim() + qkv_channel_index) * conv_state_len + source_state_index]
                } else {
                    projected_qkv[((flat_token_begin + sequence_index as usize) * core.qkv_dim()) + qkv_channel_index]
                };
                next_conv_state[(req_index * core.qkv_dim() + qkv_channel_index) * conv_state_len + state_index] = x;
            }
        }
    }

    GDNShortConvReference {
        conv_qkv,
        next_conv_state,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GDNRecurrentReference {
    pub recurrent_output: Vec<f32>,
    pub next_recurrent_state: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct GDNRecurrentReferenceInput<'a> {
    pub cu_tokens: &'a [u32],
    pub source_recurrent_state: &'a [f32],
    pub conv_qkv: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub a_log_decay: &'a [f32],
    pub dt_bias: &'a [f32],
}

pub fn gdn_recurrent_reference(core: &GDNCore, input: GDNRecurrentReferenceInput<'_>) -> GDNRecurrentReference {
    core.validate();
    let cu_tokens = input.cu_tokens;
    let num_reqs = cu_tokens.len() - 1;
    let num_tokens = *cu_tokens.last().unwrap() as usize;
    let recurrent_state_stride = core.num_v_heads * core.v_head_dim * core.qk_head_dim;
    assert_eq!(input.source_recurrent_state.len(), num_reqs * recurrent_state_stride);
    assert_eq!(input.conv_qkv.len(), num_tokens * core.qkv_dim());
    assert_eq!(input.a.len(), num_tokens * core.num_v_heads);
    assert_eq!(input.b.len(), num_tokens * core.num_v_heads);
    assert_eq!(input.a_log_decay.len(), core.num_v_heads);
    assert_eq!(input.dt_bias.len(), core.num_v_heads);

    let mut recurrent_output = vec![0.0; num_tokens * core.v_dim()];
    let mut next_recurrent_state = input.source_recurrent_state.to_vec();
    let num_v_heads_per_qk_head = core.num_v_heads / core.num_qk_heads;
    let k_base = core.qk_dim();
    let v_base = k_base + core.qk_dim();

    for req_index in 0..num_reqs {
        for flat_token_index in cu_tokens[req_index] as usize..cu_tokens[req_index + 1] as usize {
            for v_head_index in 0..core.num_v_heads {
                let qk_head_index = v_head_index / num_v_heads_per_qk_head;
                let q_inv_norm = inverse_l2_norm(
                    input.conv_qkv,
                    flat_token_index,
                    core.qkv_dim(),
                    qk_head_index,
                    core.qk_head_dim,
                    0,
                ) * core.q_scale;
                let k_inv_norm = inverse_l2_norm(
                    input.conv_qkv,
                    flat_token_index,
                    core.qkv_dim(),
                    qk_head_index,
                    core.qk_head_dim,
                    k_base,
                );
                let gate_offset = flat_token_index * core.num_v_heads + v_head_index;
                let beta_t = 1.0 / (1.0 + (-input.b[gate_offset]).exp());
                let dt = input.a[gate_offset] + input.dt_bias[v_head_index];
                let decay = (input.a_log_decay[v_head_index] * softplus_reference(dt)).exp();
                for v_dim_index in 0..core.v_head_dim {
                    let v_t = input.conv_qkv
                        [flat_token_index * core.qkv_dim() + v_base + v_head_index * core.v_head_dim + v_dim_index];
                    let state_base = req_index * recurrent_state_stride
                        + (v_head_index * core.v_head_dim + v_dim_index) * core.qk_head_dim;
                    let mut state_k_dot = 0.0;
                    for qk_dim_index in 0..core.qk_head_dim {
                        let k_norm = input.conv_qkv[flat_token_index * core.qkv_dim()
                            + k_base
                            + qk_head_index * core.qk_head_dim
                            + qk_dim_index]
                            * k_inv_norm;
                        let state_index = state_base + qk_dim_index;
                        let decayed_state = next_recurrent_state[state_index] * decay;
                        next_recurrent_state[state_index] = decayed_state;
                        state_k_dot += decayed_state * k_norm;
                    }
                    let delta = (v_t - state_k_dot) * beta_t;
                    let mut recurrent_output_value = 0.0;
                    for qk_dim_index in 0..core.qk_head_dim {
                        let q_norm = input.conv_qkv
                            [flat_token_index * core.qkv_dim() + qk_head_index * core.qk_head_dim + qk_dim_index]
                            * q_inv_norm;
                        let k_norm = input.conv_qkv[flat_token_index * core.qkv_dim()
                            + k_base
                            + qk_head_index * core.qk_head_dim
                            + qk_dim_index]
                            * k_inv_norm;
                        let state_index = state_base + qk_dim_index;
                        next_recurrent_state[state_index] += k_norm * delta;
                        recurrent_output_value += next_recurrent_state[state_index] * q_norm;
                    }
                    recurrent_output[flat_token_index * core.v_dim() + v_head_index * core.v_head_dim + v_dim_index] =
                        recurrent_output_value;
                }
            }
        }
    }

    GDNRecurrentReference {
        recurrent_output,
        next_recurrent_state,
    }
}

fn inverse_l2_norm(
    conv_qkv: &[f32],
    flat_token_index: usize,
    qkv_dim: usize,
    qk_head_index: usize,
    qk_head_dim: usize,
    qkv_base: usize,
) -> f32 {
    let mut square_sum = 0.0;
    for qk_dim_index in 0..qk_head_dim {
        let value = conv_qkv[flat_token_index * qkv_dim + qkv_base + qk_head_index * qk_head_dim + qk_dim_index];
        square_sum += value * value;
    }
    (square_sum + 1.0e-6).sqrt().recip()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core() -> GDNCore {
        GDNCore {
            model_layer_index: 0,
            hidden_dim: 4,
            num_qk_heads: 1,
            qk_head_dim: 2,
            num_v_heads: 1,
            v_head_dim: 2,
            conv_kernel_size: 3,
            q_scale: 1.0,
        }
    }

    #[test]
    fn test_short_conv() {
        let core = core();
        let output = gdn_short_conv_reference(&core, &[0, 1], &[1.0; 12], &[0.5; 6], &[1.0; 18]);

        assert_eq!(output.conv_qkv.len(), 6);
        assert_eq!(output.next_conv_state.len(), 12);
        assert_eq!(&output.next_conv_state[0..2], &[1.0, 0.5]);
    }

    #[test]
    fn test_recurrent() {
        let core = core();
        let output = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &[0, 1],
                source_recurrent_state: &[0.0; 4],
                conv_qkv: &[0.0, 1.0, 0.0, 1.0, 2.0, -1.0],
                a: &[0.0],
                b: &[0.0],
                a_log_decay: &[0.0],
                dt_bias: &[0.0],
            },
        );

        assert_eq!(output.recurrent_output.len(), 2);
        assert_eq!(output.next_recurrent_state.len(), 4);
        assert!(output.recurrent_output[0] > 0.0);
    }
}
