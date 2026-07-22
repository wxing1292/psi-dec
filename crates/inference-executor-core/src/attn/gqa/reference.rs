//! CPU reference implementation for already-projected GQA attention tests.

use crate::attn::gqa::GQACore;
use crate::reference::softmax_reference;

pub struct GQAReferenceInput<'a> {
    pub cu_tokens: &'a [u32],
    pub token_indices: &'a [u32],
    pub q: &'a [f32],
    pub context_k_by_req: &'a [&'a [f32]],
    pub context_v_by_req: &'a [&'a [f32]],
}

pub fn projected_gqa_reference(core: &GQACore, input: GQAReferenceInput<'_>) -> Vec<f32> {
    core.validate();
    let num_reqs = input.token_indices.len();
    assert_eq!(input.cu_tokens.len(), num_reqs + 1);
    assert_eq!(input.context_k_by_req.len(), num_reqs);
    assert_eq!(input.context_v_by_req.len(), num_reqs);
    let total_tokens = *input.cu_tokens.last().unwrap() as usize;
    assert_eq!(input.q.len(), total_tokens * core.num_q_heads * core.head_dim);
    let heads_per_kv_head = core.num_q_heads / core.num_kv_heads;
    let mut output = vec![0.0; total_tokens * core.num_q_heads * core.head_dim];

    for req_index in 0..num_reqs {
        let req_start = input.cu_tokens[req_index] as usize;
        let req_end = input.cu_tokens[req_index + 1] as usize;
        let num_req_tokens = req_end - req_start;
        let context_len = input.token_indices[req_index] as usize + num_req_tokens;
        assert_eq!(
            input.context_k_by_req[req_index].len(),
            context_len * core.num_kv_heads * core.head_dim
        );
        assert_eq!(
            input.context_v_by_req[req_index].len(),
            context_len * core.num_kv_heads * core.head_dim
        );
        for local_token_index in 0..num_req_tokens {
            let flat_token_index = req_start + local_token_index;
            let visible_len = input.token_indices[req_index] as usize + local_token_index + 1;
            for q_head_index in 0..core.num_q_heads {
                let kv_head_index = q_head_index / heads_per_kv_head;
                let mut logits = vec![0.0; visible_len];
                for (context_token, logit) in logits.iter_mut().enumerate() {
                    let mut dot = 0.0;
                    for dim in 0..core.head_dim {
                        let q = input.q[(flat_token_index * core.num_q_heads + q_head_index) * core.head_dim + dim];
                        let k = input.context_k_by_req[req_index]
                            [(context_token * core.num_kv_heads + kv_head_index) * core.head_dim + dim];
                        dot += q * k;
                    }
                    *logit = dot * core.scale;
                }
                let probs = softmax_reference(&logits);
                for dim in 0..core.head_dim {
                    let mut acc = 0.0;
                    for (context_token, prob) in probs.iter().enumerate() {
                        let v = input.context_v_by_req[req_index]
                            [(context_token * core.num_kv_heads + kv_head_index) * core.head_dim + dim];
                        acc += prob * v;
                    }
                    output[(flat_token_index * core.num_q_heads + q_head_index) * core.head_dim + dim] = acc;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal() {
        let core = GQACore::new(0, 4, 2, 2, 1, 1.0);
        let output = projected_gqa_reference(
            &core,
            GQAReferenceInput {
                cu_tokens: &[0, 2],
                token_indices: &[0],
                q: &[1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 2.0],
                context_k_by_req: &[&[1.0, 0.0, 0.0, 1.0]],
                context_v_by_req: &[&[10.0, 0.0, 0.0, 20.0]],
            },
        );

        assert_eq!(&output[0..4], &[10.0, 0.0, 10.0, 0.0]);
        assert!(output[4] > 1.0 && output[4] < 9.0);
        assert!(output[5] > 1.0 && output[5] < 19.0);
    }
}
