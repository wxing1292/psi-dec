//! CPU reference implementation for already-projected GQA attention tests.

use crate::attn::gqa::GQACore;
use crate::reference::softmax_reference;

#[derive(Clone, Copy, Debug)]
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
    assert_eq!(input.cu_tokens.first(), Some(&0));
    assert!(input.cu_tokens.windows(2).all(|tokens| tokens[0] <= tokens[1]));
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
        let context_length = input.token_indices[req_index] as usize + num_req_tokens;
        assert_eq!(
            input.context_k_by_req[req_index].len(),
            context_length * core.num_kv_heads * core.head_dim
        );
        assert_eq!(
            input.context_v_by_req[req_index].len(),
            context_length * core.num_kv_heads * core.head_dim
        );
        for token_index_in_req in 0..num_req_tokens {
            let token_index = req_start + token_index_in_req;
            let visible_context_length = input.token_indices[req_index] as usize + token_index_in_req + 1;
            for query_head_index in 0..core.num_q_heads {
                let key_value_head_index = query_head_index / heads_per_kv_head;
                let mut logits = vec![0.0; visible_context_length];
                for (context_token_index, logit) in logits.iter_mut().enumerate() {
                    let mut dot_product = 0.0;
                    for head_dim_index in 0..core.head_dim {
                        let query = input.q
                            [(token_index * core.num_q_heads + query_head_index) * core.head_dim + head_dim_index];
                        let key = input.context_k_by_req[req_index][(context_token_index * core.num_kv_heads
                            + key_value_head_index)
                            * core.head_dim
                            + head_dim_index];
                        dot_product += query * key;
                    }
                    *logit = dot_product * core.scale;
                }
                let probabilities = softmax_reference(&logits);
                for head_dim_index in 0..core.head_dim {
                    let mut weighted_value = 0.0;
                    for (context_token_index, probability) in probabilities.iter().enumerate() {
                        let value = input.context_v_by_req[req_index][(context_token_index * core.num_kv_heads
                            + key_value_head_index)
                            * core.head_dim
                            + head_dim_index];
                        weighted_value += probability * value;
                    }
                    output[(token_index * core.num_q_heads + query_head_index) * core.head_dim + head_dim_index] =
                        weighted_value;
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
    fn test_causal_attention() {
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

        let first_token_score = (-2.0_f32).exp();
        let score_total = first_token_score + 1.0;
        let first_token_weight = first_token_score / score_total;
        let second_token_weight = 1.0 / score_total;
        let expected = [
            10.0,
            0.0,
            10.0,
            0.0,
            10.0 * first_token_weight,
            20.0 * second_token_weight,
            10.0 * first_token_weight,
            20.0 * second_token_weight,
        ];
        for (&actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }
}
