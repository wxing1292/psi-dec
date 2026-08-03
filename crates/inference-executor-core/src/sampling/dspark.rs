//! Backend-neutral CPU reference for DSpark Markov sampling and confidence.

use half::bf16;

use crate::mlp::dense::reference::QuantizedAffineReferenceShape;
use crate::mlp::dense::reference::quantized_affine_reference;
use crate::mlp::dense::reference::quantized_affine_weight_row_reference;
use crate::sampling::SamplerConfig;
use crate::sampling::SamplingDomain;
use crate::sampling::reference::ReferenceSampleRow;
use crate::sampling::reference::sparse_sample_row_with_domain_reference;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkMarkovReferenceConfig {
    pub block_size: usize,
    pub vocab_size: usize,
    pub rank: usize,
    pub w1_group_size: usize,
    pub w1_bits: usize,
    pub w2_group_size: usize,
    pub w2_bits: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DSparkMarkovReferenceWeights<'a> {
    pub w1_weight: &'a [u8],
    pub w1_scales: &'a [f32],
    pub w1_biases: &'a [f32],
    pub w2_weight: &'a [u8],
    pub w2_scales: &'a [f32],
    pub w2_biases: &'a [f32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkConfidenceReferenceConfig {
    pub hidden_dim: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DSparkConfidenceReferenceWeights<'a> {
    pub weight: &'a [f32],
    pub bias: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DSparkMarkovReferenceProposal {
    pub requests: Vec<Vec<ReferenceSampleRow>>,
}

pub fn dspark_markov_reference(
    config: DSparkMarkovReferenceConfig,
    weights: DSparkMarkovReferenceWeights<'_>,
    anchor_token_ids: &[u32],
    anchor_positions: &[u32],
    sampler_configs: &[SamplerConfig],
    base_logits: &[f32],
    distribution_len: usize,
) -> DSparkMarkovReferenceProposal {
    config.validate();
    assert!(!anchor_token_ids.is_empty());
    assert_eq!(anchor_token_ids.len(), anchor_positions.len());
    assert_eq!(anchor_token_ids.len(), sampler_configs.len());
    assert!(distribution_len > 0);
    assert_eq!(
        base_logits.len(),
        config.block_size * anchor_token_ids.len() * config.vocab_size
    );

    let w1_shape = config.w1_shape();
    let w2_shape = config.w2_shape();
    let num_requests = anchor_token_ids.len();
    let mut requests = (0..num_requests)
        .map(|_| Vec::with_capacity(config.block_size))
        .collect::<Vec<_>>();
    let mut input_token_ids = anchor_token_ids.to_vec();

    for step_index in 0..config.block_size {
        for request_index in 0..num_requests {
            let latent = quantized_affine_weight_row_reference(
                w1_shape,
                input_token_ids[request_index] as usize,
                weights.w1_weight,
                weights.w1_scales,
                weights.w1_biases,
            )
            .into_iter()
            .map(round_bf16)
            .collect::<Vec<_>>();
            let correction = quantized_affine_reference(
                w2_shape,
                &latent,
                weights.w2_weight,
                weights.w2_scales,
                weights.w2_biases,
            );
            let base_row = (step_index * num_requests + request_index) * config.vocab_size;
            let corrected_logits = correction
                .into_iter()
                .enumerate()
                .map(|(token_id, value)| {
                    let correction = round_bf16(value);
                    let base = round_bf16(base_logits[base_row + token_id]);
                    round_bf16(base + correction)
                })
                .collect::<Vec<_>>();
            let sample_position = anchor_positions[request_index]
                .checked_add(
                    u32::try_from(step_index)
                        .expect("DSpark reference step index must fit u32")
                        .checked_add(1)
                        .expect("DSpark reference step offset must fit u32"),
                )
                .expect("DSpark reference sample position must fit u32");
            let sample = sparse_sample_row_with_domain_reference(
                &sampler_configs[request_index],
                &corrected_logits,
                distribution_len,
                sample_position,
                SamplingDomain::Draft,
            );
            input_token_ids[request_index] = sample.sampled_token;
            requests[request_index].push(sample);
        }
    }

    DSparkMarkovReferenceProposal { requests }
}

#[allow(clippy::too_many_arguments)]
pub fn dspark_confidence_reference(
    markov_config: DSparkMarkovReferenceConfig,
    markov_weights: DSparkMarkovReferenceWeights<'_>,
    confidence_config: DSparkConfidenceReferenceConfig,
    confidence_weights: DSparkConfidenceReferenceWeights<'_>,
    anchor_token_ids: &[u32],
    proposal: &DSparkMarkovReferenceProposal,
    hidden: &[f32],
) -> Vec<Vec<f32>> {
    markov_config.validate();
    confidence_config.validate(markov_config.rank);
    assert!(!anchor_token_ids.is_empty());
    assert_eq!(anchor_token_ids.len(), proposal.requests.len());
    assert!(
        proposal
            .requests
            .iter()
            .all(|request| request.len() == markov_config.block_size)
    );
    assert_eq!(
        hidden.len(),
        markov_config.block_size * anchor_token_ids.len() * confidence_config.hidden_dim
    );
    assert_eq!(
        confidence_weights.weight.len(),
        confidence_config.input_dim(markov_config.rank)
    );

    let w1_shape = markov_config.w1_shape();
    let num_requests = anchor_token_ids.len();
    let mut confidences = (0..num_requests)
        .map(|_| Vec::with_capacity(markov_config.block_size))
        .collect::<Vec<_>>();
    for step_index in 0..markov_config.block_size {
        for request_index in 0..num_requests {
            let input_token_id = if step_index == 0 {
                anchor_token_ids[request_index]
            } else {
                proposal.requests[request_index][step_index - 1].sampled_token
            };
            let hidden_begin = (step_index * num_requests + request_index) * confidence_config.hidden_dim;
            let hidden_end = hidden_begin + confidence_config.hidden_dim;
            let mut raw = round_bf16(confidence_weights.bias);
            for (&value, &weight) in hidden[hidden_begin..hidden_end]
                .iter()
                .zip(&confidence_weights.weight[..confidence_config.hidden_dim])
            {
                raw += round_bf16(value) * round_bf16(weight);
            }
            let latent = quantized_affine_weight_row_reference(
                w1_shape,
                input_token_id as usize,
                markov_weights.w1_weight,
                markov_weights.w1_scales,
                markov_weights.w1_biases,
            );
            for (&value, &weight) in latent.iter().zip(
                &confidence_weights.weight
                    [confidence_config.hidden_dim..confidence_config.hidden_dim + markov_config.rank],
            ) {
                raw += round_bf16(value) * round_bf16(weight);
            }
            confidences[request_index].push(1.0 / (1.0 + (-raw).exp()));
        }
    }
    confidences
}

impl DSparkMarkovReferenceConfig {
    fn validate(self) {
        assert!(self.block_size > 0);
        assert!(self.vocab_size > 0);
        assert!(self.rank > 0);
        self.w1_shape().validate();
        self.w2_shape().validate();
    }

    fn w1_shape(self) -> QuantizedAffineReferenceShape {
        QuantizedAffineReferenceShape {
            num_rows: 1,
            output_dim: self.vocab_size,
            input_dim: self.rank,
            group_size: self.w1_group_size,
            bits: self.w1_bits,
        }
    }

    fn w2_shape(self) -> QuantizedAffineReferenceShape {
        QuantizedAffineReferenceShape {
            num_rows: 1,
            output_dim: self.vocab_size,
            input_dim: self.rank,
            group_size: self.w2_group_size,
            bits: self.w2_bits,
        }
    }
}

impl DSparkConfidenceReferenceConfig {
    fn validate(self, rank: usize) {
        assert!(self.hidden_dim > 0);
        let _ = self.input_dim(rank);
    }

    fn input_dim(self, rank: usize) -> usize {
        self.hidden_dim
            .checked_add(rank)
            .expect("DSpark confidence reference input dimension must fit usize")
    }
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_token_drives_the_next_markov_step() {
        const VOCAB: usize = 64;
        const RANK: usize = 32;
        let config = DSparkMarkovReferenceConfig {
            block_size: 3,
            vocab_size: VOCAB,
            rank: RANK,
            w1_group_size: RANK,
            w1_bits: 8,
            w2_group_size: RANK,
            w2_bits: 8,
        };
        let mut w1_weight = vec![0_u8; VOCAB * RANK];
        let mut w2_weight = vec![0_u8; VOCAB * RANK];
        for token_id in 0..RANK - 1 {
            w1_weight[token_id * RANK + token_id] = 1;
            w2_weight[(token_id + 1) * RANK + token_id] = 16;
        }
        let unit_affine = vec![1.0; VOCAB];
        let zero_affine = vec![0.0; VOCAB];
        let proposal = dspark_markov_reference(
            config,
            DSparkMarkovReferenceWeights {
                w1_weight: &w1_weight,
                w1_scales: &unit_affine,
                w1_biases: &zero_affine,
                w2_weight: &w2_weight,
                w2_scales: &unit_affine,
                w2_biases: &zero_affine,
            },
            &[1, 5],
            &[10, 20],
            &[SamplerConfig {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                seed: 42,
            }; 2],
            &vec![0.0; config.block_size * 2 * VOCAB],
            1,
        );

        assert_eq!(
            proposal
                .requests
                .iter()
                .map(|steps| steps.iter().map(|step| step.sampled_token).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![2, 3, 4], vec![6, 7, 8]]
        );
    }

    #[test]
    fn confidence_uses_each_indexed_markov_input_token() {
        const VOCAB: usize = 64;
        const RANK: usize = 32;
        const HIDDEN: usize = 2;
        let config = DSparkMarkovReferenceConfig {
            block_size: 3,
            vocab_size: VOCAB,
            rank: RANK,
            w1_group_size: RANK,
            w1_bits: 8,
            w2_group_size: RANK,
            w2_bits: 8,
        };
        let mut w1_weight = vec![0_u8; VOCAB * RANK];
        let mut w2_weight = vec![0_u8; VOCAB * RANK];
        for token_id in 0..RANK - 1 {
            w1_weight[token_id * RANK + token_id] = 1;
            w2_weight[(token_id + 1) * RANK + token_id] = 16;
        }
        let unit_affine = vec![1.0; VOCAB];
        let zero_affine = vec![0.0; VOCAB];
        let weights = DSparkMarkovReferenceWeights {
            w1_weight: &w1_weight,
            w1_scales: &unit_affine,
            w1_biases: &zero_affine,
            w2_weight: &w2_weight,
            w2_scales: &unit_affine,
            w2_biases: &zero_affine,
        };
        let proposal = dspark_markov_reference(
            config,
            weights,
            &[1],
            &[10],
            &[SamplerConfig {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                seed: 42,
            }],
            &vec![0.0; config.block_size * VOCAB],
            1,
        );
        let mut confidence_weight = vec![0.0; HIDDEN + RANK];
        for rank_index in 0..RANK {
            confidence_weight[HIDDEN + rank_index] = rank_index as f32;
        }

        let confidences = dspark_confidence_reference(
            config,
            weights,
            DSparkConfidenceReferenceConfig { hidden_dim: HIDDEN },
            DSparkConfidenceReferenceWeights {
                weight: &confidence_weight,
                bias: 0.0,
            },
            &[1],
            &proposal,
            &vec![0.0; config.block_size * HIDDEN],
        );

        assert_eq!(
            proposal.requests[0][..2]
                .iter()
                .map(|row| row.sampled_token)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert!((confidences[0][0] - sigmoid(1.0)).abs() < 1.0e-6);
        assert!((confidences[0][1] - sigmoid(2.0)).abs() < 1.0e-6);
        assert!((confidences[0][2] - sigmoid(3.0)).abs() < 1.0e-6);
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }
}
