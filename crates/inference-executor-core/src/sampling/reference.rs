//! CPU reference sampling routines for executor and backend tests.

use crate::sampling::SamplerConfig;
use crate::sampling::SamplingDomain;

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceSampleRow {
    pub sampled_token: u32,
    pub sampled_prob: f32,
    pub prob_token_ids: Vec<i32>,
    pub prob_values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRejectionDecision {
    pub accepted_tokens: Vec<u32>,
    pub accepted_probs: Vec<f32>,
    pub sampled_token: u32,
    pub sampled_prob: f32,
}

pub fn sparse_sample_row_reference(
    config: &SamplerConfig,
    logits: &[f32],
    distribution_len: usize,
    sample_position: u32,
) -> ReferenceSampleRow {
    sparse_sample_row_with_domain_reference(
        config,
        logits,
        distribution_len,
        sample_position,
        SamplingDomain::Target,
    )
}

pub fn sparse_sample_row_with_domain_reference(
    config: &SamplerConfig,
    logits: &[f32],
    distribution_len: usize,
    sample_position: u32,
    domain: SamplingDomain,
) -> ReferenceSampleRow {
    assert!(!logits.is_empty());
    assert!(distribution_len > 0);
    assert!(i32::try_from(logits.len()).is_ok());
    let mut candidates = logits
        .iter()
        .enumerate()
        .filter_map(|(token_index, &logit)| logit.is_finite().then_some((token_index as i32, logit)))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap().then_with(|| left.0.cmp(&right.0)));
    candidates.truncate(distribution_len);
    if candidates.is_empty() {
        return build_sparse_sample_row(0, 1.0, &[(0, 1.0)], distribution_len);
    }
    if config.is_greedy() || candidates.len() == 1 {
        return build_sparse_sample_row(candidates[0].0 as u32, 1.0, &[(candidates[0].0, 1.0)], distribution_len);
    }

    let temperature = config.temperature.max(1.0e-6);
    let max_scaled_logit = candidates[0].1 / temperature;
    let weights = candidates
        .iter()
        .map(|(_, logit)| ((*logit / temperature) - max_scaled_logit).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return build_sparse_sample_row(candidates[0].0 as u32, 1.0, &[(candidates[0].0, 1.0)], distribution_len);
    }

    let mut kept_total = 0.0f32;
    let mut kept_count = 0usize;
    for weight in &weights {
        kept_total += *weight;
        kept_count += 1;
        if config.top_p < 1.0 && kept_total >= config.top_p * total {
            break;
        }
    }
    let kept = candidates
        .iter()
        .zip(weights.iter())
        .take(kept_count)
        .map(|((token_index, _), weight)| (*token_index, *weight / kept_total))
        .collect::<Vec<_>>();

    let draw = sampling_uniform(config.seed(), sample_position, domain)
        * kept.iter().map(|(_, probability)| *probability).sum::<f32>();
    let mut cumulative = 0.0f32;
    let mut selected = kept_count - 1;
    for (slot, (_, probability)) in kept.iter().enumerate() {
        cumulative += *probability;
        if cumulative >= draw {
            selected = slot;
            break;
        }
    }

    build_sparse_sample_row(kept[selected].0 as u32, kept[selected].1, &kept, distribution_len)
}

pub fn rejection_sample_reference(
    draft_tokens: &[u32],
    target_prob_rows: &[Vec<f32>],
    draft_prob_rows: &[Vec<f32>],
    seed: u32,
    sample_position: u32,
) -> ReferenceRejectionDecision {
    assert!(u32::try_from(draft_tokens.len()).is_ok());
    assert!(sample_position <= u32::MAX - draft_tokens.len() as u32);
    assert_eq!(target_prob_rows.len(), draft_tokens.len() + 1);
    assert_eq!(draft_prob_rows.len(), draft_tokens.len());
    let mut accepted_tokens = Vec::new();
    let mut accepted_probs = Vec::new();
    for (spec_offset, &draft_token) in draft_tokens.iter().enumerate() {
        let target_probs = &target_prob_rows[spec_offset];
        let draft_probs = &draft_prob_rows[spec_offset];
        assert_eq!(target_probs.len(), draft_probs.len());
        let token_index = draft_token as usize;
        assert!(token_index < target_probs.len());
        let target_prob = target_probs[token_index].max(0.0);
        let draft_prob = draft_probs[token_index].max(0.0);
        let accept_prob = if draft_prob > 0.0 {
            (target_prob / draft_prob).min(1.0)
        } else {
            0.0
        };
        let token_position = sample_position + spec_offset as u32;
        if sampling_uniform(seed, token_position, SamplingDomain::Accept) <= accept_prob {
            accepted_tokens.push(draft_token);
            accepted_probs.push(target_prob);
            continue;
        }

        let residual = target_probs
            .iter()
            .zip(draft_probs.iter())
            .map(|(&target_prob, &draft_prob)| (target_prob.max(0.0) - draft_prob.max(0.0)).max(0.0))
            .collect::<Vec<_>>();
        let (sampled_token, sampled_prob) = sample_probability_row_reference(
            &residual,
            sampling_uniform(seed, token_position, SamplingDomain::Resample),
            target_probs,
        );
        return ReferenceRejectionDecision {
            accepted_tokens,
            accepted_probs,
            sampled_token,
            sampled_prob,
        };
    }

    let final_target = target_prob_rows.last().unwrap();
    let final_position = sample_position + draft_tokens.len() as u32;
    let (sampled_token, sampled_prob) = sample_probability_row_reference(
        final_target,
        sampling_uniform(seed, final_position, SamplingDomain::Target),
        final_target,
    );
    ReferenceRejectionDecision {
        accepted_tokens,
        accepted_probs,
        sampled_token,
        sampled_prob,
    }
}

fn build_sparse_sample_row(
    sampled_token: u32,
    sampled_prob: f32,
    kept: &[(i32, f32)],
    distribution_len: usize,
) -> ReferenceSampleRow {
    let mut prob_token_ids = vec![-1; distribution_len];
    let mut prob_values = vec![0.0; distribution_len];
    for (slot, &(token_index, probability)) in kept.iter().take(distribution_len).enumerate() {
        prob_token_ids[slot] = token_index;
        prob_values[slot] = probability;
    }
    ReferenceSampleRow {
        sampled_token,
        sampled_prob,
        prob_token_ids,
        prob_values,
    }
}

fn sample_probability_row_reference(probabilities: &[f32], uniform: f32, reported_probs: &[f32]) -> (u32, f32) {
    assert_eq!(probabilities.len(), reported_probs.len());
    assert!(u32::try_from(probabilities.len()).is_ok());
    let total = probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return (0, 0.0);
    }
    let draw = uniform * total;
    let mut cumulative = 0.0f32;
    for (token_index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability.max(0.0);
        if cumulative >= draw {
            return (token_index as u32, reported_probs[token_index].max(0.0));
        }
    }
    probabilities
        .iter()
        .enumerate()
        .rev()
        .find(|(_, probability)| **probability > 0.0)
        .map(|(token_index, _)| (token_index as u32, reported_probs[token_index].max(0.0)))
        .unwrap_or((0, 0.0))
}

fn mix_u32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

fn sampling_uniform(seed: u32, sample_position: u32, domain: SamplingDomain) -> f32 {
    uniform01(mix_u32(
        seed ^ mix_u32(sample_position.wrapping_add(0x9e37_79b9)) ^ domain as u32,
    ))
}

fn uniform01(h: u32) -> f32 {
    ((h & 0x00ff_ffff) as f32 + 0.5) * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_distribution() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            seed: 12345,
        };

        let row = sparse_sample_row_reference(&config, &[0.1, 2.0, 1.5, 0.2, 3.0, 0.0, 2.5, 1.0], 4, 0);

        assert_eq!(row.prob_token_ids, vec![4, 6, 1, 2]);
        assert!(row.prob_values.iter().sum::<f32>() > 0.999);
        assert!(row.prob_values.iter().sum::<f32>() < 1.001);
    }

    #[test]
    fn test_greedy() {
        let config = SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
        };

        let row = sparse_sample_row_reference(&config, &[0.1, 2.0, 1.5, 0.2], 1, 0);

        assert_eq!(row.sampled_token, 1);
        assert_eq!(row.sampled_prob, 1.0);
        assert_eq!(row.prob_token_ids, vec![1]);
        assert_eq!(row.prob_values, vec![1.0]);
    }

    #[test]
    fn test_rejection_all_accepts() {
        let decision = rejection_sample_reference(
            &[1, 2],
            &[one_hot(4, 1), one_hot(4, 2), one_hot(4, 3)],
            &[one_hot(4, 1), one_hot(4, 2)],
            7,
            0,
        );

        assert_eq!(decision.accepted_tokens, vec![1, 2]);
        assert_eq!(decision.sampled_token, 3);
        assert_eq!(decision.sampled_prob, 1.0);
    }

    #[test]
    fn test_rejection_resamples() {
        let decision =
            rejection_sample_reference(&[1], &[vec![0.0, 0.0, 1.0, 0.0], one_hot(4, 3)], &[one_hot(4, 1)], 7, 0);

        assert_eq!(decision.accepted_tokens, Vec::<u32>::new());
        assert_eq!(decision.sampled_token, 2);
        assert_eq!(decision.sampled_prob, 1.0);
    }

    #[test]
    fn test_sampling_domains_are_disjoint() {
        let seed = 7;
        let position = 11;
        let target = sampling_uniform(seed, position, SamplingDomain::Target);

        assert_eq!(target, sampling_uniform(seed, position, SamplingDomain::Target));
        assert_ne!(target, sampling_uniform(seed, position, SamplingDomain::Draft));
        assert_ne!(target, sampling_uniform(seed, position, SamplingDomain::Accept));
        assert_ne!(target, sampling_uniform(seed, position, SamplingDomain::Resample));
    }

    fn one_hot(vocab_size: usize, token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; vocab_size];
        values[token_index] = 1.0;
        values
    }
}
