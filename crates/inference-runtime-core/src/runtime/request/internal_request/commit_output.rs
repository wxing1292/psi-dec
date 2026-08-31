use std::iter;

use crate::compute::SampledTokens;
use crate::runtime::Token;
use crate::runtime::request::CompletionReason;
use crate::runtime::request::TokenProbs;
use crate::runtime::request::internal_request::stop_sequence::StopSequences;

pub fn prepare_commit_output<I>(
    stop_sequences: &[Vec<Token>],
    remaining_sampled_tokens: usize,
    remaining_context_tokens: usize,
    history_rev: I,
    sampled_tokens: &mut SampledTokens,
) -> (TokenProbs, Option<CompletionReason>)
where
    I: Iterator<Item = Token> + Clone,
{
    let stop_sequences = StopSequences::new(stop_sequences);
    let stop_match = stop_sequences.match_decode(history_rev.clone(), sampled_tokens);
    let SampledTokens::Decode {
        validated_tokens,
        validated_probs,
        sampled_token,
        sampled_prob,
        spec_tokens,
        spec_probs,
        spec_confidences,
        ..
    } = sampled_tokens
    else {
        return (
            TokenProbs {
                tokens: Vec::new(),
                probs: Vec::new(),
            },
            None,
        );
    };

    let num_committed_tokens = validated_tokens.len() + 1;
    debug_assert!(
        num_committed_tokens <= remaining_sampled_tokens,
        "a decode response must fit the remaining sampled-token budget"
    );
    debug_assert!(
        num_committed_tokens <= remaining_context_tokens,
        "a decode response must fit the remaining context window"
    );
    debug_assert!(
        !stop_match.matched() || stop_match.num_suffix_tokens() == num_committed_tokens,
        "a stop sequence must end at the final sampled token"
    );

    let token_probs = TokenProbs {
        tokens: validated_tokens
            .iter()
            .copied()
            .chain(iter::once(*sampled_token))
            .collect(),
        probs: validated_probs
            .iter()
            .copied()
            .chain(iter::once(*sampled_prob))
            .collect(),
    };
    let completion = if num_committed_tokens == remaining_context_tokens {
        Some(CompletionReason::ContextLimit)
    } else if stop_match.matched() && num_committed_tokens == stop_match.num_suffix_tokens() {
        Some(CompletionReason::StopSequence)
    } else if num_committed_tokens == remaining_sampled_tokens {
        Some(CompletionReason::LengthLimit)
    } else {
        None
    };

    let max_spec_tokens = if completion.is_none() {
        stop_sequences
            .num_spec_tokens_before_stop(
                iter::once(*sampled_token)
                    .chain(validated_tokens.iter().rev().copied())
                    .chain(history_rev),
                spec_tokens,
            )
            .min(remaining_sampled_tokens - num_committed_tokens - 1)
            .min(remaining_context_tokens - num_committed_tokens - 1)
    } else {
        0
    };
    spec_tokens.truncate(max_spec_tokens);
    spec_probs.truncate(max_spec_tokens);
    spec_confidences.truncate(max_spec_tokens);
    (token_probs, completion)
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::*;

    #[test]
    fn test_prepare_commit_output_prefill() {
        let mut sampled_tokens = SampledTokens::Prefill { epoch: 0 };

        let (token_probs, completion) = prepare_commit_output(&[], 1, 1, std::iter::empty(), &mut sampled_tokens);

        assert!(token_probs.is_empty());
        assert_eq!(completion, None);
    }

    #[test]
    fn test_prepare_commit_output_continue() {
        let mut sampled_tokens = decode_tokens(&[10], 11, &[12, 13]);

        let (token_probs, completion) = prepare_commit_output(&[], 10, 10, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(token_probs.tokens, tokens(&[10, 11]));
        assert_eq!(token_probs.probs, probabilities(2));
        assert_eq!(completion, None);
        assert_eq!(sampled_tokens, decode_tokens(&[10], 11, &[12, 13]));
    }

    #[test]
    fn test_prepare_commit_output_length_limit() {
        let mut sampled_tokens = decode_tokens(&[], 10, &[12]);

        let (token_probs, completion) = prepare_commit_output(&[], 1, 10, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(token_probs.tokens, tokens(&[10]));
        assert_eq!(token_probs.probs, probabilities(1));
        assert_eq!(completion, Some(CompletionReason::LengthLimit));
        assert_eq!(sampled_tokens, decode_tokens(&[], 10, &[]));
    }

    #[test]
    fn test_prepare_commit_output_context_limit_precedence() {
        let stop_token = Token::new(10);
        let mut sampled_tokens = decode_tokens(&[], 10, &[20]);

        let (token_probs, completion) =
            prepare_commit_output(&[vec![stop_token]], 1, 1, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(token_probs.tokens, vec![stop_token]);
        assert_eq!(token_probs.probs, probabilities(1));
        assert_eq!(completion, Some(CompletionReason::ContextLimit));
        assert_eq!(sampled_tokens, decode_tokens(&[], 10, &[]));
    }

    #[test]
    fn test_prepare_commit_output_stop_sequence() {
        let stop_sequence = tokens(&[9, 10]);
        let mut sampled_tokens = decode_tokens(&[9], 10, &[12]);

        let (token_probs, completion) = prepare_commit_output(
            std::slice::from_ref(&stop_sequence),
            10,
            10,
            std::iter::empty(),
            &mut sampled_tokens,
        );

        assert_eq!(token_probs.tokens, stop_sequence);
        assert_eq!(token_probs.probs, probabilities(2));
        assert_eq!(completion, Some(CompletionReason::StopSequence));
        assert_eq!(sampled_tokens, decode_tokens(&[9], 10, &[]));
    }

    #[test]
    fn test_prepare_commit_output_stop_proposal_reserves_final_token() {
        let mut sampled_tokens = decode_tokens(&[], 8, &[9, 10, 11]);

        let (token_probs, completion) =
            prepare_commit_output(&[tokens(&[9, 10])], 10, 10, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(token_probs.tokens, tokens(&[8]));
        assert_eq!(completion, None);
        assert_eq!(sampled_tokens, decode_tokens(&[], 8, &[9]));
    }

    #[test]
    fn test_prepare_commit_output_length_proposal_reserves_sampled_token() {
        let mut sampled_tokens = decode_tokens(&[], 10, &[11, 12, 13]);

        let (_, completion) = prepare_commit_output(&[], 3, 10, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(completion, None);
        assert_eq!(sampled_tokens, decode_tokens(&[], 10, &[11]));
    }

    #[test]
    fn test_prepare_commit_output_context_proposal_reserves_sampled_token() {
        let mut sampled_tokens = decode_tokens(&[], 10, &[11, 12, 13]);

        let (_, completion) = prepare_commit_output(&[], 10, 3, std::iter::empty(), &mut sampled_tokens);

        assert_eq!(completion, None);
        assert_eq!(sampled_tokens, decode_tokens(&[], 10, &[11]));
    }

    fn decode_tokens(validated: &[u32], sampled: u32, spec: &[u32]) -> SampledTokens {
        SampledTokens::Decode {
            epoch: 0,
            validated_tokens: tokens(validated),
            validated_probs: probabilities(validated.len()),
            sampled_token: Token::new(sampled),
            sampled_prob: probability(),
            spec_tokens: tokens(spec),
            spec_probs: probabilities(spec.len()),
            spec_confidences: probabilities(spec.len()),
        }
    }

    fn tokens(values: &[u32]) -> Vec<Token> {
        values.iter().copied().map(Token::new).collect()
    }

    fn probabilities(count: usize) -> Vec<NotNan<f32>> {
        vec![probability(); count]
    }

    fn probability() -> NotNan<f32> {
        NotNan::new(0.5).unwrap()
    }
}
