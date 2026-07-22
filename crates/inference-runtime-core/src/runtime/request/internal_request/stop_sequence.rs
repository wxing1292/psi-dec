use std::iter::once;

use crate::compute::SampledTokens;
use crate::runtime::Token;
use crate::runtime::request::TokenProbs;

pub struct StopSequences<'a> {
    stop_seqs: &'a [Vec<Token>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StopSequenceMatch {
    num_visible_tokens: usize,
    matched: bool,
}

impl<'a> StopSequences<'a> {
    pub fn new(stop_seqs: &'a [Vec<Token>]) -> Self {
        Self { stop_seqs }
    }

    pub fn match_decode<I>(&self, prefix_rev: I, sampled_tokens: &SampledTokens) -> StopSequenceMatch
    where
        I: Iterator<Item = Token> + Clone,
    {
        let suffix_len = suffix_len(sampled_tokens);
        if suffix_len == 0 {
            return StopSequenceMatch::no_match(0);
        }
        let max_stop_len = self.stop_seqs.iter().map(Vec::len).max().unwrap_or(0);
        let mut tokens = prefix_rev.take(max_stop_len.saturating_sub(1)).collect::<Vec<_>>();
        tokens.reverse();
        let num_prefix_tokens = tokens.len();
        tokens.extend(suffix_tokens(sampled_tokens));

        for end in (num_prefix_tokens + 1)..=tokens.len() {
            for stop_seq in self.stop_seqs {
                debug_assert!(!stop_seq.is_empty(), "stop sequences must not be empty");
                if tokens[..end].ends_with(stop_seq) {
                    return StopSequenceMatch {
                        num_visible_tokens: end - num_prefix_tokens,
                        matched: true,
                    };
                }
            }
        }
        StopSequenceMatch::no_match(suffix_len)
    }
}

impl StopSequenceMatch {
    fn no_match(num_visible_tokens: usize) -> Self {
        Self {
            num_visible_tokens,
            matched: false,
        }
    }

    pub fn matched(&self) -> bool {
        self.matched
    }

    pub fn visible_token_probs(&self, sampled_tokens: &SampledTokens) -> Option<TokenProbs> {
        let SampledTokens::Decode {
            validated_tokens,
            validated_probs,
            sampled_token,
            sampled_prob,
            ..
        } = sampled_tokens
        else {
            return None;
        };
        debug_assert_eq!(
            validated_tokens.len(),
            validated_probs.len(),
            "validated token and probability counts must match"
        );

        let num_suffix_tokens = validated_tokens.len() + 1;
        debug_assert!(
            self.num_visible_tokens <= num_suffix_tokens,
            "visible decode tokens must be a prefix of committed decode tokens"
        );

        Some(TokenProbs {
            tokens: validated_tokens
                .iter()
                .copied()
                .chain(once(*sampled_token))
                .take(self.num_visible_tokens)
                .collect(),
            probs: validated_probs
                .iter()
                .copied()
                .chain(once(*sampled_prob))
                .take(self.num_visible_tokens)
                .collect(),
        })
    }
}

fn suffix_tokens(sampled_tokens: &SampledTokens) -> impl Iterator<Item = Token> + '_ {
    let (validated_tokens, sampled_token) = match sampled_tokens {
        SampledTokens::Decode {
            validated_tokens,
            sampled_token,
            ..
        } => (validated_tokens.as_slice(), Some(*sampled_token)),
        SampledTokens::Prefill { .. } => (&[][..], None),
    };
    validated_tokens.iter().copied().chain(sampled_token)
}

fn suffix_len(sampled_tokens: &SampledTokens) -> usize {
    match sampled_tokens {
        SampledTokens::Decode { validated_tokens, .. } => validated_tokens.len() + 1,
        SampledTokens::Prefill { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::*;

    #[test]
    fn test_match_single_token_0() {
        assert_eq!(
            match_decode(&[tokens(&[7])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 1,
                matched: true,
            }
        );
    }

    #[test]
    fn test_match_single_token_1() {
        assert_eq!(
            match_decode(&[tokens(&[8])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 2,
                matched: true,
            }
        );
    }

    #[test]
    fn test_match_single_token_2() {
        assert_eq!(
            match_decode(&[tokens(&[9])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 3,
                matched: true,
            }
        );
    }

    #[test]
    fn test_mismatch_single_token() {
        assert_eq!(
            match_decode(&[tokens(&[10])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 3,
                matched: false,
            }
        );
    }

    #[test]
    fn test_match_multi_token_0() {
        assert_eq!(
            match_decode(&[tokens(&[6, 7])], &tokens(&[6]), &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 1,
                matched: true,
            }
        );
    }

    #[test]
    fn test_match_multi_token_1() {
        assert_eq!(
            match_decode(&[tokens(&[7, 8])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 2,
                matched: true,
            }
        );
    }

    #[test]
    fn test_match_multi_token_2() {
        assert_eq!(
            match_decode(&[tokens(&[8, 9])], &[], &[7, 8, 9, 10]),
            StopSequenceMatch {
                num_visible_tokens: 3,
                matched: true,
            }
        );
    }

    #[test]
    fn test_match_multi_token_3() {
        assert_eq!(
            match_decode(&[tokens(&[8, 9])], &[], &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 3,
                matched: true,
            }
        );
    }

    #[test]
    fn test_mismatch_multi_token() {
        assert_eq!(
            match_decode(&[tokens(&[6, 8])], &tokens(&[6]), &[7, 8, 9]),
            StopSequenceMatch {
                num_visible_tokens: 3,
                matched: false,
            }
        );
    }

    #[test]
    fn test_visible_token_probs_truncate() {
        let sampled_tokens = sampled_tokens(&[10, 11, 12, 13]);
        let token_probs = StopSequenceMatch {
            num_visible_tokens: 2,
            matched: true,
        }
        .visible_token_probs(&sampled_tokens)
        .expect("decode token probs should exist");

        assert_eq!(token_probs.tokens, tokens(&[10, 11]));
        assert_eq!(token_probs.probs, vec![prob(0.1), prob(0.2)]);
    }

    #[test]
    fn test_visible_token_probs_no_match() {
        let sampled_tokens = sampled_tokens(&[10, 11]);
        let token_probs = StopSequenceMatch {
            num_visible_tokens: 2,
            matched: false,
        }
        .visible_token_probs(&sampled_tokens)
        .expect("decode token probs should exist");

        assert_eq!(token_probs.tokens, tokens(&[10, 11]));
        assert_eq!(token_probs.probs, vec![prob(0.1), prob(0.2)]);
    }

    fn token(value: u32) -> Token {
        Token::new(value)
    }

    fn tokens(values: &[u32]) -> Vec<Token> {
        values.iter().copied().map(token).collect()
    }

    fn prob(value: f32) -> NotNan<f32> {
        NotNan::new(value).expect("test probability must not be NaN")
    }

    fn sampled_tokens(values: &[u32]) -> SampledTokens {
        assert!(!values.is_empty(), "test sampled tokens must not be empty");
        let sampled_index = values.len() - 1;
        SampledTokens::Decode {
            epoch: 7,
            validated_tokens: tokens(&values[..sampled_index]),
            validated_probs: (0..sampled_index)
                .map(|index| prob((index + 1) as f32 / 10.0))
                .collect(),
            sampled_token: token(values[sampled_index]),
            sampled_prob: prob(values.len() as f32 / 10.0),
            spec_tokens: vec![token(100), token(101)],
            spec_probs: vec![prob(0.8), prob(0.9)],
        }
    }

    fn match_decode(stop_seqs: &[Vec<Token>], previous_tokens: &[Token], response_tokens: &[u32]) -> StopSequenceMatch {
        StopSequences::new(stop_seqs)
            .match_decode(previous_tokens.iter().rev().copied(), &sampled_tokens(response_tokens))
    }
}
