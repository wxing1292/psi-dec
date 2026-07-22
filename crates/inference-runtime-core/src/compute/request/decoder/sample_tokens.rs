use ordered_float::NotNan;

use crate::runtime::Token;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SampledTokens {
    Prefill {
        epoch: usize,
    },
    Decode {
        epoch: usize,
        validated_tokens: Vec<Token>,
        validated_probs: Vec<NotNan<f32>>,
        sampled_token: Token,
        sampled_prob: NotNan<f32>,
        spec_tokens: Vec<Token>,
        spec_probs: Vec<NotNan<f32>>,
    },
}

impl SampledTokens {
    pub fn epoch(&self) -> usize {
        match self {
            Self::Prefill { epoch, .. } => *epoch,
            Self::Decode { epoch, .. } => *epoch,
        }
    }
}
