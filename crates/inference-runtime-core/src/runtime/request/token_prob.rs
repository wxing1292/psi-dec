use ordered_float::NotNan;

use crate::runtime::Token;

#[derive(Debug)]
pub struct TokenProbs {
    pub tokens: Vec<Token>,
    pub probs: Vec<NotNan<f32>>,
}

impl TokenProbs {
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
