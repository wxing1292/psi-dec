use ordered_float::NotNan;

use crate::runtime::Token;

#[derive(Debug)]
pub struct TokenProbs {
    pub tokens: Vec<Token>,
    pub probs: Vec<NotNan<f32>>,
}
