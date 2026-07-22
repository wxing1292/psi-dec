use ordered_float::NotNan;

use crate::runtime::Token;

pub struct TokenProbs {
    pub tokens: Vec<Token>,
    pub probs: Vec<NotNan<f32>>,
}
