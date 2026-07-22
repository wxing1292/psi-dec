use std::sync::Arc;

use smallvec::SmallVec;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TrieEdge {
    annotations: SmallVec<[BlockAnnotation; 1]>,
    tokens: Arc<[Token]>,
}

impl TrieEdge {
    pub fn new(annotations: SmallVec<[BlockAnnotation; 1]>, tokens: Arc<[Token]>) -> Self {
        Self { annotations, tokens }
    }

    pub fn into_inner(self) -> (SmallVec<[BlockAnnotation; 1]>, Arc<[Token]>) {
        (self.annotations, self.tokens)
    }

    pub fn annotations(&self) -> &SmallVec<[BlockAnnotation; 1]> {
        &self.annotations
    }

    pub fn tokens(&self) -> &Arc<[Token]> {
        &self.tokens
    }
}
