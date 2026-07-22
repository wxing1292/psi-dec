use std::sync::Arc;

use smallvec::SmallVec;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMetadata<const N: usize> {
    parent_trie_node_key: Option<TrieNodeKey>,
    annotations: SmallVec<[BlockAnnotation; 1]>,
    tokens: Arc<[Token]>,
}

impl<const N: usize> BlockMetadata<N> {
    pub fn new(
        parent_trie_node_key: Option<TrieNodeKey>,
        annotations: SmallVec<[BlockAnnotation; 1]>,
        tokens: Arc<[Token]>,
    ) -> Self {
        debug_assert!(tokens.len() <= N);
        Self {
            parent_trie_node_key,
            annotations,
            tokens,
        }
    }

    pub fn parent_trie_node_key(&self) -> Option<TrieNodeKey> {
        self.parent_trie_node_key
    }

    pub fn annotations(&self) -> &SmallVec<[BlockAnnotation; 1]> {
        &self.annotations
    }

    pub fn tokens(&self) -> &Arc<[Token]> {
        &self.tokens
    }

    pub fn into_inner(self) -> (Option<TrieNodeKey>, SmallVec<[BlockAnnotation; 1]>, Arc<[Token]>) {
        (self.parent_trie_node_key, self.annotations, self.tokens)
    }
}
