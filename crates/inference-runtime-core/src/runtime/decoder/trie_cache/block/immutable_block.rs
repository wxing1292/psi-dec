use std::sync::Arc;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::trie_cache::DataStoreValueRef;
use crate::runtime::decoder::trie_cache::Trie;
use crate::runtime::decoder::trie_cache::TrieNode;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::block::DecoderBlock;

#[derive(Debug)]
pub struct ImmutableBlock<const P: usize> {
    trie: Arc<Trie<P>>,
    trie_node_key: TrieNodeKey,
    tokens: Arc<[Token]>,
}

impl<const P: usize> ImmutableBlock<P> {
    pub fn new(trie: Arc<Trie<P>>, trie_node_key: TrieNodeKey, tokens: Arc<[Token]>) -> Self {
        Self {
            trie,
            trie_node_key,
            tokens,
        }
    }

    pub fn trie_node_key(&self) -> TrieNodeKey {
        self.trie_node_key
    }

    pub fn trie_node_ref(&self) -> DataStoreValueRef<'_, TrieNodeKey, TrieNode> {
        self.trie.peek_ref_by_key(self.trie_node_key)
    }

    pub fn annotations(&self) -> Vec<BlockAnnotation> {
        self.trie.peek_ref_by_key(self.trie_node_key).annotations().to_vec()
    }

    pub fn tokens(&self) -> &Arc<[Token]> {
        &self.tokens
    }
}

impl<const P: usize> Drop for ImmutableBlock<P> {
    fn drop(&mut self) {
        self.trie.external_unpin_by_key(self.trie_node_key);
    }
}

impl<const P: usize> DecoderBlock for ImmutableBlock<P> {
    fn cached_tokens(&self) -> &[Token] {
        self.tokens.as_ref()
    }

    fn scheduled_tokens(&self) -> &[Token] {
        &[]
    }

    fn ready_tokens(&self) -> &[Token] {
        &[]
    }

    fn total_tokens(&self) -> &[Token] {
        self.tokens.as_ref()
    }

    fn ready_token_slots(&self) -> usize {
        0
    }

    fn cache_tokens(&mut self, tokens: &[Token]) {
        unreachable!()
    }

    fn schedule_tokens(&mut self, num_tokens: usize) -> &[Token] {
        unreachable!()
    }

    fn unschedule_tokens(&mut self, tokens: &[Token]) {
        unreachable!()
    }
}
