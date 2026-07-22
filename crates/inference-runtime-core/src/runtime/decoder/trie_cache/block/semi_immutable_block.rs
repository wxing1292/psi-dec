use std::cmp::min;
use std::sync::Arc;

use inference_runtime_macro::sanity_check;
use smallvec::SmallVec;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::block::DecoderBlock;

#[derive(Debug)]
pub struct SemiImmutableBlock<const N: usize> {
    parent_trie_node_key: Option<TrieNodeKey>,
    annotations: SmallVec<[BlockAnnotation; 1]>,
    tokens: Arc<[Token]>,
    scheduled_token_index: usize,
    ready_token_index: usize,
    kv_placement: KVBlockPlacement,
    state_placement: StateBlockPlacement,
}

impl<const N: usize> SemiImmutableBlock<N> {
    pub fn new(
        parent_trie_node_key: Option<TrieNodeKey>,
        annotations: SmallVec<[BlockAnnotation; 1]>,
        tokens: Arc<[Token]>,
        scheduled_token_index: usize,
        ready_token_index: usize,
        kv_placement: KVBlockPlacement,
        state_placement: StateBlockPlacement,
    ) -> Self {
        debug_assert!(scheduled_token_index <= ready_token_index);
        debug_assert!(ready_token_index <= tokens.len());
        debug_assert_eq!(N, tokens.len());
        Self {
            parent_trie_node_key,
            annotations,
            tokens,
            scheduled_token_index,
            ready_token_index,
            kv_placement,
            state_placement,
        }
    }

    pub fn parent_trie_node_key(&self) -> Option<TrieNodeKey> {
        self.parent_trie_node_key
    }

    pub fn annotations(&self) -> &SmallVec<[BlockAnnotation; 1]> {
        &self.annotations
    }

    pub fn kv_placement(&self) -> &KVBlockPlacement {
        &self.kv_placement
    }

    pub fn state_placement(&self) -> &StateBlockPlacement {
        &self.state_placement
    }

    #[allow(clippy::type_complexity)]
    pub fn into_inner(
        self,
    ) -> (
        Option<TrieNodeKey>,
        SmallVec<[BlockAnnotation; 1]>,
        Arc<[Token]>,
        usize,
        usize,
        KVBlockPlacement,
        StateBlockPlacement,
    ) {
        (
            self.parent_trie_node_key,
            self.annotations,
            self.tokens,
            self.scheduled_token_index,
            self.ready_token_index,
            self.kv_placement,
            self.state_placement,
        )
    }

    fn sanity_check(&self) {
        debug_assert!(self.scheduled_token_index <= self.ready_token_index);
        debug_assert!(self.ready_token_index <= self.tokens.len());
        debug_assert!(self.tokens.len() <= N);
    }
}

impl<const N: usize> DecoderBlock for SemiImmutableBlock<N> {
    fn cached_tokens(&self) -> &[Token] {
        &self.tokens[..self.scheduled_token_index]
    }

    fn scheduled_tokens(&self) -> &[Token] {
        &self.tokens[self.scheduled_token_index..self.ready_token_index]
    }

    fn ready_tokens(&self) -> &[Token] {
        &self.tokens[self.ready_token_index..]
    }

    fn total_tokens(&self) -> &[Token] {
        self.tokens.as_ref()
    }

    fn ready_token_slots(&self) -> usize {
        N - self.ready_token_index
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn cache_tokens(&mut self, tokens: &[Token]) {
        let num_tokens = tokens.len();
        let scheduled_token_index = self.scheduled_token_index;
        self.scheduled_token_index += num_tokens;
        debug_assert_eq!(
            &self.total_tokens()[scheduled_token_index..self.scheduled_token_index],
            tokens,
        );
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn schedule_tokens(&mut self, num_tokens: usize) -> &[Token] {
        let num_tokens = min(num_tokens, self.tokens.len() - self.ready_token_index);
        let ready_token_index = self.ready_token_index;
        self.ready_token_index += num_tokens;
        &self.total_tokens()[ready_token_index..self.ready_token_index]
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn unschedule_tokens(&mut self, tokens: &[Token]) {
        let num_tokens = tokens.len();
        let ready_token_index = self.ready_token_index;
        self.ready_token_index -= num_tokens;
        debug_assert_eq!(&self.total_tokens()[self.ready_token_index..ready_token_index], tokens);
    }
}
