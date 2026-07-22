use std::cmp::min;

use ahash::HashSet;
use inference_runtime_macro::sanity_check;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::block::DecoderBlock;

#[derive(Debug)]
pub struct MutableBlock<const N: usize> {
    annotations: HashSet<BlockAnnotation>,
    tokens: Vec<Token>,
    scheduled_token_index: usize,
    ready_token_index: usize,
    kv_placement: KVBlockPlacement,
    state_placement: StateBlockPlacement,
}

impl<const N: usize> MutableBlock<N> {
    pub fn new(
        annotations: HashSet<BlockAnnotation>,
        tokens: Vec<Token>,
        scheduled_token_index: usize,
        ready_token_index: usize,
        kv_placement: KVBlockPlacement,
        state_placement: StateBlockPlacement,
    ) -> Self {
        debug_assert!(scheduled_token_index <= ready_token_index);
        debug_assert!(ready_token_index <= tokens.len());
        debug_assert!(tokens.len() <= N);
        Self {
            annotations,
            tokens,
            scheduled_token_index,
            ready_token_index,
            kv_placement,
            state_placement,
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn insert_annotations(&mut self, annotations: impl IntoIterator<Item = BlockAnnotation>) {
        for annotation in annotations {
            // duplicates are allowed, since self can be created by
            // * alloc_mutable_kv_block
            // * reserve_semi_immutable_kv_block
            self.annotations.insert(annotation);
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn remove_annotations<'a>(&mut self, annotations: impl IntoIterator<Item = &'a BlockAnnotation>) {
        for annotation in annotations {
            // missing entries are tolerated.
            self.annotations.remove(annotation);
        }
    }

    pub fn annotations(&self) -> Vec<BlockAnnotation> {
        let mut annotations: Vec<_> = self.annotations.iter().cloned().collect();
        annotations.sort_unstable();
        annotations
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn push_tokens(&mut self, tokens: Vec<Token>) -> Vec<Token> {
        let size = min(N - self.tokens.len(), tokens.len());
        let mut tokens = tokens.into_iter();
        self.tokens.extend(tokens.by_ref().take(size));
        tokens.collect()
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn pop_tokens(&mut self, tokens: &[Token]) {
        let num_tokens = tokens.len();
        debug_assert_eq!(&self.tokens[self.tokens.len() - num_tokens..], tokens);
        self.tokens.truncate(self.tokens.len() - num_tokens);
    }

    pub fn kv_placement(&self) -> &KVBlockPlacement {
        &self.kv_placement
    }

    pub fn state_placement(&self) -> &StateBlockPlacement {
        &self.state_placement
    }

    pub fn into_inner(
        self,
    ) -> (
        Vec<BlockAnnotation>,
        Vec<Token>,
        usize,
        usize,
        KVBlockPlacement,
        StateBlockPlacement,
    ) {
        let mut annotations: Vec<_> = self.annotations.into_iter().collect();
        annotations.sort_unstable();
        (
            annotations,
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

impl<const N: usize> DecoderBlock for MutableBlock<N> {
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
        &self.tokens
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
