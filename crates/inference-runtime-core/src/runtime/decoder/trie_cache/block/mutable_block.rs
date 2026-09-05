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
    tokens: [Token; N],
    num_tokens: usize,
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
        let num_tokens = tokens.len();
        let mut storage = [Token::default(); N];
        storage[..num_tokens].copy_from_slice(&tokens);
        Self {
            annotations,
            tokens: storage,
            num_tokens,
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
    pub fn write_tokens(&mut self, token_index: usize, tokens: &[Token]) {
        let start_index = token_index;
        let end_index = start_index + tokens.len();
        debug_assert!(start_index <= self.num_tokens);
        debug_assert!(end_index <= N);
        self.tokens[start_index..end_index].copy_from_slice(tokens);
        self.num_tokens = self.num_tokens.max(end_index);
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
            self.tokens[..self.num_tokens].to_vec(),
            self.scheduled_token_index,
            self.ready_token_index,
            self.kv_placement,
            self.state_placement,
        )
    }

    fn sanity_check(&self) {
        debug_assert!(self.scheduled_token_index <= self.ready_token_index);
        debug_assert!(self.ready_token_index <= self.num_tokens);
        debug_assert!(self.num_tokens <= N);
        debug_assert!(
            self.tokens[..self.scheduled_token_index]
                .iter()
                .all(|token| *token != Token::default())
        );
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
        &self.tokens[self.ready_token_index..self.num_tokens]
    }

    fn total_tokens(&self) -> &[Token] {
        &self.tokens[..self.num_tokens]
    }

    fn ready_token_slots(&self) -> usize {
        N - self.ready_token_index
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn cache_tokens(&mut self, num_tokens: usize) {
        self.scheduled_token_index += num_tokens;
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn schedule_tokens(&mut self, num_tokens: usize) -> &[Token] {
        let num_tokens = min(num_tokens, self.num_tokens - self.ready_token_index);
        let ready_token_index = self.ready_token_index;
        self.ready_token_index += num_tokens;
        &self.total_tokens()[ready_token_index..self.ready_token_index]
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn unschedule_tokens(&mut self, num_tokens: usize) {
        self.ready_token_index -= num_tokens;
    }
}
