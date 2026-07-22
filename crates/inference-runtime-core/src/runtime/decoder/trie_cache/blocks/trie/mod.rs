use std::collections::VecDeque;
use std::sync::Arc;

use crate::compute::DecoderSyncBlocks;
use crate::runtime::Token;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;

mod api;
mod token;

pub struct TrieDecoderBlocks<const N: usize, const P: usize, const L: usize, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    block_cache: Arc<BC>,
    immutable_blocks: Vec<[ImmutableBlock<P>; L]>,
    semi_immutable_blocks: VecDeque<[SemiImmutableBlock<N>; L]>,
    mutable_blocks: VecDeque<[MutableBlock<N>; L]>,

    num_history_tokens: usize,
    num_prompt_tokens: usize,
    queued_tokens: VecDeque<Token>,
    spec_tokens: Vec<Token>,

    num_in_sync_blocks: usize,

    epoch: usize, // TODO there is no test
}

impl<const N: usize, const P: usize, const L: usize, BC> TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    const _ASSERT_N_GT_0: () = assert!(N > 0);

    pub fn new(
        block_cache: Arc<BC>,
        history_tokens: impl IntoIterator<Item = Token>,
        prompt_tokens: impl IntoIterator<Item = Token>,
        sampled_tokens: impl IntoIterator<Item = Token>,
    ) -> Self {
        let mut queued_tokens: VecDeque<Token> = history_tokens.into_iter().collect();
        let num_history_tokens = queued_tokens.len();

        queued_tokens.extend(prompt_tokens);
        let num_prompt_tokens = queued_tokens.len() - num_history_tokens;

        queued_tokens.extend(sampled_tokens);
        assert!(!queued_tokens.is_empty());
        assert!(L - 1 <= queued_tokens.len());
        Self {
            block_cache,

            immutable_blocks: vec![],
            semi_immutable_blocks: VecDeque::new(),
            mutable_blocks: VecDeque::new(),

            num_history_tokens,
            num_prompt_tokens,
            queued_tokens,
            spec_tokens: vec![],

            num_in_sync_blocks: 0,

            epoch: 0,
        }
    }

    #[cfg(debug_assertions)]
    pub fn sanity_check(&self) {
        let mut cached_tokens = vec![];
        let mut scheduled_tokens = vec![];
        let mut ready_tokens = vec![];
        let mut queued_tokens = vec![];
        let mut spec_tokens = vec![];
        let mut cache_lane_total_tokens = std::array::from_fn(|_| vec![]);

        for block_vec in self.immutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        for block_vec in self.semi_immutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        for block_vec in self.mutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        queued_tokens.extend(self.queued_tokens.iter().copied());
        spec_tokens.extend(self.spec_tokens.iter().copied());

        debug_assert_eq!(self.num_cached_tokens(), cached_tokens.len());
        debug_assert_eq!(self.num_scheduled_tokens(), scheduled_tokens.len());
        debug_assert_eq!(self.num_ready_tokens(), ready_tokens.len());
        debug_assert_eq!(self.num_queued_tokens(), queued_tokens.len());
        debug_assert_eq!(self.num_spec_tokens(), spec_tokens.len());

        debug_assert_eq!(self.cached_tokens().collect::<Vec<_>>(), cached_tokens);
        debug_assert_eq!(self.scheduled_tokens().collect::<Vec<_>>(), scheduled_tokens);
        debug_assert_eq!(self.ready_tokens().collect::<Vec<_>>(), ready_tokens);
        debug_assert_eq!(self.queued_tokens().collect::<Vec<_>>(), queued_tokens);
        debug_assert_eq!(self.spec_tokens().collect::<Vec<_>>(), spec_tokens);

        debug_assert_eq!(
            self.num_history_tokens() + self.num_prompt_tokens() + self.num_sampled_tokens(),
            self.num_total_tokens(),
        );
        debug_assert_eq!(
            self.num_total_tokens(),
            cached_tokens.len() + scheduled_tokens.len() + ready_tokens.len() + queued_tokens.len()
        );
        debug_assert_eq!(self.num_spec_tokens(), spec_tokens.len());

        let total_tokens = self.total_tokens().collect::<Vec<_>>();
        debug_assert_eq!(
            self.history_tokens()
                .chain(self.prompt_tokens())
                .chain(self.sampled_tokens())
                .collect::<Vec<_>>(),
            total_tokens,
        );
        debug_assert_eq!(
            total_tokens,
            cached_tokens
                .iter()
                .chain(scheduled_tokens.iter())
                .chain(ready_tokens.iter())
                .chain(queued_tokens.iter())
                .copied()
                .collect::<Vec<_>>(),
        );

        sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens);
    }

    pub fn prepare_blocks(&mut self) -> DecoderSyncBlocks {
        let num_in_sync_blocks = self.num_in_sync_blocks;

        let total_num_blocks =
            self.immutable_blocks.len() + self.semi_immutable_blocks.len() + self.mutable_blocks.len();
        debug_assert!(
            num_in_sync_blocks <= total_num_blocks,
            "num_in_sync_blocks={} exceeds total_num_blocks={}",
            num_in_sync_blocks,
            total_num_blocks
        );
        let num_blocks = total_num_blocks - num_in_sync_blocks;
        let mut kv_page_ids: Vec<Vec<Vec<u32>>> = (0..L).map(|_| Vec::with_capacity(num_blocks)).collect();
        let mut state_page_ids: Vec<Vec<Vec<u32>>> = (0..L).map(|_| Vec::with_capacity(num_blocks)).collect();
        if num_blocks != 0 {
            let immutable_start = num_in_sync_blocks.min(self.immutable_blocks.len());
            let semi_immutable_start = num_in_sync_blocks
                .saturating_sub(self.immutable_blocks.len())
                .min(self.semi_immutable_blocks.len());
            let mutable_start = num_in_sync_blocks
                .saturating_sub(self.immutable_blocks.len() + self.semi_immutable_blocks.len())
                .min(self.mutable_blocks.len());

            for block_vec in &self.immutable_blocks[immutable_start..] {
                for (lane, block) in block_vec.iter().enumerate() {
                    let block_ref = block.trie_node_ref();
                    kv_page_ids[lane].push(block_ref.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block_ref.state_placement().page_ids().to_vec());
                }
            }
            for block_vec in self.semi_immutable_blocks.iter().skip(semi_immutable_start) {
                for (lane, block) in block_vec.iter().enumerate() {
                    kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
                }
            }
            for block_vec in self.mutable_blocks.iter().skip(mutable_start) {
                for (lane, block) in block_vec.iter().enumerate() {
                    kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
                }
            }
        }
        self.num_in_sync_blocks = num_in_sync_blocks + num_blocks;
        DecoderSyncBlocks::new(num_in_sync_blocks, kv_page_ids, state_page_ids)
    }

    pub fn cancel_blocks(&mut self, sync_blocks: DecoderSyncBlocks) {
        self.num_in_sync_blocks = sync_blocks.block_index();
    }

    pub fn commit_blocks(&mut self, _sync_blocks: DecoderSyncBlocks) {
        // noop
    }
}

#[cfg(debug_assertions)]
fn sanity_check_materialized_block_vec<B, const L: usize>(
    block_vec: &[B; L],
    cached_tokens: &mut Vec<Token>,
    scheduled_tokens: &mut Vec<Token>,
    ready_tokens: &mut Vec<Token>,
    cache_lane_total_tokens: &mut [Vec<Token>; L],
) where
    B: crate::runtime::decoder::trie_cache::DecoderBlock,
{
    let main_block = &block_vec[0];
    match (
        cached_tokens.is_empty(),
        scheduled_tokens.is_empty(),
        ready_tokens.is_empty(),
    ) {
        (_, true, true) => { /* noop */ },
        (_, false, true) => {
            debug_assert!(main_block.cached_tokens().is_empty());
        },
        (_, _, false) => {
            debug_assert!(main_block.cached_tokens().is_empty());
            debug_assert!(main_block.scheduled_tokens().is_empty());
        },
    }

    cached_tokens.extend_from_slice(main_block.cached_tokens());
    scheduled_tokens.extend_from_slice(main_block.scheduled_tokens());
    ready_tokens.extend_from_slice(main_block.ready_tokens());

    for (lane, block) in block_vec.iter().enumerate() {
        debug_assert_eq!(main_block.cached_tokens().len(), block.cached_tokens().len());
        debug_assert_eq!(main_block.scheduled_tokens().len(), block.scheduled_tokens().len());
        debug_assert_eq!(main_block.ready_tokens().len(), block.ready_tokens().len());
        debug_assert_eq!(main_block.total_tokens().len(), block.total_tokens().len());

        cache_lane_total_tokens[lane].extend_from_slice(block.total_tokens());
    }
}

#[cfg(debug_assertions)]
fn sanity_check_cache_lane_token_windows<const L: usize>(
    cache_lane_total_tokens: &[Vec<Token>; L],
    total_tokens: &[Token],
) {
    for (lane, cache_tokens) in cache_lane_total_tokens.iter().enumerate() {
        debug_assert!(cache_tokens.len() <= total_tokens.len().saturating_sub(lane));
        if !cache_tokens.is_empty() {
            debug_assert_eq!(cache_tokens, &total_tokens[lane..lane + cache_tokens.len()]);
        }
    }
}

impl<const N: usize, const P: usize, const L: usize, BC> Drop for TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    fn drop(&mut self) {
        for block_vec in self.mutable_blocks.drain(..).rev() {
            self.block_cache.free_mutable_block(block_vec);
        }
        for block_vec in self.semi_immutable_blocks.drain(..).rev() {
            self.block_cache.free_semi_immutable_block(block_vec);
        }
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use crate::runtime::Token;

    #[test]
    fn test_sanity_check_cache_lane_token_windows_enough_token() {
        let cache_lane_total_tokens = [
            vec![Token::new(0), Token::new(1), Token::new(2)],
            vec![Token::new(1), Token::new(2), Token::new(3)],
            vec![Token::new(2), Token::new(3), Token::new(4)],
            vec![Token::new(3), Token::new(4), Token::new(5)],
        ];
        let total_tokens = vec![
            Token::new(0),
            Token::new(1),
            Token::new(2),
            Token::new(3),
            Token::new(4),
            Token::new(5),
        ];
        sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens);
    }

    #[test]
    fn test_sanity_check_cache_lane_token_windows_not_enough_token() {
        let cache_lane_total_tokens = [vec![], vec![], vec![], vec![]];

        let total_tokens = vec![Token::new(0), Token::new(1), Token::new(2)];
        sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens);

        let total_tokens = vec![Token::new(0), Token::new(1)];
        sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens);

        let total_tokens = vec![Token::new(0)];
        sanity_check_cache_lane_token_windows(&cache_lane_total_tokens, &total_tokens);
    }
}
