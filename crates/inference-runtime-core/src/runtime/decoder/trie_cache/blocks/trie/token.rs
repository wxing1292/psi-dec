use crate::runtime::Token;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;

impl<const N: usize, const P: usize, const L: usize, BC> TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    pub fn num_cached_tokens(&self) -> usize {
        self.immutable_blocks
            .iter()
            .map(|immutable_block_vec| immutable_block_vec[0].cached_tokens().len())
            .sum::<usize>()
            + self
                .semi_immutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].cached_tokens().len())
                .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].cached_tokens().len())
                .sum::<usize>()
    }

    pub fn cached_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.immutable_blocks
            .iter()
            .flat_map(|immutable_block_vec| immutable_block_vec[0].cached_tokens())
            .chain(
                self.semi_immutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].cached_tokens()),
            )
            .chain(
                self.mutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].cached_tokens()),
            )
            .copied()
    }

    pub fn num_scheduled_tokens(&self) -> usize {
        self.semi_immutable_blocks
            .iter()
            .map(|block_vec| block_vec[0].scheduled_tokens().len())
            .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].scheduled_tokens().len())
                .sum::<usize>()
    }

    pub fn scheduled_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.semi_immutable_blocks
            .iter()
            .flat_map(|block_vec| block_vec[0].scheduled_tokens())
            .chain(
                self.mutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].scheduled_tokens()),
            )
            .copied()
    }

    pub fn num_ready_tokens(&self) -> usize {
        self.semi_immutable_blocks
            .iter()
            .map(|block_vec| block_vec[0].ready_tokens().len())
            .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].ready_tokens().len())
                .sum::<usize>()
    }

    pub fn ready_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.semi_immutable_blocks
            .iter()
            .flat_map(|block_vec| block_vec[0].ready_tokens())
            .chain(
                self.mutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].ready_tokens()),
            )
            .copied()
    }

    pub fn num_queued_tokens(&self) -> usize {
        self.queued_tokens.len()
    }

    pub fn queued_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.queued_tokens.iter().copied()
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.spec_tokens.len()
    }

    pub fn spec_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.spec_tokens.iter().copied()
    }

    pub fn num_history_tokens(&self) -> usize {
        self.num_history_tokens
    }

    pub fn history_tokens(&self) -> impl Iterator<Item = Token> + Clone {
        self.total_tokens().take(self.num_history_tokens)
    }

    pub fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
    }

    pub fn prompt_tokens(&self) -> impl Iterator<Item = Token> + Clone {
        self.total_tokens()
            .skip(self.num_history_tokens)
            .take(self.num_prompt_tokens)
    }

    pub fn num_sampled_tokens(&self) -> usize {
        self.num_total_tokens()
            .saturating_sub(self.num_history_tokens)
            .saturating_sub(self.num_prompt_tokens)
    }

    pub fn sampled_tokens(&self) -> impl Iterator<Item = Token> + Clone {
        self.total_tokens()
            .skip(self.num_history_tokens)
            .skip(self.num_prompt_tokens)
    }

    pub fn sampled_tokens_rev(&self) -> impl Iterator<Item = Token> + Clone {
        self.total_tokens().rev().take(self.num_sampled_tokens())
    }

    pub fn num_total_tokens(&self) -> usize {
        self.immutable_blocks
            .iter()
            .map(|immutable_block_vec| immutable_block_vec[0].total_tokens().len())
            .sum::<usize>()
            + self
                .semi_immutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].total_tokens().len())
                .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].total_tokens().len())
                .sum::<usize>()
            + self.queued_tokens.len()
    }

    pub fn total_tokens(&self) -> impl DoubleEndedIterator<Item = Token> + Clone {
        self.immutable_blocks
            .iter()
            .flat_map(|immutable_block_vec| immutable_block_vec[0].total_tokens())
            .chain(
                self.semi_immutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].total_tokens()),
            )
            .chain(
                self.mutable_blocks
                    .iter()
                    .flat_map(|block_vec| block_vec[0].total_tokens()),
            )
            .chain(self.queued_tokens.iter())
            .copied()
    }
}
