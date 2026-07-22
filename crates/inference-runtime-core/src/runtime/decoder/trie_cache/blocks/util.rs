use std::cmp::min;
use std::collections::VecDeque;

use crate::runtime::Token;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::MutableBlock;

pub fn cache_tokens<const N: usize, const L: usize, B>(
    block_vec: &mut [B; L],
    tokens: &[Token],
    index_start: usize,
    index_end: usize,
) where
    B: DecoderBlock,
{
    if index_start == index_end {
        return;
    }
    for (lane, block) in block_vec.iter_mut().enumerate() {
        let scheduled_tokens = block.scheduled_tokens()[..(index_end - index_start)].to_vec();
        debug_assert_eq!(&tokens[index_start + lane..index_end + lane], &scheduled_tokens);
        block.cache_tokens(&scheduled_tokens);
    }
}

pub fn schedule_tokens<const N: usize, const L: usize, B>(block_vec: &mut [B; L], num_tokens: usize) -> &[Token]
where
    B: DecoderBlock,
{
    if num_tokens == 0 {
        return &[];
    }
    let (main_block, mtp_blocks) = block_vec.split_first_mut().unwrap();
    for mtp_block in mtp_blocks {
        mtp_block.schedule_tokens(num_tokens);
    }
    main_block.schedule_tokens(num_tokens)
}

pub fn unschedule_tokens<const N: usize, const L: usize, B>(
    block_vec: &mut [B; L],
    tokens: &[Token],
    index_start: usize,
    index_end: usize,
) where
    B: DecoderBlock,
{
    if index_start == index_end {
        return;
    }
    for (lane, block) in block_vec.iter_mut().enumerate() {
        let scheduled_tokens =
            block.scheduled_tokens()[block.scheduled_tokens().len() - (index_end - index_start)..].to_vec();
        debug_assert_eq!(&tokens[index_start + lane..index_end + lane], &scheduled_tokens);
        block.unschedule_tokens(&scheduled_tokens);
    }
}

pub fn push_tokens<const N: usize, const L: usize>(block_vec: &mut [MutableBlock<N>; L], tokens: &[Token]) {
    if tokens.is_empty() {
        return;
    }
    let window = tokens.len() - (L - 1);
    for (block, tokens) in block_vec.iter_mut().zip(tokens.windows(window)) {
        let remaining = block.push_tokens(tokens.to_vec());
        debug_assert!(remaining.is_empty());
    }
}

pub fn pop_front_queued_tokens<const L: usize>(queued_tokens: &mut VecDeque<Token>, num_tokens: usize) -> Vec<Token> {
    debug_assert!(L >= 1);
    debug_assert!((L - 1) <= queued_tokens.len());

    let num_tokens = min(queued_tokens.len() - (L - 1), num_tokens);
    let tokens = queued_tokens.iter().take(num_tokens + L - 1).copied().collect();
    queued_tokens.drain(0..num_tokens);
    tokens
}

pub fn push_front_queued_tokens<const L: usize>(
    queued_tokens: &mut VecDeque<Token>,
    tokens: impl DoubleEndedIterator<Item = Token>,
) {
    debug_assert!(L >= 1);

    for token in tokens.rev().skip(L - 1) {
        queued_tokens.push_front(token);
    }
}
