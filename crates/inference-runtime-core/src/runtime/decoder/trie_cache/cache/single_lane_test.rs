use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures_lite::pin;
use futures_util::task::noop_waker_ref;
use rand::RngExt;

use super::*;
use crate::channel::Shutdown;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::ResourceDigest;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::TPStateBlockAllocator;
use crate::runtime::decoder::allocator::TPKVBlockAllocator;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::block::DecoderBlock;

const NUM_KV_PAGES_PER_BLOCK: usize = 1;
const NUM_STATE_PAGES_PER_BLOCK: usize = 1;
const BLOCK_CACHE_CAPACITY: usize = 1024;

const NUM_TOKEN_PER_BLOCK: usize = 4;

#[test]
fn test_mutable_block_alloc_mutable_free() {
    let _rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_resource_limit_free() {
    let _rng = rand::rng();

    let max_page_ids = 2;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block: block_0 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::ResourceLimitExceeded = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block_0);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block_1);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_mutable_commit_immutable() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { mut block } = result else {
        unreachable!()
    };
    block.insert_annotations(annotations.clone());
    assert_eq!(Vec::<Token>::new(), block.push_tokens(tokens.clone()));
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_mutable_block(None, block);
    let CommitSingleLaneMutableBlockResult::Immutable { block } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_mutable_commit_immutable_collision() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block: mut block_0 } = result else {
        unreachable!()
    };
    block_0.insert_annotations(annotations.clone());
    assert_eq!(Vec::<Token>::new(), block_0.push_tokens(tokens.clone()));
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block: mut block_1 } = result else {
        unreachable!()
    };
    block_1.insert_annotations(annotations.clone());
    assert_eq!(Vec::<Token>::new(), block_1.push_tokens(tokens.clone()));
    assert_eq!(4, page_id_allocator.used());
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_0.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_0.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_mutable_block(None, block_0);
    let CommitSingleLaneMutableBlockResult::Immutable { block: block_0 } = result else {
        unreachable!()
    };
    assert_eq!(4, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let trie_node_key = block_0.trie_node_key();
    drop(block_0);
    assert_eq!(4, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_1.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_1.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_mutable_block(None, block_1);
    let CommitSingleLaneMutableBlockResult::ImmutableCollision { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(trie_node_key, block_1.trie_node_key());
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_1);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_free() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let tokens: Arc<[Token]> = tokens.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_resource_limit_free() {
    let mut rng = rand::rng();

    let max_page_ids = 2;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations_0 = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens_0: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let token_0: Arc<[Token]> = tokens_0.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations_0.clone().into(),
        token_0.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block: block_0 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let annotations_1 = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens_1: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let tokens_1: Arc<[Token]> = tokens_1.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations_1.clone().into(),
        tokens_1.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::ResourceLimitExceeded = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_0);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        vec![].into(),
        tokens_1.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_1);
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_commit_immutable() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let tokens: Arc<[Token]> = tokens.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { mut block } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_semi_immutable_block(block);
    let CommitSingleLaneSemiImmutableBlockResult::Immutable { block } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_commit_immutable_collision() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateSingleLaneMutableBlockResult::Mutable { block: mut block_0 } = result else {
        unreachable!()
    };
    block_0.insert_annotations(annotations.clone());
    assert_eq!(Vec::<Token>::new(), block_0.push_tokens(tokens.clone()));
    let tokens: Arc<[Token]> = tokens.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block: mut block_1 } = result else {
        unreachable!()
    };
    assert_eq!(4, page_id_allocator.used());
    assert_eq!(1, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_0.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_0.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_mutable_block(None, block_0);
    let CommitSingleLaneMutableBlockResult::Immutable { block: block_0 } = result else {
        unreachable!()
    };
    assert_eq!(4, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_1.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_1.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_semi_immutable_block(block_1);
    let CommitSingleLaneSemiImmutableBlockResult::ImmutableCollision { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_0);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_1);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_immutable() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let tokens: Arc<[Token]> = tokens.into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block: mut block_0 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_0.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_0.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_semi_immutable_block(block_0);
    let CommitSingleLaneSemiImmutableBlockResult::Immutable { block: block_0 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::Immutable { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_0);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_1);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_reservation_collision() {
    let mut rng = rand::rng();

    let max_page_ids = 1024;
    let cache_capacity = 1024;
    let (page_id_allocator, block_cache) = initialize_block_cache(max_page_ids, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let tokens: Arc<[Token]> = tokens.clone().into();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block: mut block_0 } = result else {
        unreachable!()
    };
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::Wait { wait: wait_1 } = result else {
        unreachable!()
    };
    pin!(wait_1);
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(wait_1.as_mut().poll(&mut cx), Poll::Pending));
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(1, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let scheduled_tokens = block_0.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_0.cache_tokens(&scheduled_tokens);
    let result = block_cache.commit_semi_immutable_block(block_0);
    let CommitSingleLaneSemiImmutableBlockResult::Immutable { block: block_0 } = result else {
        unreachable!()
    };
    assert!(matches!(wait_1.as_mut().poll(&mut cx), Poll::Ready(_)));
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(BlockMetadata::new(
        None,
        annotations.clone().into(),
        tokens.clone(),
    ));
    let ReserveSingleLaneSemiImmutableBlockResult::Immutable { block: block_1 } = result else {
        unreachable!()
    };
    assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_0);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(1, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_1);
    assert_eq!(2, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(0, page_id_allocator.used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[allow(clippy::type_complexity)]
fn initialize_block_cache(
    max_pages: usize,
    _cache_capacity: usize,
) -> (
    Arc<U32IDAllocator>,
    SingleLaneTrieBlockCache<NUM_TOKEN_PER_BLOCK, TPKVBlockAllocator, TPStateBlockAllocator>,
) {
    let page_id_allocator = Arc::new(U32IDAllocator::new(max_pages as u64));
    let kv_block_allocator = TPKVBlockAllocator::new(NUM_KV_PAGES_PER_BLOCK, page_id_allocator.clone());
    let state_block_allocator = TPStateBlockAllocator::new(NUM_STATE_PAGES_PER_BLOCK, page_id_allocator.clone());
    let block_cache = SingleLaneTrieBlockCache::new(
        kv_block_allocator,
        state_block_allocator,
        BLOCK_CACHE_CAPACITY,
        Shutdown::new(),
    );
    (page_id_allocator, block_cache)
}
