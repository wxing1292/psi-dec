use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures_util::pin_mut;
use futures_util::task::noop_waker_ref;
use itertools::Itertools;
use rand::RngExt;
use smallvec::SmallVec;

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
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::block::DecoderBlock;

const NUM_KV_PAGES_PER_BLOCK: usize = 1;
const NUM_STATE_PAGES_PER_BLOCK: usize = 1;
const BLOCK_CACHE_CAPACITY: usize = 1024;

const NUM_TOKEN_PER_BLOCK: usize = 4;
const NUM_TRIE_PARTITION: usize = 4;
const NUM_CACHE_LANE: usize = 2;

#[test]
fn test_mutable_block_alloc_mutable_free() {
    let _rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block_vec);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_resource_limit_free() {
    let _rng = rand::rng();

    let max_pages_per_lane = [2; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::ResourceLimitExceeded = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block_vec_0);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_mutable_block(block_vec_1);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_mutable_commit_immutable() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec {
        block.insert_annotations(annotations.clone());
        assert_eq!(Vec::<Token>::new(), block.push_tokens(tokens.clone()));
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_mutable_block_alloc_mutable_commit_immutable_collision() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))];
    let tokens: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(4, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_0 {
        block.insert_annotations(annotations.clone());
        assert_eq!(Vec::<Token>::new(), block.push_tokens(tokens.clone()));
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_0);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_1 {
        block.insert_annotations(annotations.clone());
        assert_eq!(Vec::<Token>::new(), block.push_tokens(tokens.clone()));
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_1);
    let CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    for (block_0, block_1) in block_vec_0.iter().zip(block_vec_1.iter()) {
        assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    }
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_free() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_vec);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_resource_limit_free() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [2; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations_0: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_0: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec_0 =
        std::array::from_fn(|_| BlockMetadata::new(None, annotations_0.clone(), tokens_0.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let annotations_1: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_1: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec_1 =
        std::array::from_fn(|_| BlockMetadata::new(None, annotations_1.clone(), tokens_1.clone()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::ResourceLimitExceeded {
        block_metadata_vec: block_metadata_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(block_metadata_vec_1_clone, block_metadata_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_vec_0);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_vec_1);
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_commit_immutable() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { mut block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_semi_immutable_commit_immutable_collision() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let result = block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>();
    let AllocateMultiLaneMutableBlockResult::Mutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    let block_metadata_vec_1 = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_0 {
        block.insert_annotations(annotations.clone());
        assert_eq!(Vec::<Token>::new(), block.push_tokens(tokens.iter().copied().collect()));
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_0);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_1 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_1);
    let CommitMultiLaneSemiImmutableBlockResult::ImmutableCollision { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    for (block_0, block_1) in block_vec_0.iter().zip(block_vec_1.iter()) {
        assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    }
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_immutable() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec_0 = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_0 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_0);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let block_metadata_vec_1 = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    for (block_0, block_1) in block_vec_0.iter().zip(block_vec_1.iter()) {
        assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    }
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_reservation_collision() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens: Arc<[Token]> = (0..NUM_TOKEN_PER_BLOCK)
        .map(|_| Token(rng.random::<u32>()))
        .collect::<Vec<_>>()
        .into();
    let block_metadata_vec_0 = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    let block_metadata_vec_1 = std::array::from_fn(|_| BlockMetadata::new(None, annotations.clone(), tokens.clone()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Wait {
        block_metadata_vec: block_metadata_vec_1,
        wait: wait_1,
    } = result
    else {
        unreachable!()
    };
    pin_mut!(wait_1);
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(wait_1.as_mut().poll(&mut cx), Poll::Pending));
    assert_eq!(block_metadata_vec_1_clone, block_metadata_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_0 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_0);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert!(matches!(wait_1.as_mut().poll(&mut cx), Poll::Ready(_)));
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    for (block_0, block_1) in block_vec_0.iter().zip(block_vec_1.iter()) {
        assert_eq!(block_0.trie_node_key(), block_1.trie_node_key());
    }
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_mutable_commit_immutable_0() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations_common: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_common: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();

    let annotations_diff_0: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_0: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_0 = [annotations_common.clone(), annotations_diff_0];
    let tokens_0 = [tokens_common.clone(), tokens_diff_0];
    let block_metadata_vec_0 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_0[i].clone(), tokens_0[i].clone().into()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };

    let annotations_diff_1: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_1: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_1 = [annotations_common.clone(), annotations_diff_1];
    let tokens_1 = [tokens_common.clone(), tokens_diff_1];
    let block_metadata_vec_1 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_1[i].clone(), tokens_1[i].clone().into()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Mutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for (block, block_metadata) in block_vec_1.iter_mut().zip(block_metadata_vec_1_clone.iter()) {
        assert_eq!(
            Vec::<Token>::new(),
            block.push_tokens(block_metadata.tokens().as_ref().to_vec())
        );
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_1);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    block_cache.free_semi_immutable_block(block_vec_0);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_mutable_commit_immutable_1() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations_common: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_common: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();

    let annotations_diff_0: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_0: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_0 = [annotations_common.clone(), annotations_diff_0];
    let tokens_0 = [tokens_common.clone(), tokens_diff_0];
    let block_metadata_vec_0 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_0[i].clone(), tokens_0[i].clone().into()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    for block in &mut block_vec_0 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_0);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let annotations_diff_1: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_1: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_1 = [annotations_common.clone(), annotations_diff_1];
    let tokens_1 = [tokens_common.clone(), tokens_diff_1];
    let block_metadata_vec_1 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_1[i].clone(), tokens_1[i].clone().into()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Mutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    while block_cache.try_evict() {}
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for (block, block_metadata) in block_vec_1.iter_mut().zip(block_metadata_vec_1_clone.iter()) {
        assert_eq!(
            Vec::<Token>::new(),
            block.push_tokens(block_metadata.tokens().as_ref().to_vec())
        );
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_1);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(2, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_mutable_commit_immutable_collision_0() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations_common: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_common: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();

    let annotations_diff_0: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_0: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_0 = [annotations_common.clone(), annotations_diff_0];
    let tokens_0 = [tokens_common.clone(), tokens_diff_0];
    let block_metadata_vec_0 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_0[i].clone(), tokens_0[i].clone().into()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };

    let annotations_diff_1: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_1: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_1 = [annotations_common.clone(), annotations_diff_1];
    let tokens_1 = [tokens_common.clone(), tokens_diff_1];
    let block_metadata_vec_1 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_1[i].clone(), tokens_1[i].clone().into()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Mutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(2, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for block in &mut block_vec_0 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_0);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for (block, block_metadata) in block_vec_1.iter_mut().zip(block_metadata_vec_1_clone.iter()) {
        assert_eq!(
            Vec::<Token>::new(),
            block.push_tokens(block_metadata.tokens().as_ref().to_vec())
        );
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_1);
    let CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(3, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(3, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[test]
fn test_semi_immutable_block_reserve_mutable_commit_immutable_collision_1() {
    let mut rng = rand::rng();

    let max_pages_per_lane = [1024; NUM_CACHE_LANE];
    let cache_capacity = 1024;
    let (page_id_allocators, block_cache) = initialize_block_cache(max_pages_per_lane, cache_capacity);

    let annotations_common: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_common: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();

    let annotations_diff_0: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_0: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_0 = [annotations_common.clone(), annotations_diff_0];
    let tokens_0 = [tokens_common.clone(), tokens_diff_0];
    let block_metadata_vec_0 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_0[i].clone(), tokens_0[i].clone().into()));
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_0);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable {
        block_vec: mut block_vec_0,
    } = result
    else {
        unreachable!()
    };
    for block in &mut block_vec_0 {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_semi_immutable_block(block_vec_0);
    let CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec: block_vec_0 } = result else {
        unreachable!()
    };
    assert_eq!(
        2,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    let annotations_diff_1: SmallVec<[BlockAnnotation; 1]> = vec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest(rng.random()),
        rng.random(),
    ))]
    .into();
    let tokens_diff_1: Vec<Token> = (0..NUM_TOKEN_PER_BLOCK).map(|_| Token(rng.random::<u32>())).collect();
    let annotations_1 = [annotations_common.clone(), annotations_diff_1];
    let tokens_1 = [tokens_common.clone(), tokens_diff_1];
    let block_metadata_vec_1 =
        std::array::from_fn(|i| BlockMetadata::new(None, annotations_1[i].clone(), tokens_1[i].clone().into()));
    let block_metadata_vec_1_clone = block_metadata_vec_1.clone();
    let result = block_cache.reserve_semi_immutable_block::<NUM_TOKEN_PER_BLOCK>(block_metadata_vec_1);
    let ReserveMultiLaneSemiImmutableBlockResult::Mutable {
        block_vec: mut block_vec_1,
    } = result
    else {
        unreachable!()
    };
    assert_eq!(
        4,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(2, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    for (block, block_metadata) in block_vec_1.iter_mut().zip(block_metadata_vec_1_clone.iter()) {
        assert_eq!(
            Vec::<Token>::new(),
            block.push_tokens(block_metadata.tokens().as_ref().to_vec())
        );
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(&scheduled_tokens);
    }
    let result = block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec_1);
    let CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec: block_vec_1 } = result else {
        unreachable!()
    };
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(3, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());

    drop(block_vec_0);
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(2, block_cache.num_pinned_immutable_block());
    assert_eq!(1, block_cache.num_unpinned_immutable_block());

    drop(block_vec_1);
    assert_eq!(2, page_id_allocators.len());
    assert_eq!(2, page_id_allocators[0].used());
    assert_eq!(4, page_id_allocators[1].used());
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(3, block_cache.num_unpinned_immutable_block());

    while block_cache.try_evict() {}
    assert_eq!(
        0,
        page_id_allocators
            .iter()
            .map(|page_id_allocator| page_id_allocator.used())
            .all_equal_value()
            .unwrap()
    );
    assert_eq!(0, block_cache.num_mutable_block());
    assert_eq!(0, block_cache.num_semi_immutable_block());
    assert_eq!(0, block_cache.num_pinned_immutable_block());
    assert_eq!(0, block_cache.num_unpinned_immutable_block());
}

#[allow(clippy::type_complexity)]
fn initialize_block_cache(
    max_pages_per_lane: [usize; NUM_CACHE_LANE],
    _cache_capacity: usize,
) -> (
    Vec<Arc<U32IDAllocator>>,
    MultiLaneTrieBlockCache<NUM_TRIE_PARTITION, NUM_CACHE_LANE, TPKVBlockAllocator, TPStateBlockAllocator>,
) {
    let page_id_allocators: [Arc<U32IDAllocator>; NUM_CACHE_LANE] =
        std::array::from_fn(|lane| Arc::new(U32IDAllocator::new(max_pages_per_lane[lane] as u64)));
    let block_cache_vec: [Arc<SingleLaneTrieBlockCache<NUM_TRIE_PARTITION, TPKVBlockAllocator, TPStateBlockAllocator>>;
        NUM_CACHE_LANE] = std::array::from_fn(|lane| {
        let kv_block_allocator = TPKVBlockAllocator::new(NUM_KV_PAGES_PER_BLOCK, page_id_allocators[lane].clone());
        let state_block_allocator =
            TPStateBlockAllocator::new(NUM_STATE_PAGES_PER_BLOCK, page_id_allocators[lane].clone());
        Arc::new(SingleLaneTrieBlockCache::new(
            kv_block_allocator,
            state_block_allocator,
            BLOCK_CACHE_CAPACITY,
            Shutdown::new(),
        ))
    });
    (
        page_id_allocators.to_vec(),
        MultiLaneTrieBlockCache::new(block_cache_vec),
    )
}
