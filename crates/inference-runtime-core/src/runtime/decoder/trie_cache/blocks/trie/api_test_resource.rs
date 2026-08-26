use std::sync::Arc;

use smallvec::SmallVec;

use super::api_test_support::concrete_resource;
use crate::channel::Shutdown;
use crate::compute::QueryTokens;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::TPStateBlockAllocator;
use crate::runtime::decoder::allocator::TPKVBlockAllocator;
use crate::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::resource::Resource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourcePlacement;
use crate::runtime::resource::ResourceTypeID;
use crate::runtime::resource::ResourceURI;
use crate::runtime::resource::SymbolicResource;

const NUM_TOKEN_PER_BLOCK: usize = 4;
const NUM_TRIE_PARTITION: usize = 4;

type TestSingleLaneBlockCache = SingleLaneTrieBlockCache<NUM_TRIE_PARTITION, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestMultiLaneBlockCache<const L: usize> =
    MultiLaneTrieBlockCache<NUM_TRIE_PARTITION, L, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestBlocks<const L: usize> =
    TrieDecoderBlocks<NUM_TOKEN_PER_BLOCK, NUM_TRIE_PARTITION, L, TestMultiLaneBlockCache<L>>;

#[test]
fn test_resource_annotations_block_zero_l1() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    assert_eq!(
        [annotations([resource(resource_id, 3, 10, 1)])],
        resource_annotations::<1>(resource_id, 0)
    );
}

#[test]
fn test_resource_annotations_block_nonzero_l1() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    assert_eq!(
        [annotations([resource(resource_id, 0, 11, 3)])],
        resource_annotations::<1>(resource_id, 1)
    );
}

#[test]
fn test_resource_annotations_block_zero_l2() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    assert_eq!(
        [
            annotations([resource(resource_id, 3, 10, 1)]),
            annotations([
                resource(resource_id, 2, 10, 2),
                BlockAnnotation::prefix_tokens(vec![Token::new(0)].into()),
            ]),
        ],
        resource_annotations::<2>(resource_id, 0)
    );
}

#[test]
fn test_resource_annotations_block_nonzero_l2() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    assert_eq!(
        [
            annotations([resource(resource_id, 0, 11, 3)]),
            annotations([resource(resource_id, 0, 12, 2)]),
        ],
        resource_annotations::<2>(resource_id, 1)
    );
}

#[test]
fn test_device_resource_placements_w_query_intersection() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 1, 2), (6, 4, 2)], 9);
    let (resource, _resource_allocator) = concrete_resource(resource_id, 8);
    let blocks = TestBlocks::<1>::new(
        block_cache::<1>(),
        vec![resource],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        (0..9).map(Token::new),
    );
    let query_tokens = QueryTokens::Prefill {
        epoch: 0,
        token_index: 2,
        tokens: vec![Token::new(2)],
        window: 1,
    };

    let placements = blocks.device_resource_placements(&query_tokens);

    assert_eq!(1, placements.len());
    assert_eq!(0, placements[0].arena_offset_bytes());
    assert_eq!(32, placements[0].arena_len_bytes());
    assert_eq!(&[(1, 1, 2), (6, 4, 2)], placements[0].placements());
}

#[test]
fn test_device_resource_placements_wo_query_intersection() {
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 1, 2), (6, 4, 2)], 9);
    let (resource, _resource_allocator) = concrete_resource(resource_id, 8);
    let blocks = TestBlocks::<1>::new(
        block_cache::<1>(),
        vec![resource],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        (0..9).map(Token::new),
    );
    let query_tokens = QueryTokens::Prefill {
        epoch: 0,
        token_index: 3,
        tokens: vec![Token::new(3), Token::new(4)],
        window: 2,
    };

    assert!(blocks.device_resource_placements(&query_tokens).is_empty());
}

fn resource_annotations<const L: usize>(
    resource_id: ResourceID,
    block_index: usize,
) -> [SmallVec<[BlockAnnotation; 1]>; L] {
    let blocks = TestBlocks::new(
        block_cache::<L>(),
        vec![Resource::Symbolic(SymbolicResource::new(
            resource_id,
            ResourceURI::new("test://resource".to_string()),
        ))],
        vec![ResourcePlacement::new(resource_id, vec![(3, 10, 4)], 9)],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        (0..9).map(Token::new),
    );
    blocks.block_annotation_vec(block_index)
}

fn block_cache<const L: usize>() -> Arc<TestMultiLaneBlockCache<L>> {
    let block_cache_vec = std::array::from_fn(|_| {
        let page_id_allocator = Arc::new(U32IDAllocator::new(16));
        Arc::new(TestSingleLaneBlockCache::new(
            TPKVBlockAllocator::new(1, page_id_allocator.clone()),
            TPStateBlockAllocator::new(1, page_id_allocator),
            16,
            Shutdown::new(),
        ))
    });
    Arc::new(TestMultiLaneBlockCache::new(block_cache_vec))
}

fn resource(resource_id: ResourceID, local_token_index: u16, resource_index: u32, len: u16) -> BlockAnnotation {
    BlockAnnotation::resource(ResourceSegment::new(
        resource_id,
        local_token_index,
        resource_index,
        len,
    ))
}

fn annotations(items: impl IntoIterator<Item = BlockAnnotation>) -> SmallVec<[BlockAnnotation; 1]> {
    items.into_iter().collect()
}
