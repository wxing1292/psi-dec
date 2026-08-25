use std::sync::Arc;

use smallvec::SmallVec;

use crate::channel::Shutdown;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::TPStateBlockAllocator;
use crate::runtime::decoder::allocator::TPKVBlockAllocator;
use crate::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourcePlacement;
use crate::runtime::resource::ResourceTypeID;

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

fn resource_annotations<const L: usize>(
    resource_id: ResourceID,
    block_index: usize,
) -> [SmallVec<[BlockAnnotation; 1]>; L] {
    let block_cache_vec = std::array::from_fn(|_| {
        let page_id_allocator = Arc::new(U32IDAllocator::new(16));
        Arc::new(TestSingleLaneBlockCache::new(
            TPKVBlockAllocator::new(1, page_id_allocator.clone()),
            TPStateBlockAllocator::new(1, page_id_allocator),
            16,
            Shutdown::new(),
        ))
    });
    let blocks = TestBlocks::new(
        Arc::new(TestMultiLaneBlockCache::new(block_cache_vec)),
        vec![ResourcePlacement::new(resource_id, vec![(3, 10, 4)], 9)],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        (0..9).map(Token::new),
    );
    blocks.block_annotation_vec(block_index)
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
