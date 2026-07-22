use std::sync::Arc;

use smallvec::SmallVec;
use smallvec::smallvec;

use crate::channel::Shutdown;
use crate::memory::DeviceBlock;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::ResourceDigest;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::S3FIFOClient;
use crate::runtime::decoder::trie_cache::Trie;
use crate::runtime::decoder::trie_cache::TrieEdge;

pub const TEST_PARTITIONS: usize = 4;

pub fn new_trie() -> Arc<Trie<TEST_PARTITIONS>> {
    Arc::new(Trie::new(Arc::new(S3FIFOClient::new(32, Shutdown::new()))))
}

pub fn new_annotations(seed: u8) -> SmallVec<[BlockAnnotation; 1]> {
    smallvec![BlockAnnotation::resource(ResourceSegment::new(
        ResourceDigest([seed; 32]),
        seed as u16,
    ))]
}

pub fn new_tokens(values: &[u32]) -> Arc<[Token]> {
    values.iter().copied().map(Token::new).collect::<Vec<_>>().into()
}

pub fn new_edge(seed: u8, values: &[u32]) -> TrieEdge {
    TrieEdge::new(new_annotations(seed), new_tokens(values))
}

pub fn new_kv_placement() -> KVBlockPlacement {
    let allocator = Arc::new(U32IDAllocator::new(8));
    let page_ids = allocator.alloc_many(1).unwrap();
    KVBlockPlacement::Device {
        block: DeviceBlock::tp(allocator, page_ids),
    }
}

pub fn new_state_placement() -> StateBlockPlacement {
    let allocator = Arc::new(U32IDAllocator::new(8));
    let page_ids = allocator.alloc_many(1).unwrap();
    StateBlockPlacement::Device {
        block: DeviceBlock::tp(allocator, page_ids),
    }
}
