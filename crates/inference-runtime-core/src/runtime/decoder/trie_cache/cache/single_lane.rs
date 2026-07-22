use event_listener::EventListener;

use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

#[derive(Debug)]
pub enum AllocateSingleLaneMutableBlockResult<const N: usize> {
    Mutable { block: MutableBlock<N> },
    ResourceLimitExceeded,
}

#[derive(Debug)]
pub enum CommitSingleLaneMutableBlockResult<const P: usize> {
    Immutable { block: ImmutableBlock<P> },
    ImmutableCollision { block: ImmutableBlock<P> },
}

#[derive(Debug)]
pub enum ReserveSingleLaneSemiImmutableBlockResult<const N: usize, const P: usize> {
    Immutable { block: ImmutableBlock<P> },
    SemiImmutable { block: SemiImmutableBlock<N> },
    Wait { wait: EventListener },
    ResourceLimitExceeded,
}

#[derive(Debug)]
pub enum CommitSingleLaneSemiImmutableBlockResult<const P: usize> {
    Immutable { block: ImmutableBlock<P> },
    ImmutableCollision { block: ImmutableBlock<P> },
}

#[mockall::automock]
pub trait SingleLaneBlockCache<const P: usize>: Send + Sync + 'static {
    fn try_evict(&self) -> bool;

    fn alloc_mutable_block<const N: usize>(&self) -> AllocateSingleLaneMutableBlockResult<N>;
    fn free_mutable_block<const N: usize>(&self, block: MutableBlock<N>);
    fn commit_mutable_block<const N: usize>(
        &self,
        parent_trie_node_key: Option<TrieNodeKey>,
        block: MutableBlock<N>,
    ) -> CommitSingleLaneMutableBlockResult<P>;

    fn reserve_semi_immutable_block<const N: usize>(
        &self,
        block_metadata: BlockMetadata<N>,
    ) -> ReserveSingleLaneSemiImmutableBlockResult<N, P>;
    fn free_semi_immutable_block<const N: usize>(&self, block: SemiImmutableBlock<N>);
    fn commit_semi_immutable_block<const N: usize>(
        &self,
        block: SemiImmutableBlock<N>,
    ) -> CommitSingleLaneSemiImmutableBlockResult<P>;

    fn num_mutable_block(&self) -> usize;
    fn num_semi_immutable_block(&self) -> usize;
    fn num_pinned_immutable_block(&self) -> usize;
    fn num_unpinned_immutable_block(&self) -> usize;
}
