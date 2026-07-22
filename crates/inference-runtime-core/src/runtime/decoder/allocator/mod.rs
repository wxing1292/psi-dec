mod tp_kv_block_allocator;
pub use tp_kv_block_allocator::TPKVBlockAllocator;

mod tp_state_block_allocator;
pub use tp_state_block_allocator::TPStateBlockAllocator;

mod kv_block_placement;
pub use kv_block_placement::KVBlockPlacement;

mod state_block_placement;
pub use state_block_placement::StateBlockPlacement;

pub enum KVBlockAllocationResult {
    Ok { block_placement: KVBlockPlacement },
    ResourceLimitExceeded,
}

pub trait KVBlockAllocator: Send + Sync + 'static {
    fn allocate(&self) -> KVBlockAllocationResult;

    fn num_raw_free_blocks(&self) -> usize;
    fn num_raw_used_blocks(&self) -> usize;
    fn num_raw_total_blocks(&self) -> usize;
}

pub enum StateBlockAllocationResult {
    Ok { block_placement: StateBlockPlacement },
    ResourceLimitExceeded,
}

pub trait StateBlockAllocator: Send + Sync + 'static {
    fn allocate(&self) -> StateBlockAllocationResult;

    fn num_raw_free_blocks(&self) -> usize;
    fn num_raw_used_blocks(&self) -> usize;
    fn num_raw_total_blocks(&self) -> usize;
}
