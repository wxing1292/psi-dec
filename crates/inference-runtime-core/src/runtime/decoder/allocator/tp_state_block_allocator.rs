use std::sync::Arc;

use crate::memory::DeviceBlock;
use crate::memory::U32IDAllocator;
use crate::runtime::decoder::allocator::StateBlockAllocationResult;
use crate::runtime::decoder::allocator::StateBlockAllocator;
use crate::runtime::decoder::allocator::StateBlockPlacement;

#[derive(Clone, Debug)]
pub struct TPStateBlockAllocator {
    num_pages_per_state_block: usize,
    allocator: Arc<U32IDAllocator>,
}

impl TPStateBlockAllocator {
    pub fn new(num_pages_per_state_block: usize, allocator: Arc<U32IDAllocator>) -> Self {
        Self {
            num_pages_per_state_block,
            allocator,
        }
    }
}

impl StateBlockAllocator for TPStateBlockAllocator {
    fn allocate(&self) -> StateBlockAllocationResult {
        match self.allocator.alloc_many(self.num_pages_per_state_block) {
            Ok(page_ids) => {
                StateBlockAllocationResult::Ok {
                    block_placement: StateBlockPlacement::Device {
                        block: DeviceBlock::tp(self.allocator.clone(), page_ids),
                    },
                }
            },
            Err(_) => StateBlockAllocationResult::ResourceLimitExceeded,
        }
    }

    fn num_raw_free_blocks(&self) -> usize {
        if self.num_pages_per_state_block == 0 {
            return usize::MAX;
        }
        self.allocator.free() / self.num_pages_per_state_block
    }

    fn num_raw_used_blocks(&self) -> usize {
        if self.num_pages_per_state_block == 0 {
            return 0;
        }
        self.allocator.used() / self.num_pages_per_state_block
    }

    fn num_raw_total_blocks(&self) -> usize {
        if self.num_pages_per_state_block == 0 {
            return usize::MAX;
        }
        self.allocator.capacity() / self.num_pages_per_state_block
    }
}
