use std::sync::Arc;

use crate::memory::DeviceBlock;
use crate::memory::U32IDAllocator;
use crate::runtime::decoder::KVBlockAllocationResult;
use crate::runtime::decoder::KVBlockAllocator;
use crate::runtime::decoder::KVBlockPlacement;

#[derive(Clone, Debug)]
pub struct TPKVBlockAllocator {
    num_pages_per_kv_block: usize,
    allocator: Arc<U32IDAllocator>,
}

impl TPKVBlockAllocator {
    pub fn new(num_pages_per_kv_block: usize, allocator: Arc<U32IDAllocator>) -> Self {
        Self {
            num_pages_per_kv_block,
            allocator,
        }
    }
}

impl KVBlockAllocator for TPKVBlockAllocator {
    fn allocate(&self) -> KVBlockAllocationResult {
        match self.allocator.alloc_many(self.num_pages_per_kv_block) {
            Ok(page_ids) => {
                KVBlockAllocationResult::Ok {
                    block_placement: KVBlockPlacement::Device {
                        block: DeviceBlock::tp(self.allocator.clone(), page_ids),
                    },
                }
            },
            Err(_) => KVBlockAllocationResult::ResourceLimitExceeded,
        }
    }

    fn num_raw_free_blocks(&self) -> usize {
        if self.num_pages_per_kv_block == 0 {
            return usize::MAX;
        }
        self.allocator.free() / self.num_pages_per_kv_block
    }

    fn num_raw_used_blocks(&self) -> usize {
        if self.num_pages_per_kv_block == 0 {
            return 0;
        }
        self.allocator.used() / self.num_pages_per_kv_block
    }

    fn num_raw_total_blocks(&self) -> usize {
        if self.num_pages_per_kv_block == 0 {
            return usize::MAX;
        }
        self.allocator.capacity() / self.num_pages_per_kv_block
    }
}
