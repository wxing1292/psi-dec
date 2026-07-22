use std::sync::Arc;

use crate::memory::U32IDAllocator;

#[derive(Debug)]
pub enum DeviceBlock {
    TP {
        allocator: Arc<U32IDAllocator>,
        page_ids: Vec<u32>,
    },
}

impl DeviceBlock {
    pub fn tp(allocator: Arc<U32IDAllocator>, page_ids: Vec<u32>) -> Self {
        Self::TP { allocator, page_ids }
    }

    pub fn page_ids(&self) -> &[u32] {
        match self {
            Self::TP { page_ids, .. } => page_ids,
        }
    }
}

impl PartialEq for DeviceBlock {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::TP {
                    allocator: l_allocator,
                    page_ids: l_page_ids,
                },
                Self::TP {
                    allocator: r_allocator,
                    page_ids: r_page_ids,
                },
            ) => Arc::ptr_eq(l_allocator, r_allocator) && l_page_ids == r_page_ids,
        }
    }
}

impl Eq for DeviceBlock {}

impl Drop for DeviceBlock {
    fn drop(&mut self) {
        match self {
            Self::TP { allocator, page_ids } => {
                allocator.free_many(page_ids.drain(..));
            },
        }
    }
}
