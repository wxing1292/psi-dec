use std::fmt;
use std::sync::Arc;

use super::ResourceID;
use super::ResourceURI;
use super::SymbolicResource;
use crate::memory::BlockAllocator;
use crate::memory::OffsetAllocation;

pub struct ConcreteResource {
    id: ResourceID,
    uri: ResourceURI,
    allocator: Arc<dyn BlockAllocator<BlockSegment = OffsetAllocation>>,
    source: Option<OffsetAllocation>,
    num_resource_tokens: u32,
}

impl ConcreteResource {
    pub fn new(
        id: ResourceID,
        uri: ResourceURI,
        allocator: Arc<dyn BlockAllocator<BlockSegment = OffsetAllocation>>,
        source: OffsetAllocation,
        num_resource_tokens: u32,
    ) -> Self {
        debug_assert!(
            num_resource_tokens != 0,
            "concrete resource must contain at least one resource token"
        );
        Self {
            id,
            uri,
            allocator,
            source: Some(source),
            num_resource_tokens,
        }
    }

    pub const fn id(&self) -> ResourceID {
        self.id
    }

    pub const fn uri(&self) -> &ResourceURI {
        &self.uri
    }

    pub fn source(&self) -> &OffsetAllocation {
        self.source.as_ref().unwrap()
    }

    pub const fn num_resource_tokens(&self) -> u32 {
        self.num_resource_tokens
    }

    pub fn into_symbolic(self) -> SymbolicResource {
        SymbolicResource::new(self.id, self.uri.clone())
    }
}

impl fmt::Debug for ConcreteResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConcreteResource")
            .field("id", &self.id)
            .field("uri", &self.uri)
            .field("source", &self.source())
            .field("num_resource_tokens", &self.num_resource_tokens)
            .finish()
    }
}

impl Drop for ConcreteResource {
    fn drop(&mut self) {
        self.allocator.free_segment(self.source.take().unwrap());
    }
}
