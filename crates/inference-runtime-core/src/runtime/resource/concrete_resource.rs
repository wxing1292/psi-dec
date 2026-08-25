use super::ResourceID;
use super::ResourceURI;
use super::SymbolicResource;
use crate::memory::OffsetAllocation;

#[derive(Debug)]
pub struct ConcreteResource {
    id: ResourceID,
    uri: ResourceURI,
    source: OffsetAllocation,
    num_resource_tokens: u32,
}

impl ConcreteResource {
    pub fn new(id: ResourceID, uri: ResourceURI, source: OffsetAllocation, num_resource_tokens: u32) -> Self {
        debug_assert!(
            num_resource_tokens != 0,
            "concrete resource must contain at least one resource token"
        );
        Self {
            id,
            uri,
            source,
            num_resource_tokens,
        }
    }

    pub const fn id(&self) -> ResourceID {
        self.id
    }

    pub const fn uri(&self) -> &ResourceURI {
        &self.uri
    }

    pub const fn source(&self) -> &OffsetAllocation {
        &self.source
    }

    pub const fn num_resource_tokens(&self) -> u32 {
        self.num_resource_tokens
    }

    pub fn into_symbolic(self) -> SymbolicResource {
        SymbolicResource::new(self.id, self.uri)
    }
}
