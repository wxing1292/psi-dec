use super::ConcreteResource;
use super::ResourceID;
use super::ResourceURI;
use crate::memory::OffsetAllocation;

#[derive(Debug)]
pub struct SymbolicResource {
    id: ResourceID,
    uri: ResourceURI,
}

impl SymbolicResource {
    pub const fn new(id: ResourceID, uri: ResourceURI) -> Self {
        Self { id, uri }
    }

    pub const fn id(&self) -> ResourceID {
        self.id
    }

    pub const fn uri(&self) -> &ResourceURI {
        &self.uri
    }

    pub fn into_concrete(self, source: OffsetAllocation, num_resource_tokens: u32) -> ConcreteResource {
        ConcreteResource::new(self.id, self.uri, source, num_resource_tokens)
    }
}
