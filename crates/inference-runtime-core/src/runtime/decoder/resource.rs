use std::sync::Arc;

use crate::runtime::ResourceID;
use crate::runtime::Token;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceSegment {
    resource_id: ResourceID,
    local_token_index: u16,
    resource_index: u32,
    len: u16,
}

impl ResourceSegment {
    pub fn new(resource_id: ResourceID, local_token_index: u16, resource_index: u32, len: u16) -> Self {
        debug_assert!(len > 0, "resource segment must have a positive length");
        Self {
            resource_id,
            local_token_index,
            resource_index,
            len,
        }
    }

    #[inline]
    pub const fn resource_id(&self) -> ResourceID {
        self.resource_id
    }

    #[inline]
    pub const fn local_token_index(&self) -> u16 {
        self.local_token_index
    }

    #[inline]
    pub const fn resource_index(&self) -> u32 {
        self.resource_index
    }

    #[inline]
    pub const fn len(&self) -> u16 {
        self.len
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BlockAnnotation {
    Resource(ResourceSegment),
    PrefixTokens(Arc<[Token]>),
}

impl BlockAnnotation {
    pub const fn resource(resource_segment: ResourceSegment) -> Self {
        Self::Resource(resource_segment)
    }

    pub fn prefix_tokens(tokens: Arc<[Token]>) -> Self {
        Self::PrefixTokens(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::resource::ResourceTypeID;

    #[test]
    fn test_resource_segment_preserves_each_coordinate() {
        let resource_id = ResourceID::new(ResourceTypeID::new(7));
        let segment = ResourceSegment::new(resource_id, 3, 11, 5);

        assert_eq!(segment.resource_id(), resource_id);
        assert_eq!(segment.local_token_index(), 3);
        assert_eq!(segment.resource_index(), 11);
        assert_eq!(segment.len(), 5);
    }
}
