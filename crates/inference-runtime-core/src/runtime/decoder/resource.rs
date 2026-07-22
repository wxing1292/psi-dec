use std::sync::Arc;

use crate::runtime::Token;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceDigest(pub [u8; 32]);
impl AsRef<[u8]> for ResourceDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceSegment {
    resource_digest: ResourceDigest,

    block_offset: u16,
}

impl ResourceSegment {
    pub fn new(resource_digest: ResourceDigest, block_offset: u16) -> Self {
        Self {
            resource_digest,
            block_offset,
        }
    }

    #[inline]
    pub fn resource_digest(&self) -> &ResourceDigest {
        &self.resource_digest
    }

    #[inline]
    pub fn block_offset(&self) -> u16 {
        self.block_offset
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
