use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use uuid::Uuid;

use crate::Error;
use crate::Result;

mod concrete_resource;
pub use concrete_resource::ConcreteResource;

pub mod processor;

mod symbolic_resource;
pub use symbolic_resource::SymbolicResource;

#[derive(Debug)]
pub enum Resource {
    Symbolic(SymbolicResource),
    Concrete(ConcreteResource),
}

impl Resource {
    pub const fn id(&self) -> ResourceID {
        match self {
            Self::Symbolic(resource) => resource.id(),
            Self::Concrete(resource) => resource.id(),
        }
    }

    pub const fn resource_type(&self) -> ResourceTypeID {
        self.id().resource_type()
    }

    pub const fn is_symbolic(&self) -> bool {
        matches!(self, Self::Symbolic(_))
    }

    pub const fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    pub const fn uri(&self) -> &ResourceURI {
        match self {
            Self::Symbolic(resource) => resource.uri(),
            Self::Concrete(resource) => resource.uri(),
        }
    }

    pub fn into_concrete(
        self,
        allocator: Arc<dyn crate::memory::BlockAllocator<BlockSegment = crate::memory::OffsetAllocation>>,
        source: crate::memory::OffsetAllocation,
        num_resource_tokens: u32,
    ) -> Self {
        debug_assert!(self.is_symbolic(), "only a symbolic resource can become concrete");
        match self {
            Self::Symbolic(resource) => Self::Concrete(resource.into_concrete(allocator, source, num_resource_tokens)),
            Self::Concrete(_) => panic!("only a symbolic resource can become concrete"),
        }
    }

    pub fn into_symbolic(self) -> Self {
        debug_assert!(self.is_concrete(), "only a concrete resource can become symbolic");
        match self {
            Self::Symbolic(_) => panic!("only a concrete resource can become symbolic"),
            Self::Concrete(resource) => Self::Symbolic(resource.into_symbolic()),
        }
    }

    pub const fn concrete(&self) -> Option<&ConcreteResource> {
        match self {
            Self::Symbolic(_) => None,
            Self::Concrete(resource) => Some(resource),
        }
    }

    pub const fn symbolic(&self) -> Option<&SymbolicResource> {
        match self {
            Self::Symbolic(resource) => Some(resource),
            Self::Concrete(_) => None,
        }
    }
}

const UUID_VERSION_8: u8 = 8;
const UUID_VERSION_BYTE_INDEX: usize = 6;
const UUID_VARIANT_BYTE_INDEX: usize = 8;
const UUID_VERSION_MASK: u8 = 0x0f;
const UUID_VARIANT_MASK: u8 = 0x3f;
const UUID_RFC_9562_VARIANT: u8 = 0x80;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceTypeID(u16);

impl ResourceTypeID {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u16 {
        self.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceID(Uuid);

impl ResourceID {
    pub fn new(resource_type: ResourceTypeID) -> Self {
        let mut bytes = *Uuid::new_v4().as_bytes();
        bytes[..2].copy_from_slice(&resource_type.value().to_be_bytes());
        bytes[UUID_VERSION_BYTE_INDEX] = (bytes[UUID_VERSION_BYTE_INDEX] & UUID_VERSION_MASK) | (UUID_VERSION_8 << 4);
        bytes[UUID_VARIANT_BYTE_INDEX] = (bytes[UUID_VARIANT_BYTE_INDEX] & UUID_VARIANT_MASK) | UUID_RFC_9562_VARIANT;
        Self(Uuid::from_bytes(bytes))
    }

    pub fn from_uuid(uuid: Uuid) -> Result<Self> {
        let bytes = uuid.as_bytes();
        if bytes[UUID_VERSION_BYTE_INDEX] >> 4 != UUID_VERSION_8
            || bytes[UUID_VARIANT_BYTE_INDEX] & !UUID_VARIANT_MASK != UUID_RFC_9562_VARIANT
        {
            return Err(Error::invalid_argument(format!(
                "resource ID must use the project UUIDv8 layout, got {uuid}"
            )));
        }
        Ok(Self(uuid))
    }

    pub const fn resource_type(self) -> ResourceTypeID {
        let bytes = self.0.as_bytes();
        ResourceTypeID::new(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub const fn uuid(self) -> Uuid {
        self.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResourceURI(String);

impl ResourceURI {
    pub fn new(value: String) -> Self {
        assert!(!value.is_empty(), "resource URI must not be empty");
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourcePlacement {
    resource_id: ResourceID,

    /// Each tuple is `(token_index, resource_index, num_resource_tokens)`.
    ///
    /// `token_index` is the absolute index in the initial request token sequence.
    /// `resource_index` is the logical index in the resource embedding sequence.
    /// `num_resource_tokens` is the number of consecutive tokens in both sequences.
    placements: Vec<(usize, usize, usize)>,
}

impl ResourcePlacement {
    pub fn new(resource_id: ResourceID, placements: Vec<(usize, usize, usize)>, initial_token_count: usize) -> Self {
        debug_assert!(
            !placements.is_empty(),
            "resource placement must contain at least one mapping"
        );
        debug_assert!(placements.is_sorted(), "resource placement mappings must be sorted");
        let mut previous_token_end = 0;
        for (placement_index, &(token_index, resource_index, num_resource_tokens)) in placements.iter().enumerate() {
            debug_assert!(
                num_resource_tokens != 0,
                "resource placement mapping {placement_index} must have a positive token count"
            );
            debug_assert!(
                token_index <= initial_token_count && num_resource_tokens <= initial_token_count - token_index,
                "resource placement mapping {placement_index} token range exceeds initial token \
                 count={initial_token_count}"
            );
            debug_assert!(
                num_resource_tokens <= usize::MAX - resource_index,
                "resource placement mapping {placement_index} exceeds the resource-index domain"
            );
            debug_assert!(
                placement_index == 0 || token_index >= previous_token_end,
                "resource placement mapping {placement_index} overlaps the previous token range"
            );
            previous_token_end = token_index + num_resource_tokens;
        }
        Self {
            resource_id,
            placements,
        }
    }

    pub const fn resource_id(&self) -> ResourceID {
        self.resource_id
    }

    pub fn placements(&self) -> &[(usize, usize, usize)] {
        &self.placements
    }

    /// Returns the smallest request-token range that contains all placements.
    /// Tokens inside this range do not have to belong to the resource.
    pub fn token_index_range(&self) -> Range<usize> {
        let &(token_start, ..) = self.placements.first().unwrap();
        let &(last_token_start, _, last_num_resource_tokens) = self.placements.last().unwrap();
        token_start..last_token_start + last_num_resource_tokens
    }
}

pub fn validate_resource_placements(placements: &[ResourcePlacement]) -> Result<()> {
    let mut resource_ids = HashSet::with_capacity(placements.len());
    let mut token_ranges = placements
        .iter()
        .flat_map(|placement| {
            placement
                .placements()
                .iter()
                .map(|&(token_index, _, num_resource_tokens)| (token_index, token_index + num_resource_tokens))
        })
        .collect::<Vec<_>>();

    for placement in placements {
        if !resource_ids.insert(placement.resource_id()) {
            return Err(Error::invalid_argument(format!(
                "request contains multiple placement groups for resource ID {}",
                placement.resource_id().uuid()
            )));
        }
    }

    token_ranges.sort_unstable();
    for ranges in token_ranges.windows(2) {
        let [previous, current] = ranges else { unreachable!() };
        assert!(
            current.0 >= previous.1,
            "request resource placements must not contain overlapping token ranges"
        );
    }
    Ok(())
}

pub fn validate_resources(
    resources: &[Resource],
    placements: &[ResourcePlacement],
    initial_token_count: usize,
) -> Result<()> {
    validate_resource_placements(placements)?;

    let resource_by_id = resources
        .iter()
        .map(|resource| (resource.id(), resource))
        .collect::<std::collections::HashMap<_, _>>();
    if resource_by_id.len() != resources.len() {
        return Err(Error::invalid_argument("request contains duplicate resource IDs"));
    }
    if resources.len() != placements.len() {
        return Err(Error::invalid_argument(
            "request resources and resource placement groups must have a one-to-one relation",
        ));
    }

    for placement in placements {
        let resource = resource_by_id.get(&placement.resource_id()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "resource placement references unknown resource ID {}",
                placement.resource_id().uuid()
            ))
        })?;
        for &(token_index, _, num_resource_tokens) in placement.placements() {
            if token_index > initial_token_count || num_resource_tokens > initial_token_count - token_index {
                return Err(Error::invalid_argument(format!(
                    "resource placement token range exceeds request initial token count={initial_token_count}"
                )));
            }
        }
        let Some(concrete) = resource.concrete() else {
            continue;
        };
        let concrete_num_resource_tokens = concrete.num_resource_tokens() as usize;
        for &(_, resource_index, num_resource_tokens) in placement.placements() {
            if resource_index > concrete_num_resource_tokens
                || num_resource_tokens > concrete_num_resource_tokens - resource_index
            {
                return Err(Error::invalid_argument(format!(
                    "resource placement range exceeds concrete resource token count={} for resource ID {}",
                    concrete.num_resource_tokens(),
                    placement.resource_id().uuid()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod resource_id_test;

#[cfg(test)]
mod resource_placement_test;
