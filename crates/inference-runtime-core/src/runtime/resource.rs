use std::collections::HashSet;

use uuid::Uuid;

use crate::Error;
use crate::Result;

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
        if current.0 < previous.1 {
            return Err(Error::invalid_argument(
                "request resource placements contain overlapping token ranges",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_id_round_trip_preserves_type() {
        let resource_type = ResourceTypeID::new(0x1234);
        let resource_id = ResourceID::new(resource_type);

        assert_eq!(resource_id.resource_type(), resource_type);
        assert_eq!(ResourceID::from_uuid(resource_id.uuid()).unwrap(), resource_id);
        assert_eq!(
            resource_id.uuid().as_bytes()[UUID_VERSION_BYTE_INDEX] >> 4,
            UUID_VERSION_8
        );
    }

    #[test]
    fn resource_id_rejects_other_uuid_versions() {
        let uuid = Uuid::new_v4();
        assert!(matches!(ResourceID::from_uuid(uuid), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn resource_placement_preserves_mapping_order() {
        let resource_id = ResourceID::new(ResourceTypeID::new(1));
        let placement = ResourcePlacement::new(resource_id, vec![(2, 0, 3), (8, 4, 2)], 16);

        assert_eq!(placement.resource_id(), resource_id);
        assert_eq!(placement.placements(), &[(2, 0, 3), (8, 4, 2)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn resource_placement_asserts_invalid_mapping_contracts() {
        let resource_id = ResourceID::new(ResourceTypeID::new(1));
        assert!(std::panic::catch_unwind(|| ResourcePlacement::new(resource_id, Vec::new(), 16)).is_err());
        assert!(
            std::panic::catch_unwind(|| ResourcePlacement::new(resource_id, vec![(8, 0, 1), (2, 1, 1)], 16)).is_err()
        );
        assert!(std::panic::catch_unwind(|| ResourcePlacement::new(resource_id, vec![(0, 0, 0)], 16)).is_err());
        assert!(std::panic::catch_unwind(|| ResourcePlacement::new(resource_id, vec![(15, 0, 2)], 16)).is_err());
        assert!(
            std::panic::catch_unwind(|| ResourcePlacement::new(resource_id, vec![(0, usize::MAX, 1)], 16)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { ResourcePlacement::new(resource_id, vec![(1, 0, 3), (3, 4, 2)], 16) })
                .is_err()
        );
    }

    #[test]
    fn request_placements_reject_duplicate_resource_and_cross_resource_overlap() {
        let first_id = ResourceID::new(ResourceTypeID::new(1));
        let second_id = ResourceID::new(ResourceTypeID::new(2));
        let first = ResourcePlacement::new(first_id, vec![(2, 0, 3)], 16);
        let duplicate = ResourcePlacement::new(first_id, vec![(8, 3, 2)], 16);
        assert!(validate_resource_placements(&[first.clone(), duplicate]).is_err());

        let overlap = ResourcePlacement::new(second_id, vec![(4, 0, 3)], 16);
        assert!(validate_resource_placements(&[first, overlap]).is_err());
    }

    #[test]
    fn request_placements_accept_disjoint_resources() {
        let first = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(1)), vec![(2, 0, 3)], 16);
        let second = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(2)), vec![(8, 4, 2)], 16);

        validate_resource_placements(&[first, second]).unwrap();
    }
}
