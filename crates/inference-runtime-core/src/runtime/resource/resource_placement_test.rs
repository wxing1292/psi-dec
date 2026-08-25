use super::ResourceID;
use super::ResourcePlacement;
use super::ResourceTypeID;
use super::validate_resource_placements;

#[test]
fn test_resource_placement_order() {
    let resource_id = ResourceID::new(ResourceTypeID::new(1));
    let placement = ResourcePlacement::new(resource_id, vec![(2, 0, 3), (8, 4, 2)], 16);

    assert_eq!(placement.resource_id(), resource_id);
    assert_eq!(placement.placements(), &[(2, 0, 3), (8, 4, 2)]);
}

#[test]
#[should_panic(expected = "request resource placements must not contain overlapping token ranges")]
fn test_resource_placement_w_overlap() {
    let first = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(1)), vec![(2, 0, 3)], 16);
    let second = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(2)), vec![(4, 0, 3)], 16);

    let _ = validate_resource_placements(&[first, second]);
}

#[test]
fn test_resource_placement_wo_overlap() {
    let first = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(1)), vec![(2, 0, 3)], 16);
    let second = ResourcePlacement::new(ResourceID::new(ResourceTypeID::new(2)), vec![(8, 4, 2)], 16);

    validate_resource_placements(&[first, second]).unwrap();
}
