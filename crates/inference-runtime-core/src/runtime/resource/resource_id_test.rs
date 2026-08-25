use super::ResourceID;
use super::ResourceTypeID;
use super::UUID_VERSION_8;
use super::UUID_VERSION_BYTE_INDEX;

#[test]
fn test_resource_id_encode_decode() {
    let resource_type = ResourceTypeID::new(0x1234);
    let resource_id = ResourceID::new(resource_type);

    assert_eq!(resource_id.resource_type(), resource_type);
    assert_eq!(ResourceID::from_uuid(resource_id.uuid()).unwrap(), resource_id);
    assert_eq!(
        resource_id.uuid().as_bytes()[UUID_VERSION_BYTE_INDEX] >> 4,
        UUID_VERSION_8
    );
}
