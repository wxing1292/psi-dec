use super::client::S3FIFOClient;
use crate::channel::Shutdown;

#[test]
fn test_shutdown() {
    let shutdown = Shutdown::new();
    let client = S3FIFOClient::<u8>::new(1, shutdown.clone());
    shutdown.shutdown();
    drop(client);
}

#[test]
fn test_insert_is_fire_and_forget() {
    let client = S3FIFOClient::<u8>::new(1, Shutdown::new());
    let eviction = client.new_eviction(1);
    client.insert(eviction.clone());
    eviction.unpin();

    assert_eq!(Some(1), client.evict_candidate());
    client.accept_candidate(1);
}
