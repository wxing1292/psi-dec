use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crossbeam_channel::bounded;
use map_macro::hash_map;
use rand::RngExt;

use super::*;
use crate::runtime::pin_cache::NoopCallback;
use crate::runtime::pin_cache::PinCacheResult;
use crate::runtime::pin_cache::UnitWeight;

#[test]
fn test_insert_get_brand_new_success() {
    let mut rng = rand::rng();

    let key = rng.random();
    let value = rng.random();

    const SHARDS: usize = 1;
    let capacity = 1;
    let weight_fn = Arc::new(UnitWeight);
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard } = cache.insert(key, value) else {
        unreachable!()
    };
    assert_eq!(key, *pin_guard.key());
    assert_eq!(value, *pin_guard.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(1, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key, value, Some(&pin_guard));
    drop(pin_guard);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key, value, None);
    assert_eq!(1, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert { key, value },
            CallbackEvent::Get { key, value },
            CallbackEvent::Get { key, value },
            CallbackEvent::Delete { key, value },
        ],
        events,
    );
}

#[test]
fn test_insert_get_brand_new_evict_success() {
    let mut rng = rand::rng();

    let key_0 = rng.random();
    let value_0 = rng.random();
    let key_1 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 1;
    let weight_fn = Arc::new(UnitWeight);
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key_0, value_0) else {
        unreachable!()
    };
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    let PinCacheResult::Ok { entry: pin_guard_1 } = cache.insert(key_1, value_1) else {
        unreachable!()
    };
    assert_cache_miss(&cache, key_0);
    assert_eq!(key_1, *pin_guard_1.key());
    assert_eq!(value_1, *pin_guard_1.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(1, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key_1, value_1, Some(&pin_guard_1));
    drop(pin_guard_1);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key_1, value_1, None);
    assert_eq!(1, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert {
                key: key_0,
                value: value_0
            },
            CallbackEvent::Delete {
                key: key_0,
                value: value_0
            },
            CallbackEvent::Insert {
                key: key_1,
                value: value_1
            },
            CallbackEvent::Get {
                key: key_1,
                value: value_1
            },
            CallbackEvent::Get {
                key: key_1,
                value: value_1
            },
            CallbackEvent::Delete {
                key: key_1,
                value: value_1
            },
        ],
        events,
    );
}

#[test]
fn test_insert_get_brand_new_evict_fail() {
    let mut rng = rand::rng();

    let key_0 = rng.random();
    let value_0 = rng.random();
    let key_1 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 1;
    let weight_fn = Arc::new(UnitWeight);
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key_0, value_0) else {
        unreachable!()
    };
    if let PinCacheResult::Full { value } = cache.insert(key_1, value_1) {
        assert_eq!(value_1, value);
    } else {
        unreachable!()
    };
    assert_eq!(key_0, *pin_guard_0.key());
    assert_eq!(value_0, *pin_guard_0.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(1, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key_0, value_0, Some(&pin_guard_0));
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key_0, value_0, None);
    assert_eq!(1, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert {
                key: key_0,
                value: value_0
            },
            CallbackEvent::Get {
                key: key_0,
                value: value_0
            },
            CallbackEvent::Get {
                key: key_0,
                value: value_0
            },
            CallbackEvent::Delete {
                key: key_0,
                value: value_0
            },
        ],
        events,
    );
}

#[test]
fn test_insert_get_collision_pin_fail() {
    let mut rng = rand::rng();

    let key = rng.random();
    let value_0 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 2;
    let weight_fn = Arc::new(Weighter::new(hash_map! {
        value_0 => 2,
        value_1 => 1,
    }));
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key, value_0) else {
        unreachable!()
    };
    if let PinCacheResult::Pin { old_entry, new_value } = cache.insert(key, value_1) {
        assert_eq!(key, *old_entry.key());
        assert_eq!(value_0, *old_entry.value());
        assert_eq!(value_1, new_value)
    } else {
        unreachable!()
    };
    assert_eq!(key, *pin_guard_0.key());
    assert_eq!(value_0, *pin_guard_0.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(2, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    assert_cache_contains(&cache, key, value_0, Some(&pin_guard_0));
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(2, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    assert_cache_contains(&cache, key, value_0, None);
    assert_eq!(2, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert { key, value: value_0 },
            CallbackEvent::Get { key, value: value_0 },
            CallbackEvent::Get { key, value: value_0 },
            CallbackEvent::Delete { key, value: value_0 },
        ],
        events,
    );
}

#[test]
fn test_insert_get_collision_unpin_success() {
    let mut rng = rand::rng();

    let key = rng.random();
    let value_0 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 2;
    let weight_fn = Arc::new(Weighter::new(hash_map! {
        value_0 => 1,
        value_1 => 2,
    }));
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key, value_0) else {
        unreachable!()
    };
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    let PinCacheResult::Ok { entry: pin_guard_1 } = cache.insert(key, value_1) else {
        unreachable!()
    };
    assert_eq!(key, *pin_guard_1.key());
    assert_eq!(value_1, *pin_guard_1.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(2, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    assert_cache_contains(&cache, key, value_1, Some(&pin_guard_1));
    drop(pin_guard_1);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(2, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    assert_cache_contains(&cache, key, value_1, None);
    assert_eq!(2, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert { key, value: value_0 },
            CallbackEvent::Delete { key, value: value_0 },
            CallbackEvent::Insert { key, value: value_1 },
            CallbackEvent::Get { key, value: value_1 },
            CallbackEvent::Get { key, value: value_1 },
            CallbackEvent::Delete { key, value: value_1 },
        ],
        events,
    );
}

#[test]
fn test_insert_get_collision_unpin_evict_success() {
    let mut rng = rand::rng();

    let key_0 = rng.random();
    let value_00 = rng.random();
    let value_01 = rng.random();
    let key_1 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 2;
    let weight_fn = Arc::new(Weighter::new(hash_map! {
        value_00 => 1,
        value_01 => 2,
        value_1 => 1,
    }));
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key_0, value_00) else {
        unreachable!()
    };
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    let PinCacheResult::Ok { entry: pin_guard_1 } = cache.insert(key_1, value_1) else {
        unreachable!()
    };
    drop(pin_guard_1);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(2, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key_0, value_01) else {
        unreachable!()
    };
    assert_cache_miss(&cache, key_1);
    assert_eq!(key_0, *pin_guard_0.key());
    assert_eq!(value_01, *pin_guard_0.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(2, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(2, cache.total_weight());
    assert_cache_contains(&cache, key_0, value_01, Some(&pin_guard_0));
    drop(pin_guard_0);
    assert_cache_contains(&cache, key_0, value_01, None);
    assert_eq!(2, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert {
                key: key_0,
                value: value_00,
            },
            CallbackEvent::Insert {
                key: key_1,
                value: value_1,
            },
            CallbackEvent::Delete {
                key: key_0,
                value: value_00,
            },
            CallbackEvent::Delete {
                key: key_1,
                value: value_1,
            },
            CallbackEvent::Insert {
                key: key_0,
                value: value_01,
            },
            CallbackEvent::Get {
                key: key_0,
                value: value_01,
            },
            CallbackEvent::Get {
                key: key_0,
                value: value_01,
            },
            CallbackEvent::Delete {
                key: key_0,
                value: value_01,
            },
        ],
        events,
    );
}

#[test]
fn test_insert_get_collision_unpin_evict_fail() {
    let mut rng = rand::rng();

    let key_0 = rng.random();
    let value_00 = rng.random();
    let value_01 = rng.random();
    let key_1 = rng.random();
    let value_1 = rng.random();

    const SHARDS: usize = 1;
    let capacity = 2;
    let weight_fn = Arc::new(Weighter::new(hash_map! {
        value_00 => 1,
        value_01 => 2,
        value_1 => 1,
    }));
    let callback_fn = Arc::new(CallbackEvents::new());

    let cache = PinCacheImpl::<SHARDS, i64, u64, _, _>::new(capacity, weight_fn, callback_fn.clone());
    let PinCacheResult::Ok { entry: pin_guard_0 } = cache.insert(key_0, value_00) else {
        unreachable!()
    };
    drop(pin_guard_0);
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(1, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    let PinCacheResult::Ok { entry: pin_guard_1 } = cache.insert(key_1, value_1) else {
        unreachable!()
    };
    if let PinCacheResult::Full { value } = cache.insert(key_0, value_01) {
        assert_eq!(value_01, value)
    } else {
        unreachable!()
    };
    assert_cache_miss(&cache, key_0);
    assert_eq!(key_1, *pin_guard_1.key());
    assert_eq!(value_1, *pin_guard_1.value());
    assert_eq!(0, cache.evict_all());
    assert_eq!(1, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(1, cache.total_weight());
    assert_cache_contains(&cache, key_1, value_1, Some(&pin_guard_1));
    drop(pin_guard_1);
    assert_cache_contains(&cache, key_1, value_1, None);
    assert_eq!(1, cache.evict_all());
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());

    drop(cache);
    let callback_fn = Arc::try_unwrap(callback_fn).unwrap();
    let events = callback_fn.into_inner();
    assert_eq!(
        vec![
            CallbackEvent::Insert {
                key: key_0,
                value: value_00,
            },
            CallbackEvent::Insert {
                key: key_1,
                value: value_1,
            },
            CallbackEvent::Delete {
                key: key_0,
                value: value_00,
            },
            CallbackEvent::Get {
                key: key_1,
                value: value_1,
            },
            CallbackEvent::Get {
                key: key_1,
                value: value_1,
            },
            CallbackEvent::Delete {
                key: key_1,
                value: value_1,
            },
        ],
        events,
    );
}

#[test]
fn test_concurrency() {
    const SHARDS: usize = 16;
    let capacity = 4096;
    let weight_fn = Arc::new(UnitWeight);
    let callback_fn = Arc::new(NoopCallback);

    let cache = Arc::new(PinCacheImpl::<SHARDS, i64, u64, _, _>::new(
        capacity,
        weight_fn,
        callback_fn.clone(),
    ));

    let threads = 128;
    let iterations = 4096;
    let (start_tx, start_rx) = bounded::<()>(threads);
    let (ready_tx, ready_rx) = bounded::<()>(1);

    let mut join_handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let start_tx = start_tx.clone();
        let ready_rx = ready_rx.clone();
        let cache = cache.clone();
        let join_handle = std::thread::spawn(move || {
            let mut rng = rand::rng();
            start_tx.send(()).unwrap();
            let _ = ready_rx.recv();

            let mut pin_guards = VecDeque::with_capacity(iterations);
            for _ in 0..iterations {
                let key = rng.random();
                let value = rng.random();

                match cache.insert(key, value) {
                    PinCacheResult::Ok { entry: pin_guard } => {
                        pin_guards.push_back(pin_guard);
                    },
                    _ => {
                        let size = pin_guards.len();
                        if size > 0 {
                            let size = rng.random_range(0..size);
                            for pin_guard in pin_guards.drain(0..size) {
                                assert_cache_contains(&cache, *pin_guard.key(), *pin_guard.value(), Some(&pin_guard));
                            }
                        }
                    },
                }
            }
            let size = pin_guards.len();
            if size > 0 {
                let size = rng.random_range(0..size);
                for pin_guard in pin_guards.drain(0..size) {
                    assert_cache_contains(&cache, *pin_guard.key(), *pin_guard.value(), Some(&pin_guard));
                }
            }
        });
        join_handles.push(join_handle);
    }

    for _ in 0..threads {
        start_rx.recv().unwrap();
    }
    drop(ready_tx);
    for join_handle in join_handles {
        join_handle.join().unwrap();
    }

    cache.evict_all();
    assert_eq!(0, cache.pinned_weight());
    assert_eq!(0, cache.unpinned_weight());
    assert_eq!(0, cache.total_weight());
}

#[test]
fn weight_overflow_is_reported_as_full() {
    let weight = Arc::new(|_key: &i64, _value: &u64| usize::MAX);
    let cache = PinCacheImpl::<1, i64, u64, _, _>::new(usize::MAX, weight, Arc::new(NoopCallback));
    let PinCacheResult::Ok { entry } = cache.insert(1, 1) else {
        panic!("first maximum-weight entry should fit");
    };

    let result = cache.insert(2, 2);

    assert!(matches!(result, PinCacheResult::Full { value: 2 }));
    assert_eq!(usize::MAX, cache.total_weight());
    drop(entry);
}

#[test]
fn entry_weight_is_frozen_at_insertion() {
    let current_weight = Arc::new(AtomicUsize::new(1));
    let weight = {
        let current_weight = current_weight.clone();
        Arc::new(move |_key: &i64, _value: &u64| current_weight.load(Ordering::Relaxed))
    };
    let cache = PinCacheImpl::<1, i64, u64, _, _>::new(2, weight, Arc::new(NoopCallback));
    let PinCacheResult::Ok { entry } = cache.insert(1, 1) else {
        panic!("entry should fit");
    };
    drop(entry);

    current_weight.store(2, Ordering::Relaxed);

    assert_eq!(1, cache.evict_one());
    assert_eq!(0, cache.total_weight());
}

fn assert_cache_contains<const S: usize, W, C, H>(
    pin_cache: &PinCacheImpl<S, i64, u64, W, C, H>,
    key: i64,
    value: u64,
    expected_pin_guard: Option<&PinGuard<i64, u64, W, H>>,
) where
    W: Weight<i64, u64>,
    C: Callback<i64, u64>,
    H: BuildHasher + Default + Send + Sync + 'static,
{
    match pin_cache.get(&key) {
        Some(actual_pin_guard) => {
            assert_eq!(key, *actual_pin_guard.key());
            assert_eq!(value, *actual_pin_guard.value());
            if let Some(expected_pin_guard) = expected_pin_guard {
                assert_eq!(expected_pin_guard.node_ptr(), actual_pin_guard.node_ptr());
            }
        },
        None => panic!("unable to find pin guard for key: {key}"),
    }
}

fn assert_cache_miss<const S: usize, W, C, H>(pin_cache: &PinCacheImpl<S, i64, u64, W, C, H>, key: i64)
where
    W: Weight<i64, u64>,
    C: Callback<i64, u64>,
    H: BuildHasher + Default + Send + Sync + 'static,
{
    match pin_cache.get(&key) {
        Some(actual_pin_guard) => panic!("unexpected pin guard for key: {key}"),
        None => { /* noop */ },
    }
}

#[derive(Debug)]
struct Weighter {
    map: HashMap<u64, usize>,
}

impl Weighter {
    fn new(map: HashMap<u64, usize>) -> Self {
        Self { map }
    }
}

impl Weight<i64, u64> for Weighter {
    fn weight(&self, key: &i64, value: &u64) -> usize {
        *self.map.get(value).expect("unknown weight")
    }
}

#[derive(Debug, Eq, PartialEq)]
enum CallbackEvent<K, V>
where
    K: Eq + Debug,
    V: Eq + Debug,
{
    Insert { key: K, value: V },
    Get { key: K, value: V },
    Delete { key: K, value: V },
}

#[derive(Debug)]
struct CallbackEvents<K, V>
where
    K: Eq + Clone + Debug + Send + Sync + 'static,
    V: Eq + Clone + Debug + Send + Sync + 'static,
{
    events: Mutex<Vec<CallbackEvent<K, V>>>,
}

impl<K, V> CallbackEvents<K, V>
where
    K: Eq + Clone + Debug + Send + Sync + 'static,
    V: Eq + Clone + Debug + Send + Sync + 'static,
{
    fn new() -> Self {
        Self {
            events: Mutex::new(vec![]),
        }
    }

    fn into_inner(self) -> Vec<CallbackEvent<K, V>> {
        self.events.into_inner().unwrap()
    }
}

impl<K, V> Callback<K, V> for CallbackEvents<K, V>
where
    K: Eq + Clone + Debug + Send + Sync + 'static,
    V: Eq + Clone + Debug + Send + Sync + 'static,
{
    fn on_insert(&self, key: &K, value: &V) {
        self.events.lock().unwrap().push(CallbackEvent::Insert {
            key: key.clone(),
            value: value.clone(),
        });
    }

    fn on_get(&self, key: &K, value: &V) {
        self.events.lock().unwrap().push(CallbackEvent::Get {
            key: key.clone(),
            value: value.clone(),
        });
    }

    fn on_delete(&self, key: K, value: V) {
        self.events.lock().unwrap().push(CallbackEvent::Delete { key, value });
    }
}
