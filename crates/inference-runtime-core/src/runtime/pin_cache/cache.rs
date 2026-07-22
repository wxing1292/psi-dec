#![allow(clippy::type_complexity)]

use std::borrow::Borrow;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use ahash::RandomState;
use crossbeam_utils::CachePadded;

use crate::runtime::pin_cache::Callback;
use crate::runtime::pin_cache::Key;
use crate::runtime::pin_cache::PinCache;
use crate::runtime::pin_cache::PinCacheResult;
use crate::runtime::pin_cache::Value;
use crate::runtime::pin_cache::Weight;
use crate::runtime::pin_cache::guard::PinGuard;
use crate::runtime::pin_cache::shard::Shard;

#[derive(Debug)]
pub struct PinCacheImpl<const S: usize, K, V, W, C, H = RandomState>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    C: Callback<K, V>,
    H: BuildHasher + Default + Send + Sync + 'static,
{
    weight: Arc<W>,
    callback: Arc<C>,
    total_capacity: usize,
    total_weight: Arc<AtomicUsize>,
    hasher: H,
    shards: [CachePadded<Arc<Mutex<Shard<K, V, W, H>>>>; S],
}

impl<const S: usize, K, V, W, C, H> PinCacheImpl<S, K, V, W, C, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    C: Callback<K, V>,
    H: BuildHasher + Default + Send + Sync + 'static,
{
    pub fn new(total_capacity: usize, weight: Arc<W>, callback: Arc<C>) -> Self {
        assert!(S.is_power_of_two(), "N must be a power of two");
        let total_weight = Arc::new(AtomicUsize::new(0));
        let shards = std::array::from_fn(|_| {
            let shard = Shard::new(total_capacity, total_weight.clone(), weight.clone());
            CachePadded::new(Arc::new(Mutex::new(shard)))
        });
        Self {
            weight,
            callback,
            total_capacity,
            total_weight,
            hasher: H::default(),
            shards,
        }
    }

    #[inline]
    fn shard_index<Q>(&self, query: &Q) -> usize
    where
        Q: ?Sized + Hash + Eq,
        K: Borrow<Q>,
    {
        (self.hasher.hash_one(query) as usize) & (S - 1)
    }
}

impl<const S: usize, K, V, W, C, H> PinCache<K, V, W, C, H> for PinCacheImpl<S, K, V, W, C, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    C: Callback<K, V>,
    H: BuildHasher + Default + Send + Sync + 'static,
{
    fn insert(&self, key: K, value: V) -> PinCacheResult<PinGuard<K, V, W, H>, V> {
        let index = self.shard_index(&key);
        let shard = self.shards[index].clone().into_inner();

        let mut guard = self.shards[index].lock().unwrap();
        let (result, evict_key_values) = guard.insert_pin(key, value);
        drop(guard);

        for (key, value) in evict_key_values {
            self.callback.on_delete(key, value);
        }
        match result {
            PinCacheResult::Ok { entry: node_nn } => {
                unsafe {
                    self.callback
                        .on_insert(node_nn.as_ref().key(), node_nn.as_ref().value());
                }
                let pin_guard = PinGuard::new(shard, node_nn);
                PinCacheResult::Ok { entry: pin_guard }
            },
            PinCacheResult::Pin { old_entry, new_value } => {
                PinCacheResult::Pin {
                    old_entry: PinGuard::new(shard, old_entry),
                    new_value,
                }
            },
            PinCacheResult::Full { value } => PinCacheResult::Full { value },
        }
    }

    fn get<Q>(&self, query: &Q) -> Option<PinGuard<K, V, W, H>>
    where
        Q: ?Sized + Hash + Eq,
        K: Borrow<Q>,
    {
        let index = self.shard_index(query);
        let shard = self.shards[index].clone().into_inner();

        let mut guard = self.shards[index].lock().unwrap();
        let node_nn = guard.get_pin(query)?;
        drop(guard);

        unsafe {
            self.callback.on_get(node_nn.as_ref().key(), node_nn.as_ref().value());
        }
        let pin_guard = PinGuard::new(shard, node_nn);
        Some(pin_guard)
    }

    fn evict_one(&self) -> usize {
        let mut evict_weight = 0;
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap();
            let (weight, evict_key_values) = guard.evict_one();
            drop(guard);

            evict_weight += weight;
            for (key, value) in evict_key_values {
                self.callback.on_delete(key, value);
            }
        }
        evict_weight
    }

    fn evict_all(&self) -> usize {
        let mut evict_weight = 0;
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap();
            let (weight, evict_key_values) = guard.evict_all();
            drop(guard);

            evict_weight += weight;
            for (key, value) in evict_key_values {
                self.callback.on_delete(key, value);
            }
        }
        evict_weight
    }

    fn pinned_weight(&self) -> usize {
        self.total_weight
            .load(Ordering::SeqCst)
            .saturating_sub(self.unpinned_weight())
    }

    fn unpinned_weight(&self) -> usize {
        let mut unpinned_weight = 0;
        for shard in &self.shards {
            let guard = shard.lock().unwrap();
            unpinned_weight += guard.unpinned_weight();
            drop(guard);
        }
        unpinned_weight
    }

    fn total_weight(&self) -> usize {
        self.total_weight.load(Ordering::SeqCst)
    }

    fn total_capacity(&self) -> usize {
        self.total_capacity
    }
}

#[cfg(test)]
#[path = "./cache_test.rs"]
mod cache_test;
