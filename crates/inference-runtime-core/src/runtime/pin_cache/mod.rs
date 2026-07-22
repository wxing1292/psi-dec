use std::borrow::Borrow;
use std::hash::BuildHasher;
use std::hash::Hash;

mod entry;
use entry::Entry;

mod list;
use list::LinkedList;
use list::LinkedNode;

mod guard;
pub use guard::PinGuard;

mod shard;
mod cache;
pub use cache::PinCacheImpl;

pub trait Key: Eq + Clone + Hash + Send + Sync + 'static {}
impl<T> Key for T where T: Eq + Clone + Hash + Send + Sync + 'static {}

pub trait Value: Send + Sync + 'static {}
impl<T> Value for T where T: Send + Sync + 'static {}

pub trait Weight<K, V>: Send + Sync + 'static {
    fn weight(&self, key: &K, value: &V) -> usize;
}

impl<K, V, F> Weight<K, V> for F
where
    F: Fn(&K, &V) -> usize + Send + Sync + 'static,
{
    #[inline]
    fn weight(&self, k: &K, v: &V) -> usize {
        self(k, v)
    }
}

pub trait Callback<K, V>: Send + Sync + 'static {
    fn on_insert(&self, key: &K, value: &V);
    fn on_get(&self, key: &K, value: &V);
    fn on_delete(&self, key: K, value: V);
}

#[derive(Clone, Debug)]
pub struct UnitWeight;
impl<K, V> Weight<K, V> for UnitWeight {
    #[inline]
    fn weight(&self, _: &K, _: &V) -> usize {
        1
    }
}

#[derive(Clone, Debug)]
pub struct NoopCallback;
impl<K, V> Callback<K, V> for NoopCallback {
    fn on_insert(&self, key: &K, value: &V) {}

    fn on_get(&self, key: &K, value: &V) {}

    fn on_delete(&self, key: K, value: V) {}
}

#[derive(Debug)]
pub enum PinCacheResult<E, V> {
    Ok { entry: E },
    Pin { old_entry: E, new_value: V },
    Full { value: V },
}

pub trait PinCache<K, V, W, C, H>: Send + Sync + 'static
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    C: Callback<K, V>,
    H: BuildHasher + Default,
{
    fn insert(&self, key: K, value: V) -> PinCacheResult<PinGuard<K, V, W, H>, V>;
    fn get<Q>(&self, query: &Q) -> Option<PinGuard<K, V, W, H>>
    where
        Q: ?Sized + Hash + Eq,
        K: Borrow<Q>;

    fn evict_one(&self) -> usize;
    fn evict_all(&self) -> usize;

    fn pinned_weight(&self) -> usize;
    fn unpinned_weight(&self) -> usize;
    fn total_weight(&self) -> usize;
    fn total_capacity(&self) -> usize;
}
