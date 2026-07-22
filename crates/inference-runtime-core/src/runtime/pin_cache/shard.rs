use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::runtime::pin_cache::Entry;
use crate::runtime::pin_cache::Key;
use crate::runtime::pin_cache::LinkedList;
use crate::runtime::pin_cache::LinkedNode;
use crate::runtime::pin_cache::PinCacheResult;
use crate::runtime::pin_cache::Value;
use crate::runtime::pin_cache::Weight;

type NodeNN<K, V> = NonNull<LinkedNode<Entry<K, V>>>;

#[derive(Debug)]
pub struct Shard<K, V, W, H = ahash::RandomState>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    weight: Arc<W>,
    total_capacity: usize,
    total_weight: Arc<AtomicUsize>,

    map: HashMap<K, NodeNN<K, V>, H>,
    unpinned_list: LinkedList<Entry<K, V>>,
    unpinned_weight: usize,
}

impl<K, V, W, H> Shard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    pub fn new(total_capacity: usize, total_weight: Arc<AtomicUsize>, weight: Arc<W>) -> Self {
        Self {
            weight,
            total_capacity,
            total_weight,

            map: HashMap::with_hasher(H::default()),
            unpinned_list: LinkedList::new(),
            unpinned_weight: 0,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert_pin(&mut self, key: K, value: V) -> (PinCacheResult<NodeNN<K, V>, V>, Vec<(K, V)>) {
        let mut evict_key_values = vec![];
        let weight = self.weight.weight(&key, &value);

        if let Some(old_node_nn) = self.map.remove(&key) {
            let old_node_ref = unsafe { old_node_nn.as_ref() };
            if old_node_ref.pin_count() > 0 {
                let collision = self.map.insert(key, old_node_nn);
                debug_assert!(collision.is_none());
                old_node_ref.pin();
                return (
                    PinCacheResult::Pin {
                        old_entry: old_node_nn,
                        new_value: value,
                    },
                    evict_key_values,
                );
            } else {
                let old_weight = old_node_ref.weight();
                self.total_weight.fetch_sub(old_weight, Ordering::SeqCst);
                if old_node_ref.is_attached() {
                    unsafe {
                        self.unpinned_list.detach(old_node_nn);
                        self.unpinned_weight -= old_weight;
                    }
                }
                unsafe {
                    let old_entry = Box::from_raw(old_node_nn.as_ptr()).into_inner();
                    let (old_key, old_value) = old_entry.into_key_value();
                    evict_key_values.push((old_key, old_value));
                }
            }
        }

        loop {
            match self
                .total_weight
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |weight_before| {
                    let weight_after = weight_before.checked_add(weight)?;
                    if weight_after <= self.total_capacity {
                        Some(weight_after)
                    } else {
                        None
                    }
                }) {
                Ok(_) => break,
                Err(_) => {
                    if let Some((evict_key, evict_value, evict_weight)) = self.do_evict() {
                        self.total_weight.fetch_sub(evict_weight, Ordering::SeqCst);
                        self.unpinned_weight -= evict_weight;
                        evict_key_values.push((evict_key, evict_value));
                        continue;
                    } else {
                        return (PinCacheResult::Full { value }, evict_key_values);
                    }
                },
            }
        }

        let new_entry = Entry::new(1, weight, key.clone(), value);
        let new_node_nn = unsafe { NonNull::new_unchecked(Box::into_raw(LinkedNode::new(new_entry))) };
        let collision = self.map.insert(key, new_node_nn);
        debug_assert!(collision.is_none());
        (PinCacheResult::Ok { entry: new_node_nn }, evict_key_values)
    }

    pub fn get_pin<Q>(&mut self, query: &Q) -> Option<NodeNN<K, V>>
    where
        Q: ?Sized + Hash + Eq,
        K: Borrow<Q>,
    {
        let node_nn = *self.map.get_mut(query)?;
        unsafe {
            let node_ref = node_nn.as_ref();
            if node_ref.pin_count() == 0 && node_ref.is_attached() {
                let weight = node_ref.weight();
                self.unpinned_list.detach(node_nn);
                self.unpinned_weight -= weight;
            }
            node_ref.pin();
            Some(node_nn)
        }
    }

    #[inline]
    pub fn detach_unpin(&mut self, node_nn: NodeNN<K, V>) {
        let weight;
        unsafe {
            let node_ref = node_nn.as_ref();
            if node_ref.pin_count() != 0 {
                return;
            }
            debug_assert_eq!(0, node_ref.pin_count());
            debug_assert_eq!(node_nn.as_ptr(), self.map.get(node_ref.key()).unwrap().as_ptr());
            debug_assert!(!node_ref.is_attached());
            weight = node_ref.weight();
        }
        self.unpinned_list.push_front(node_nn);
        self.unpinned_weight += weight;
    }

    #[inline]
    pub fn evict_one(&mut self) -> (usize, Vec<(K, V)>) {
        let Some((key, value, evict_weight)) = self.do_evict() else {
            return (0, vec![]);
        };
        // self.callback_fn.on_delete(key, value);
        self.total_weight.fetch_sub(evict_weight, Ordering::SeqCst);
        self.unpinned_weight -= evict_weight;
        (evict_weight, vec![(key, value)])
    }

    #[inline]
    pub fn evict_all(&mut self) -> (usize, Vec<(K, V)>) {
        let mut evict_weight = 0;
        let mut evict_key_values = vec![];
        while let Some((key, value, weight)) = self.do_evict() {
            evict_weight += weight;
            // self.callback_fn.on_delete(key, value);
            evict_key_values.push((key, value));
        }
        self.total_weight.fetch_sub(evict_weight, Ordering::SeqCst);
        self.unpinned_weight -= evict_weight;
        (evict_weight, evict_key_values)
    }

    #[inline]
    fn do_evict(&mut self) -> Option<(K, V, usize)> {
        let node_nn = self.unpinned_list.pop_back()?;
        unsafe {
            let node_ref = node_nn.as_ref();
            debug_assert_eq!(0, node_ref.pin_count());
            debug_assert_eq!(node_nn.as_ptr(), self.map.remove(node_ref.key()).unwrap().as_ptr());

            let entry = Box::from_raw(node_nn.as_ptr()).into_inner();
            let weight = entry.weight();
            let (key, value) = entry.into_key_value();
            Some((key, value, weight))
        }
    }

    #[inline]
    pub fn unpinned_weight(&self) -> usize {
        self.unpinned_weight
    }
}

impl<K, V, W, H> Drop for Shard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        {
            for node_nn in self.map.values() {
                unsafe {
                    let node_ref = node_nn.as_ref();
                    debug_assert_eq!(0, node_ref.pin_count(),);
                }
            }
        }

        self.unpinned_weight = 0;
        while let Some(node_nn) = self.unpinned_list.pop_front() {
            unsafe {
                let node_ref = node_nn.as_ref();
                debug_assert_eq!(0, node_ref.pin_count(),);
            }
        }
        for (_, node_nn) in self.map.drain() {
            unsafe {
                let node_ref = node_nn.as_ref();
                debug_assert!(!node_ref.is_attached());
                drop(Box::from_raw(node_nn.as_ptr()));
            }
        }
    }
}

unsafe impl<K, V, W, H> Send for Shard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
}

unsafe impl<K, V, W, H> Sync for Shard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
}
