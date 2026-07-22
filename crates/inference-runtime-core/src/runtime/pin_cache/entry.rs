use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

#[derive(Debug)]
pub struct Entry<K, V> {
    pin_count: AtomicU64,
    weight: usize,

    key: K,
    value: V,
}

impl<K, V> Entry<K, V> {
    #[inline]
    pub fn new(pin_count: u64, weight: usize, key: K, value: V) -> Self {
        Self {
            pin_count: AtomicU64::new(pin_count),
            weight,

            key,
            value,
        }
    }

    #[inline]
    pub fn pin(&self) -> u64 {
        self.pin_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    #[inline]
    pub fn unpin(&self) -> u64 {
        self.pin_count.fetch_sub(1, Ordering::SeqCst) - 1
    }

    #[inline]
    pub fn pin_count(&self) -> u64 {
        self.pin_count.load(Ordering::SeqCst)
    }

    #[inline]
    pub fn weight(&self) -> usize {
        self.weight
    }

    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    #[inline]
    pub fn into_key_value(self) -> (K, V) {
        debug_assert_eq!(0, self.pin_count());
        (self.key, self.value)
    }
}
