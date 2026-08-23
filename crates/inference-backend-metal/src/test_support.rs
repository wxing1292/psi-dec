use std::collections::HashMap;
use std::hash::Hash;

use crate::metal::ReplayProgram;

pub struct ReplayTestCache<K> {
    entries: HashMap<K, ReplayProgram>,
}

impl<K> ReplayTestCache<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn record(&mut self, key: K, record: impl FnOnce() -> ReplayProgram) -> (&ReplayProgram, bool) {
        let cache_hit = self.entries.contains_key(&key);
        if !cache_hit {
            assert!(self.entries.insert(key.clone(), record()).is_none());
        }
        (self.entries.get(&key).unwrap(), cache_hit)
    }
}
