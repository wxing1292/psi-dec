use std::cell::Cell;
use std::hash::Hash;
use std::sync::Arc;

use intrusive_collections::LinkedListLink;
use intrusive_collections::intrusive_adapter;

use super::eviction::Eviction;
use super::queue::Queue;

pub struct Entry<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub eviction: Arc<Eviction<K>>,
    pub queue_link: LinkedListLink,
    pub queue: Cell<Queue>,
    pub is_candidate: Cell<bool>,
}

intrusive_adapter!(pub EntryLink<K> = Arc<Entry<K>>: Entry<K> { queue_link => LinkedListLink } where K: Copy + Eq + Hash + Send + Sync + 'static);
