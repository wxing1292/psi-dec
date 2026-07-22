use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use ahash::RandomState as AHashRandomState;
use dashmap::DashMap;
use rand::rngs::SmallRng;

use crate::runtime::decoder::trie_cache::S3FIFOClient;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::TrieNodeStore;

mod edge;
pub use edge::TrieEdge;

mod by_key;
mod by_parent;
mod pin;
mod alloc_free;

#[derive(Debug)]
pub enum InsertNodeResult {
    Success { trie_node_key: TrieNodeKey },
    Collision { trie_node_key: TrieNodeKey },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryEvictByKeyResult {
    Success,
    Missing,
    Rejected,
}

// TODO use a non random partition picker
thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(rand::make_rng());
}

#[derive(Debug)]
pub struct Trie<const P: usize> {
    trie_roots: DashMap<TrieEdge, TrieNodeKey, AHashRandomState>,
    trie_nodes: TrieNodeStore<P>,
    s3_fifo: Arc<S3FIFOClient<TrieNodeKey>>,
    num_pinned_trie_nodes: AtomicUsize,
}

impl<const P: usize> Trie<P> {
    pub fn new(s3_fifo: Arc<S3FIFOClient<TrieNodeKey>>) -> Self {
        Self {
            trie_roots: DashMap::with_hasher(AHashRandomState::new()),
            trie_nodes: TrieNodeStore::new(),
            s3_fifo,
            num_pinned_trie_nodes: AtomicUsize::new(0),
        }
    }

    pub fn s3_fifo(&self) -> Arc<S3FIFOClient<TrieNodeKey>> {
        self.s3_fifo.clone()
    }

    fn register_eviction_by_key(&self, trie_node_key: TrieNodeKey) {
        let eviction = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("register_eviction_by_key: trie node must exist")
            .register_eviction();
        if let Some(eviction) = eviction {
            self.s3_fifo.insert(eviction);
        }
    }

    pub fn node_count(&self) -> usize {
        self.trie_nodes.len()
    }

    pub fn num_pinned_trie_node(&self) -> usize {
        self.num_pinned_trie_nodes.load(Ordering::SeqCst)
    }

    pub fn num_total_trie_node(&self) -> usize {
        self.trie_nodes.len()
    }
}

#[path = "./alloc_free_test.rs"]
#[cfg(test)]
mod alloc_free_test;

#[path = "./by_key_test.rs"]
#[cfg(test)]
mod by_key_test;

#[path = "./by_parent_test.rs"]
#[cfg(test)]
mod by_parent_test;

#[path = "./pin_test.rs"]
#[cfg(test)]
mod pin_test;

#[path = "./test_utils.rs"]
#[cfg(test)]
mod test_utils;
