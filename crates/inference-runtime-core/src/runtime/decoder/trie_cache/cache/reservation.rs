use event_listener::Event;
use event_listener::EventListener;

use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReservationKey {
    parent_trie_node_key: Option<TrieNodeKey>,
    trie_edge: TrieEdge,
}

impl ReservationKey {
    pub fn new(parent_trie_node_key: Option<TrieNodeKey>, trie_edge: TrieEdge) -> Self {
        Self {
            parent_trie_node_key,
            trie_edge,
        }
    }
}

pub struct Reservation {
    event: Event,
}

impl Reservation {
    pub fn new() -> Self {
        Self { event: Event::new() }
    }

    pub fn listen(&self) -> EventListener {
        self.event.listen()
    }

    pub fn notify(&self) {
        self.event.notify(usize::MAX);
    }
}
