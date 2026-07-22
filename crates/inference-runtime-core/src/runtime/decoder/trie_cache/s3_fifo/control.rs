use std::hash::Hash;

use crossbeam_channel::Sender;

pub enum Control<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    Reevaluate { key: K },
    EvictCandidate { reply: Sender<Option<K>> },
    RejectCandidate { key: K, reply: Sender<()> },
    AcceptCandidate { key: K, reply: Sender<()> },
    Shutdown,
}
