//! # S3FIFO contract
//!
//! This implementation has an S queue and an M queue. It intentionally has no
//! ghost queue yet.
//!
//! ```text
//! Trie / cache threads                         S3FIFO worker thread
//! ────────────────────                         ────────────────────
//! DashMap<key, Arc<Eviction>>                   Entry
//!   inserted while pinned                         eviction: Arc<Eviction>
//! Arc<Eviction>                                  queue_link
//!   pinned: AtomicBool                            queue: S | M
//!   count: AtomicU8                               is_candidate: bool
//!   signal sender
//!
//! pin / unpin / touch / untouch                owns private entries and S/M queues
//! only update atomics or enqueue Reevaluate.    mutates queue links and policy state.
//! ```
//!
//! `queue_link.is_linked()` is the physical fact: attached entries are in an
//! intrusive S or M queue; detached entries are not. `queue` remembers the
//! queue to use when an eligible detached entry is attached again.
//!
//! ```text
//! client API
//! ─────────────────────────────────────────────────────────────────
//! S3FIFOClient::new(capacity, shutdown)
//!   Runtime init creates one client per cache lane. Its worker constructs the
//!   S3FIFOServer, which owns and monitors the shutdown receiver.
//!
//! new_eviction(key) -> Arc<Eviction>
//!   Creates the caller-owned handle with pinned = true and count = 1.
//!
//! insert(eviction)
//!   Inserts a brand-new pinned handle with count = 1 directly into the shared
//!   registry. It does not wait for or notify the worker.
//!
//! Eviction::pin / unpin
//!   Change the eligibility flag and signal the worker to re-evaluate.
//!   They are edge notifications, not a reference count.
//!
//! Eviction::touch / untouch
//!   Change the saturating S3FIFO policy count only. They do not move links.
//!
//! evict_candidate() -> Option<key>
//!   Detaches and reserves one policy candidate for Trie-level eviction.
//!
//! reject_candidate(key)
//!   Trie declined the proposed eviction for any reason. Clear the reservation,
//!   then re-evaluate current pin state: unpinned reattaches; pinned remains
//!   detached.
//!
//! accept_candidate(key)
//!   Trie successfully evicted the node. The entry must still be reserved and
//!   must not be pinned; the worker permanently removes it.
//! ```
//!
//! ```text
//! lifecycle
//! ─────────────────────────────────────────────────────────────────
//! insert
//!   pinned = true, count = 1, no worker Entry yet
//!
//! unpin -> Reevaluate
//!   materialize the worker Entry, then attach it to remembered S/M queue
//!   when unpinned
//!
//! pin -> Reevaluate
//!   pinned && attached                => detach from S/M queue
//!
//! eviction scan
//!   S, count == 1  => detach and reserve candidate
//!   S, count > 1   => queue = M, count = 1, attach only if unpinned
//!   M, count == 1  => detach and reserve candidate
//!   M, count > 1   => count -= 1, attach only if unpinned
//!
//! candidate resolution
//!   accept_candidate => permanently remove Entry
//!   reject_candidate => clear reservation, then attach only if unpinned
//!
//! runtime shutdown
//!   server receives shutdown => stop event loop
//!   later Eviction pin/unpin signals => no-op during teardown
//! ```

mod client;
mod control;
mod entry;
mod eviction;
mod queue;
mod server;

pub use client::S3FIFOClient;
pub use eviction::Eviction;

#[cfg(test)]
mod client_test;

#[cfg(test)]
mod server_test;
