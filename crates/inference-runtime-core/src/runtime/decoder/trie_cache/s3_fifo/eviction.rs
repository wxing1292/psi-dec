use std::hash::Hash;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use crossbeam_channel::Sender;

use super::control::Control;
use crate::channel::Shutdown;

const MAX_COUNT: u8 = 3;

#[derive(Debug)]
pub struct Eviction<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    key: K,
    pinned: AtomicBool,
    count: AtomicU8,
    signal_sender: Sender<Control<K>>,
    shutdown: Shutdown,
}

impl<K> Eviction<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn new(key: K, signal_sender: Sender<Control<K>>, shutdown: Shutdown) -> Self {
        Self {
            key,
            pinned: AtomicBool::new(true),
            count: AtomicU8::new(1),
            signal_sender,
            shutdown,
        }
    }

    pub fn key(&self) -> K {
        self.key
    }

    pub fn pin(&self) {
        debug_assert!(
            !self.pinned.swap(true, Ordering::AcqRel),
            "pin: entry must not already be pinned"
        );
        self.signal();
    }

    pub fn unpin(&self) {
        debug_assert!(self.pinned.swap(false, Ordering::AcqRel), "unpin: entry must be pinned");
        self.signal();
    }

    pub fn touch(&self) {
        self.inc_count();
    }

    pub fn untouch(&self) {
        debug_assert!(self.dec_count(), "untouch: count must be greater than one");
    }

    pub fn inc_count(&self) {
        loop {
            let count = self.count.load(Ordering::Relaxed);
            if count == MAX_COUNT {
                return;
            }
            if self
                .count
                .compare_exchange_weak(count, count + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    pub fn dec_count(&self) -> bool {
        loop {
            let count = self.count.load(Ordering::Relaxed);
            if count <= 1 {
                return false;
            }
            if self
                .count
                .compare_exchange_weak(count, count - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Acquire)
    }

    pub fn reset_count(&self) {
        self.count.store(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u8 {
        self.count.load(Ordering::Relaxed)
    }

    fn signal(&self) {
        if self.shutdown.is_shutdown() {
            return;
        }
        self.signal_sender
            .send(Control::Reevaluate { key: self.key })
            .expect("signal: S3FIFO worker must be running");
    }
}
