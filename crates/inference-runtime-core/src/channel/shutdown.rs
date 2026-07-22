use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use async_channel::Receiver as AsyncReceiver;
use async_channel::Sender as AsyncSender;
use async_channel::bounded as async_bounded;
use crossbeam_channel::Receiver as SyncReceiver;
use crossbeam_channel::Sender as SyncSender;
use crossbeam_channel::bounded as sync_bounded;

#[derive(Clone, Debug)]
pub struct Shutdown {
    inner: Arc<ShutdownInner>,
}

#[derive(Debug)]
pub struct ShutdownGuard {
    shutdown: Shutdown,
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ShutdownInner::new()),
        }
    }

    delegate::delegate! {
        to self.inner {
            pub fn shutdown(&self);
            pub fn is_shutdown(&self) -> bool;
            pub fn sync_rx(&self) -> &SyncReceiver<()>;
            pub fn async_rx(&self) -> &AsyncReceiver<()>;
        }
    }
}

impl ShutdownGuard {
    pub fn new(shutdown: Shutdown) -> Self {
        Self { shutdown }
    }
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        self.shutdown.shutdown();
    }
}

#[derive(Debug)]
struct ShutdownInner {
    flag: AtomicBool,

    sync_tx: Mutex<Option<SyncSender<()>>>,
    async_tx: Mutex<Option<AsyncSender<()>>>,
    sync_rx: SyncReceiver<()>,
    async_rx: AsyncReceiver<()>,
}

impl ShutdownInner {
    fn new() -> Self {
        let (sync_tx, sync_rx) = sync_bounded(1);
        let (async_tx, async_rx) = async_bounded(1);
        Self {
            flag: AtomicBool::new(false),

            sync_tx: Mutex::new(Some(sync_tx)),
            async_tx: Mutex::new(Some(async_tx)),
            sync_rx,
            async_rx,
        }
    }

    fn shutdown(&self) {
        if self
            .flag
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let _ = self.sync_tx.lock().unwrap().take();
        let _ = self.async_tx.lock().unwrap().take();
    }

    fn is_shutdown(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    fn sync_rx(&self) -> &SyncReceiver<()> {
        &self.sync_rx
    }

    fn async_rx(&self) -> &AsyncReceiver<()> {
        &self.async_rx
    }
}

#[cfg(test)]
mod tests {
    use futures_lite::future::block_on;

    use super::*;

    #[test]
    fn test_shutdown_atomic() {
        let shutdown = Shutdown::new();
        assert!(!shutdown.is_shutdown());

        shutdown.shutdown();
        assert!(shutdown.is_shutdown());
    }

    #[test]
    fn shutdown_guard_shutdowns_on_drop() {
        let shutdown = Shutdown::new();
        {
            let _guard = ShutdownGuard::new(shutdown.clone());
            assert!(!shutdown.is_shutdown());
        }
        assert!(shutdown.is_shutdown());
    }

    #[test]
    fn shutdown_guard_shutdowns_during_unwind() {
        let shutdown = Shutdown::new();
        let result = std::panic::catch_unwind({
            let shutdown = shutdown.clone();
            move || {
                let _guard = ShutdownGuard::new(shutdown);
                panic!("test panic");
            }
        });
        assert!(result.is_err());
        assert!(shutdown.is_shutdown());
    }

    #[test]
    fn test_sync_rx_unblocks_after_shutdown() {
        let shutdown = Shutdown::new();
        let shutdown_rx = shutdown.sync_rx().clone();
        match shutdown_rx.try_recv() {
            Ok(()) => panic!("expect empty err"),
            Err(crossbeam_channel::TryRecvError::Empty) => { /* noop */ },
            Err(crossbeam_channel::TryRecvError::Disconnected) => panic!("expect empty err"),
        }

        shutdown.shutdown();
        match shutdown_rx.try_recv() {
            Ok(()) => panic!("expect disconnected err"),
            Err(crossbeam_channel::TryRecvError::Empty) => panic!("expect disconnected err"),
            Err(crossbeam_channel::TryRecvError::Disconnected) => { /* noop */ },
        }
        match shutdown_rx.recv() {
            Ok(()) => panic!("expect err"),
            Err(crossbeam_channel::RecvError) => { /* noop */ },
        }
    }

    #[test]
    fn test_async_rx_unblocks_after_shutdown() {
        let shutdown = Shutdown::new();
        let shutdown_rx = shutdown.async_rx().clone();
        match shutdown_rx.try_recv() {
            Ok(()) => panic!("expect empty err"),
            Err(async_channel::TryRecvError::Empty) => { /* noop */ },
            Err(async_channel::TryRecvError::Closed) => panic!("expect empty err"),
        }

        shutdown.shutdown();
        match shutdown_rx.try_recv() {
            Ok(()) => panic!("expect closed err"),
            Err(async_channel::TryRecvError::Empty) => panic!("expect closed err"),
            Err(async_channel::TryRecvError::Closed) => { /* noop */ },
        }
        block_on(async {
            match shutdown_rx.recv().await {
                Ok(()) => panic!("expect err"),
                Err(async_channel::RecvError) => { /* noop */ },
            }
        });
    }
}
