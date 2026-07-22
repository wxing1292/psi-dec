use std::sync::Arc;

use crossbeam_channel::Receiver as SyncReceiver;
use crossbeam_channel::Sender as SyncSender;
use crossbeam_channel::TryRecvError as SyncTryRecvError;
use crossbeam_channel::bounded as sync_channel;
use crossbeam_utils::atomic::AtomicCell;
use tokio::sync::mpsc::Receiver as AsyncReceiver;
use tokio::sync::mpsc::Sender as AsyncSender;
use tokio::sync::mpsc::channel as async_channel;
use tokio::sync::mpsc::error::TryRecvError as AsyncTryRecvError;

#[derive(Clone, Debug)]
pub struct SignalSender<T>
where
    T: PartialEq + Eq + Copy + Send + Sync + 'static,
{
    value: Arc<AtomicCell<T>>,

    sync_tx: SyncSender<()>,
    async_tx: AsyncSender<()>,
}

#[derive(Debug)]
pub struct SignalReceiver<T>
where
    T: PartialEq + Eq + Copy + Send + Sync + 'static,
{
    value: Arc<AtomicCell<T>>,

    sync_tx: SyncSender<()>,
    async_tx: AsyncSender<()>,

    sync_rx: SyncReceiver<()>,
    async_rx: AsyncReceiver<()>,
}

pub fn signal_channel<T>(init_value: T) -> (SignalSender<T>, SignalReceiver<T>)
where
    T: PartialEq + Eq + Copy + Send + Sync + 'static,
{
    let init_value = Arc::new(AtomicCell::new(init_value));
    let (sync_tx, sync_rx) = sync_channel(1);
    let (async_tx, async_rx) = async_channel(1);

    (
        SignalSender {
            value: init_value.clone(),
            sync_tx: sync_tx.clone(),
            async_tx: async_tx.clone(),
        },
        SignalReceiver {
            value: init_value,
            sync_tx,
            async_tx,
            sync_rx,
            async_rx,
        },
    )
}

impl<T> SignalSender<T>
where
    T: PartialEq + Eq + Copy + Send + Sync + 'static,
{
    pub fn get(&self) -> T {
        self.value.load()
    }

    pub fn set(&self, new_value: T) {
        let old_value = self.value.swap(new_value);
        if old_value != new_value {
            self.notify();
        }
    }

    pub fn compare_exchange(&self, old_value: T, new_value: T) -> bool {
        match self.value.compare_exchange(old_value, new_value) {
            Ok(_) => {
                if old_value != new_value {
                    self.notify();
                }
                true
            },
            Err(_) => false,
        }
    }

    fn notify(&self) {
        let _ = self.sync_tx.try_send(());
        let _ = self.async_tx.try_send(());
    }
}

impl<T> SignalReceiver<T>
where
    T: PartialEq + Eq + Copy + Send + Sync + 'static,
{
    pub fn get(&self) -> T {
        self.value.load()
    }

    pub fn notify(&self) {
        let _ = self.sync_tx.try_send(());
        let _ = self.async_tx.try_send(());
    }

    pub fn sync_try_wait(&self) -> bool {
        match self.sync_rx.try_recv() {
            Ok(()) => true,
            Err(SyncTryRecvError::Empty) => false,
            Err(SyncTryRecvError::Disconnected) => unreachable!(),
        }
    }

    pub fn async_try_wait(&mut self) -> bool {
        match self.async_rx.try_recv() {
            Ok(()) => true,
            Err(AsyncTryRecvError::Empty) => false,
            Err(AsyncTryRecvError::Disconnected) => unreachable!(),
        }
    }

    pub fn sync_wait(&self) {
        let _ = self.sync_rx.recv();
    }

    pub async fn async_wait(&mut self) {
        let _ = self.async_rx.recv().await;
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::*;

    #[test]
    fn test_set_noop() {
        let mut rng = rand::rng();

        let value = rng.random::<u64>();
        let (signal_tx, mut signal_rx) = signal_channel(value);
        assert_wo_signal(&mut signal_rx);
        signal_tx.set(value);
        assert_wo_signal(&mut signal_rx);
        assert_eq!(value, signal_rx.get());
    }

    #[test]
    fn test_set_signal() {
        let mut rng = rand::rng();

        let value = rng.random::<u64>();
        let (signal_tx, mut signal_rx) = signal_channel(value);

        assert_wo_signal(&mut signal_rx);
        signal_tx.set(value + 1);
        assert_w_signal(&mut signal_rx);
        assert_eq!(value + 1, signal_rx.get());
    }

    #[test]
    fn test_set_signal_dedup() {
        let mut rng = rand::rng();

        let value = rng.random::<u64>();
        let (signal_tx, mut signal_rx) = signal_channel(value);

        assert_wo_signal(&mut signal_rx);
        signal_tx.set(value + 1);
        assert_w_signal(&mut signal_rx);
        signal_tx.set(value + 1);
        assert_wo_signal(&mut signal_rx);
        assert_eq!(value + 1, signal_rx.get());
    }

    #[test]
    fn test_compare_exchange_noop() {
        let mut rng = rand::rng();

        let value = rng.random::<u64>();
        let (signal_tx, mut signal_rx) = signal_channel(value);

        assert_wo_signal(&mut signal_rx);
        let success = signal_tx.compare_exchange(value - 1, value + 1);
        assert!(!success);
        assert_wo_signal(&mut signal_rx);
        assert_eq!(value, signal_rx.get());
    }

    #[test]
    fn test_compare_exchange_signal() {
        let mut rng = rand::rng();

        let value = rng.random::<u64>();
        let (signal_tx, mut signal_rx) = signal_channel(value);

        assert_wo_signal(&mut signal_rx);
        let success = signal_tx.compare_exchange(value, value + 1);
        assert!(success);
        assert_w_signal(&mut signal_rx);
        assert_eq!(value + 1, signal_rx.get());
    }

    pub fn assert_wo_signal<T>(signal_rx: &mut SignalReceiver<T>)
    where
        T: PartialEq + Eq + Copy + Send + Sync + 'static,
    {
        match signal_rx.sync_rx.try_recv() {
            Ok(()) => unreachable!(),
            Err(SyncTryRecvError::Empty) => {},
            Err(SyncTryRecvError::Disconnected) => unreachable!(),
        }
        match signal_rx.async_rx.try_recv() {
            Ok(()) => unreachable!(),
            Err(AsyncTryRecvError::Empty) => {},
            Err(AsyncTryRecvError::Disconnected) => unreachable!(),
        }
    }

    pub fn assert_w_signal<T>(signal_rx: &mut SignalReceiver<T>)
    where
        T: PartialEq + Eq + Copy + Send + Sync + 'static,
    {
        match signal_rx.sync_rx.try_recv() {
            Ok(()) => {},
            Err(SyncTryRecvError::Empty) => unreachable!(),
            Err(SyncTryRecvError::Disconnected) => unreachable!(),
        }
        match signal_rx.async_rx.try_recv() {
            Ok(()) => {},
            Err(AsyncTryRecvError::Empty) => unreachable!(),
            Err(AsyncTryRecvError::Disconnected) => unreachable!(),
        }
    }
}
