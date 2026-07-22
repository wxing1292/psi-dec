use std::panic::AssertUnwindSafe;

use async_channel::Receiver;
use async_channel::Sender;
use async_channel::TrySendError;
use async_trait::async_trait;
use futures_lite::future::Boxed;
use futures_util::FutureExt;

use crate::Result;
use crate::channel::Shutdown;

#[async_trait]
pub trait AsyncTask: Send + 'static {
    type Output: Send + 'static;

    async fn run(self) -> Self::Output;
}

pub struct AsyncTaskPool<T>
where
    T: AsyncTask,
{
    task_in_rx: Receiver<T>,
    task_out_tx: Sender<T::Output>,
    shutdown: Shutdown,
    num_runners: usize,
}

impl<T> AsyncTaskPool<T>
where
    T: AsyncTask,
{
    pub fn new(
        task_in_rx: Receiver<T>,
        task_out_tx: Sender<T::Output>,
        shutdown: Shutdown,
        num_runners: usize,
    ) -> Self {
        assert!(0 < num_runners);
        Self {
            task_in_rx,
            task_out_tx,
            shutdown,
            num_runners,
        }
    }

    pub async fn event_loop(self) -> Result<()> {
        let span = tracing::info_span!("async task pool");
        let _enter = span.enter();
        tracing::info!("started with {} runners", self.num_runners);

        let shutdown_rx = self.shutdown.async_rx().clone();
        let mut runners = tokio::task::JoinSet::new();
        for runner_id in 0..self.num_runners {
            runners.spawn(async_task_runner_guard(
                runner_id,
                self.task_in_rx.clone(),
                self.task_out_tx.clone(),
                self.shutdown.clone(),
            ));
        }

        let _ = shutdown_rx.recv().await;
        tracing::info!("received shutdown signal, stopping");

        while let Some(result) = runners.join_next().await {
            if let Err(err) = result {
                tracing::info!("async task runner failed during shutdown, err: {err}");
            }
        }
        tracing::info!("stopped");
        Ok(())
    }
}

async fn async_task_runner_guard<T>(
    runner_id: usize,
    task_in_rx: Receiver<T>,
    task_out_tx: Sender<T::Output>,
    shutdown: Shutdown,
) where
    T: AsyncTask,
{
    let shutdown_on_panic = shutdown.clone();
    let result = AssertUnwindSafe(async_task_runner_loop(runner_id, task_in_rx, task_out_tx, shutdown))
        .catch_unwind()
        .await;
    if let Err(payload) = result {
        tracing::info!("async task runner panicked, stopping");
        shutdown_on_panic.shutdown();
        std::panic::resume_unwind(payload);
    }
}

async fn async_task_runner_loop<T>(
    runner_id: usize,
    task_in_rx: Receiver<T>,
    task_out_tx: Sender<T::Output>,
    shutdown: Shutdown,
) where
    T: AsyncTask,
{
    let span = tracing::info_span!("async task runner", runner_id);
    let _enter = span.enter();
    tracing::info!("started");

    let shutdown_rx = shutdown.async_rx().clone();
    'event_loop: while !shutdown.is_shutdown() {
        let task = tokio::select! {
            shutdown = shutdown_rx.recv() => {
                let _ = shutdown;
                break 'event_loop;
            },
            task = task_in_rx.recv() => {
                match task {
                    Ok(task) => task,
                    Err(err) => {
                        tracing::info!("unable to receive async task, err: {err}, stopping");
                        break 'event_loop;
                    },
                }
            },
        };

        let output = tokio::select! {
            biased;
            shutdown = shutdown_rx.recv() => {
                let _ = shutdown;
                break 'event_loop;
            },
            output = task.run() => output,
        };
        match task_out_tx.try_send(output) {
            Ok(()) => {},
            Err(TrySendError::Full(_)) => {
                panic!("async task output channel is full")
            },
            Err(TrySendError::Closed(_)) => {
                tracing::info!("unable to send async task output, channel closed, stopping");
                shutdown.shutdown();
                break 'event_loop;
            },
        }
    }

    shutdown.shutdown();
    tracing::info!("stopped");
}

pub enum SwapOutTask<UserReq> {
    AwaitReservation { user_req: UserReq, wait: Boxed<()> },
}

pub enum SwapInTask<UserReq> {
    Ready(UserReq),
}

#[async_trait]
impl<UserReq> AsyncTask for SwapOutTask<UserReq>
where
    UserReq: Send + 'static,
{
    type Output = SwapInTask<UserReq>;

    async fn run(self) -> Self::Output {
        match self {
            SwapOutTask::AwaitReservation { user_req, wait } => {
                wait.await;
                SwapInTask::Ready(user_req)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use async_channel::bounded;

    use super::*;

    struct PendingTask {
        started_tx: Sender<()>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for PendingTask {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    #[async_trait]
    impl AsyncTask for PendingTask {
        type Output = ();

        async fn run(self) {
            let started_tx = self.started_tx.clone();
            let task = self;
            started_tx
                .send(())
                .await
                .expect("async task cancellation test start receiver should remain open");
            std::future::pending::<()>().await;
            drop(task);
        }
    }

    #[tokio::test]
    async fn shutdown_cancels_running_task() {
        let (task_in_tx, task_in_rx) = bounded(1);
        let (task_out_tx, task_out_rx) = bounded(1);
        let (started_tx, started_rx) = bounded(1);
        let dropped = Arc::new(AtomicBool::new(false));
        let shutdown = Shutdown::new();
        let pool = AsyncTaskPool::new(task_in_rx, task_out_tx, shutdown.clone(), 1);
        let pool_task = tokio::spawn(pool.event_loop());

        task_in_tx
            .send(PendingTask {
                started_tx,
                dropped: dropped.clone(),
            })
            .await
            .expect("async task cancellation test input should remain open");
        tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
            .await
            .expect("async task should start before the test timeout")
            .expect("async task start sender should remain open");

        shutdown.shutdown();
        tokio::time::timeout(std::time::Duration::from_secs(1), pool_task)
            .await
            .expect("async task pool should stop after shutdown")
            .expect("async task pool should not panic")
            .expect("async task pool should stop successfully");

        assert!(dropped.load(Ordering::Acquire));
        assert!(task_out_rx.try_recv().is_err());
    }
}
