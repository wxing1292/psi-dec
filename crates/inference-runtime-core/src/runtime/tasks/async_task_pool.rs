use std::future::Future;
use std::panic::AssertUnwindSafe;

use async_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TrySendError;
use futures_util::FutureExt;
use tracing::Instrument;

use crate::Result;
use crate::channel::Shutdown;

pub trait AsyncTask: Send + 'static {
    type Output: Send + 'static;

    fn run(self) -> impl Future<Output = Self::Output> + Send;
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
        let span = tracing::info_span!("async-task-pool", num_runners = self.num_runners);
        async move {
            tracing::info!("started");

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

            while let Some(result) = runners.join_next().await {
                if let Err(err) = result {
                    tracing::error!(error = %err, "async task runner failed during shutdown");
                }
            }
            while self.task_in_rx.try_recv().is_ok() {}
            tracing::info!("stopped");
            Ok(())
        }
        .instrument(span)
        .await
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
        tracing::error!(runner_id, "async task runner panicked");
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
    let span = tracing::info_span!("async-task-runner", runner_id);
    async move {
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
                            tracing::debug!(error = %err, "async task input channel closed");
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
                Err(TrySendError::Disconnected(_)) => {
                    tracing::debug!("async task output channel closed");
                    shutdown.shutdown();
                    break 'event_loop;
                },
            }
        }

        shutdown.shutdown();
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use async_channel::bounded as async_bounded;
    use crossbeam_channel::bounded as sync_bounded;

    use super::*;

    type TaskFuture = Pin<Box<dyn Future<Output = usize> + Send>>;

    mockall::mock! {
        Task {
            fn run(self) -> TaskFuture;
        }
    }

    impl AsyncTask for MockTask {
        type Output = usize;

        fn run(self) -> impl Future<Output = Self::Output> + Send {
            MockTask::run(self)
        }
    }

    struct DropMarker {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    #[tokio::test]
    async fn test_execute() {
        let (task_in_tx, task_in_rx) = async_bounded(1);
        let (task_out_tx, task_out_rx) = sync_bounded(1);
        let shutdown = Shutdown::new();
        let pool = AsyncTaskPool::new(task_in_rx, task_out_tx, shutdown, 1);
        let pool_task = tokio::spawn(pool.event_loop());

        let mut task = MockTask::new();
        task.expect_run().once().return_once(|| Box::pin(async { 7 }));
        task_in_tx.send(task).await.unwrap();
        drop(task_in_tx);

        tokio::time::timeout(std::time::Duration::from_secs(1), pool_task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(task_out_rx.try_recv(), Ok(7));
    }

    #[tokio::test]
    async fn test_shutdown() {
        let (task_in_tx, task_in_rx) = async_bounded(3);
        let (task_out_tx, task_out_rx) = sync_bounded(3);
        let (started_tx, started_rx) = async_bounded(1);
        let running_dropped = Arc::new(AtomicBool::new(false));
        let queued_dropped = [Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false))];
        let shutdown = Shutdown::new();
        let pool = AsyncTaskPool::new(task_in_rx, task_out_tx, shutdown.clone(), 1);
        let pool_task = tokio::spawn(pool.event_loop());

        let running_marker = DropMarker {
            dropped: running_dropped.clone(),
        };
        let mut running_task = MockTask::new();
        running_task.expect_run().once().return_once(move || {
            Box::pin(async move {
                let _marker = running_marker;
                started_tx.send(()).await.unwrap();
                std::future::pending::<usize>().await
            })
        });
        task_in_tx.send(running_task).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
            .await
            .unwrap()
            .unwrap();
        for dropped in &queued_dropped {
            task_in_tx.send(new_task(dropped.clone())).await.unwrap();
        }

        shutdown.shutdown();
        tokio::time::timeout(std::time::Duration::from_secs(1), pool_task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        assert!(running_dropped.load(Ordering::Acquire));
        assert!(queued_dropped.iter().all(|dropped| dropped.load(Ordering::Acquire)));
        assert!(task_in_tx.is_empty());
        assert!(task_out_rx.try_recv().is_err());
    }

    fn new_task(dropped: Arc<AtomicBool>) -> MockTask {
        let marker = DropMarker { dropped };
        let mut task = MockTask::new();
        task.expect_run().never().return_once(move || {
            drop(marker);
            unreachable!("queued async task should not execute")
        });
        task
    }
}
