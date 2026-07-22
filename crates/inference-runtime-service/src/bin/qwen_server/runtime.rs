use std::net::SocketAddr;
use std::sync::Arc;

use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::compute::ReplayableModelBatchExecutor;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::config::ServiceConfig;
use inference_runtime_service::consts::NUM_TRIE_PARTITION;
use inference_runtime_service::executor::ReplayableModelExecutorLoop;
use inference_runtime_service::runtime::InferenceRuntime;
use inference_runtime_service::service::run_rpc_server;

pub fn run_replay_model_runtime<const N: usize, const L: usize, M>(
    listen_addr: SocketAddr,
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    service_config: ServiceConfig,
    model: M,
    debug_logging: bool,
) where
    M: ReplayableModelBatchExecutor,
{
    let shutdown = Shutdown::new();
    let default_stop_sequences = model.default_stop_sequences();
    let runtime = Arc::new(InferenceRuntime::<N, L, NUM_TRIE_PARTITION>::new(
        model_runtime_config,
        scheduler_config,
        service_config,
        shutdown.clone(),
    ));
    let server_runtime = runtime.clone();
    let server_shutdown = shutdown.clone();
    let server_tokio_runtime = tokio::runtime::Runtime::new().expect("tokio runtime should initialize");
    let server_thread = std::thread::Builder::new()
        .name("inference-rpc-server".to_string())
        .spawn(move || {
            let _shutdown_guard = ShutdownGuard::new(server_shutdown.clone());
            server_tokio_runtime.block_on(run_rpc_server(
                listen_addr,
                server_runtime,
                default_stop_sequences,
                server_shutdown,
            ))
        })
        .expect("inference RPC server thread should start");

    let executor = ReplayableModelExecutorLoop::new(
        runtime.batch_device_request_rx(),
        runtime.batch_device_response_tx(),
        runtime.request_slot_reset_notifier(),
        runtime.request_slot_reset_rx(),
        shutdown,
        model,
    )
    .with_debug_logging(debug_logging);
    let executor_result = executor.event_loop();
    runtime.shutdown();

    server_thread
        .join()
        .expect("inference RPC server thread panicked")
        .expect("inference runtime RPC server failed");
    executor_result.expect("inference replay executor loop failed");
}
