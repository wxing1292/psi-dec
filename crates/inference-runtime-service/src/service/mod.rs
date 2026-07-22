use std::net::SocketAddr;
use std::sync::Arc;

use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::runtime::Token;
use inference_runtime_proto::inference_runtime_service::DecodeRequest;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntime;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;

use crate::consts::NUM_CACHE_LANE;
use crate::consts::NUM_TRIE_PARTITION;
use crate::runtime::InferenceRuntime as ServiceRuntime;

pub type InferenceServiceRuntimeN8 = ServiceRuntime<8, NUM_CACHE_LANE, NUM_TRIE_PARTITION>;
pub type InferenceServiceRuntimeN16 = ServiceRuntime<16, NUM_CACHE_LANE, NUM_TRIE_PARTITION>;
pub type InferenceServiceRuntimeN8L2 = ServiceRuntime<8, 2, NUM_TRIE_PARTITION>;
pub type InferenceServiceRuntimeN16L2 = ServiceRuntime<16, 2, NUM_TRIE_PARTITION>;

mod decode;
use decode::DecodeStream;

pub async fn run_rpc_server<const N: usize, const L: usize, const P: usize>(
    listen_addr: SocketAddr,
    runtime: Arc<ServiceRuntime<N, L, P>>,
    default_stop_sequences: Vec<Vec<Token>>,
    shutdown: Shutdown,
) -> Result<(), tonic::transport::Error> {
    let service = InferenceRuntimeService::new(runtime, default_stop_sequences);
    tracing::info!(%listen_addr, "inference runtime service: starting RPC server");
    Server::builder()
        .add_service(InferenceRuntimeServer::new(service))
        .serve_with_shutdown(listen_addr, rpc_shutdown(shutdown))
        .await
}

#[derive(Clone)]
pub struct InferenceRuntimeService<const N: usize, const L: usize, const P: usize> {
    runtime: Arc<ServiceRuntime<N, L, P>>,
    default_stop_sequences: Vec<Vec<Token>>,
}

impl<const N: usize, const L: usize, const P: usize> InferenceRuntimeService<N, L, P> {
    pub fn new(runtime: Arc<ServiceRuntime<N, L, P>>, default_stop_sequences: Vec<Vec<Token>>) -> Self {
        debug_assert!(
            default_stop_sequences
                .iter()
                .all(|stop_sequence| !stop_sequence.is_empty()),
            "default stop sequences must not be empty"
        );
        Self {
            runtime,
            default_stop_sequences,
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::SignalKind;

        let mut terminate =
            tokio::signal::unix::signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install shutdown handler");
    }

    tracing::info!("inference runtime service: shutdown signal received");
}

async fn rpc_shutdown(shutdown: Shutdown) {
    let shutdown_rx = shutdown.async_rx().clone();
    tokio::select! {
        _ = shutdown_signal() => shutdown.shutdown(),
        _ = shutdown_rx.recv() => {},
    }
}

#[async_trait::async_trait]
impl<const N: usize, const L: usize, const P: usize> InferenceRuntime for InferenceRuntimeService<N, L, P> {
    type DecodeStream = DecodeStream;

    async fn decode(&self, request: Request<DecodeRequest>) -> Result<Response<Self::DecodeStream>, Status> {
        self.decode(request).await
    }
}
