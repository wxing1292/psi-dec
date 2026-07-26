use std::sync::Arc;

use inference_runtime_core::Error;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use tonic::Status;
use tonic::transport::Server;

use crate::api::Inference;

mod decode;

#[derive(Clone)]
struct GRPCServer<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
}

impl<const N: usize, const L: usize, const P: usize> GRPCServer<N, L, P> {
    fn new(inference: Arc<Inference<N, L, P>>) -> Self {
        Self { inference }
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    inference: Arc<Inference<N, L, P>>,
    shutdown: Shutdown,
) -> Result<(), tonic::transport::Error> {
    tracing::info!(%listen_addr, "inference runtime service: starting gRPC server");
    Server::builder()
        .add_service(InferenceRuntimeServer::new(GRPCServer::new(inference)))
        .serve_with_shutdown(listen_addr, async move {
            let _ = shutdown.async_rx().recv().await;
        })
        .await
}

fn map_error(error: Error) -> Status {
    match error {
        Error::InvalidArgument(message) => Status::invalid_argument(message),
        Error::ResourceExhausted(message) => Status::resource_exhausted(message),
        Error::Cancelled(message) => Status::cancelled(message),
        Error::DeadlineExceeded(message) => Status::deadline_exceeded(message),
        Error::Aborted(message) => Status::aborted(message),
        Error::Unavailable(message) => Status::unavailable(message),
        Error::Internal(message) => Status::internal(message),
    }
}
