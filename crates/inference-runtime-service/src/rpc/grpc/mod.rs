use std::sync::Arc;

use inference_runtime_core::Error;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use tonic::Status;
use tonic::transport::Server;
use tracing::Instrument;

use crate::api::Inference;
use crate::api::messages::MessageGenerator;
use crate::codec::qwen::QwenCodec;

mod generation;

#[derive(Clone)]
struct GRPCServer<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    message_generator: Option<MessageGenerator<N, L, P>>,
}

impl<const N: usize, const L: usize, const P: usize> GRPCServer<N, L, P> {
    fn new(inference: Arc<Inference<N, L, P>>, qwen_codec: Option<Arc<QwenCodec>>) -> Self {
        let message_generator = qwen_codec.map(|codec| MessageGenerator::new(inference.clone(), codec));
        Self {
            inference,
            message_generator,
        }
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    inference: Arc<Inference<N, L, P>>,
    qwen_codec: Option<Arc<QwenCodec>>,
    shutdown: Shutdown,
) -> Result<(), tonic::transport::Error> {
    async move {
        tracing::info!("started");
        let result = Server::builder()
            .add_service(InferenceRuntimeServer::new(GRPCServer::new(inference, qwen_codec)))
            .serve_with_shutdown(listen_addr, async move {
                let _ = shutdown.async_rx().recv().await;
            })
            .await;
        tracing::info!("stopped");
        result
    }
    .instrument(tracing::info_span!("grpc-server", %listen_addr))
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
