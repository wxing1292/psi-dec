use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::post;
use inference_runtime_core::channel::Shutdown;

use crate::api::Inference;
use crate::codec::qwen::QwenCodec;

mod chat_completions;
mod error;

const MAX_REQUEST_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
struct HTTPServer<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    qwen_codec: Arc<QwenCodec>,
}

impl<const N: usize, const L: usize, const P: usize> HTTPServer<N, L, P> {
    fn new(inference: Arc<Inference<N, L, P>>, qwen_codec: Arc<QwenCodec>) -> Self {
        Self { inference, qwen_codec }
    }

    fn router(self) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(chat_completions::handle::<N, L, P>))
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(self)
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    inference: Arc<Inference<N, L, P>>,
    qwen_codec: Arc<QwenCodec>,
    shutdown: Shutdown,
) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    tracing::info!(%listen_addr, "inference runtime service: starting HTTP server");
    axum::serve(listener, HTTPServer::new(inference, qwen_codec).router())
        .with_graceful_shutdown(async move {
            let _ = shutdown.async_rx().recv().await;
        })
        .await
}
