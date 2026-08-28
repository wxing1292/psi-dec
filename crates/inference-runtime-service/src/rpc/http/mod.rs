use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::post;
use inference_runtime_core::channel::Shutdown;
use tracing::Instrument;

use crate::api::Inference;
use crate::asr::Qwen3ASRService;
use crate::codec::qwen::QwenCodec;

mod chat_completions;
mod error;
mod transcriptions;

const MAX_CHAT_COMPLETIONS_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_TRANSCRIPTIONS_BODY_BYTES: usize = 32 * 1024 * 1024;

pub enum HTTPService {
    ChatCompletions(Arc<QwenCodec>),
    Transcriptions(Arc<Qwen3ASRService>),
}

#[derive(Clone)]
struct ChatCompletionsServer<const N: usize, const L: usize, const P: usize> {
    model_name: Arc<str>,
    inference: Arc<Inference<N, L, P>>,
    qwen_codec: Arc<QwenCodec>,
}

impl<const N: usize, const L: usize, const P: usize> ChatCompletionsServer<N, L, P> {
    fn new(model_name: String, inference: Arc<Inference<N, L, P>>, qwen_codec: Arc<QwenCodec>) -> Self {
        Self {
            model_name: model_name.into(),
            inference,
            qwen_codec,
        }
    }

    fn router(self) -> Router {
        Router::<Self>::new()
            .route("/v1/chat/completions", post(chat_completions::handle::<N, L, P>))
            .layer(DefaultBodyLimit::max(MAX_CHAT_COMPLETIONS_BODY_BYTES))
            .with_state(self)
    }
}

#[derive(Clone)]
struct TranscriptionsServer<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    asr: Arc<Qwen3ASRService>,
}

impl<const N: usize, const L: usize, const P: usize> TranscriptionsServer<N, L, P> {
    fn new(inference: Arc<Inference<N, L, P>>, asr: Arc<Qwen3ASRService>) -> Self {
        Self { inference, asr }
    }

    fn router(self) -> Router {
        Router::<Self>::new()
            .route("/v1/audio/transcriptions", post(transcriptions::handle::<N, L, P>))
            .layer(DefaultBodyLimit::max(MAX_TRANSCRIPTIONS_BODY_BYTES))
            .with_state(self)
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    model_name: String,
    inference: Arc<Inference<N, L, P>>,
    http_service: HTTPService,
    shutdown: Shutdown,
) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    let router = match http_service {
        HTTPService::ChatCompletions(qwen_codec) => {
            ChatCompletionsServer::new(model_name, inference, qwen_codec).router()
        },
        HTTPService::Transcriptions(asr) => TranscriptionsServer::new(inference, asr).router(),
    };
    async move {
        tracing::info!("started");
        let result = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown.async_rx().recv().await;
            })
            .await;
        tracing::info!("stopped");
        result
    }
    .instrument(tracing::info_span!("http-server", %listen_addr))
    .await
}
