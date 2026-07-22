use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::config::MAX_SAMPLING_TOP_K;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::RequestStatus;
use inference_runtime_core::runtime::Token;
use inference_runtime_proto::inference_runtime_service::DecodeChunkResponse;
use inference_runtime_proto::inference_runtime_service::DecodeRequest;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use crate::service::InferenceRuntimeService;

pub type DecodeStream = ReceiverStream<Result<DecodeChunkResponse, Status>>;

impl<const N: usize, const L: usize, const P: usize> InferenceRuntimeService<N, L, P> {
    pub async fn decode(&self, request: Request<DecodeRequest>) -> Result<Response<DecodeStream>, Status> {
        let request = request.into_inner();
        tracing::debug!(
            target: "inference-runtime-service::rpc",
            phase = "decode.request.received",
            request_id = request.request_id,
            token_count = request.tokens.len(),
            max_sampled_tokens = request.max_sampled_tokens,
            temperature = request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
            top_k = request.top_k.unwrap_or(DEFAULT_SAMPLING_TOP_K as u32),
            top_p = request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
            seed = request.seed,
            default_stop_sequence_count = self.default_stop_sequences.len(),
            stop_sequence_count = request.stop_sequences.len(),
            "decode request received"
        );
        self.validate(&request)?;
        let request = self.submit(request)?;
        self.wait(request)
    }

    fn validate(&self, request: &DecodeRequest) -> Result<(), Status> {
        if request.request_id == 0 {
            return Err(Status::invalid_argument(
                "inference runtime service: decode request must include a valid request_id",
            ));
        }
        if request.tokens.is_empty() {
            return Err(Status::invalid_argument(
                "inference runtime service: decode request must include at least one token",
            ));
        }
        if request.max_sampled_tokens == 0 {
            return Err(Status::invalid_argument(
                "inference runtime service: decode request max_sampled_tokens must be greater than 0",
            ));
        }
        let temperature = request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE);
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(Status::invalid_argument(format!(
                "inference runtime service: decode request temperature must be finite and non-negative, got \
                 {temperature}",
            )));
        }
        let top_k = request.top_k.unwrap_or(DEFAULT_SAMPLING_TOP_K as u32);
        if top_k == 0 || top_k as usize > MAX_SAMPLING_TOP_K {
            return Err(Status::invalid_argument(format!(
                "inference runtime service: decode request top_k must be in [1, {MAX_SAMPLING_TOP_K}], got {top_k}",
            )));
        }
        let top_p = request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P);
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(Status::invalid_argument(format!(
                "inference runtime service: decode request top_p must be finite and in [0, 1], got {top_p}",
            )));
        }
        if request
            .stop_sequences
            .iter()
            .any(|stop_sequence| stop_sequence.tokens.is_empty())
        {
            return Err(Status::invalid_argument(
                "inference runtime service: decode request stop_sequences must not include empty sequences",
            ));
        }
        Ok(())
    }

    fn submit(&self, request: DecodeRequest) -> Result<ExternalRequest, Status> {
        let request_id = request.request_id;
        let mut stop_sequences = self.default_stop_sequences.clone();
        stop_sequences.extend(
            request
                .stop_sequences
                .into_iter()
                .map(|stop_sequence| stop_sequence.tokens.into_iter().map(Token::new).collect::<Vec<_>>()),
        );
        stop_sequences.sort_unstable();
        stop_sequences.dedup();
        let sampling_config = SamplingConfig {
            max_sampled_tokens: usize::try_from(request.max_sampled_tokens).map_err(|_| {
                Status::invalid_argument(format!(
                    "inference runtime service: max_sampled_tokens {} exceeds usize",
                    request.max_sampled_tokens
                ))
            })?,
            temperature: request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
            top_k: request.top_k.unwrap_or(DEFAULT_SAMPLING_TOP_K as u32) as usize,
            top_p: request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
            seed: request.seed,
            stop_sequences,
        };
        let tokens = request.tokens.into_iter().map(Token::new).collect();
        let (internal_request, external_request) = self.runtime.initialize_req(request_id, tokens, sampling_config)?;
        self.runtime.submit_req(internal_request)?;
        Ok(external_request)
    }

    fn wait(&self, request: ExternalRequest) -> Result<Response<DecodeStream>, Status> {
        let (tx, rx) = mpsc::channel(128);

        let request_id = request.req_id();
        let token_prob_rx = request.token_prob_rx().clone();
        tokio::spawn(async move {
            while let Ok(token_prob) = token_prob_rx.recv().await {
                let response = DecodeChunkResponse {
                    tokens: token_prob.tokens.into_iter().map(|token| token.value()).collect(),
                    probs: token_prob.probs.into_iter().map(|prob| prob.into_inner()).collect(),
                };

                if tx.send(Ok(response)).await.is_err() {
                    tracing::debug!(
                        target: "inference-runtime-service::rpc",
                        phase = "decode.stream.receiver_dropped",
                        request_id,
                        "decode stream receiver dropped; cancelling request"
                    );
                    let _ = request.store_cancelled();
                    return;
                }
            }

            let request_status = request.status();
            if let Some(status) = terminal_status(&request) {
                tracing::warn!(
                    target: "inference-runtime-service::rpc",
                    phase = "decode.request.failed",
                    request_id,
                    ?request_status,
                    code = ?status.code(),
                    "decode request finished with error"
                );
                let _ = tx.send(Err(status)).await;
            } else {
                tracing::debug!(
                    target: "inference-runtime-service::rpc",
                    phase = "decode.request.completed",
                    request_id,
                    "decode request completed"
                );
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn terminal_status(request: &ExternalRequest) -> Option<Status> {
    match request.status() {
        RequestStatus::Completed => None,
        RequestStatus::Cancelled => {
            Some(Status::cancelled(
                "inference runtime service: decode request was cancelled",
            ))
        },
        RequestStatus::TimedOut => {
            Some(Status::deadline_exceeded(
                "inference runtime service: decode request timed out",
            ))
        },
        RequestStatus::Aborted => Some(Status::internal("inference runtime service: decode request aborted")),
        _ => {
            let _ = request.store_cancelled();
            Some(Status::cancelled(
                "inference runtime service: decode request was cancelled",
            ))
        },
    }
}
