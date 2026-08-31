use std::pin::Pin;

use inference_runtime_core::Result as RuntimeResult;
use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::Token;
use inference_runtime_proto::inference_runtime_service::CompletionReason as ProtoCompletionReason;
use inference_runtime_proto::inference_runtime_service::DecodeChunk;
use inference_runtime_proto::inference_runtime_service::DecodeCompletion as ProtoDecodeCompletion;
use inference_runtime_proto::inference_runtime_service::DecodeRequest as ProtoDecodeRequest;
use inference_runtime_proto::inference_runtime_service::DecodeResponse as ProtoDecodeResponse;
use inference_runtime_proto::inference_runtime_service::decode_response::Event;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntime;
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::Streaming;

use crate::api::Inference;
use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;
use crate::rpc::grpc::GRPCServer;
use crate::rpc::grpc::map_error;

type ProtoDecodeResponseStream = Pin<Box<dyn Stream<Item = Result<ProtoDecodeResponse, Status>> + Send>>;

#[async_trait::async_trait]
impl<const N: usize, const L: usize, const P: usize> InferenceRuntime for GRPCServer<N, L, P> {
    type DecodeStream = ProtoDecodeResponseStream;
    type DecodeStreamStream = ProtoDecodeResponseStream;

    async fn decode(&self, request: Request<ProtoDecodeRequest>) -> Result<Response<Self::DecodeStream>, Status> {
        let response = start_decode(&self.inference, request.into_inner())?;
        Ok(Response::new(response))
    }

    async fn decode_stream(
        &self,
        request: Request<Streaming<ProtoDecodeRequest>>,
    ) -> Result<Response<Self::DecodeStreamStream>, Status> {
        let inference = self.inference.clone();
        let mut requests = request.into_inner();
        let first_request = requests
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("decode stream requires at least one request"))?;
        let mut session = inference
            .create_session(map_request(first_request)?)
            .map_err(map_error)?;
        let responses = async_stream::try_stream! {
            while {
                loop {
                    let event = session
                        .next_event()
                        .await
                        .ok_or_else(|| Status::internal("decode session ended without a turn completion"))?;
                    let turn_completed = matches!(event, Ok(DecodeEvent::Completed { .. }));
                    yield map_response(session.request_id() as u64, event)?;
                    if turn_completed {
                        break;
                    }
                }

                match requests.message().await? {
                    Some(request) => {
                        inference
                        .continue_session(&session, map_request(request)?)
                        .map_err(map_error)?;
                        true
                    },
                    None => false,
                }
            } {}
        };
        Ok(Response::new(Box::pin(responses)))
    }
}

fn start_decode<const N: usize, const L: usize, const P: usize>(
    inference: &Inference<N, L, P>,
    request: ProtoDecodeRequest,
) -> Result<ProtoDecodeResponseStream, Status> {
    let request = map_request(request)?;
    let response = inference.decode(request).map_err(map_error)?;
    let request_id = response.request_id() as u64;
    Ok(Box::pin(response.map(move |event| map_response(request_id, event))))
}

fn map_request(request: ProtoDecodeRequest) -> Result<DecodeRequest, Status> {
    let sampling = SamplingConfig {
        max_sampled_tokens: usize::try_from(request.max_sampled_tokens)
            .map_err(|_| Status::invalid_argument("max_sampled_tokens does not fit usize"))?,
        temperature: request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
        top_k: request.top_k.unwrap_or(DEFAULT_SAMPLING_TOP_K as u32) as usize,
        top_p: request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
        seed: request.seed,
        stop_sequences: request
            .stop_sequences
            .into_iter()
            .map(|sequence| sequence.tokens.into_iter().map(Token::new).collect())
            .collect(),
    };
    DecodeRequest::new(
        request.tokens.into_iter().map(Token::new).collect(),
        None,
        vec![],
        sampling,
    )
    .map_err(map_error)
}

fn map_response(request_id: u64, event: RuntimeResult<DecodeEvent>) -> Result<ProtoDecodeResponse, Status> {
    let event = match event.map_err(map_error)? {
        DecodeEvent::TokenProbs(token_probs) => {
            Event::Chunk(DecodeChunk {
                tokens: token_probs.tokens.into_iter().map(Token::value).collect(),
                probs: token_probs.probs.into_iter().map(|prob| prob.into_inner()).collect(),
            })
        },
        DecodeEvent::Completed {
            reason,
            num_output_tokens,
        } => {
            let reason = match reason {
                CompletionReason::StopSequence => ProtoCompletionReason::StopSequence,
                CompletionReason::LengthLimit => ProtoCompletionReason::LengthLimit,
                CompletionReason::ContextLimit => ProtoCompletionReason::ContextLimit,
            };
            Event::Completion(ProtoDecodeCompletion {
                reason: reason as i32,
                num_output_tokens: num_output_tokens as u64,
            })
        },
    };
    Ok(ProtoDecodeResponse {
        request_id,
        event: Some(event),
    })
}

#[cfg(test)]
#[path = "./decode_test.rs"]
mod decode_test;
