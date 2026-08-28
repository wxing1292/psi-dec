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

use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;
use crate::rpc::grpc::GRPCServer;
use crate::rpc::grpc::map_error;

#[async_trait::async_trait]
impl<const N: usize, const L: usize, const P: usize> InferenceRuntime for GRPCServer<N, L, P> {
    type DecodeStream = Pin<Box<dyn Stream<Item = Result<ProtoDecodeResponse, Status>> + Send>>;

    async fn decode(&self, request: Request<ProtoDecodeRequest>) -> Result<Response<Self::DecodeStream>, Status> {
        let request = request.into_inner();
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
        let request = DecodeRequest::new(
            request.tokens.into_iter().map(Token::new).collect(),
            None,
            vec![],
            sampling,
        )
        .map_err(map_error)?;
        let response = self.inference.decode(request).map_err(map_error)?;
        let request_id = response.request_id() as u64;
        let response = response.map(move |event| map_response(request_id, event));
        Ok(Response::new(Box::pin(response)))
    }
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
