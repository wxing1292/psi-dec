use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures_util::stream;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::config::MAX_SAMPLING_TOP_K;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RequestEvent;
use inference_runtime_core::runtime::RequestStatus;
use inference_runtime_core::runtime::RequestTokenPositions;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::TokenProbs;
use tokio_stream::Stream;

use crate::api::Inference;

impl<const N: usize, const L: usize, const P: usize> Inference<N, L, P> {
    pub fn decode(&self, request: DecodeRequest) -> Result<DecodeResponse> {
        self.create_session(request).map(DecodeResponse::new)
    }

    pub fn create_session(&self, request: DecodeRequest) -> Result<DecodeSession<N, L, P>> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        assert_ne!(request_id, 0, "decode request ID allocator wrapped to zero");

        let DecodeRequest {
            tokens,
            token_positions,
            resource_entries,
            mut sampling,
        } = request;
        merge_stop_sequences(&mut sampling.stop_sequences, &self.default_stop_sequences);

        let external_request = self.runtime.create_session(
            request_id,
            vec![],
            tokens,
            vec![],
            token_positions,
            resource_entries,
            sampling,
        )?;
        Ok(DecodeSession::new(external_request))
    }

    pub fn continue_session(&self, session: &DecodeSession<N, L, P>, request: DecodeRequest) -> Result<()> {
        let DecodeRequest {
            tokens,
            token_positions,
            resource_entries,
            mut sampling,
        } = request;
        if token_positions.is_some() {
            return Err(Error::invalid_argument(
                "decode session continuation does not support explicit token positions",
            ));
        }
        if !resource_entries.is_empty() {
            return Err(Error::invalid_argument(
                "decode session continuation does not support new resource entries",
            ));
        }
        merge_stop_sequences(&mut sampling.stop_sequences, &self.default_stop_sequences);
        self.runtime.continue_session(&session.request, tokens, sampling)
    }
}

pub struct DecodeRequest {
    tokens: Vec<Token>,
    token_positions: Option<RequestTokenPositions>,
    resource_entries: Vec<(Resource, ResourcePlacement)>,
    sampling: SamplingConfig,
}

impl DecodeRequest {
    pub fn new(
        tokens: Vec<Token>,
        token_positions: Option<RequestTokenPositions>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling: SamplingConfig,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::invalid_argument(
                "decode request must include at least one token",
            ));
        }
        if sampling.max_sampled_tokens == 0 {
            return Err(Error::invalid_argument("max_sampled_tokens must be greater than 0"));
        }
        if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
            return Err(Error::invalid_argument(format!(
                "temperature must be finite and non-negative, got {}",
                sampling.temperature
            )));
        }
        if !(1..=MAX_SAMPLING_TOP_K).contains(&sampling.top_k) {
            return Err(Error::invalid_argument(format!(
                "top_k must be in [1, {MAX_SAMPLING_TOP_K}], got {}",
                sampling.top_k
            )));
        }
        if !sampling.top_p.is_finite() || !(0.0..=1.0).contains(&sampling.top_p) {
            return Err(Error::invalid_argument(format!(
                "top_p must be finite and in [0, 1], got {}",
                sampling.top_p
            )));
        }
        if sampling.stop_sequences.iter().any(Vec::is_empty) {
            return Err(Error::invalid_argument(
                "stop sequences must not include empty sequences",
            ));
        }
        Ok(Self {
            tokens,
            token_positions,
            resource_entries,
            sampling,
        })
    }
}

pub struct DecodeResponse {
    request_id: RawRequestID,
    stream: Pin<Box<dyn Stream<Item = Result<DecodeEvent>> + Send>>,
}

impl DecodeResponse {
    fn new<const N: usize, const L: usize, const P: usize>(session: DecodeSession<N, L, P>) -> Self {
        let request_id = session.request_id();
        let stream = stream::unfold((session, false), |(mut session, closed)| {
            async move {
                if closed {
                    return None;
                }
                let event = session.next_event().await?;
                let closed = matches!(event, Ok(DecodeEvent::Completed { .. }) | Err(_));
                Some((event, (session, closed)))
            }
        });
        Self {
            request_id,
            stream: Box::pin(stream),
        }
    }

    pub fn request_id(&self) -> RawRequestID {
        self.request_id
    }
}

pub struct DecodeSession<const N: usize, const L: usize, const P: usize> {
    request: ExternalRequest,
    num_output_tokens: usize,
    closed: bool,
}

impl<const N: usize, const L: usize, const P: usize> DecodeSession<N, L, P> {
    fn new(request: ExternalRequest) -> Self {
        Self {
            request,
            num_output_tokens: 0,
            closed: false,
        }
    }

    pub fn request_id(&self) -> RawRequestID {
        self.request.req_id()
    }

    pub async fn next_event(&mut self) -> Option<Result<DecodeEvent>> {
        if self.closed {
            return None;
        }
        match self.request.event_rx().recv().await {
            Ok(RequestEvent::TokenProbs(token_probs)) => {
                debug_assert!(!token_probs.tokens.is_empty());
                debug_assert_eq!(token_probs.tokens.len(), token_probs.probs.len());
                self.num_output_tokens += token_probs.tokens.len();
                Some(Ok(DecodeEvent::TokenProbs(token_probs)))
            },
            Ok(RequestEvent::TurnCompleted(reason)) => {
                let num_output_tokens = std::mem::take(&mut self.num_output_tokens);
                Some(Ok(DecodeEvent::Completed {
                    reason,
                    num_output_tokens,
                }))
            },
            Err(_) => {
                self.closed = true;
                Some(match self.request.status() {
                    RequestStatus::Completed(reason) => {
                        Ok(DecodeEvent::Completed {
                            reason,
                            num_output_tokens: std::mem::take(&mut self.num_output_tokens),
                        })
                    },
                    RequestStatus::Cancelled => Err(Error::cancelled("request was cancelled")),
                    RequestStatus::TimedOut => Err(Error::deadline_exceeded("request deadline exceeded")),
                    RequestStatus::Aborted => Err(Error::aborted("request was aborted")),
                    status => panic!("request output closed in non-terminal state: {status:?}"),
                })
            },
        }
    }
}

impl Stream for DecodeResponse {
    type Item = Result<DecodeEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

#[derive(Debug)]
pub enum DecodeEvent {
    TokenProbs(TokenProbs),
    Completed {
        reason: CompletionReason,
        num_output_tokens: usize,
    },
}

fn merge_stop_sequences(input: &mut Vec<Vec<Token>>, defaults: &[Vec<Token>]) {
    input.extend(defaults.iter().cloned());
    input.sort_unstable();
    input.dedup();
}

#[cfg(test)]
#[path = "./decode_test.rs"]
mod decode_test;
