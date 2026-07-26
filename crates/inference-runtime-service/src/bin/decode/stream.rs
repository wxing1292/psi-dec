use std::io::Write;
use std::time::Duration;
use std::time::Instant;

use inference_runtime_core::runtime::Token;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use inference_runtime_core::tokenizer::huggingface::IncrementalDecoder;
use inference_runtime_proto::inference_runtime_service::CompletionReason as ProtoCompletionReason;
use inference_runtime_proto::inference_runtime_service::DecodeRequest as ProtoDecodeRequest;
use inference_runtime_proto::inference_runtime_service::decode_response::Event;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use inference_runtime_service::perf_metrics::DecodePerfMetrics;
use tonic::Request;
use tonic::transport::Channel;

use crate::config::RuntimeConfig;
use crate::error::DecodeCliResult;
use crate::executor::DecodeRequest;

pub struct DecodeStreamExecutor<'a> {
    client: &'a mut InferenceRuntimeClient<Channel>,
    tokenizer: &'a HFTokenizer,
    runtime: &'a RuntimeConfig,
    stream_stdout: bool,
}

#[derive(Debug)]
pub struct DecodeStreamResult {
    pub text: String,
    pub metrics: DecodePerfMetrics,
    pub streamed: bool,
}

impl<'a> DecodeStreamExecutor<'a> {
    pub fn new(
        client: &'a mut InferenceRuntimeClient<Channel>,
        tokenizer: &'a HFTokenizer,
        runtime: &'a RuntimeConfig,
        stream_stdout: bool,
    ) -> Self {
        Self {
            client,
            tokenizer,
            runtime,
            stream_stdout,
        }
    }

    pub async fn execute(&mut self, request: &DecodeRequest) -> DecodeCliResult<DecodeStreamResult> {
        let proto_request = ProtoDecodeRequest {
            tokens: request.tokens.iter().map(|token| token.value()).collect(),
            max_sampled_tokens: request.max_sampled_tokens,
            stop_sequences: Vec::new(),
            temperature: Some(request.temperature),
            top_k: Some(u32::try_from(request.top_k).map_err(|_| "top_k does not fit u32")?),
            top_p: Some(request.top_p),
            seed: request.seed,
        };
        let input_tokens = proto_request.tokens.len();
        let max_sampled_tokens = proto_request.max_sampled_tokens;
        let mut tonic_request = Request::new(proto_request);
        if let Some(timeout_ms) = self.runtime.timeout_ms() {
            tonic_request.set_timeout(Duration::from_millis(timeout_ms));
        }
        let mut stream = self
            .client
            .decode(tonic_request)
            .await
            .map_err(|err| format!("decode RPC failed: {err}"))?
            .into_inner();

        let started_at = Instant::now();
        let mut state = DecodeStreamState::new(input_tokens, max_sampled_tokens, started_at);
        let mut response_request_id = None;
        let mut decoder = IncrementalDecoder::without_special_tokens(self.tokenizer);
        let mut stream_decode_failed = false;
        let mut completed = false;

        while let Some(response) = stream
            .message()
            .await
            .map_err(|err| format!("decode stream failed: {err}"))?
        {
            if response.request_id == 0 {
                return Err("decode stream returned reserved request ID zero".into());
            }
            if response_request_id
                .replace(response.request_id)
                .is_some_and(|id| id != response.request_id)
            {
                return Err("decode stream changed server request ID".into());
            }
            if completed {
                return Err("decode stream emitted an event after its completion event".into());
            }
            match response.event {
                Some(Event::Chunk(chunk)) => {
                    if chunk.tokens.is_empty() || chunk.tokens.len() != chunk.probs.len() {
                        return Err("decode stream returned an invalid token chunk".into());
                    }
                    let tokens = chunk.tokens;
                    let mut decoded_text = String::new();
                    if !stream_decode_failed {
                        let decoded_tokens = tokens.iter().copied().map(Token::new).collect::<Vec<_>>();
                        match decoder.decode(&decoded_tokens) {
                            Ok(Some(piece)) => decoded_text.push_str(&piece),
                            Ok(None) => {},
                            Err(err) => {
                                stream_decode_failed = true;
                                tracing::warn!(
                                    num_tokens = decoded_tokens.len(),
                                    error = ?err,
                                    "streaming token decode failed"
                                );
                            },
                        }
                    }
                    if self.stream_stdout && !decoded_text.is_empty() {
                        print!("{decoded_text}");
                        std::io::stdout()
                            .flush()
                            .map_err(|err| format!("unable to flush streamed output: {err}"))?;
                    }
                    state.observe_chunk(tokens, decoded_text);
                },
                Some(Event::Completion(event)) => {
                    match ProtoCompletionReason::try_from(event.reason) {
                        Ok(
                            ProtoCompletionReason::StopSequence
                            | ProtoCompletionReason::LengthLimit
                            | ProtoCompletionReason::ContextLimit,
                        ) => {},
                        Ok(ProtoCompletionReason::Unspecified) | Err(_) => {
                            return Err(
                                format!("decode stream returned unknown completion reason {}", event.reason).into(),
                            );
                        },
                    }
                    if event.num_output_tokens as usize != state.sampled_tokens.len() {
                        return Err("decode completion count did not match token chunks".into());
                    }
                    completed = true;
                },
                None => return Err("decode stream event envelope was empty".into()),
            }
        }
        if !completed {
            return Err("decode stream ended before a completion event".into());
        }
        let request_id = response_request_id.expect("a completion event must carry a non-zero request ID");
        let final_text = self.decode_full_response_text(&state)?;
        if self.stream_stdout {
            finish_stream_stdout(&state, &final_text)?;
        }
        Ok(DecodeStreamResult {
            text: final_text,
            metrics: state.finish(Instant::now(), request_id),
            streamed: self.stream_stdout,
        })
    }

    fn decode_full_response_text(&self, state: &DecodeStreamState) -> DecodeCliResult<String> {
        if state.sampled_tokens.is_empty() {
            return Ok(String::new());
        }
        self.tokenizer
            .decode_without_special_tokens(&state.sampled_tokens)
            .map_err(|error| format!("unable to decode final response: {error:?}").into())
    }
}

fn finish_stream_stdout(state: &DecodeStreamState, final_text: &str) -> DecodeCliResult<()> {
    if let Some(suffix) = final_text.strip_prefix(&state.streamed_text) {
        print!("{suffix}");
    } else if final_text != state.streamed_text {
        if !state.streamed_text.is_empty() && !state.streamed_text.ends_with('\n') {
            println!();
        }
        print!("{final_text}");
    }
    if !final_text.is_empty() {
        if !final_text.ends_with('\n') {
            println!();
        }
        std::io::stdout()
            .flush()
            .map_err(|err| format!("unable to flush final output: {err}"))?;
    }
    Ok(())
}

#[derive(Debug)]
struct DecodeStreamState {
    input_tokens: usize,
    max_sampled_tokens: u32,
    started_at: Instant,
    first_chunk_at: Option<Instant>,
    last_chunk_at: Option<Instant>,
    inter_chunk_latencies: Vec<Duration>,
    streamed_text: String,
    sampled_tokens: Vec<Token>,
    chunk_count: usize,
}

impl DecodeStreamState {
    fn new(input_tokens: usize, max_sampled_tokens: u32, started_at: Instant) -> Self {
        Self {
            input_tokens,
            max_sampled_tokens,
            started_at,
            first_chunk_at: None,
            last_chunk_at: None,
            inter_chunk_latencies: Vec::new(),
            streamed_text: String::new(),
            sampled_tokens: Vec::new(),
            chunk_count: 0,
        }
    }

    fn observe_chunk(&mut self, tokens: Vec<u32>, decoded_text: String) {
        let now = Instant::now();
        if self.first_chunk_at.is_none() {
            self.first_chunk_at = Some(now);
        }
        if let Some(previous) = self.last_chunk_at.replace(now) {
            self.inter_chunk_latencies.push(now.duration_since(previous));
        }
        self.chunk_count += 1;
        self.sampled_tokens.extend(tokens.into_iter().map(Token::new));
        self.streamed_text.push_str(&decoded_text);
    }

    fn finish(self, finished_at: Instant, request_id: u64) -> DecodePerfMetrics {
        let ttft = self.first_chunk_at.map(|start| start.duration_since(self.started_at));
        let decode_elapsed = self.first_chunk_at.map(|start| finished_at.duration_since(start));
        DecodePerfMetrics {
            request_id,
            input_tokens: self.input_tokens,
            max_sampled_tokens: self.max_sampled_tokens,
            sampled_tokens: self.sampled_tokens.len(),
            chunk_count: self.chunk_count,
            elapsed: finished_at.duration_since(self.started_at),
            ttft,
            decode_elapsed,
            inter_chunk_latencies: self.inter_chunk_latencies,
        }
    }
}
