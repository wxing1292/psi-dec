use std::io::Write;
use std::time::Duration;
use std::time::Instant;

use inference_runtime_core::tokenizer::HFTokenizer;
use inference_runtime_proto::inference_runtime_service::DecodeRequest as ProtoDecodeRequest;
use inference_runtime_proto::inference_runtime_service::TokenSequence as ProtoTokenSequence;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use inference_runtime_service::perf_metrics::DecodePerfMetrics;
use tonic::Request;
use tonic::transport::Channel;

use crate::config::RuntimeConfig;
use crate::error::DecodeCliResult;
use crate::req_resp::DecodeRequest;
use crate::req_resp::DecodeResponse;

pub struct DecodeStreamExecutor<'a> {
    client: &'a mut InferenceRuntimeClient<Channel>,
    tokenizer: &'a HFTokenizer,
    runtime: &'a RuntimeConfig,
    raw: bool,
    stream_stdout: bool,
}

impl<'a> DecodeStreamExecutor<'a> {
    pub fn new(
        client: &'a mut InferenceRuntimeClient<Channel>,
        tokenizer: &'a HFTokenizer,
        runtime: &'a RuntimeConfig,
        raw: bool,
        stream_stdout: bool,
    ) -> Self {
        Self {
            client,
            tokenizer,
            runtime,
            raw,
            stream_stdout,
        }
    }

    pub async fn execute(&mut self, request: &DecodeRequest) -> DecodeCliResult<DecodeResponse> {
        let proto_request = ProtoDecodeRequest {
            request_id: request.request_id(),
            tokens: request.tokens().to_vec(),
            max_sampled_tokens: request.max_sampled_tokens(),
            stop_sequences: default_stop_sequences(self.tokenizer),
            temperature: Some(request.temperature()),
            top_k: Some(
                u32::try_from(request.top_k())
                    .map_err(|_| format!("decode request top_k {} exceeds u32", request.top_k()))?,
            ),
            top_p: Some(request.top_p()),
            seed: request.seed(),
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
            .map_err(|err| format!("decode RPC failed for request_id={}: {err}", request.request_id()))?
            .into_inner();

        let started_at = Instant::now();
        let mut state = DecodeStreamState::new(request.request_id(), input_tokens, max_sampled_tokens, started_at);
        let skip_special_tokens = !self.raw;
        let mut decode_stream = self.tokenizer.decode_stream(skip_special_tokens);
        let mut stream_decode_failed = false;

        while let Some(chunk) = stream
            .message()
            .await
            .map_err(|err| format!("decode stream failed for request_id={}: {err}", request.request_id()))?
        {
            if chunk.tokens.len() != chunk.probs.len() {
                return Err(format!(
                    "decode stream returned mismatched tokens/probs for request_id={}: tokens={} probs={}",
                    request.request_id(),
                    chunk.tokens.len(),
                    chunk.probs.len(),
                )
                .into());
            }

            let tokens = chunk.tokens;
            let mut decoded_text = String::new();
            if !stream_decode_failed {
                for &token in &tokens {
                    match decode_stream.step(token) {
                        Ok(Some(piece)) => decoded_text.push_str(&piece),
                        Ok(None) => {},
                        Err(err) => {
                            stream_decode_failed = true;
                            tracing::warn!(
                                request_id = request.request_id(),
                                token,
                                error = ?err,
                                "streaming token decode failed; keeping token ids for final full decode"
                            );
                            if self.stream_stdout {
                                eprintln!(
                                    "warning: streaming decode failed for token {token}; final output/file will still \
                                     use full-token decode"
                                );
                            }
                            break;
                        },
                    }
                }
            }

            if self.stream_stdout && !decoded_text.is_empty() {
                print!("{decoded_text}");
                std::io::stdout()
                    .flush()
                    .map_err(|err| format!("unable to flush streamed decode output: {err}"))?;
            }

            state.observe_chunk(tokens, decoded_text);
        }

        let final_text = self.decode_full_response_text(&state)?;
        if self.stream_stdout {
            self.finish_stream_stdout(&state, &final_text, stream_decode_failed)?;
        }
        Ok(state.finish(Instant::now(), final_text))
    }

    fn finish_stream_stdout(
        &self,
        state: &DecodeStreamState,
        final_text: &str,
        stream_decode_failed: bool,
    ) -> DecodeCliResult<()> {
        if stream_decode_failed {
            if let Some(suffix) = final_text.strip_prefix(&state.streamed_text) {
                if !suffix.is_empty() {
                    print!("{suffix}");
                }
            } else if !final_text.is_empty() {
                eprintln!("warning: streamed text diverged from full decode; reprinting full final output");
                print!("{final_text}");
            }
        }

        let printed_text = if stream_decode_failed {
            final_text
        } else {
            &state.streamed_text
        };
        if !printed_text.is_empty() {
            if !printed_text.ends_with('\n') {
                println!();
            }
            std::io::stdout()
                .flush()
                .map_err(|err| format!("unable to flush final decode output: {err}"))?;
        }
        Ok(())
    }

    fn decode_full_response_text(&self, state: &DecodeStreamState) -> DecodeCliResult<String> {
        if state.sampled_token_ids.is_empty() {
            return Ok(String::new());
        }
        match self.tokenizer.decode(&state.sampled_token_ids, !self.raw) {
            Ok(text) => Ok(text),
            Err(err) => {
                tracing::warn!(
                    error = ?err,
                    "falling back to incremental streamed text after full decode failed"
                );
                Ok(state.streamed_text.clone())
            },
        }
    }
}

fn default_stop_sequences(tokenizer: &HFTokenizer) -> Vec<ProtoTokenSequence> {
    let mut stop_token_ids = ["<|im_end|>", "<|endoftext|>"]
        .into_iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect::<Vec<_>>();
    stop_token_ids.sort_unstable();
    stop_token_ids.dedup();
    stop_token_ids
        .into_iter()
        .map(|token_id| ProtoTokenSequence { tokens: vec![token_id] })
        .collect()
}

#[derive(Debug)]
struct DecodeStreamState {
    request_id: u64,
    input_tokens: usize,
    max_sampled_tokens: u32,
    started_at: Instant,
    first_chunk_at: Option<Instant>,
    last_chunk_at: Option<Instant>,
    inter_chunk_latencies: Vec<Duration>,
    streamed_text: String,
    sampled_token_ids: Vec<u32>,
    chunk_count: usize,
}

impl DecodeStreamState {
    fn new(request_id: u64, input_tokens: usize, max_sampled_tokens: u32, started_at: Instant) -> Self {
        Self {
            request_id,
            input_tokens,
            max_sampled_tokens,
            started_at,
            first_chunk_at: None,
            last_chunk_at: None,
            inter_chunk_latencies: Vec::new(),
            streamed_text: String::new(),
            sampled_token_ids: Vec::new(),
            chunk_count: 0,
        }
    }

    fn observe_chunk(&mut self, tokens: Vec<u32>, decoded_text: String) {
        if !tokens.is_empty() {
            let now = Instant::now();
            if self.first_chunk_at.is_none() {
                self.first_chunk_at = Some(now);
            }
            if let Some(previous) = self.last_chunk_at.replace(now) {
                self.inter_chunk_latencies.push(now.duration_since(previous));
            }
        }

        self.chunk_count += 1;
        self.sampled_token_ids.extend_from_slice(&tokens);
        self.streamed_text.push_str(&decoded_text);
    }

    fn finish(self, finished_at: Instant, final_text: String) -> DecodeResponse {
        let ttft = self
            .first_chunk_at
            .map(|first_chunk_at| first_chunk_at.duration_since(self.started_at));
        let decode_elapsed = self
            .first_chunk_at
            .map(|first_chunk_at| finished_at.duration_since(first_chunk_at));
        DecodeResponse::new(
            final_text,
            DecodePerfMetrics {
                request_id: self.request_id,
                input_tokens: self.input_tokens,
                max_sampled_tokens: self.max_sampled_tokens,
                sampled_tokens: self.sampled_token_ids.len(),
                chunk_count: self.chunk_count,
                elapsed: finished_at.duration_since(self.started_at),
                ttft,
                decode_elapsed,
                // The shared service metrics type still names this field inter_token_latencies.
                // The decode RPC currently streams token chunks, so this records chunk-arrival
                // intervals without inventing fake per-token timings inside a multi-token chunk.
                inter_token_latencies: self.inter_chunk_latencies,
            },
        )
    }
}
