use std::time::Duration;

use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ModelOutputTiming;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::SampledTokens;
use serde_json::json;

#[derive(Clone, Debug)]
pub struct ExecutorBatchPerfMetrics {
    pub sampled_rows: usize,
    pub do_sample: bool,
    pub model_output_timing: Option<ModelOutputTiming>,
    pub total_elapsed: Duration,
    pub prepare_batch_elapsed: Duration,
    pub input_elapsed: Duration,
    pub model_elapsed: Duration,
    pub output_elapsed: Duration,
    pub commit_batch_elapsed: Duration,
}

#[derive(Clone, Debug)]
pub struct BatchRequestSummary {
    pub num_reqs: usize,
    pub prefill_reqs: usize,
    pub decode_reqs: usize,
    pub query_tokens: usize,
    pub max_query_tokens: usize,
    pub mtp_decode_reqs_with_spec: usize,
    pub mtp_input_spec_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct BatchResponseSummary {
    pub mtp_accepted_tokens: usize,
    pub sampled_tokens: usize,
    pub mtp_output_spec_tokens: usize,
}

pub fn summarize_batch_device_request(batch: &BatchDeviceRequest) -> BatchRequestSummary {
    let mut summary = BatchRequestSummary {
        num_reqs: batch.dev_reqs.len(),
        prefill_reqs: 0,
        decode_reqs: 0,
        query_tokens: 0,
        max_query_tokens: 0,
        mtp_decode_reqs_with_spec: 0,
        mtp_input_spec_tokens: 0,
    };

    for request in &batch.dev_reqs {
        let tokens = request.decoder_query_tokens.token_consumption();
        summary.query_tokens += tokens;
        summary.max_query_tokens = summary.max_query_tokens.max(tokens);
        match &request.decoder_query_tokens {
            QueryTokens::Prefill { .. } => summary.prefill_reqs += 1,
            QueryTokens::Decode { spec_tokens, .. } => {
                summary.decode_reqs += 1;
                if !spec_tokens.is_empty() {
                    summary.mtp_decode_reqs_with_spec += 1;
                    summary.mtp_input_spec_tokens += spec_tokens.len();
                }
            },
        }
    }

    summary
}

pub fn summarize_batch_device_response(batch: &BatchDeviceResponse) -> BatchResponseSummary {
    let mut summary = BatchResponseSummary {
        mtp_accepted_tokens: 0,
        sampled_tokens: 0,
        mtp_output_spec_tokens: 0,
    };

    for response in &batch.dev_resps {
        if let SampledTokens::Decode {
            validated_tokens,
            spec_tokens,
            ..
        } = &response.sampled_tokens
        {
            summary.mtp_accepted_tokens += validated_tokens.len();
            summary.sampled_tokens += validated_tokens.len() + 1;
            summary.mtp_output_spec_tokens += spec_tokens.len();
        }
    }

    summary
}

pub fn emit_executor_batch_perf_metrics(
    debug_logging: bool,
    model_name: &str,
    batch_seq: u64,
    batch_summary: BatchRequestSummary,
    response_summary: BatchResponseSummary,
    metrics: ExecutorBatchPerfMetrics,
) {
    if debug_logging {
        emit_executor_batch_perf_debug(model_name, batch_seq, &batch_summary, &response_summary, &metrics);
    } else {
        emit_executor_batch_perf_info(model_name, batch_seq, &batch_summary, &response_summary, &metrics);
    }
}

fn emit_executor_batch_perf_info(
    model_name: &str,
    batch_seq: u64,
    batch_summary: &BatchRequestSummary,
    response_summary: &BatchResponseSummary,
    metrics: &ExecutorBatchPerfMetrics,
) {
    let acceptance_rate = ratio(
        response_summary.mtp_accepted_tokens,
        batch_summary.mtp_input_spec_tokens,
    );
    tracing::info!(
        target: "inference-runtime-service::perf",
        phase = "executor.batch.perf",
        model = model_name,
        batch_seq,
        num_reqs = batch_summary.num_reqs,
        num_tokens = batch_summary.query_tokens,
        num_spec_tokens = batch_summary.mtp_input_spec_tokens,
        num_accepted_tokens = response_summary.mtp_accepted_tokens,
        num_sampled_tokens = response_summary.sampled_tokens,
        acceptance_rate,
        latency_ms = ms(metrics.total_elapsed),
        "executor batch perf"
    );
}

fn emit_executor_batch_perf_debug(
    model_name: &str,
    batch_seq: u64,
    batch_summary: &BatchRequestSummary,
    response_summary: &BatchResponseSummary,
    metrics: &ExecutorBatchPerfMetrics,
) {
    let mtp_rejected_tokens = batch_summary
        .mtp_input_spec_tokens
        .saturating_sub(response_summary.mtp_accepted_tokens);
    let acceptance_rate = ratio(
        response_summary.mtp_accepted_tokens,
        batch_summary.mtp_input_spec_tokens,
    );
    let rejection_rate = ratio(mtp_rejected_tokens, batch_summary.mtp_input_spec_tokens);

    tracing::debug!(
        target: "inference-runtime-service::perf",
        phase = "executor.batch.perf",
        model = model_name,
        batch_seq,
        num_reqs = batch_summary.num_reqs,
        num_tokens = batch_summary.query_tokens,
        num_spec_tokens = batch_summary.mtp_input_spec_tokens,
        num_accepted_tokens = response_summary.mtp_accepted_tokens,
        num_sampled_tokens = response_summary.sampled_tokens,
        acceptance_rate,
        latency_ms = ms(metrics.total_elapsed),
        num_prefill_reqs = batch_summary.prefill_reqs,
        num_decode_reqs = batch_summary.decode_reqs,
        max_num_tokens = batch_summary.max_query_tokens,
        num_mtp_reqs = batch_summary.mtp_decode_reqs_with_spec,
        num_rejected_tokens = mtp_rejected_tokens,
        num_output_spec_tokens = response_summary.mtp_output_spec_tokens,
        rejection_rate,
        sampled_rows = metrics.sampled_rows,
        do_sample = metrics.do_sample,
        model_output_main_replay_ms = metrics.model_output_timing.map(|timing| ms(timing.main_replay_elapsed)),
        model_output_main_output_replay_ms =
            metrics.model_output_timing.map(|timing| ms(timing.main_output_replay_elapsed)),
        model_output_sample_read_ms = metrics.model_output_timing.map(|timing| ms(timing.sample_read_elapsed)),
        model_output_rejection_build_ms =
            metrics.model_output_timing.map(|timing| ms(timing.rejection_build_elapsed)),
        model_output_rejection_read_ms =
            metrics.model_output_timing.map(|timing| ms(timing.rejection_read_elapsed)),
        model_output_mtp_build_ms = metrics.model_output_timing.map(|timing| ms(timing.mtp_build_elapsed)),
        model_output_mtp_replay_ms = metrics.model_output_timing.map(|timing| ms(timing.mtp_replay_elapsed)),
        model_output_mtp_read_ms = metrics.model_output_timing.map(|timing| ms(timing.mtp_read_elapsed)),
        model_output_mtp_modules = metrics.model_output_timing.map(|timing| timing.mtp_modules),
        prepare_batch_ms = ms(metrics.prepare_batch_elapsed),
        input_ms = ms(metrics.input_elapsed),
        model_ms = ms(metrics.model_elapsed),
        output_ms = ms(metrics.output_elapsed),
        commit_batch_ms = ms(metrics.commit_batch_elapsed),
        "executor batch perf"
    );
}

#[derive(Clone, Debug)]
pub struct DecodePerfMetrics {
    pub request_id: u64,
    pub input_tokens: usize,
    pub max_sampled_tokens: u32,
    pub sampled_tokens: usize,
    pub chunk_count: usize,
    pub elapsed: Duration,
    pub ttft: Option<Duration>,
    pub decode_elapsed: Option<Duration>,
    pub inter_token_latencies: Vec<Duration>,
}

impl DecodePerfMetrics {
    pub fn prompt_tokens_per_s(&self) -> Option<f64> {
        rate(self.input_tokens, self.ttft?)
    }

    pub fn overall_tokens_per_s(&self) -> Option<f64> {
        rate(self.sampled_tokens, self.elapsed)
    }

    pub fn decode_tokens_per_s(&self) -> Option<f64> {
        rate(self.sampled_tokens, self.decode_elapsed?)
    }

    pub fn avg_inter_token_ms(&self) -> Option<f64> {
        if self.inter_token_latencies.is_empty() {
            return None;
        }
        Some(
            self.inter_token_latencies
                .iter()
                .map(|duration| duration.as_secs_f64() * 1000.0)
                .sum::<f64>()
                / self.inter_token_latencies.len() as f64,
        )
    }

    pub fn p50_inter_token_ms(&self) -> Option<f64> {
        percentile_ms(&self.inter_token_latencies, 0.50)
    }

    pub fn p95_inter_token_ms(&self) -> Option<f64> {
        percentile_ms(&self.inter_token_latencies, 0.95)
    }

    pub fn json_line(&self) -> String {
        json!({
            "type": "decode_perf",
            "request_id": self.request_id,
            "input_tokens": self.input_tokens,
            "max_sampled_tokens": self.max_sampled_tokens,
            "sampled_tokens": self.sampled_tokens,
            "chunk_count": self.chunk_count,
            "elapsed_ms": ms(self.elapsed),
            "ttft_ms": self.ttft.map(ms),
            "decode_elapsed_ms": self.decode_elapsed.map(ms),
            "prompt_tokens_per_s": self.prompt_tokens_per_s(),
            "overall_tokens_per_s": self.overall_tokens_per_s(),
            "decode_tokens_per_s": self.decode_tokens_per_s(),
            "inter_token_avg_ms": self.avg_inter_token_ms(),
            "inter_token_p50_ms": self.p50_inter_token_ms(),
            "inter_token_p95_ms": self.p95_inter_token_ms(),
        })
        .to_string()
    }
}

pub fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn rate(tokens: usize, duration: Duration) -> Option<f64> {
    let seconds = duration.as_secs_f64();
    (seconds > 0.0).then_some(tokens as f64 / seconds)
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile_ms(values: &[Duration], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut millis = values.iter().map(|duration| ms(*duration)).collect::<Vec<_>>();
    millis.sort_by(|left, right| left.total_cmp(right));
    let index = ((millis.len() - 1) as f64 * percentile).round() as usize;
    millis.get(index).copied()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use inference_runtime_core::compute::BatchDeviceRequest;
    use inference_runtime_core::compute::BatchDeviceResponse;
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::DeviceResponse;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::compute::SampledTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::Token;
    use ordered_float::NotNan;

    use super::*;

    #[test]
    fn percentile_uses_nearest_rank_index() {
        let values = [1, 2, 3, 4, 5]
            .into_iter()
            .map(Duration::from_millis)
            .collect::<Vec<_>>();

        assert_eq!(percentile_ms(&values, 0.50), Some(3.0));
        assert_eq!(percentile_ms(&values, 0.95), Some(5.0));
    }

    #[test]
    fn json_contains_core_decode_fields() {
        let metrics = DecodePerfMetrics {
            request_id: 7,
            input_tokens: 3,
            max_sampled_tokens: 2,
            sampled_tokens: 2,
            chunk_count: 2,
            elapsed: Duration::from_millis(100),
            ttft: Some(Duration::from_millis(30)),
            decode_elapsed: Some(Duration::from_millis(70)),
            inter_token_latencies: vec![Duration::from_millis(20)],
        };

        let value: serde_json::Value = serde_json::from_str(&metrics.json_line()).unwrap();
        assert_eq!(value["type"], "decode_perf");
        assert_eq!(value["request_id"], 7);
        assert_eq!(value["sampled_tokens"], 2);
        assert_eq!(value["prompt_tokens_per_s"], 100.0);
    }

    #[test]
    fn summarizes_mtp_acceptance_fields() {
        let request = BatchDeviceRequest::new(
            0,
            [DeviceRequest::new(
                7,
                0,
                QueryTokens::Decode {
                    epoch: 0,
                    token_index: 3,
                    tokens: vec![Token::new(10)],
                    spec_tokens: vec![Token::new(11), Token::new(12)],
                },
                DecoderSyncBlocks::new(0, vec![], vec![]),
                SamplingConfig::default(),
            )],
        );
        let response = BatchDeviceResponse::new(
            0,
            [DeviceResponse {
                req_id: 7,
                query_tokens: QueryTokens::Decode {
                    epoch: 0,
                    token_index: 3,
                    tokens: vec![Token::new(10)],
                    spec_tokens: vec![Token::new(11), Token::new(12)],
                },
                sampled_tokens: SampledTokens::Decode {
                    epoch: 0,
                    validated_tokens: vec![Token::new(11)],
                    validated_probs: vec![NotNan::new(0.9).unwrap()],
                    sampled_token: Token::new(13),
                    sampled_prob: NotNan::new(0.8).unwrap(),
                    spec_tokens: vec![Token::new(14), Token::new(15)],
                    spec_probs: vec![NotNan::new(0.6).unwrap(), NotNan::new(0.7).unwrap()],
                },
            }],
        );

        let request_summary = summarize_batch_device_request(&request);
        let response_summary = summarize_batch_device_response(&response);

        assert_eq!(request_summary.mtp_decode_reqs_with_spec, 1);
        assert_eq!(request_summary.mtp_input_spec_tokens, 2);
        assert_eq!(response_summary.mtp_accepted_tokens, 1);
        assert_eq!(response_summary.sampled_tokens, 2);
        assert_eq!(response_summary.mtp_output_spec_tokens, 2);
        assert_eq!(
            ratio(
                response_summary.mtp_accepted_tokens,
                request_summary.mtp_input_spec_tokens
            ),
            0.5
        );
    }
}
