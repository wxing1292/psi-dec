use std::time::Duration;

use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::SampledTokens;
use serde_json::json;

#[derive(Clone, Debug)]
pub struct ExecutorBatchPerfMetrics {
    pub main_elapsed: Duration,
    pub spec_elapsed: Duration,
    pub spec_passes: usize,
    pub main_gpu_elapsed: Option<Duration>,
    pub rejection_gpu_elapsed: Option<Duration>,
    pub spec_prepare_gpu_elapsed: Option<Duration>,
    pub spec_prefill_gpu_elapsed: Option<Duration>,
    pub spec_decode_gpu_elapsed: Option<Duration>,
    pub spec_gpu_elapsed: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct BatchRequestSummary {
    pub num_reqs: usize,
    pub num_input_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct BatchResponseSummary {
    pub num_spec_tokens: usize,
    pub num_verified_tokens: usize,
    pub num_spec_token_by_index: Vec<usize>,
    pub num_verified_token_by_index: Vec<usize>,
}

pub fn summarize_batch_device_request(batch: &BatchDeviceRequest) -> BatchRequestSummary {
    let mut summary = BatchRequestSummary {
        num_reqs: batch.dev_reqs.len(),
        num_input_tokens: 0,
    };

    for request in &batch.dev_reqs {
        summary.num_input_tokens += request.decoder_query_tokens.token_consumption();
    }

    summary
}

pub fn summarize_batch_device_response(batch: &BatchDeviceResponse) -> BatchResponseSummary {
    let mut summary = BatchResponseSummary {
        num_spec_tokens: 0,
        num_verified_tokens: 0,
        num_spec_token_by_index: Vec::new(),
        num_verified_token_by_index: Vec::new(),
    };

    for response in &batch.dev_resps {
        if let (QueryTokens::Decode { spec_tokens, .. }, SampledTokens::Decode { validated_tokens, .. }) =
            (&response.query_tokens, &response.sampled_tokens)
        {
            let num_spec_tokens = spec_tokens.len();
            let num_verified_tokens = validated_tokens.len();
            debug_assert!(
                num_verified_tokens <= num_spec_tokens,
                "executor response cannot verify more speculative tokens than the request provides"
            );

            summary.num_spec_tokens += num_spec_tokens;
            summary.num_verified_tokens += num_verified_tokens;
            summary
                .num_spec_token_by_index
                .resize(summary.num_spec_token_by_index.len().max(num_spec_tokens), 0);
            summary
                .num_verified_token_by_index
                .resize(summary.num_verified_token_by_index.len().max(num_spec_tokens), 0);
            for spec_token_index in 0..num_spec_tokens {
                if spec_token_index > num_verified_tokens {
                    break;
                }
                summary.num_spec_token_by_index[spec_token_index] += 1;
                if spec_token_index < num_verified_tokens {
                    summary.num_verified_token_by_index[spec_token_index] += 1;
                }
            }
        }
    }

    summary
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
    pub inter_chunk_latencies: Vec<Duration>,
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

    pub fn avg_inter_chunk_ms(&self) -> Option<f64> {
        if self.inter_chunk_latencies.is_empty() {
            return None;
        }
        Some(
            self.inter_chunk_latencies
                .iter()
                .map(|duration| duration.as_secs_f64() * 1000.0)
                .sum::<f64>()
                / self.inter_chunk_latencies.len() as f64,
        )
    }

    pub fn p50_inter_chunk_ms(&self) -> Option<f64> {
        percentile_ms(&self.inter_chunk_latencies, 0.50)
    }

    pub fn p95_inter_chunk_ms(&self) -> Option<f64> {
        percentile_ms(&self.inter_chunk_latencies, 0.95)
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
            "inter_chunk_avg_ms": self.avg_inter_chunk_ms(),
            "inter_chunk_p50_ms": self.p50_inter_chunk_ms(),
            "inter_chunk_p95_ms": self.p95_inter_chunk_ms(),
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

    use inference_runtime_core::compute::BatchDeviceResponse;
    use inference_runtime_core::compute::DeviceResponse;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::compute::SampledTokens;
    use inference_runtime_core::runtime::Token;
    use ordered_float::NotNan;

    use super::*;

    #[test]
    fn test_percentile_uses_nearest_rank_index() {
        let values = [1, 2, 3, 4, 5]
            .into_iter()
            .map(Duration::from_millis)
            .collect::<Vec<_>>();

        assert_eq!(percentile_ms(&values, 0.50), Some(3.0));
        assert_eq!(percentile_ms(&values, 0.95), Some(5.0));
    }

    #[test]
    fn test_json_contains_core_decode_fields() {
        let metrics = DecodePerfMetrics {
            request_id: 7,
            input_tokens: 3,
            max_sampled_tokens: 2,
            sampled_tokens: 2,
            chunk_count: 2,
            elapsed: Duration::from_millis(100),
            ttft: Some(Duration::from_millis(30)),
            decode_elapsed: Some(Duration::from_millis(70)),
            inter_chunk_latencies: vec![Duration::from_millis(20)],
        };

        let value: serde_json::Value = serde_json::from_str(&metrics.json_line()).unwrap();
        assert_eq!(value["type"], "decode_perf");
        assert_eq!(value["request_id"], 7);
        assert_eq!(value["sampled_tokens"], 2);
        assert_eq!(value["prompt_tokens_per_s"], 100.0);
    }

    #[test]
    fn test_summarizes_conditional_spec_acceptance_by_index() {
        let response = BatchDeviceResponse::new(
            0,
            [
                new_decode_response(7, &[11, 12, 13], 3),
                new_decode_response(8, &[21, 22], 1),
                new_decode_response(9, &[31, 32, 33], 0),
            ],
        );
        let response_summary = summarize_batch_device_response(&response);

        assert_eq!(response_summary.num_spec_tokens, 8);
        assert_eq!(response_summary.num_verified_tokens, 4);
        assert_eq!(response_summary.num_spec_token_by_index, vec![3, 2, 1]);
        assert_eq!(response_summary.num_verified_token_by_index, vec![2, 1, 1]);
    }

    fn new_decode_response(req_id: usize, spec_token_ids: &[u32], num_verified_tokens: usize) -> DeviceResponse {
        assert!(num_verified_tokens <= spec_token_ids.len());
        let spec_tokens = spec_token_ids.iter().copied().map(Token::new).collect::<Vec<_>>();
        DeviceResponse {
            req_id,
            query_tokens: QueryTokens::Decode {
                epoch: 0,
                token_index: 0,
                tokens: vec![Token::new(10)],
                spec_tokens: spec_tokens.clone(),
            },
            sampled_tokens: SampledTokens::Decode {
                epoch: 0,
                validated_tokens: spec_tokens[..num_verified_tokens].to_vec(),
                validated_probs: vec![NotNan::new(0.9).unwrap(); num_verified_tokens],
                sampled_token: Token::new(99),
                sampled_prob: NotNan::new(0.8).unwrap(),
                spec_tokens: Vec::new(),
                spec_probs: Vec::new(),
                spec_confidences: Vec::new(),
            },
        }
    }
}
