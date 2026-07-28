use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DevReq;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::runtime::RawComputeSlotSeq;
use inference_runtime_core::runtime::Token;
use ordered_float::NotNan;

use crate::sampling::SamplerConfig;

/// Qwen3 target payload for one executor request, independent of its compute sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Microbatch {
    // One batch with prefill request 0 and decode request 1 can look like:
    //
    // cu_tokens:         [0, 3, 4]
    // flat index:         0    1    2 | 3  (a coordinate, not stored)
    // flat_token_ids:   101  102  103 | 201
    // token_indices:     10             | 20  (one request-absolute start per request)
    // decode_req_indices:                1
    //
    // GQA expands the request starts into flat_token_indices:
    // [10, 11, 12, 20].
    req_slots: Vec<u32>,
    token_indices: Vec<u32>,
    flat_token_ids: Vec<i32>,
    cu_tokens: Vec<u32>,
    sampler_configs: Vec<SamplerConfig>,
    decode_req_indices: Vec<usize>,
}

impl Qwen3Microbatch {
    fn new(
        req_slots: Vec<u32>,
        token_indices: Vec<u32>,
        flat_token_ids: Vec<i32>,
        cu_tokens: Vec<u32>,
        sampler_configs: Vec<SamplerConfig>,
        decode_req_indices: Vec<usize>,
    ) -> Self {
        validate_batch_fields(&req_slots, &token_indices, &flat_token_ids, &cu_tokens);
        assert_eq!(
            sampler_configs.len(),
            req_slots.len(),
            "Qwen3 request requires one sampler_config entry per request"
        );
        validate_decode_req_indices(req_slots.len(), &decode_req_indices);
        Self {
            req_slots,
            token_indices,
            flat_token_ids,
            cu_tokens,
            sampler_configs,
            decode_req_indices,
        }
    }

    fn from_requests(requests: &[DeviceRequest], sampler_configs: Vec<SamplerConfig>) -> Self {
        assert_eq!(
            sampler_configs.len(),
            requests.len(),
            "Qwen3 request requires one sampler config per request"
        );

        let mut req_slots = Vec::with_capacity(requests.len());
        let mut token_indices = Vec::with_capacity(requests.len());
        let mut flat_token_ids = Vec::new();
        let mut cu_tokens = Vec::with_capacity(
            requests
                .len()
                .checked_add(1)
                .expect("Qwen3 cumulative-token capacity must fit usize"),
        );
        let mut decode_req_indices = Vec::new();
        cu_tokens.push(0);

        for (req_index, request) in requests.iter().enumerate() {
            let token_index: u32 = request
                .decoder_query_tokens
                .token_index()
                .try_into()
                .expect("Qwen3 request token index must fit u32");
            let q_len: u32 = request
                .token_cost()
                .try_into()
                .expect("Qwen3 request q_len must fit u32");
            match &request.decoder_query_tokens {
                QueryTokens::Prefill { .. } => {},
                QueryTokens::Decode { spec_tokens, .. } => {
                    assert!(
                        spec_tokens.is_empty(),
                        "Qwen3 target-only batch does not accept speculative input tokens"
                    );
                    decode_req_indices.push(req_index);
                },
            }
            assert!(q_len > 0, "Qwen3 batch requires positive q_len");

            let token_ids: Vec<i32> = request
                .decoder_query_tokens
                .token_ids_by_lane(0)
                .map(|token_id| {
                    token_id
                        .try_into()
                        .expect("Qwen3 input token ID must fit the model i32 token domain")
                })
                .collect::<Vec<_>>();
            assert_eq!(
                token_ids.len(),
                q_len as usize,
                "Qwen3 token lane must match request width"
            );
            flat_token_ids.extend(token_ids);
            cu_tokens.push(
                flat_token_ids
                    .len()
                    .try_into()
                    .expect("Qwen3 cumulative token count must fit u32"),
            );
            req_slots.push(request.req_slot);
            token_indices.push(token_index);
        }

        Self::new(
            req_slots,
            token_indices,
            flat_token_ids,
            cu_tokens,
            sampler_configs,
            decode_req_indices,
        )
    }

    pub fn req_slots(&self) -> &[u32] {
        &self.req_slots
    }

    pub fn token_indices(&self) -> &[u32] {
        &self.token_indices
    }

    pub fn flat_token_ids(&self) -> &[i32] {
        &self.flat_token_ids
    }

    pub fn cu_tokens(&self) -> &[u32] {
        &self.cu_tokens
    }

    pub fn total_tokens(&self) -> usize {
        self.flat_token_ids.len()
    }
}

/// One runtime-scheduled Qwen3 target batch.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ModelBatchRequest {
    compute_seq: RawComputeSlotSeq,
    microbatch: Qwen3Microbatch,
}

impl Qwen3ModelBatchRequest {
    pub fn from_core_batch(core_batch_req: &BatchDeviceRequest, sampler_configs: Vec<SamplerConfig>) -> Self {
        Self {
            compute_seq: core_batch_req.seq,
            microbatch: Qwen3Microbatch::from_requests(&core_batch_req.dev_reqs, sampler_configs),
        }
    }

    pub fn compute_seq(&self) -> RawComputeSlotSeq {
        self.compute_seq
    }

    pub fn microbatch(&self) -> &Qwen3Microbatch {
        &self.microbatch
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3SampledTokens {
    token_ids: Vec<i32>,
    token_probs: Vec<f32>,
}

impl Qwen3SampledTokens {
    pub fn new(token_ids: Vec<i32>, token_probs: Vec<f32>) -> Self {
        assert_eq!(
            token_ids.len(),
            token_probs.len(),
            "Qwen3 sampled tokens require one probability per token"
        );
        Self { token_ids, token_probs }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3DecodeDecision {
    pub sampled_token: u32,
    pub sampled_prob: f32,
}

pub fn gather_flat_indices(microbatch: &Qwen3Microbatch) -> Vec<u32> {
    microbatch
        .decode_req_indices
        .iter()
        .map(|&req_index| {
            microbatch.cu_tokens[req_index + 1]
                .checked_sub(1)
                .expect("Qwen3 decode request must contain a token")
        })
        .collect()
}

pub fn num_target_hidden_states(microbatch: &Qwen3Microbatch) -> usize {
    microbatch.decode_req_indices.len()
}

/// Returns the logical output-token position for each target hidden state.
pub fn sample_token_positions(microbatch: &Qwen3Microbatch) -> Vec<u32> {
    microbatch
        .decode_req_indices
        .iter()
        .map(|&req_index| {
            let q_len = microbatch.cu_tokens[req_index + 1] - microbatch.cu_tokens[req_index];
            microbatch.token_indices[req_index]
                .checked_add(q_len)
                .expect("Qwen3 sample position overflow")
        })
        .collect()
}

pub fn sample_sampler_configs(microbatch: &Qwen3Microbatch) -> Vec<SamplerConfig> {
    let configs = microbatch
        .decode_req_indices
        .iter()
        .map(|&req_index| microbatch.sampler_configs[req_index])
        .collect::<Vec<_>>();
    assert!(
        !configs.is_empty(),
        "Qwen3 sampler configs require target hidden states"
    );
    configs
}

pub fn sample_decisions_from_sampled_tokens(sampled_tokens: &Qwen3SampledTokens) -> Vec<Qwen3DecodeDecision> {
    sampled_tokens
        .token_ids
        .iter()
        .zip(&sampled_tokens.token_probs)
        .map(|(&token_id, &token_prob)| {
            Qwen3DecodeDecision {
                sampled_token: token_id
                    .try_into()
                    .expect("Qwen3 sampled token ID must be non-negative and fit u32"),
                sampled_prob: token_prob,
            }
        })
        .collect()
}

pub fn to_core_batch_resp(
    core_batch_req: BatchDeviceRequest,
    decisions: Vec<Qwen3DecodeDecision>,
) -> BatchDeviceResponse {
    let seq = core_batch_req.seq;
    let mut decisions = decisions.into_iter();
    let core_resps = core_batch_req
        .dev_reqs
        .into_iter()
        .map(|core_req| {
            let sampled_tokens = match &core_req.decoder_query_tokens {
                QueryTokens::Prefill { .. } => {
                    SampledTokens::Prefill {
                        epoch: core_req.decoder_query_tokens.epoch(),
                    }
                },
                QueryTokens::Decode { .. } => {
                    let decision = decisions
                        .next()
                        .expect("Qwen3 service requires one decision per sampled request");
                    SampledTokens::Decode {
                        epoch: core_req.decoder_query_tokens.epoch(),
                        validated_tokens: Vec::new(),
                        validated_probs: Vec::new(),
                        sampled_token: Token::new(decision.sampled_token),
                        sampled_prob: finite_probability(decision.sampled_prob),
                        spec_tokens: Vec::new(),
                        spec_probs: Vec::new(),
                    }
                },
            };
            DeviceResponse {
                req_id: core_req.req_id,
                query_tokens: core_req.decoder_query_tokens.clone(),
                sampled_tokens,
            }
        })
        .collect::<Vec<_>>();
    assert!(
        decisions.next().is_none(),
        "Qwen3 service received more decisions than decode requests"
    );
    BatchDeviceResponse::new(seq, core_resps)
}

fn finite_probability(probability: f32) -> NotNan<f32> {
    NotNan::new(probability).expect("Qwen3 probability should be finite and non-NaN")
}

fn validate_batch_fields(req_slots: &[u32], token_indices: &[u32], flat_token_ids: &[i32], cu_tokens: &[u32]) {
    let num_reqs = req_slots.len();
    assert_eq!(
        token_indices.len(),
        num_reqs,
        "Qwen3 request requires one token_index entry per request"
    );
    assert_eq!(
        cu_tokens.len(),
        num_reqs + 1,
        "Qwen3 request requires cu_tokens length to equal num_reqs + 1"
    );
    assert_eq!(cu_tokens[0], 0, "Qwen3 request requires cu_tokens[0] == 0");
    assert_eq!(
        *cu_tokens.last().expect("Qwen3 request requires cu_tokens"),
        u32::try_from(flat_token_ids.len()).expect("Qwen3 flat token count must fit u32"),
        "Qwen3 request requires cu_tokens.last() == flat_token_ids.len()"
    );
    for req_index in 0..num_reqs {
        let start = cu_tokens[req_index];
        let end = cu_tokens[req_index + 1];
        assert!(
            start < end,
            "Qwen3 request requires strictly increasing cu_tokens, req_index={req_index}, start={start}, end={end}"
        );
    }
}

fn validate_decode_req_indices(num_reqs: usize, decode_req_indices: &[usize]) {
    let mut previous_req_index = None;
    for &req_index in decode_req_indices {
        assert!(
            req_index < num_reqs,
            "Qwen3 decode request index {req_index} exceeds batch request count {num_reqs}"
        );
        if let Some(previous_req_index) = previous_req_index {
            assert!(
                previous_req_index < req_index,
                "Qwen3 decode request indices must be strictly increasing"
            );
        }
        previous_req_index = Some(req_index);
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
