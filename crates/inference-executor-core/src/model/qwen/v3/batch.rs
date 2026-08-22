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
use crate::sampling::SpecMicrobatch;

/// Qwen3 Main payload for one executor request, independent of its compute sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Microbatch {
    // One batch with prefill request 0 and decode request 1 can look like:
    //
    // cu_tokens:         [0, 3, 4]
    // flat index:         0    1    2 | 3  (a coordinate, not stored)
    // flat_token_ids:   101  102  103 | 201
    // token_indices:     10             | 20  (one request-absolute start per request)
    // flat_sample_mask:  F    F    F   |  T
    //
    // GQA expands the request starts into flat_token_indices:
    // [10, 11, 12, 20].
    req_slots: Vec<u32>,
    token_indices: Vec<u32>,
    flat_token_ids: Vec<i32>,
    cu_tokens: Vec<u32>,
    sampler_configs: Vec<SamplerConfig>,
    num_spec_tokens: Vec<u32>,
    flat_sample_mask: Vec<bool>,
}

impl SpecMicrobatch for Qwen3Microbatch {
    fn num_reqs(&self) -> usize {
        self.num_reqs()
    }

    fn is_decode_req(&self, req_index: usize) -> bool {
        self.is_decode_req(req_index)
    }

    fn num_spec_tokens(&self, req_index: usize) -> u32 {
        self.num_spec_tokens(req_index)
    }

    fn req_slots(&self) -> &[u32] {
        self.req_slots()
    }

    fn token_indices(&self) -> &[u32] {
        self.token_indices()
    }

    fn cu_tokens(&self) -> &[u32] {
        self.cu_tokens()
    }

    fn flat_token_ids(&self) -> &[i32] {
        self.flat_token_ids()
    }
}

impl Qwen3Microbatch {
    fn new(
        req_slots: Vec<u32>,
        token_indices: Vec<u32>,
        flat_token_ids: Vec<i32>,
        cu_tokens: Vec<u32>,
        sampler_configs: Vec<SamplerConfig>,
        num_spec_tokens: Vec<u32>,
        flat_sample_mask: Vec<bool>,
    ) -> Self {
        validate_batch_fields(&req_slots, &token_indices, &flat_token_ids, &cu_tokens);
        assert_eq!(
            sampler_configs.len(),
            req_slots.len(),
            "Qwen3 request requires one sampler_config entry per request"
        );
        assert_eq!(
            num_spec_tokens.len(),
            req_slots.len(),
            "Qwen3 request requires one speculative-token count per request"
        );
        validate_flat_sample_mask(&cu_tokens, &num_spec_tokens, &flat_sample_mask);
        Self {
            req_slots,
            token_indices,
            flat_token_ids,
            cu_tokens,
            sampler_configs,
            num_spec_tokens,
            flat_sample_mask,
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
        let mut num_spec_tokens = Vec::with_capacity(requests.len());
        let mut flat_sample_mask = Vec::new();
        cu_tokens.push(0);

        for request in requests {
            let token_index: u32 = request
                .decoder_query_tokens
                .token_index()
                .try_into()
                .expect("Qwen3 request token index must fit u32");
            let q_len: u32 = request
                .token_cost()
                .try_into()
                .expect("Qwen3 request q_len must fit u32");
            let num_req_spec_tokens: u32 = request
                .decoder_query_tokens
                .num_spec_tokens()
                .try_into()
                .expect("Qwen3 request speculative-token count must fit u32");
            let num_sample_rows = match request.decoder_query_tokens {
                QueryTokens::Prefill { .. } => 0,
                QueryTokens::Decode { .. } => {
                    num_req_spec_tokens
                        .checked_add(1)
                        .expect("Qwen3 Main sample-row count must fit u32")
                },
            };
            assert!(
                num_sample_rows <= q_len,
                "Qwen3 Main sample rows must fit the request query width"
            );
            let first_sample_offset = q_len
                .checked_sub(num_sample_rows)
                .expect("Qwen3 Main sample suffix must fit q_len");
            flat_sample_mask.extend((0..q_len).map(|token_offset| token_offset >= first_sample_offset));
            if matches!(request.decoder_query_tokens, QueryTokens::Prefill { .. }) {
                assert_eq!(
                    num_req_spec_tokens, 0,
                    "Qwen3 prefill request must not contain speculative tokens"
                );
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
            num_spec_tokens.push(num_req_spec_tokens);
        }

        Self::new(
            req_slots,
            token_indices,
            flat_token_ids,
            cu_tokens,
            sampler_configs,
            num_spec_tokens,
            flat_sample_mask,
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

    pub fn num_reqs(&self) -> usize {
        self.req_slots.len()
    }

    pub fn q_len(&self, req_index: usize) -> u32 {
        self.cu_tokens[req_index + 1] - self.cu_tokens[req_index]
    }

    pub fn sampler_configs(&self) -> &[SamplerConfig] {
        &self.sampler_configs
    }

    pub fn num_spec_tokens(&self, req_index: usize) -> u32 {
        self.num_spec_tokens[req_index]
    }

    pub fn flat_sample_mask(&self) -> &[bool] {
        &self.flat_sample_mask
    }

    pub fn num_main_hidden_states_for_req(&self, req_index: usize) -> usize {
        let token_start = self.cu_tokens[req_index] as usize;
        let token_end = self.cu_tokens[req_index + 1] as usize;
        self.flat_sample_mask[token_start..token_end]
            .iter()
            .filter(|&&sample| sample)
            .count()
    }

    pub fn is_decode_req(&self, req_index: usize) -> bool {
        let token_end = self.cu_tokens[req_index + 1] as usize;
        self.flat_sample_mask[token_end - 1]
    }

    pub fn has_spec_tokens(&self) -> bool {
        self.num_spec_tokens.iter().any(|&count| count > 0)
    }

    pub fn total_tokens(&self) -> usize {
        self.flat_token_ids.len()
    }
}

/// One runtime-scheduled Qwen3 Main batch.
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen3DecodeDecision {
    pub validated_tokens: Vec<u32>,
    pub validated_probs: Vec<f32>,
    pub sampled_token: u32,
    pub sampled_prob: f32,
    pub spec_tokens: Vec<u32>,
    pub spec_probs: Vec<f32>,
    pub spec_confidences: Vec<f32>,
}

pub fn gather_flat_indices(microbatch: &Qwen3Microbatch) -> Vec<u32> {
    microbatch
        .flat_sample_mask
        .iter()
        .enumerate()
        .filter(|&(_, &sample)| sample)
        .map(|(flat_index, _)| {
            flat_index
                .try_into()
                .expect("Qwen3 gathered flat token index must fit u32")
        })
        .collect()
}

pub fn num_main_output_rows(microbatch: &Qwen3Microbatch) -> usize {
    microbatch.flat_sample_mask.iter().filter(|&&sample| sample).count()
}

/// Returns the logical output-token position for each Main output row.
pub fn sample_token_positions(microbatch: &Qwen3Microbatch) -> Vec<u32> {
    let mut positions = Vec::with_capacity(num_main_output_rows(microbatch));
    for req_index in 0..microbatch.num_reqs() {
        let token_start = microbatch.cu_tokens[req_index] as usize;
        let token_end = microbatch.cu_tokens[req_index + 1] as usize;
        positions.extend(
            microbatch.flat_sample_mask[token_start..token_end]
                .iter()
                .enumerate()
                .filter(|&(_, &sample)| sample)
                .map(|(token_offset, _)| {
                    microbatch.token_indices[req_index]
                        .checked_add(token_offset.try_into().expect("Qwen3 sample token offset must fit u32"))
                        .and_then(|position| position.checked_add(1))
                        .expect("Qwen3 sample position overflow")
                }),
        );
    }
    positions
}

pub fn sample_sampler_configs(microbatch: &Qwen3Microbatch) -> Vec<SamplerConfig> {
    let mut configs = Vec::with_capacity(num_main_output_rows(microbatch));
    for req_index in 0..microbatch.num_reqs() {
        configs.extend(std::iter::repeat_n(
            microbatch.sampler_configs[req_index],
            microbatch.num_main_hidden_states_for_req(req_index),
        ));
    }
    assert!(!configs.is_empty(), "Qwen3 sampler configs require Main output rows");
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
                ..Qwen3DecodeDecision::default()
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
                    assert_eq!(decision.spec_tokens.len(), decision.spec_probs.len());
                    assert_eq!(decision.spec_tokens.len(), decision.spec_confidences.len());
                    SampledTokens::Decode {
                        epoch: core_req.decoder_query_tokens.epoch(),
                        validated_tokens: decision.validated_tokens.into_iter().map(Token::new).collect(),
                        validated_probs: decision.validated_probs.into_iter().map(finite_probability).collect(),
                        sampled_token: Token::new(decision.sampled_token),
                        sampled_prob: finite_probability(decision.sampled_prob),
                        spec_tokens: decision.spec_tokens.into_iter().map(Token::new).collect(),
                        spec_probs: decision.spec_probs.into_iter().map(finite_probability).collect(),
                        spec_confidences: decision.spec_confidences.into_iter().map(finite_probability).collect(),
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

fn validate_flat_sample_mask(cu_tokens: &[u32], num_spec_tokens: &[u32], flat_sample_mask: &[bool]) {
    assert_eq!(
        flat_sample_mask.len(),
        *cu_tokens.last().expect("Qwen3 request requires cu_tokens") as usize,
        "Qwen3 sample mask must match flattened tokens"
    );
    for req_index in 0..num_spec_tokens.len() {
        let token_start = cu_tokens[req_index] as usize;
        let token_end = cu_tokens[req_index + 1] as usize;
        let sample_count = flat_sample_mask[token_start..token_end]
            .iter()
            .filter(|&&sample| sample)
            .count();
        assert!(
            sample_count == 0 || sample_count == num_spec_tokens[req_index] as usize + 1,
            "Qwen3 decode request requires one Main sample row per speculative token plus one final row"
        );
        assert!(
            flat_sample_mask[token_start..token_end]
                .windows(2)
                .all(|pair| !pair[0] || pair[1]),
            "Qwen3 sample mask must be a request-local suffix"
        );
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
