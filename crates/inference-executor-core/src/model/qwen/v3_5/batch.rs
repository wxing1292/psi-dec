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

use crate::attn::gdn::state::GDNStateTxn;
use crate::sampling::SamplerConfig;

/// Model payload for one executor request, independent of its compute sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Microbatch {
    // One batch with prefill request 0 and decode request 1 can look like:
    //
    // cu_tokens:        [0, 3, 6]
    // flat index:        0    1    2 | 3    4    5  (a coordinate, not stored)
    // flat_token_ids:  101  102  103 | 201  202  203
    // token_indices:    10             | 20             (one request-absolute start per request)
    // flat_sample_mask:  F    F    F   |  T    T    T
    //
    // GQA expands the request starts into flat_token_indices:
    // [10, 11, 12, 20, 21, 22].
    req_slots: Vec<u32>,
    block_indices: Vec<usize>,
    token_indices: Vec<u32>,
    flat_token_ids_by_lane: Vec<Vec<i32>>,
    cu_tokens_by_lane: Vec<Vec<u32>>,
    gdn_state_txns: Vec<GDNStateTxn>,
    gdn_state_page_ids_by_req: Vec<Vec<Vec<u32>>>,
    sampler_configs: Vec<SamplerConfig>,
    flat_sample_mask: Vec<bool>,
}

impl Qwen35Microbatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        req_slots: Vec<u32>,
        block_indices: Vec<usize>,
        token_indices: Vec<u32>,
        flat_token_ids: Vec<i32>,
        cu_tokens: Vec<u32>,
        gdn_state_txns: Vec<GDNStateTxn>,
        gdn_state_page_ids_by_req: Vec<Vec<Vec<u32>>>,
        sampler_configs: Vec<SamplerConfig>,
        flat_sample_mask: Vec<bool>,
    ) -> Self {
        validate_batch_fields(&req_slots, &block_indices, &token_indices, &flat_token_ids, &cu_tokens);
        validate_gdn_state_txns(&token_indices, &cu_tokens, &gdn_state_txns);
        assert_eq!(
            gdn_state_page_ids_by_req.len(),
            req_slots.len(),
            "qwen3.5 request requires one gdn_state_page_ids entry per request"
        );
        assert_eq!(
            sampler_configs.len(),
            req_slots.len(),
            "qwen3.5 request requires one sampler_config entry per request"
        );
        validate_flat_sample_mask(&cu_tokens, &gdn_state_txns, &flat_sample_mask);

        Self {
            req_slots,
            block_indices,
            token_indices,
            flat_token_ids_by_lane: vec![flat_token_ids],
            cu_tokens_by_lane: vec![cu_tokens],
            gdn_state_txns,
            gdn_state_page_ids_by_req,
            sampler_configs,
            flat_sample_mask,
        }
    }

    pub fn from_requests(requests: &[DeviceRequest], sampler_configs: Vec<SamplerConfig>) -> Self {
        Self::from_requests_with_mtp_modules(requests, 0, sampler_configs)
    }

    pub fn from_requests_with_mtp_modules(
        requests: &[DeviceRequest],
        num_mtp_modules: usize,
        sampler_configs: Vec<SamplerConfig>,
    ) -> Self {
        assert_eq!(
            sampler_configs.len(),
            requests.len(),
            "qwen3.5 request requires one sampler config per request"
        );

        let mut req_slots = Vec::with_capacity(requests.len());
        let mut block_indices = Vec::with_capacity(requests.len());
        let mut token_indices = Vec::with_capacity(requests.len());
        let num_lanes = num_mtp_modules
            .checked_add(1)
            .expect("qwen3.5 MTP lane count must fit usize");
        let cu_capacity = requests
            .len()
            .checked_add(1)
            .expect("qwen3.5 cumulative-token capacity must fit usize");
        let mut flat_token_ids_by_lane = vec![Vec::new(); num_lanes];
        let mut cu_tokens_by_lane = vec![Vec::with_capacity(cu_capacity); num_lanes];
        let mut gdn_state_txns = Vec::with_capacity(requests.len());
        let mut gdn_state_page_ids_by_req = Vec::with_capacity(requests.len());
        let mut flat_sample_mask = Vec::new();

        for cu_tokens in &mut cu_tokens_by_lane {
            cu_tokens.push(0);
        }
        for request in requests {
            let token_index: u32 = request
                .decoder_query_tokens
                .token_index()
                .try_into()
                .expect("qwen3.5 request token index must fit u32");
            let q_len: u32 = request
                .token_cost()
                .try_into()
                .expect("qwen3.5 request q_len must fit u32");
            let num_req_spec_tokens: u32 = request
                .decoder_query_tokens
                .num_spec_tokens()
                .try_into()
                .expect("qwen3.5 request speculative-token count must fit u32");
            let num_req_target_hidden_states = match request.decoder_query_tokens {
                QueryTokens::Prefill { .. } => 0,
                QueryTokens::Decode { .. } => {
                    num_req_spec_tokens
                        .checked_add(1)
                        .expect("qwen3.5 target hidden-state count must fit u32")
                },
            };
            let gdn_state_page_ids = request
                .decoder_sync_blocks
                .state_page_ids()
                .first()
                .cloned()
                .unwrap_or_default();

            assert!(q_len > 0, "qwen3.5 batch requires positive q_len");
            assert!(
                num_req_target_hidden_states <= q_len,
                "qwen3.5 sample tokens exceed request query tokens"
            );
            let first_target_offset = q_len
                .checked_sub(num_req_target_hidden_states)
                .expect("qwen3.5 target hidden-state suffix must fit q_len");
            flat_sample_mask.extend((0..q_len).map(|token_offset| token_offset >= first_target_offset));

            for lane in 0..=num_mtp_modules {
                let needs_runtime_lane =
                    lane == 0 || matches!(request.decoder_query_tokens, QueryTokens::Prefill { .. });
                let token_ids: Vec<i32> = if needs_runtime_lane {
                    request
                        .decoder_query_tokens
                        .token_ids_by_lane(lane)
                        .map(|token_id| {
                            token_id
                                .try_into()
                                .expect("qwen3.5 input token ID must fit the model i32 token domain")
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                if needs_runtime_lane {
                    assert_eq!(
                        u32::try_from(token_ids.len()).expect("qwen3.5 token-lane length must fit u32"),
                        q_len,
                        "qwen3.5 token lane must match request width for main and prefill MTP rows"
                    );
                }
                flat_token_ids_by_lane[lane].extend(token_ids);
                cu_tokens_by_lane[lane].push(
                    flat_token_ids_by_lane[lane]
                        .len()
                        .try_into()
                        .expect("qwen3.5 cumulative token count must fit u32"),
                );
            }

            req_slots.push(request.req_slot);
            block_indices.push(request.decoder_sync_blocks.block_index());
            token_indices.push(token_index);
            gdn_state_txns.push(GDNStateTxn::new(token_index, q_len, num_req_spec_tokens));
            gdn_state_page_ids_by_req.push(gdn_state_page_ids);
        }

        let batch = Self {
            req_slots,
            block_indices,
            token_indices,
            flat_token_ids_by_lane,
            cu_tokens_by_lane,
            gdn_state_txns,
            gdn_state_page_ids_by_req,
            sampler_configs,
            flat_sample_mask,
        };
        validate_batch_fields(
            &batch.req_slots,
            &batch.block_indices,
            &batch.token_indices,
            batch.flat_token_ids(),
            batch.cu_tokens(),
        );
        validate_gdn_state_txns(batch.token_indices(), batch.cu_tokens(), batch.gdn_state_txns());
        validate_flat_sample_mask(batch.cu_tokens(), batch.gdn_state_txns(), batch.flat_sample_mask());
        batch
    }

    pub fn num_reqs(&self) -> usize {
        self.req_slots.len()
    }

    pub fn req_slots(&self) -> &[u32] {
        &self.req_slots
    }

    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }

    pub fn token_indices(&self) -> &[u32] {
        &self.token_indices
    }

    pub fn flat_token_ids(&self) -> &[i32] {
        &self.flat_token_ids_by_lane[0]
    }

    pub fn cu_tokens(&self) -> &[u32] {
        &self.cu_tokens_by_lane[0]
    }

    pub fn token_ids_for_lane(&self, req_index: usize, lane: usize) -> &[i32] {
        assert!(
            req_index < self.num_reqs(),
            "qwen3.5 token-lane request index exceeds batch"
        );
        assert!(
            lane < self.flat_token_ids_by_lane.len(),
            "qwen3.5 token lane is unavailable"
        );
        let cu_tokens = &self.cu_tokens_by_lane[lane];
        &self.flat_token_ids_by_lane[lane][cu_tokens[req_index] as usize..cu_tokens[req_index + 1] as usize]
    }

    pub fn q_len(&self, req_index: usize) -> u32 {
        self.cu_tokens()[req_index + 1] - self.cu_tokens()[req_index]
    }

    pub fn gdn_state_txns(&self) -> &[GDNStateTxn] {
        &self.gdn_state_txns
    }

    pub fn gdn_state_page_ids_by_req(&self) -> &[Vec<Vec<u32>>] {
        &self.gdn_state_page_ids_by_req
    }

    pub fn num_spec_tokens(&self, req_index: usize) -> u32 {
        self.gdn_state_txns[req_index].num_spec_tokens
    }

    pub fn sampler_configs(&self) -> &[SamplerConfig] {
        &self.sampler_configs
    }

    pub fn flat_sample_mask(&self) -> &[bool] {
        &self.flat_sample_mask
    }

    pub fn num_target_hidden_states_for_req(&self, req_index: usize) -> usize {
        let token_start = self.cu_tokens()[req_index] as usize;
        let token_end = self.cu_tokens()[req_index + 1] as usize;
        self.flat_sample_mask[token_start..token_end]
            .iter()
            .filter(|&&sample| sample)
            .count()
    }

    pub fn is_decode_req(&self, req_index: usize) -> bool {
        let token_end = self.cu_tokens()[req_index + 1] as usize;
        self.flat_sample_mask[token_end - 1]
    }

    pub fn has_spec_tokens(&self) -> bool {
        self.gdn_state_txns.iter().any(|txn| txn.num_spec_tokens > 0)
    }

    pub fn total_tokens(&self) -> usize {
        self.flat_token_ids().len()
    }
}

/// A runtime-scheduled Qwen model request.
///
/// This transaction boundary keeps the runtime compute sequence with its
/// microbatch. The microbatch owns only Qwen token/layout data, so it can also
/// represent an internal MTP batch without inventing a compute sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35ModelBatchRequest {
    compute_seq: RawComputeSlotSeq,
    microbatch: Qwen35Microbatch,
}

impl Qwen35ModelBatchRequest {
    pub fn from_core_batch(
        core_batch_req: &BatchDeviceRequest,
        num_mtp_modules: usize,
        sampler_configs: Vec<SamplerConfig>,
    ) -> Self {
        Self {
            compute_seq: core_batch_req.seq,
            microbatch: Qwen35Microbatch::from_requests_with_mtp_modules(
                &core_batch_req.dev_reqs,
                num_mtp_modules,
                sampler_configs,
            ),
        }
    }

    pub fn compute_seq(&self) -> RawComputeSlotSeq {
        self.compute_seq
    }

    pub fn microbatch(&self) -> &Qwen35Microbatch {
        &self.microbatch
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35SampledTokens {
    token_ids: Vec<i32>,
    token_probs: Vec<f32>,
}

impl Qwen35SampledTokens {
    pub fn new(token_ids: Vec<i32>, token_probs: Vec<f32>) -> Self {
        assert_eq!(
            token_ids.len(),
            token_probs.len(),
            "qwen3.5 sampled tokens require one probability per token"
        );
        Self { token_ids, token_probs }
    }

    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub fn token_probs(&self) -> &[f32] {
        &self.token_probs
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen35DecodeDecision {
    pub validated_tokens: Vec<u32>,
    pub validated_probs: Vec<f32>,
    pub sampled_token: u32,
    pub sampled_prob: f32,
    pub spec_tokens: Vec<u32>,
    pub spec_probs: Vec<f32>,
}

pub fn gather_flat_indices(microbatch: &Qwen35Microbatch) -> Vec<u32> {
    microbatch
        .flat_sample_mask()
        .iter()
        .enumerate()
        .filter(|&(_, &sample)| sample)
        .map(|(token_index, _)| {
            token_index
                .try_into()
                .expect("qwen3.5 gathered flat token index must fit u32")
        })
        .collect()
}

pub fn num_target_hidden_states(microbatch: &Qwen35Microbatch) -> usize {
    microbatch.flat_sample_mask().iter().filter(|&&sample| sample).count()
}

/// Returns the logical output-token position for each target hidden state.
///
/// The sampler combines these positions with request seeds rather than the
/// transient compact flat index, keeping fixed-seed decoding stable when a
/// replay is reused for a differently packed batch.
pub fn sample_token_positions(microbatch: &Qwen35Microbatch) -> Vec<u32> {
    let mut positions = Vec::with_capacity(num_target_hidden_states(microbatch));
    for req_index in 0..microbatch.num_reqs() {
        let token_start = microbatch.cu_tokens()[req_index] as usize;
        let token_end = microbatch.cu_tokens()[req_index + 1] as usize;
        positions.extend(
            microbatch.flat_sample_mask()[token_start..token_end]
                .iter()
                .enumerate()
                .filter(|&(_, &sample)| sample)
                .map(|(token_offset, _)| {
                    microbatch.token_indices()[req_index]
                        .checked_add(
                            token_offset
                                .try_into()
                                .expect("qwen3.5 sample token offset must fit u32"),
                        )
                        .and_then(|position| position.checked_add(1))
                        .expect("qwen3.5 sample position overflow")
                }),
        );
    }
    positions
}

/// Returns the request-level sampling configuration expanded over target hidden states.
///
/// A request with `num_spec_tokens` speculative input tokens owns `num_spec_tokens + 1`
/// contiguous target hidden states: one validation distribution per draft token and
/// one final sample distribution.
pub fn sample_sampler_configs(microbatch: &Qwen35Microbatch) -> Vec<SamplerConfig> {
    let mut configs = Vec::with_capacity(num_target_hidden_states(microbatch));
    for req_index in 0..microbatch.num_reqs() {
        configs.extend(std::iter::repeat_n(
            microbatch.sampler_configs[req_index],
            microbatch.num_target_hidden_states_for_req(req_index),
        ));
    }
    assert!(
        !configs.is_empty(),
        "qwen3.5 sampler configs require target hidden states"
    );
    configs
}

pub fn sample_decisions_from_sampled_tokens(sampled_tokens: &Qwen35SampledTokens) -> Vec<Qwen35DecodeDecision> {
    sampled_tokens
        .token_ids
        .iter()
        .zip(&sampled_tokens.token_probs)
        .map(|(&token_id, &token_prob)| {
            Qwen35DecodeDecision {
                sampled_token: token_id
                    .try_into()
                    .expect("qwen3.5 sampled token ID must be non-negative and fit u32"),
                sampled_prob: token_prob,
                ..Qwen35DecodeDecision::default()
            }
        })
        .collect()
}

pub fn verified_state_versions(microbatch: &Qwen35Microbatch) -> Vec<u32> {
    microbatch
        .req_slots()
        .iter()
        .enumerate()
        .map(|(req_index, _)| {
            microbatch.token_indices()[req_index]
                .checked_add(microbatch.q_len(req_index))
                .expect("qwen3.5 verified state version must fit u32")
        })
        .collect()
}

pub fn verified_state_versions_for_decisions(
    microbatch: &Qwen35Microbatch,
    decisions: &[Qwen35DecodeDecision],
) -> Vec<u32> {
    let mut decision_iter = decisions.iter();
    let mut verified_state_versions = Vec::with_capacity(microbatch.num_reqs());
    for req_index in 0..microbatch.num_reqs() {
        let num_query_tokens = microbatch.q_len(req_index);
        let num_spec_tokens = microbatch.num_spec_tokens(req_index);
        let num_base_tokens = num_query_tokens
            .checked_sub(num_spec_tokens)
            .expect("qwen3.5 commit requires GDN spec suffix <= q_len");
        if num_spec_tokens == 0 {
            verified_state_versions.push(
                microbatch.token_indices()[req_index]
                    .checked_add(num_base_tokens)
                    .expect("qwen3.5 verified state version must fit u32"),
            );
            if microbatch.is_decode_req(req_index) {
                let _ = decision_iter
                    .next()
                    .expect("qwen3.5 commit requires one decision per sampled request");
            }
            continue;
        }

        let num_accepted_spec_tokens = if microbatch.is_decode_req(req_index) {
            u32::try_from(
                decision_iter
                    .next()
                    .expect("qwen3.5 commit requires one decision per sampled request")
                    .validated_tokens
                    .len(),
            )
            .expect("qwen3.5 accepted speculative-token count must fit u32")
        } else {
            0
        };
        assert!(
            num_accepted_spec_tokens <= num_spec_tokens,
            "qwen3.5 commit accepted more spec tokens than provided"
        );
        verified_state_versions.push(
            microbatch.token_indices()[req_index]
                .checked_add(num_base_tokens)
                .and_then(|version| version.checked_add(num_accepted_spec_tokens))
                .expect("qwen3.5 verified state version must fit u32"),
        );
    }
    assert!(
        decision_iter.next().is_none(),
        "qwen3.5 commit received more decisions than sampled requests"
    );
    verified_state_versions
}

pub fn to_core_batch_resp(
    core_batch_req: BatchDeviceRequest,
    decisions: Vec<Qwen35DecodeDecision>,
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
                        .expect("qwen3.5 service requires one decision per sampled request");
                    SampledTokens::Decode {
                        epoch: core_req.decoder_query_tokens.epoch(),
                        validated_tokens: decision.validated_tokens.into_iter().map(Token::new).collect(),
                        validated_probs: decision.validated_probs.into_iter().map(finite_probability).collect(),
                        sampled_token: Token::new(decision.sampled_token),
                        sampled_prob: finite_probability(decision.sampled_prob),
                        spec_tokens: decision.spec_tokens.into_iter().map(Token::new).collect(),
                        spec_probs: decision.spec_probs.into_iter().map(finite_probability).collect(),
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
        "qwen3.5 service received more decisions than decode requests"
    );
    BatchDeviceResponse::new(seq, core_resps)
}

pub fn has_synced_pages(page_ids_by_layer: &[Vec<Vec<u32>>]) -> bool {
    page_ids_by_layer
        .iter()
        .any(|page_ids_by_block| !page_ids_by_block.is_empty())
}

fn finite_probability(probability: f32) -> NotNan<f32> {
    NotNan::new(probability).expect("qwen3.5 probability should be finite and non-NaN")
}

fn validate_batch_fields(
    req_slots: &[u32],
    block_indices: &[usize],
    token_indices: &[u32],
    flat_token_ids: &[i32],
    cu_tokens: &[u32],
) {
    let num_reqs = req_slots.len();
    assert_eq!(
        block_indices.len(),
        num_reqs,
        "qwen3.5 request requires one block_index entry per request"
    );
    assert_eq!(
        token_indices.len(),
        num_reqs,
        "qwen3.5 request requires one token_index entry per request"
    );
    assert_eq!(
        cu_tokens.len(),
        num_reqs + 1,
        "qwen3.5 request requires cu_tokens length to equal num_reqs + 1"
    );
    assert_eq!(cu_tokens[0], 0, "qwen3.5 request requires cu_tokens[0] == 0");
    assert_eq!(
        *cu_tokens.last().expect("qwen3.5 request requires cu_tokens"),
        u32::try_from(flat_token_ids.len()).expect("qwen3.5 flat token count must fit u32"),
        "qwen3.5 request requires cu_tokens.last() == flat_token_ids.len()"
    );
    for req_index in 0..num_reqs {
        let start = cu_tokens[req_index];
        let end = cu_tokens[req_index + 1];
        assert!(
            start < end,
            "qwen3.5 request requires strictly increasing cu_tokens, req_index={req_index}, start={start}, end={end}"
        );
    }
}

fn validate_gdn_state_txns(token_indices: &[u32], cu_tokens: &[u32], gdn_state_txns: &[GDNStateTxn]) {
    assert_eq!(
        token_indices.len(),
        gdn_state_txns.len(),
        "qwen3.5 request requires one GDN state txn per request"
    );
    for req_index in 0..token_indices.len() {
        let txn = gdn_state_txns[req_index];
        let q_len = cu_tokens[req_index + 1]
            .checked_sub(cu_tokens[req_index])
            .expect("qwen3.5 cumulative token counts must be increasing");
        assert_eq!(
            txn.token_index, token_indices[req_index],
            "qwen3.5 GDN state txn token_index must match request token_index"
        );
        assert_eq!(
            txn.num_total_tokens, q_len,
            "qwen3.5 GDN state txn num_total_tokens must match q_len"
        );
    }
}

fn validate_flat_sample_mask(cu_tokens: &[u32], gdn_state_txns: &[GDNStateTxn], flat_sample_mask: &[bool]) {
    assert_eq!(
        flat_sample_mask.len(),
        *cu_tokens.last().expect("qwen3.5 sample mask requires cu_tokens") as usize,
        "qwen3.5 flat_sample_mask requires one entry per flat token"
    );
    for req_index in 0..gdn_state_txns.len() {
        let token_start = cu_tokens[req_index] as usize;
        let token_end = cu_tokens[req_index + 1] as usize;
        let req_flat_sample_mask = &flat_sample_mask[token_start..token_end];
        let num_req_target_hidden_states = req_flat_sample_mask.iter().filter(|&&sample| sample).count();
        if num_req_target_hidden_states == 0 {
            continue;
        }
        assert_eq!(
            num_req_target_hidden_states,
            usize::try_from(gdn_state_txns[req_index].num_spec_tokens)
                .expect("qwen3.5 speculative-token count must fit host usize")
                .checked_add(1)
                .expect("qwen3.5 target hidden-state count must fit usize"),
            "qwen3.5 decode request requires one sample token per speculative token plus one"
        );
        assert!(
            req_flat_sample_mask[..req_flat_sample_mask.len() - num_req_target_hidden_states]
                .iter()
                .all(|&sample| !sample)
                && req_flat_sample_mask[req_flat_sample_mask.len() - num_req_target_hidden_states..]
                    .iter()
                    .all(|&sample| sample),
            "qwen3.5 flat_sample_mask requires a contiguous request suffix"
        );
    }
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::RawRequestSlot;
    use inference_runtime_core::runtime::Token;

    use super::*;

    #[test]
    fn test_builds_microbatch_from_core_requests() {
        let requests = vec![
            device_request(
                10,
                0,
                QueryTokens::Prefill {
                    epoch: 1,
                    token_index: 4,
                    tokens: tokens(&[101, 102, 103]),
                    window: 3,
                },
                2,
            ),
            device_request(
                11,
                1,
                QueryTokens::Decode {
                    epoch: 1,
                    token_index: 7,
                    tokens: tokens(&[201]),
                    spec_tokens: tokens(&[202, 203]),
                },
                3,
            ),
        ];
        let batch = Qwen35Microbatch::from_requests(&requests, vec![SamplerConfig::default(); 2]);

        assert_eq!(batch.req_slots(), &[0, 1]);
        assert_eq!(batch.block_indices(), &[2, 3]);
        assert_eq!(batch.token_indices(), &[4, 7]);
        assert_eq!(batch.flat_token_ids(), &[101, 102, 103, 201, 202, 203]);
        assert_eq!(batch.cu_tokens(), &[0, 3, 6]);
        assert_eq!(batch.num_spec_tokens(0), 0);
        assert_eq!(batch.num_spec_tokens(1), 2);
        assert_eq!(batch.flat_sample_mask(), &[false, false, false, true, true, true]);
        assert!(batch.has_spec_tokens());
    }

    #[test]
    fn test_batch_request_preserves_compute_sequence() {
        let core_batch = BatchDeviceRequest::new(
            17,
            [device_request(
                10,
                0,
                QueryTokens::Decode {
                    epoch: 1,
                    token_index: 4,
                    tokens: tokens(&[101]),
                    spec_tokens: vec![],
                },
                2,
            )],
        );

        let batch = Qwen35ModelBatchRequest::from_core_batch(&core_batch, 0, vec![SamplerConfig::default()]);

        assert_eq!(batch.compute_seq(), 17);
    }

    #[test]
    fn test_equal_width_prefill_token_lanes() {
        let requests = vec![device_request(
            10,
            0,
            QueryTokens::Prefill {
                epoch: 1,
                token_index: 4,
                tokens: tokens(&[100, 101, 102, 103, 104]),
                window: 3,
            },
            2,
        )];

        let batch = Qwen35Microbatch::from_requests_with_mtp_modules(&requests, 2, vec![SamplerConfig::default()]);

        assert_eq!(batch.token_ids_for_lane(0, 0), &[100, 101, 102]);
        assert_eq!(batch.token_ids_for_lane(0, 1), &[101, 102, 103]);
        assert_eq!(batch.token_ids_for_lane(0, 2), &[102, 103, 104]);
    }

    #[test]
    fn test_decode_has_empty_executor_built_lanes() {
        let requests = vec![device_request(
            10,
            0,
            QueryTokens::Decode {
                epoch: 1,
                token_index: 4,
                tokens: tokens(&[100]),
                spec_tokens: vec![],
            },
            2,
        )];

        let batch = Qwen35Microbatch::from_requests_with_mtp_modules(&requests, 2, vec![SamplerConfig::default()]);

        assert_eq!(batch.token_ids_for_lane(0, 0), &[100]);
        assert!(batch.token_ids_for_lane(0, 1).is_empty());
        assert!(batch.token_ids_for_lane(0, 2).is_empty());
    }

    #[test]
    fn test_target_hidden_states_speculative() {
        let batch = Qwen35Microbatch::new(
            vec![0],
            vec![0],
            vec![8],
            vec![10, 11, 12, 20, 21],
            vec![0, 5],
            vec![GDNStateTxn::new(8, 5, 2)],
            vec![Vec::new()],
            vec![SamplerConfig::default()],
            vec![false, false, true, true, true],
        );

        assert_eq!(gather_flat_indices(&batch), vec![2, 3, 4]);
        assert_eq!(num_target_hidden_states(&batch), 3);
    }

    #[test]
    #[should_panic(expected = "flat_sample_mask requires a contiguous request suffix")]
    fn test_flat_sample_mask_requires_suffix() {
        let _ = Qwen35Microbatch::new(
            vec![0],
            vec![0],
            vec![8],
            vec![10, 11, 12],
            vec![0, 3],
            vec![GDNStateTxn::new(8, 3, 1)],
            vec![Vec::new()],
            vec![SamplerConfig::default()],
            vec![true, false, true],
        );
    }

    #[test]
    fn test_sample_sampler_configs_repeat_each_decode_config() {
        let first = SamplerConfig {
            temperature: 0.7,
            top_k: 64,
            top_p: 0.8,
            seed: 7,
        };
        let second = SamplerConfig {
            temperature: 1.1,
            top_k: 64,
            top_p: 0.9,
            seed: 99,
        };
        let batch = Qwen35Microbatch::new(
            vec![0, 1],
            vec![0, 1],
            vec![0, 0],
            vec![11, 12, 13, 21, 22],
            vec![0, 3, 5],
            vec![GDNStateTxn::new(0, 3, 2), GDNStateTxn::new(0, 2, 1)],
            vec![Vec::new(), Vec::new()],
            vec![first, second],
            vec![true; 5],
        );

        assert_eq!(
            sample_sampler_configs(&batch),
            vec![first, first, first, second, second]
        );
    }

    #[test]
    fn test_target_hidden_states_decode() {
        let request = Qwen35Microbatch::from_requests(
            &[
                device_request(
                    10,
                    0,
                    QueryTokens::Prefill {
                        epoch: 1,
                        token_index: 0,
                        tokens: tokens(&[1, 2, 3]),
                        window: 3,
                    },
                    0,
                ),
                device_request(
                    11,
                    1,
                    QueryTokens::Decode {
                        epoch: 1,
                        token_index: 3,
                        tokens: tokens(&[4]),
                        spec_tokens: vec![],
                    },
                    0,
                ),
                device_request(
                    12,
                    2,
                    QueryTokens::Decode {
                        epoch: 1,
                        token_index: 4,
                        tokens: tokens(&[5]),
                        spec_tokens: tokens(&[6, 7]),
                    },
                    0,
                ),
            ],
            vec![SamplerConfig::default(); 3],
        );

        assert_eq!(num_target_hidden_states(&request), 4);
        assert_eq!(gather_flat_indices(&request), vec![3, 4, 5, 6]);
        assert_eq!(sample_token_positions(&request), vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_sample_decisions_use_one_token_per_decode_request() {
        let request = Qwen35Microbatch::from_requests(
            &[
                device_request(
                    10,
                    0,
                    QueryTokens::Prefill {
                        epoch: 1,
                        token_index: 0,
                        tokens: tokens(&[1, 2]),
                        window: 2,
                    },
                    0,
                ),
                device_request(
                    11,
                    1,
                    QueryTokens::Decode {
                        epoch: 1,
                        token_index: 2,
                        tokens: tokens(&[3]),
                        spec_tokens: vec![],
                    },
                    0,
                ),
            ],
            vec![SamplerConfig::default(); 2],
        );
        let sampled_tokens = Qwen35SampledTokens::new(vec![31], vec![0.5]);

        let decisions = sample_decisions_from_sampled_tokens(&sampled_tokens);

        assert_eq!(
            decisions,
            vec![Qwen35DecodeDecision {
                sampled_token: 31,
                sampled_prob: 0.5,
                ..Qwen35DecodeDecision::default()
            }]
        );
    }

    #[test]
    fn test_verified_state_versions_follow_rejection_accept_count() {
        let request = Qwen35Microbatch::from_requests(
            &[device_request(
                10,
                0,
                QueryTokens::Decode {
                    epoch: 1,
                    token_index: 4,
                    tokens: tokens(&[11]),
                    spec_tokens: tokens(&[12, 13]),
                },
                0,
            )],
            vec![SamplerConfig::default()],
        );
        let decisions = vec![Qwen35DecodeDecision {
            validated_tokens: vec![12],
            validated_probs: vec![0.8],
            sampled_token: 99,
            sampled_prob: 0.2,
            ..Qwen35DecodeDecision::default()
        }];

        assert_eq!(verified_state_versions_for_decisions(&request, &decisions), vec![6]);
    }

    #[test]
    fn test_converts_decisions_to_core_response() {
        let core = BatchDeviceRequest::new(
            7,
            vec![
                device_request(
                    10,
                    0,
                    QueryTokens::Prefill {
                        epoch: 1,
                        token_index: 0,
                        tokens: tokens(&[1, 2]),
                        window: 2,
                    },
                    0,
                ),
                device_request(
                    11,
                    1,
                    QueryTokens::Decode {
                        epoch: 2,
                        token_index: 2,
                        tokens: tokens(&[3]),
                        spec_tokens: vec![],
                    },
                    0,
                ),
            ],
        );
        let decision = Qwen35DecodeDecision {
            validated_tokens: vec![4],
            validated_probs: vec![0.7],
            sampled_token: 5,
            sampled_prob: 0.3,
            spec_tokens: vec![6],
            spec_probs: vec![0.2],
        };

        let response = to_core_batch_resp(core, vec![decision]);

        assert_eq!(response.seq, 7);
        assert!(matches!(
            response.dev_resps[0].sampled_tokens,
            SampledTokens::Prefill { epoch: 1 }
        ));
        match &response.dev_resps[1].sampled_tokens {
            SampledTokens::Decode {
                epoch,
                validated_tokens,
                sampled_token,
                spec_tokens,
                ..
            } => {
                assert_eq!(*epoch, 2);
                assert_eq!(validated_tokens[0].value(), 4);
                assert_eq!(sampled_token.value(), 5);
                assert_eq!(spec_tokens[0].value(), 6);
            },
            SampledTokens::Prefill { .. } => panic!("expected decode sampled tokens"),
        }
    }

    fn device_request(
        req_id: usize,
        req_slot: RawRequestSlot,
        tokens: QueryTokens,
        block_index: usize,
    ) -> DeviceRequest {
        DeviceRequest::new(
            req_id,
            req_slot,
            tokens,
            DecoderSyncBlocks::new(block_index, vec![], vec![]),
            SamplingConfig::default(),
        )
    }

    fn tokens(values: &[u32]) -> Vec<Token> {
        values.iter().copied().map(Token::new).collect()
    }
}
