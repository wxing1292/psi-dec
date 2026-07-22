use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_executor_core::backend::runtime::Runtime;
use inference_executor_core::model::qwen::v3_5::Qwen35DecodeDecision;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelBatchRequest;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::Qwen35SampledTokens;
use inference_executor_core::model::qwen::v3_5::gather_flat_indices;
use inference_executor_core::model::qwen::v3_5::num_target_hidden_states;
use inference_executor_core::model::qwen::v3_5::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3_5::sample_sampler_configs;
use inference_executor_core::model::qwen::v3_5::sample_token_positions;
use inference_executor_core::model::qwen::v3_5::to_core_batch_resp;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::SparseRejectionSamplingReqParams;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingShape;
use inference_runtime_core::compute::BatchDevReq;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ModelOutputTiming;
use inference_runtime_core::compute::ReplayableModelBatchExecutor;
use inference_runtime_core::runtime::RawComputeSlotSeq;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3_5::model::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::model::Qwen35GatherUnembedArgs;
use crate::model::qwen::v3_5::model::Qwen35GatherUnembedReplayKey;
use crate::model::qwen::v3_5::model::Qwen35Main;
use crate::model::qwen::v3_5::model::Qwen35MainArgs;
use crate::model::qwen::v3_5::model::Qwen35MainEmbed;
use crate::model::qwen::v3_5::model::Qwen35MainEmbedArgs;
use crate::model::qwen::v3_5::model::Qwen35MainEmbedReplayKey;
use crate::model::qwen::v3_5::model::Qwen35MainReplayKey;
use crate::model::qwen::v3_5::mtp::Qwen35MTP;
use crate::model::qwen::v3_5::mtp::Qwen35MTPArgs;
use crate::model::qwen::v3_5::mtp::Qwen35MTPEmbed;
use crate::model::qwen::v3_5::mtp::Qwen35MTPEmbedArgs;
use crate::model::qwen::v3_5::mtp::Qwen35MTPEmbedReplayKey;
use crate::model::qwen::v3_5::mtp::Qwen35MTPReplayKey;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35RejectionSamplingInput;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35TargetRejectionReplayKey;
use crate::model::qwen::v3_5::rejection_sampling::RejectionSampling;
use crate::model::qwen::v3_5::rejection_sampling::RejectionSamplingInput;
use crate::model::qwen::v3_5::state::Qwen35GDNState;
use crate::model::qwen::v3_5::state::Qwen35GQAState;
use crate::replay::Replay;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::DraftSamplingInput;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingSparseDistributionOutput;
use crate::trace;

mod load;

pub use load::Qwen35ExecutorConfig;
use load::Qwen35ModelLayout;
pub use load::init_qwen_3_5_model;
pub use load::init_qwen_3_5_model_with_hf_mtp;

include!("batch.rs");
include!("main.rs");
include!("mtp.rs");
include!("recording.rs");
include!("sampling.rs");

pub struct Qwen35Executor {
    model_name: String,
    default_stop_sequences: Vec<Vec<Token>>,
    config: Qwen35ExecutorConfig,
    runtime: MetalRuntime,
    layout: Qwen35ModelLayout,
    token_ids: Buffer,
    token_hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    mtp_hidden_input: Option<Rc<Buffer>>,
    mtp_input_gather_flat_indices: Buffer,
    draft_distribution_indices: Buffer,
    target_distribution_indices: Buffer,
    mtp_previous_hidden: Buffer,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    main_embed: Replay<Qwen35MainEmbed>,
    main: Replay<Qwen35Main>,
    gather_unembed: Replay<Qwen35GatherUnembed>,
    sampling: Replay<Sampling>,
    mtp_embed: Option<Replay<Qwen35MTPEmbed>>,
    mtp: Option<Replay<Qwen35MTP>>,
    draft_sampling: Replay<DraftSampling>,
    rejection_sampling: Replay<RejectionSampling>,
    sampler: Rc<TopKSampling>,
    sampler_bounds: TopKSamplingBounds,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    main_gqa_state: Qwen35GQAState,
    main_gdn_state: Qwen35GDNState,
    mtp_gqa_state: Option<Qwen35GQAState>,
    spec_probs: SpecProbsStore,
    pages: PageArena,
    pending_transactions: Qwen35PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
}

pub struct Qwen35ModelRecorder {
    compute_seq: RawComputeSlotSeq,
    main_embed_key: Qwen35MainEmbedReplayKey,
    main_key: Qwen35MainReplayKey,
    main_embed_cache_hit: bool,
    main_cache_hit: bool,
    gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    main_stage_submitted: bool,
}

impl Qwen35ModelRecorder {
    fn main_replay_cache_hit(&self) -> bool {
        self.main_embed_cache_hit && self.main_cache_hit
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]

struct Qwen35MTPRequest {
    num_tokens: usize,
    current_token_ids: Vec<i32>,
    next_token_id: Option<i32>,
    decision_index: Option<usize>,
}

struct Qwen35MTPModuleBatch {
    microbatch: Qwen35Microbatch,
    input_gather_flat_indices: Vec<u32>,
    draft_distribution_indices: Vec<u32>,
    sampler_configs: Vec<SamplerConfig>,
    sample_positions: Vec<u32>,
}

fn mtp_proposal_sample_position(token_index: u32, num_tokens: usize) -> u32 {
    token_index
        .checked_add(
            num_tokens
                .try_into()
                .expect("qwen3.5 MTP request token count must fit u32"),
        )
        .and_then(|position| position.checked_add(1))
        .expect("qwen3.5 MTP proposal sample position overflow")
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35ForwardOutput;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen35DecodeOutput {
    decisions: Vec<Qwen35DecodeDecision>,
    read_sampling_output: bool,
    timing: ModelOutputTiming,
}

impl ReplayableModelBatchExecutor for Qwen35Executor {
    type ModelBatchReq = Qwen35ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResp = Qwen35ForwardOutput;
    type SampledOutput = Qwen35DecodeOutput;
    type ModelOpsRecorder = Qwen35ModelRecorder;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        Qwen35Executor::reset_req_slots(self, request_slots)
    }

    fn prepare_batch(&mut self, core_batch_req: &BatchDeviceRequest) -> Self::ModelBatchReq {
        Qwen35Executor::prepare_batch(self, core_batch_req)
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse {
        Qwen35Executor::commit_batch(self, core_batch_req, sampled_output)
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Self::ModelBatchReq) -> Self::ModelOpsRecorder {
        Qwen35Executor::begin_ops_recording(self, model_batch_request)
    }

    fn finish_ops_recording(
        &mut self,
        recorder: Self::ModelOpsRecorder,
        sampled_output: Self::SampledOutput,
    ) -> Self::SampledOutput {
        Qwen35Executor::finish_ops_recording(self, recorder, sampled_output)
    }

    fn embed(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_request: &Self::ModelBatchReq,
    ) -> Self::ModelBatchHidden {
        Qwen35Executor::embed(self, recorder, model_batch_request)
    }

    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        Qwen35Executor::forward_main(self, recorder, model_batch_req, model_batch_hidden)
    }

    fn unembed(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResp {
        Qwen35Executor::unembed(self, recorder, model_batch_req, model_batch_hidden)
    }

    fn sample(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        _model_batch_resp: &Self::ModelBatchResp,
    ) -> Self::SampledOutput {
        Qwen35Executor::sample(self, recorder, model_batch_req, _model_batch_resp)
    }

    fn rejection_sample(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        _model_batch_resp: &Self::ModelBatchResp,
    ) -> Self::SampledOutput {
        Qwen35Executor::rejection_sample(self, recorder, model_batch_req, _model_batch_resp)
    }

    fn forward_mtp(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchReq,
        model_batch_hidden: &Self::ModelBatchHidden,
        sampled_output: Self::SampledOutput,
    ) -> Self::SampledOutput {
        Qwen35Executor::forward_mtp(self, recorder, model_batch_req, model_batch_hidden, sampled_output)
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen35DecodeOutput::default()
    }

    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize {
        sampled_output.decisions.len()
    }

    fn sampled_output_timing(&self, sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        (!sampled_output.timing.is_zero()).then_some(sampled_output.timing)
    }
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(
        num_tokens_per_block > 0,
        "qwen3.5 GQA requires positive tokens per block"
    );
    assert!(num_tokens_per_page > 0, "qwen3.5 GQA requires positive tokens per page");
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "qwen3.5 GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}

fn trace_decisions(event: &str, decisions: &[Qwen35DecodeDecision]) {
    trace::qwen35_state(|| {
        let decisions = decisions
            .iter()
            .map(|decision| {
                (
                    decision.validated_tokens.as_slice(),
                    decision.sampled_token,
                    decision.spec_tokens.as_slice(),
                    decision.validated_probs.len(),
                    decision.spec_probs.len(),
                )
            })
            .collect::<Vec<_>>();
        format!("event={} decisions={:?}", event, decisions)
    });
}

fn replay_bucket_capacity(active: u32, max_capacity: u32) -> u32 {
    assert!(active > 0, "qwen3.5 replay bucket requires active work");
    assert!(active <= max_capacity, "qwen3.5 replay active work exceeds capacity");
    active
        .checked_next_power_of_two()
        .unwrap_or(max_capacity)
        .min(max_capacity)
}

fn replay_bucket_capacity_usize(active: usize, max_capacity: usize) -> usize {
    assert!(active > 0, "qwen3.5 replay bucket requires active work");
    assert!(active <= max_capacity, "qwen3.5 replay active work exceeds capacity");
    active
        .checked_next_power_of_two()
        .unwrap_or(max_capacity)
        .min(max_capacity)
}

fn replay_bucket_capacity_allow_zero(active: usize, max_capacity: usize) -> usize {
    if active == 0 {
        assert!(max_capacity > 0);
        return 0;
    }
    replay_bucket_capacity_usize(active, max_capacity)
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::GDNReplayShape;
    use inference_executor_core::attn::GQAReplayShape;
    use inference_executor_core::attn::gdn::state::GDNStateTxn;
    use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
    use inference_executor_core::sampling::SamplerConfig;

    use super::Qwen35ExecutorConfig;
    use super::Qwen35GatherUnembedReplayKey;
    use super::Qwen35MTPEmbedReplayKey;
    use super::Qwen35MainEmbedReplayKey;
    use super::Qwen35MainReplayKey;
    use super::mtp_proposal_sample_position;
    use super::replay_bucket_capacity;

    #[test]
    fn test_executor_config_supports_at_most_one_mtp_module() {
        let config = Qwen35ExecutorConfig {
            max_requests: 1,
            max_tokens: 4,
            max_tokens_per_request: 4,
            num_cache_pages: 1,
            num_tokens_per_block: 1024,
            num_mtp_modules: 1,
        };
        config.validate();

        let too_many_mtp_modules = Qwen35ExecutorConfig {
            num_mtp_modules: 2,
            ..config
        };
        assert!(std::panic::catch_unwind(|| too_many_mtp_modules.validate()).is_err());
    }

    #[test]
    fn test_mtp_proposal_sample_position_follows_single_body() {
        assert_eq!(mtp_proposal_sample_position(17, 3), 21);
    }

    #[test]
    fn test_main_key() {
        let key = Qwen35MainReplayKey::from_shapes(context_parallel_gqa_shape(), gdn_shape(1));

        assert_eq!(key.debug_parts(), (4, 4, 4, 1));
    }

    #[test]
    fn test_embed_keys_separate_token_counts() {
        assert_ne!(Qwen35MainEmbedReplayKey::new(1), Qwen35MainEmbedReplayKey::new(2));
        assert_ne!(Qwen35MTPEmbedReplayKey::new(0, 1), Qwen35MTPEmbedReplayKey::new(0, 2));
    }

    #[test]
    fn test_main_key_tiled() {
        let key = Qwen35MainReplayKey::from_shapes(tiled_gqa_shape(), gdn_shape(1));

        assert_eq!(key.debug_parts(), (4, 1, 1, 1));
    }

    #[test]
    fn test_main_key_separates_gdn_request_geometry() {
        let one_req = Qwen35MainReplayKey::from_shapes(context_parallel_gqa_shape(), gdn_shape(1));
        let two_reqs = Qwen35MainReplayKey::from_shapes(context_parallel_gqa_shape(), gdn_shape(2));

        assert_ne!(one_req, two_reqs);
    }

    #[test]
    fn test_main_key_shares_partial_output_reduce_topology() {
        let one_task_template_per_token = context_parallel_gqa_shape();
        let multiple_task_templates_per_token = GQAReplayShape {
            reduce_sdpa_partial_outputs: true,
            ..one_task_template_per_token
        };

        assert_eq!(
            Qwen35MainReplayKey::from_shapes(one_task_template_per_token, gdn_shape(1)),
            Qwen35MainReplayKey::from_shapes(multiple_task_templates_per_token, gdn_shape(1))
        );
    }

    #[test]
    fn test_gather_unembed_key_separates_target_hidden_states() {
        let one_target = one_req_batch(4, 0);
        let three_targets = one_req_batch(4, 2);

        assert_ne!(
            Qwen35GatherUnembedReplayKey::from_microbatch(&one_target),
            Qwen35GatherUnembedReplayKey::from_microbatch(&three_targets)
        );
    }

    #[test]
    fn test_bucket_capacity() {
        assert_eq!(replay_bucket_capacity(1, 48), 1);
        assert_eq!(replay_bucket_capacity(2, 48), 2);
        assert_eq!(replay_bucket_capacity(3, 48), 4);
        assert_eq!(replay_bucket_capacity(32, 48), 32);
        assert_eq!(replay_bucket_capacity(33, 48), 48);
        assert_eq!(replay_bucket_capacity(48, 48), 48);
    }

    fn one_req_batch(num_tokens: u32, num_spec_tokens: u32) -> Qwen35Microbatch {
        let num_sample_tokens = num_spec_tokens + 1;
        Qwen35Microbatch::new(
            vec![0],
            vec![0],
            vec![0],
            (0..num_tokens).map(|token| token as i32).collect(),
            vec![0, num_tokens],
            vec![GDNStateTxn::new(0, num_tokens, num_spec_tokens)],
            vec![Vec::new()],
            vec![SamplerConfig::default()],
            (0..num_tokens)
                .map(|token_offset| token_offset + num_sample_tokens >= num_tokens)
                .collect(),
        )
    }

    fn context_parallel_gqa_shape() -> GQAReplayShape {
        GQAReplayShape {
            num_tokens: 4,
            num_q_token_tiles: 4,
            total_sdpa_map_task_templates: 4,
            reduce_sdpa_partial_outputs: false,
        }
    }

    fn gdn_shape(num_reqs: u32) -> GDNReplayShape {
        GDNReplayShape {
            num_reqs,
            num_tokens: 4,
        }
    }

    fn tiled_gqa_shape() -> GQAReplayShape {
        GQAReplayShape {
            num_tokens: 4,
            num_q_token_tiles: 1,
            total_sdpa_map_task_templates: 1,
            reduce_sdpa_partial_outputs: true,
        }
    }
}
