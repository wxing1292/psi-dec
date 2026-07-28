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
use crate::def::replay_op::MetalReplaySubmission;
use crate::def::replay_op::ReplayRecorder;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3_5::main::Qwen35Main;
use crate::model::qwen::v3_5::main::Qwen35MainArgs;
use crate::model::qwen::v3_5::main::Qwen35MainReplayKey;
use crate::model::qwen::v3_5::main::embed::Qwen35MainEmbed;
use crate::model::qwen::v3_5::main::embed::Qwen35MainEmbedArgs;
use crate::model::qwen::v3_5::main::embed::Qwen35MainEmbedReplayKey;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedArgs;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedReplayKey;
use crate::model::qwen::v3_5::mtp::Qwen35MTP;
use crate::model::qwen::v3_5::mtp::Qwen35MTPArgs;
use crate::model::qwen::v3_5::mtp::Qwen35MTPReplayKey;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbed;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbedArgs;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbedReplayKey;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35PreparedRejection;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35RejectionSamplingInput;
use crate::model::qwen::v3_5::rejection_sampling::Qwen35TargetRejectionReplayKey;
use crate::model::qwen::v3_5::rejection_sampling::RejectionSampling;
use crate::model::qwen::v3_5::rejection_sampling::RejectionSamplingInput;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
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
    main_gqa_state: Qwen3xGQAState,
    main_gdn_state: Qwen3xGDNState,
    mtp_gqa_state: Option<Qwen3xGQAState>,
    spec_probs: SpecProbsStore,
    pages: PageArena,
    pending_transactions: Qwen35PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
}

pub struct Qwen35ModelOpsRecorder {
    main_embed_key: Qwen35MainEmbedReplayKey,
    main_key: Qwen35MainReplayKey,
    main_embed_cache_hit: bool,
    main_cache_hit: bool,
    gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    rejection_key: Option<Qwen35TargetRejectionReplayKey>,
    rejection_arguments: ReplayArguments,
    rejection_prepared: Option<Qwen35PreparedRejection>,
    rejection_build_elapsed: Duration,
    num_sample_tokens: usize,
    mtp_embed_key: Option<Qwen35MTPEmbedReplayKey>,
    mtp_key: Option<Qwen35MTPReplayKey>,
    mtp_microbatch: Option<Qwen35Microbatch>,
    mtp_sampler_configs: Vec<SamplerConfig>,
    mtp_sample_positions: Vec<u32>,
    mtp_embed_cache_hit: bool,
    mtp_gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    mtp_draft_sampling_key: Option<TopKSamplingReplayKey>,
    mtp_draft_sampling_arguments: ReplayArguments,
    mtp_draft_req_slots: Vec<u32>,
    mtp_draft_decision_indices: Vec<usize>,
    mtp_num_sample_tokens: usize,
    mtp_build_elapsed: Duration,
}

impl Qwen35ModelOpsRecorder {
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
pub struct Qwen35ModelBatchResponse;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen35SampledOutput {
    decisions: Vec<Qwen35DecodeDecision>,
    timing: ModelOutputTiming,
}

impl ReplayableModelBatchExecutor for Qwen35Executor {
    type ModelBatchRequest = Qwen35ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = Qwen35ModelBatchResponse;
    type SampledOutput = Qwen35SampledOutput;
    type ModelOpsRecorder = Qwen35ModelOpsRecorder;
    type Submission = MetalReplaySubmission;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.finish_cache_publish();
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
        if let Some(mtp_gqa_state) = &self.mtp_gqa_state {
            mtp_gqa_state.reset_req_slots(request_slots);
        }
        self.spec_probs.reset_req_slots(request_slots);
        self.main_gdn_state.reset_req_slots(request_slots);
    }

    fn prepare_batch(&mut self, core_batch_req: &BatchDeviceRequest) -> Self::ModelBatchRequest {
        self.finish_cache_publish();
        let batch_seq = core_batch_req.seq;
        self.validate_input(core_batch_req);
        let sampler_configs = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                let seed = self
                    .request_sampling
                    .resolve(request.req_slot, request.sampling_config.seed);
                SamplerConfig::from_runtime(&request.sampling_config, seed)
            })
            .collect();
        let model_batch_request =
            Qwen35ModelBatchRequest::from_core_batch(core_batch_req, usize::from(self.mtp.is_some()), sampler_configs);
        let microbatch = model_batch_request.microbatch();
        trace::qwen35_state(|| {
            format!(
                "event=batch_from_core seq={} req_slots={:?} token_indices={:?} num_spec_tokens={:?} seeds={:?} \
                 total_tokens={} num_reqs={}",
                batch_seq,
                microbatch.req_slots(),
                microbatch.token_indices(),
                microbatch
                    .gdn_state_txns()
                    .iter()
                    .map(|txn| txn.num_spec_tokens)
                    .collect::<Vec<_>>(),
                microbatch
                    .sampler_configs()
                    .iter()
                    .map(SamplerConfig::seed)
                    .collect::<Vec<_>>(),
                microbatch.total_tokens(),
                microbatch.num_reqs()
            )
        });
        self.write_token_ids(microbatch.flat_token_ids());
        let prepare_start = Instant::now();
        let gqa_start = Instant::now();
        self.main_gqa_state.prepare_pages(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
        );
        let gqa_elapsed = gqa_start.elapsed();
        debug_assert_eq!(gqa_shape.num_tokens as usize, microbatch.total_tokens());
        let gdn_states_start = Instant::now();
        let gdn_prepared = self.main_gdn_state.prepare_states(
            microbatch.req_slots(),
            microbatch.block_indices(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            microbatch.gdn_state_txns(),
            microbatch.gdn_state_page_ids_by_req(),
        );
        let gdn_states_elapsed = gdn_states_start.elapsed();
        let gdn_metadata_start = Instant::now();
        let gdn_shape = self
            .main_gdn_state
            .prepare_metadata(microbatch.cu_tokens(), &gdn_prepared);
        let gdn_metadata_elapsed = gdn_metadata_start.elapsed();
        debug_assert_eq!(gdn_shape.num_tokens as usize, microbatch.total_tokens());
        debug_assert_eq!(gdn_shape.num_reqs as usize, microbatch.num_reqs());
        if let Some(mtp_gqa_state) = &self.mtp_gqa_state {
            mtp_gqa_state.prepare_pages(core_batch_req);
        }
        let prepare_elapsed = prepare_start.elapsed();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_sync seq={} gqa_us={} gdn_states_us={} gdn_metadata_us={} wall_us={}",
                batch_seq,
                gqa_elapsed.as_micros(),
                gdn_states_elapsed.as_micros(),
                gdn_metadata_elapsed.as_micros(),
                prepare_elapsed.as_micros()
            )
        });
        let restore_elapsed = self.submit_gdn_state_restore();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_batch_done seq={} gdn_restore_us={}",
                batch_seq,
                restore_elapsed.as_micros()
            )
        });
        model_batch_request
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse {
        self.commit(core_batch_req.seq, &sampled_output.decisions);
        to_core_batch_resp(core_batch_req, sampled_output.decisions)
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
        let main_embed_key = Qwen35MainEmbedReplayKey::new(
            model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3.5 MainEmbed token count must fit u32"),
        );
        let main_key = Qwen35MainReplayKey::from_shapes(
            self.main_gqa_state.metadata().replay_shape(),
            self.main_gdn_state.metadata().replay_shape(),
        );
        trace::qwen35_state(|| {
            format!(
                "event=begin_ops_recording main_embed_key={:?} main_key={:?}",
                main_embed_key, main_key
            )
        });
        Qwen35ModelOpsRecorder {
            main_embed_key,
            main_key,
            main_embed_cache_hit: false,
            main_cache_hit: false,
            gather_unembed_key: None,
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            rejection_key: None,
            rejection_arguments: ReplayArguments::new(),
            rejection_prepared: None,
            rejection_build_elapsed: Duration::ZERO,
            num_sample_tokens: num_target_hidden_states(model_batch_request.microbatch()),
            mtp_embed_key: None,
            mtp_key: None,
            mtp_microbatch: None,
            mtp_sampler_configs: Vec::new(),
            mtp_sample_positions: Vec::new(),
            mtp_embed_cache_hit: false,
            mtp_gather_unembed_key: None,
            mtp_draft_sampling_key: None,
            mtp_draft_sampling_arguments: ReplayArguments::new(),
            mtp_draft_req_slots: Vec::new(),
            mtp_draft_decision_indices: Vec::new(),
            mtp_num_sample_tokens: 0,
            mtp_build_elapsed: Duration::ZERO,
        }
    }

    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_request: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden {
        let input = Qwen35MainEmbedArgs {
            num_tokens: model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3.5 MainEmbed token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &self.token_hidden_input,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, cache_hit) = self.main_embed.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_embed_key,
            "qwen3.5 MainEmbed replay input must match the prepared replay key"
        );
        recorder.main_embed_cache_hit = cache_hit;
        Rc::clone(&self.token_hidden_input)
    }

    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        let microbatch = model_batch_req.microbatch();
        assert!(
            Rc::ptr_eq(&model_batch_hidden, &self.token_hidden_input),
            "qwen3.5 Main must consume the MainEmbed hidden workspace"
        );
        let input = Qwen35MainArgs {
            num_tokens: microbatch
                .total_tokens()
                .try_into()
                .expect("qwen3.5 Main token count must fit u32"),
            hidden_input: &model_batch_hidden,
            hidden_output: &self.hidden_output,
            gqa: self.main_gqa_state.metadata(),
            gdn: self.main_gdn_state.metadata(),
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, cache_hit) = self.main.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3.5 Main replay input must match the prepared replay key"
        );
        recorder.main_cache_hit = cache_hit;
        trace::qwen35_state(|| {
            format!(
                "event=main_replays main_embed_key={:?} main_key={:?} main_embed_cache_hit={} main_cache_hit={} \
                 cache_hit={}",
                recorder.main_embed_key,
                recorder.main_key,
                recorder.main_embed_cache_hit,
                recorder.main_cache_hit,
                recorder.main_replay_cache_hit(),
            )
        });
        self.pending_transactions
            .push(model_batch_req.compute_seq(), microbatch.clone());
        Rc::clone(&self.hidden_output)
    }

    fn unembed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3.5 Output must consume the executor final-norm hidden workspace"
        );
        if num_target_hidden_states(model_batch_req.microbatch()) == 0 {
            return Qwen35ModelBatchResponse;
        }
        recorder.gather_unembed_key =
            Some(self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden));
        Qwen35ModelBatchResponse
    }

    fn sample_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        let microbatch = model_batch_req.microbatch();
        let num_sample_tokens = num_target_hidden_states(microbatch);
        assert_eq!(
            num_sample_tokens, recorder.num_sample_tokens,
            "qwen3.5 sampling rows must match the recording"
        );
        if num_sample_tokens == 0 {
            return;
        }
        if self.spec_probs.is_enabled() {
            self.record_rejection_sampling(recorder, microbatch);
        } else {
            let (sampling_key, sampling_arguments) = self.record_sampling(microbatch);
            recorder.sampling_key = Some(sampling_key);
            recorder.sampling_arguments = sampling_arguments;
        }
    }

    fn submit_main(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        if recorder.num_sample_tokens == 0 {
            return self.submit_main_recording(recorder);
        }
        self.submit_main_sampling_recording(recorder)
    }

    fn read_main(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        if recorder.num_sample_tokens == 0 {
            let timing = ModelOutputTiming {
                main_replay_elapsed: replay_elapsed,
                ..ModelOutputTiming::default()
            };
            return Qwen35SampledOutput {
                decisions: Vec::new(),
                timing,
            };
        }
        let (decisions, timing) = if self.spec_probs.is_enabled() {
            self.read_rejection_sampling(recorder, model_batch_req.microbatch(), replay_elapsed)
        } else {
            self.read_sampling(recorder.num_sample_tokens, replay_elapsed)
        };
        Qwen35SampledOutput { decisions, timing }
    }

    fn has_speculator(&self) -> bool {
        self.mtp.is_some()
    }

    fn embed_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
        sampled_output: &Self::SampledOutput,
    ) -> Self::ModelBatchHidden {
        assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3.5 MTPEmbed must consume the Main hidden workspace"
        );
        let microbatch = model_batch_req.microbatch();
        let num_decode_reqs = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .count();
        assert_eq!(
            sampled_output.decisions.len(),
            num_decode_reqs,
            "qwen3.5 speculator requires one decision per decode request"
        );
        self.record_mtp_embed(
            recorder,
            microbatch,
            Rc::clone(model_batch_hidden),
            &sampled_output.decisions,
        )
    }

    fn forward_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        self.record_mtp(recorder, model_batch_hidden)
    }

    fn unembed_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        self.record_mtp_unembed(recorder, model_batch_hidden);
        Qwen35ModelBatchResponse
    }

    fn sample_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        self.record_mtp_sampling(recorder);
    }

    fn submit_spec(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        self.submit_mtp_recording(recorder)
    }

    fn read_spec(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        mut sampled_output: Self::SampledOutput,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        let timing = self.read_mtp_batch(recorder, &mut sampled_output.decisions, replay_elapsed);
        sampled_output.timing.add_assign(timing);
        sampled_output
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen35SampledOutput::default()
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
        let key = Qwen35MainReplayKey::from_shapes(single_q_token_gqa_shape(), gdn_shape(1));

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
        let one_req = Qwen35MainReplayKey::from_shapes(single_q_token_gqa_shape(), gdn_shape(1));
        let two_reqs = Qwen35MainReplayKey::from_shapes(single_q_token_gqa_shape(), gdn_shape(2));

        assert_ne!(one_req, two_reqs);
    }

    #[test]
    fn test_main_key_shares_partial_output_reduce_topology() {
        let one_task_template_per_token = single_q_token_gqa_shape();
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

    fn single_q_token_gqa_shape() -> GQAReplayShape {
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
