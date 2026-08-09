use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_executor_core::backend::runtime::Runtime;
use inference_executor_core::model::qwen::v3_5::Qwen35DecodeDecision;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelBatchRequest;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::Qwen35SampledTokens;
use inference_executor_core::model::qwen::v3_5::gather_flat_indices;
use inference_executor_core::model::qwen::v3_5::num_main_output_rows;
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
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkExecution;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkProposalInput;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkRecording;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkContextArgs;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::replay::Replay;
use crate::sampling::rejection_replay::PreparedRejection;
use crate::sampling::rejection_replay::RejectionReplayKey;
use crate::sampling::rejection_replay::RejectionSamplerInput;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::rejection_replay::RejectionSamplingInput;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::DraftSamplingInput;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;
use crate::trace;

mod load;

pub use load::Qwen35ExecutorConfig;
use load::Qwen35ModelLayout;
pub use load::init_qwen_3_5_model;
pub use load::init_qwen_3_5_model_with_dspark;
pub use load::init_qwen_3_5_model_with_mtp;

include!("batch.rs");
include!("dspark.rs");
include!("main.rs");
include!("mtp.rs");
include!("recording.rs");
include!("sampling.rs");

#[allow(clippy::upper_case_acronyms)]
enum Qwen35Speculator {
    Vanilla,
    MTP(Box<Qwen35MTPSpeculator>),
    DSpark(Box<Qwen35DSparkSpeculator>),
}

struct Qwen35SpeculativeResources {
    rejection_sampling: Replay<RejectionSampling>,
    spec_probs: SpecProbsStore,
    target_distribution_indices: Buffer,
}

struct Qwen35MTPSpeculator {
    common: Qwen35SpeculativeResources,
    num_spec_tokens: usize,
    hidden_input: Rc<Buffer>,
    input_gather_flat_indices: Buffer,
    draft_distribution_indices: Buffer,
    previous_hidden: Buffer,
    embed: Replay<Qwen35MTPEmbed>,
    body: Replay<Qwen35MTP>,
    sampling: Replay<DraftSampling>,
    gqa_state: Qwen3xGQAState,
    execution: Qwen35MTPExecution,
}

struct Qwen35MTPExecution {
    requests: Vec<Qwen35MTPRequest>,
    draft_token_ids: Vec<i32>,
    draft_probs: Vec<f32>,
    read_elapsed: Duration,
    completed_steps: usize,
}

impl Qwen35MTPExecution {
    fn new(max_requests: usize, num_spec_tokens: usize) -> Self {
        let proposal_capacity = max_requests
            .checked_mul(num_spec_tokens)
            .expect("qwen3.5 MTP proposal capacity must fit usize");
        Self {
            requests: Vec::with_capacity(max_requests),
            draft_token_ids: Vec::with_capacity(proposal_capacity),
            draft_probs: Vec::with_capacity(proposal_capacity),
            read_elapsed: Duration::ZERO,
            completed_steps: 0,
        }
    }

    fn begin(&mut self, requests: Vec<Qwen35MTPRequest>) {
        self.requests = requests;
        self.draft_token_ids.clear();
        self.draft_probs.clear();
        self.read_elapsed = Duration::ZERO;
        self.completed_steps = 0;
    }

    fn push_step(&mut self, draft_token_ids: &[i32], draft_probs: &[f32], read_elapsed: Duration) {
        assert_eq!(draft_token_ids.len(), draft_probs.len());
        self.draft_token_ids.extend_from_slice(draft_token_ids);
        self.draft_probs.extend_from_slice(draft_probs);
        self.read_elapsed += read_elapsed;
        self.completed_steps += 1;
    }
}

struct Qwen35DSparkSpeculator {
    common: Qwen35SpeculativeResources,
    execution: Qwen3xDSparkExecution,
}

impl Qwen35Speculator {
    fn is_enabled(&self) -> bool {
        !matches!(self, Self::Vanilla)
    }

    fn is_mtp(&self) -> bool {
        matches!(self, Self::MTP(_))
    }

    fn is_dspark(&self) -> bool {
        matches!(self, Self::DSpark(_))
    }

    fn num_spec_tokens(&self) -> usize {
        match self {
            Self::Vanilla => 0,
            Self::MTP(mtp) => mtp.num_spec_tokens,
            Self::DSpark(dspark) => dspark.execution.num_spec_tokens(),
        }
    }

    fn mtp(&self) -> &Qwen35MTPSpeculator {
        match self {
            Self::MTP(mtp) => mtp,
            Self::Vanilla | Self::DSpark(_) => panic!("qwen3.5 executor has no MTP resources"),
        }
    }

    fn mtp_mut(&mut self) -> &mut Qwen35MTPSpeculator {
        match self {
            Self::MTP(mtp) => mtp,
            Self::Vanilla | Self::DSpark(_) => panic!("qwen3.5 executor has no MTP resources"),
        }
    }

    fn dspark(&self) -> &Qwen35DSparkSpeculator {
        match self {
            Self::DSpark(dspark) => dspark,
            Self::Vanilla | Self::MTP(_) => panic!("qwen3.5 executor has no DSpark resources"),
        }
    }

    fn dspark_mut(&mut self) -> &mut Qwen35DSparkSpeculator {
        match self {
            Self::DSpark(dspark) => dspark,
            Self::Vanilla | Self::MTP(_) => panic!("qwen3.5 executor has no DSpark resources"),
        }
    }

    fn common(&self) -> &Qwen35SpeculativeResources {
        match self {
            Self::Vanilla => panic!("qwen3.5 Vanilla executor has no speculative resources"),
            Self::MTP(mtp) => &mtp.common,
            Self::DSpark(dspark) => &dspark.common,
        }
    }

    fn common_mut(&mut self) -> &mut Qwen35SpeculativeResources {
        match self {
            Self::Vanilla => panic!("qwen3.5 Vanilla executor has no speculative resources"),
            Self::MTP(mtp) => &mut mtp.common,
            Self::DSpark(dspark) => &mut dspark.common,
        }
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.gqa_state.reset_req_slots(request_slots);
                mtp.common.spec_probs.reset_req_slots(request_slots);
            },
            Self::DSpark(dspark) => {
                dspark.execution.reset_req_slots(request_slots);
                dspark.common.spec_probs.reset_req_slots(request_slots);
            },
        }
    }
}

pub struct Qwen35Executor {
    model_name: String,
    default_stop_sequences: Vec<Vec<Token>>,
    config: Qwen35ExecutorConfig,
    runtime: MetalRuntime,
    layout: Qwen35ModelLayout,
    token_ids: Buffer,
    token_hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    main_embed: Replay<Qwen35MainEmbed>,
    main: Replay<Qwen35Main>,
    gather_unembed: Replay<Qwen35GatherUnembed>,
    sampling: Replay<Sampling>,
    sampler: Rc<TopKSampling>,
    sampler_bounds: TopKSamplingBounds,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    main_gqa_state: Qwen3xGQAState,
    main_gdn_state: Qwen3xGDNState,
    speculator: Qwen35Speculator,
    pages: PageArena,
    pending_transactions: Qwen35PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
    num_runtime_page_ids_per_main_block: usize,
}

pub struct Qwen35ModelOpsRecorder {
    main_embed_key: Qwen35MainEmbedReplayKey,
    main_embed_arguments: ReplayArguments,
    main_key: Qwen35MainReplayKey,
    main_arguments: ReplayArguments,
    main_embed_cache_hit: bool,
    main_cache_hit: bool,
    main_gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    main_gather_unembed_arguments: ReplayArguments,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    rejection_key: Option<RejectionReplayKey>,
    rejection_arguments: ReplayArguments,
    rejection_prepared: Option<PreparedRejection>,
    rejection_build_elapsed: Duration,
    num_main_sample_rows: usize,
    mtp_embed_key: Option<Qwen35MTPEmbedReplayKey>,
    mtp_embed_arguments: ReplayArguments,
    mtp_key: Option<Qwen35MTPReplayKey>,
    mtp_gqa_shape: Option<GQAReplayShape>,
    mtp_gqa_topology: Option<crate::attn::gqa::backend::GQAReplayTopology>,
    mtp_microbatch: Option<Qwen35Microbatch>,
    mtp_sampler_configs: Vec<SamplerConfig>,
    mtp_sample_positions: Vec<u32>,
    mtp_embed_cache_hit: bool,
    mtp_gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    mtp_gather_unembed_arguments: ReplayArguments,
    mtp_sampling_key: Option<TopKSamplingReplayKey>,
    mtp_sampling_arguments: ReplayArguments,
    mtp_sample_req_slots: Vec<u32>,
    mtp_sample_decision_indices: Vec<usize>,
    mtp_build_elapsed: Duration,
    dspark: Qwen3xDSparkRecording,
}

impl Qwen35ModelOpsRecorder {
    fn main_replay_cache_hit(&self) -> bool {
        self.main_embed_cache_hit && self.main_cache_hit
    }

    fn num_mtp_sample_rows(&self) -> usize {
        let num_rows = self.mtp_sample_req_slots.len();
        debug_assert_eq!(self.mtp_sample_decision_indices.len(), num_rows);
        debug_assert_eq!(self.mtp_sampler_configs.len(), num_rows);
        debug_assert_eq!(self.mtp_sample_positions.len(), num_rows);
        num_rows
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]

struct Qwen35MTPRequest {
    num_tokens: usize,
    current_token_ids: Vec<i32>,
    prefill_token_ids_by_step: Vec<Vec<i32>>,
    next_token_id: Option<i32>,
    decision_index: Option<usize>,
}

struct Qwen35MTPBatch {
    microbatch: Qwen35Microbatch,
    input_gather_flat_indices: Vec<u32>,
    draft_distribution_indices: Vec<u32>,
    sampler_configs: Vec<SamplerConfig>,
    sample_positions: Vec<u32>,
}

fn mtp_proposal_sample_position(token_index: u32, num_tokens: usize, step_index: usize) -> u32 {
    token_index
        .checked_add(
            num_tokens
                .try_into()
                .expect("qwen3.5 MTP request token count must fit u32"),
        )
        .and_then(|position| position.checked_add(step_index.try_into().expect("qwen3.5 MTP step index must fit u32")))
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
        self.speculator.reset_req_slots(request_slots);
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
        let num_spec_tokens = match &self.speculator {
            Qwen35Speculator::MTP(mtp) => mtp.num_spec_tokens,
            Qwen35Speculator::Vanilla | Qwen35Speculator::DSpark(_) => 0,
        };
        let model_batch_request =
            Qwen35ModelBatchRequest::from_core_batch(core_batch_req, num_spec_tokens, sampler_configs);
        let microbatch = model_batch_request.microbatch();
        trace::qwen35_state(|| {
            format!(
                "event=batch_from_core seq={} req_slots={:?} token_indices={:?} num_spec_tokens={:?} seeds={:?} \
                 total_tokens={} num_reqs={}",
                batch_seq,
                microbatch.req_slots(),
                microbatch.token_indices(),
                microbatch
                    .req_slots()
                    .iter()
                    .enumerate()
                    .map(|(req_index, _)| microbatch.num_spec_tokens(req_index))
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
        let num_main_active_tokens = microbatch
            .total_tokens()
            .try_into()
            .expect("qwen3.5 Main token count must fit u32");
        let num_main_total_tokens = self.main.component().replay_token_capacity(num_main_active_tokens);
        let prepare_start = Instant::now();
        let gqa_start = Instant::now();
        if self.speculator.is_dspark() {
            self.main_gqa_state
                .prepare_page_span(core_batch_req, self.num_runtime_page_ids_per_main_block, 0);
            self.speculator.dspark().execution.prepare_page_span(
                core_batch_req,
                self.num_runtime_page_ids_per_main_block,
                self.num_main_gqa_page_ids_per_block(),
            );
        } else {
            self.main_gqa_state.prepare_pages(core_batch_req);
        }
        let gqa_shape = self.main_gqa_state.prepare_metadata_bucketed_with_token_capacity(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            num_main_total_tokens,
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
        let gdn_shape = self.main_gdn_state.prepare_metadata_bucketed_with_token_capacity(
            microbatch.cu_tokens(),
            &gdn_prepared,
            num_main_total_tokens,
        );
        let gdn_metadata_elapsed = gdn_metadata_start.elapsed();
        debug_assert_eq!(gdn_shape.num_tokens as usize, microbatch.total_tokens());
        debug_assert_eq!(gdn_shape.num_reqs as usize, microbatch.num_reqs());
        if self.speculator.is_mtp() {
            self.speculator.mtp().body.component().prepare_pages(core_batch_req);
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
        let num_main_active_tokens = model_batch_request
            .microbatch()
            .total_tokens()
            .try_into()
            .expect("qwen3.5 Main token count must fit u32");
        let (main_embed_key, main_embed_arguments) = self.main_embed.component().prepare_replay(num_main_active_tokens);
        let (main_key, mut main_arguments) = self.main.component().prepare_replay(
            num_main_active_tokens,
            self.main_gqa_state.metadata().replay_shape(),
            self.main_gqa_state.replay_topology(),
            self.main_gdn_state.metadata().replay_shape(),
            self.main_gdn_state.replay_topology(),
        );
        self.main_gqa_state.add_private_replay_arguments(&mut main_arguments);
        self.main_gdn_state.add_private_replay_arguments(&mut main_arguments);
        trace::qwen35_state(|| {
            format!(
                "event=begin_ops_recording main_embed_key={:?} main_key={:?}",
                main_embed_key, main_key
            )
        });
        Qwen35ModelOpsRecorder {
            main_embed_key,
            main_embed_arguments,
            main_key,
            main_arguments,
            main_embed_cache_hit: false,
            main_cache_hit: false,
            dspark: Qwen3xDSparkRecording::new(),
            main_gather_unembed_key: None,
            main_gather_unembed_arguments: ReplayArguments::new(),
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            rejection_key: None,
            rejection_arguments: ReplayArguments::new(),
            rejection_prepared: None,
            rejection_build_elapsed: Duration::ZERO,
            num_main_sample_rows: num_main_output_rows(model_batch_request.microbatch()),
            mtp_embed_key: None,
            mtp_embed_arguments: ReplayArguments::new(),
            mtp_key: None,
            mtp_gqa_shape: None,
            mtp_gqa_topology: None,
            mtp_microbatch: None,
            mtp_sampler_configs: Vec::new(),
            mtp_sample_positions: Vec::new(),
            mtp_embed_cache_hit: false,
            mtp_gather_unembed_key: None,
            mtp_gather_unembed_arguments: ReplayArguments::new(),
            mtp_sampling_key: None,
            mtp_sampling_arguments: ReplayArguments::new(),
            mtp_sample_req_slots: Vec::new(),
            mtp_sample_decision_indices: Vec::new(),
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
            gqa_replay_topology: self.main_gqa_state.replay_topology(),
            gdn: self.main_gdn_state.metadata(),
            gdn_replay_topology: self.main_gdn_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, cache_hit) = self.main.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3.5 Main replay input must match the prepared replay key"
        );
        recorder.main_cache_hit = cache_hit;
        if self.speculator.is_dspark() {
            let context_input = Qwen3xDSparkContextArgs {
                num_tokens: microbatch
                    .total_tokens()
                    .try_into()
                    .expect("qwen3.5 DSpark context token count must fit u32"),
                req_slots: self.main_gqa_state.metadata().req_slots(),
                flat_token_indices: self.main_gqa_state.metadata().flat_token_indices(),
                pages: self.pages.buffer(),
            };
            self.speculator
                .dspark_mut()
                .execution
                .record_context(&runtime, &context_input, &mut recorder.dspark);
        }
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
        if num_main_output_rows(model_batch_req.microbatch()) == 0 {
            return Qwen35ModelBatchResponse;
        }
        let (gather_unembed_key, gather_unembed_arguments) =
            self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden);
        recorder.main_gather_unembed_key = Some(gather_unembed_key);
        recorder.main_gather_unembed_arguments = gather_unembed_arguments;
        Qwen35ModelBatchResponse
    }

    fn sample_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        let microbatch = model_batch_req.microbatch();
        let num_main_sample_rows = num_main_output_rows(microbatch);
        assert_eq!(
            num_main_sample_rows, recorder.num_main_sample_rows,
            "qwen3.5 sampling rows must match the recording"
        );
        if num_main_sample_rows == 0 {
            return;
        }
        if self.speculator.is_enabled() {
            self.record_rejection_sampling(recorder, microbatch);
        } else {
            let (sampling_key, sampling_arguments) = self.record_sampling(microbatch);
            recorder.sampling_key = Some(sampling_key);
            recorder.sampling_arguments = sampling_arguments;
        }
    }

    fn submit_main(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        self.submit_main_recording(recorder)
    }

    fn read_main(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        if recorder.num_main_sample_rows == 0 {
            let timing = ModelOutputTiming {
                main_replay_elapsed: replay_elapsed,
                ..ModelOutputTiming::default()
            };
            return Qwen35SampledOutput {
                decisions: Vec::new(),
                timing,
            };
        }
        let (decisions, timing) = if self.speculator.is_enabled() {
            self.read_rejection_sampling(recorder, model_batch_req.microbatch(), replay_elapsed)
        } else {
            self.read_sampling(recorder.num_main_sample_rows, replay_elapsed)
        };
        Qwen35SampledOutput { decisions, timing }
    }

    fn run_spec(&self, _model_batch_req: &Self::ModelBatchRequest, sampled_output: &Self::SampledOutput) -> bool {
        self.speculator.is_mtp() || (self.speculator.is_dspark() && !sampled_output.decisions.is_empty())
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
            "qwen3.5 speculator must follow the Main hidden workspace"
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
        if self.speculator.is_dspark() {
            self.record_dspark_embed(recorder, microbatch, &sampled_output.decisions)
        } else {
            self.record_mtp_embed(
                recorder,
                microbatch,
                Rc::clone(model_batch_hidden),
                &sampled_output.decisions,
            )
        }
    }

    fn forward_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        if self.speculator.is_dspark() {
            self.record_dspark(recorder, model_batch_hidden)
        } else {
            self.record_mtp(recorder, model_batch_hidden)
        }
    }

    fn unembed_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        if self.speculator.is_dspark() {
            self.record_dspark_gather_unembed(recorder, model_batch_hidden);
        } else {
            self.record_mtp_gather_unembed(recorder, model_batch_hidden);
        }
        Qwen35ModelBatchResponse
    }

    fn sample_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        if self.speculator.is_dspark() {
            self.record_dspark_sampling(recorder);
        } else {
            self.record_mtp_sampling(recorder);
        }
    }

    fn submit_spec(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        if self.speculator.is_dspark() {
            let runtime = self.replay_runtime();
            self.speculator.dspark().execution.submit(&runtime, &recorder.dspark)
        } else {
            self.submit_mtp_recording(recorder)
        }
    }

    fn read_spec(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        mut sampled_output: Self::SampledOutput,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        if self.speculator.is_dspark() {
            sampled_output.timing.spec_replay_elapsed += replay_elapsed;
            sampled_output.timing.spec_passes += 1;
            let read_start = Instant::now();
            sampled_output.decisions = self.read_dspark_proposal(recorder, sampled_output.decisions);
            sampled_output.timing.spec_read_elapsed += read_start.elapsed();
            sampled_output
        } else {
            let timing = self.read_mtp_proposal(recorder, &mut sampled_output.decisions, replay_elapsed);
            sampled_output.timing.add_assign(timing);
            sampled_output
        }
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
    use std::rc::Rc;

    use inference_backend_metal::components::GQAComputePath;
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::Stream;
    use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
    use inference_executor_core::attn::GDNReplayShape;
    use inference_executor_core::attn::GQAReplayShape;
    use inference_executor_core::attn::gdn::state::GDNStateTxn;
    use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
    use inference_executor_core::sampling::SamplerConfig;
    use inference_executor_core::sampling::TopKSamplingBounds;

    use super::DraftSampling;
    use super::DraftSamplingInput;
    use super::MetalReplayRuntime;
    use super::Qwen35GatherUnembedReplayKey;
    use super::Qwen35MainReplayKey;
    use super::Replay;
    use super::TopKSampling;
    use super::TopKSamplingOutputBuffers;
    use super::TopKSamplingWriteDistributionOutput;
    use super::mtp_proposal_sample_position;
    use super::mtp_sample_replay_shape;
    use super::replay_bucket_capacity;
    use crate::attn::gdn::backend::GDNReplayTopology;
    use crate::attn::gqa::backend::GQAReplayTopology;

    #[test]
    fn test_mtp_proposal_sample_position_advances_per_step() {
        assert_eq!(mtp_proposal_sample_position(17, 3, 0), 21);
        assert_eq!(mtp_proposal_sample_position(17, 3, 3), 24);
    }

    #[test]
    fn test_mtp_sample_replay_shape_caps_bucket_at_max_requests() {
        let config = SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        };
        let bounds = TopKSamplingBounds::from_config(&config, 8, 8).unwrap();
        let configs = vec![config; 5];
        let shape = mtp_sample_replay_shape(bounds, &configs, 6);
        assert_eq!(shape.num_active_sampling_inputs, 5);
        assert_eq!(shape.num_total_sampling_inputs, 6);

        let device = Device::system_default();
        let sampler = Rc::new(TopKSampling::new(&device, bounds));
        let output = TopKSamplingOutputBuffers::new(&device, bounds);
        let logits_elements = bounds.max_sampling_inputs as usize * bounds.vocab_size as usize;
        let logits = Buffer::new_zeroed_elements(&device, logits_elements, Dtype::Bfloat16);
        let distribution_token_ids = Buffer::new_zeroed_elements(&device, 6, Dtype::Int32);
        let distribution_probs = Buffer::new_zeroed_elements(&device, 6, Dtype::Float32);
        let distribution_indices = Buffer::new_zeroed_elements(&device, 6, Dtype::Uint32);
        let input = DraftSamplingInput {
            shape,
            logits: &logits,
            output: output.as_output(),
            sparse: TopKSamplingWriteDistributionOutput {
                token_ids: &distribution_token_ids,
                probs: &distribution_probs,
                output_distribution_indices: &distribution_indices,
                max_k: 1,
                num_output_distributions: 6,
            },
        };
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let mut replay = Replay::new("qwen3.5 MTP sampling capacity test", DraftSampling { sampler });

        let (key, cache_hit) = replay.record(&runtime, &input);

        assert!(!cache_hit);
        assert_eq!(key.num_sampling_input_capacity, 6);
    }

    #[test]
    fn test_main_key() {
        let topology = single_gqa_topology();
        let gdn_topology = gdn_topology();
        let key = Qwen35MainReplayKey::from_shapes(single_q_token_gqa_shape(), topology, gdn_shape(1), gdn_topology);

        assert_eq!(key.debug_parts(), (4, 4, 4, 4, 1, 4, topology, gdn_topology));
    }

    #[test]
    fn test_main_key_tiled() {
        let topology = tiled_gqa_topology();
        let gdn_topology = gdn_topology();
        let key = Qwen35MainReplayKey::from_shapes(tiled_gqa_shape(), topology, gdn_shape(1), gdn_topology);

        assert_eq!(key.debug_parts(), (4, 4, 1, 1, 1, 4, topology, gdn_topology));
    }

    #[test]
    fn test_main_key_uses_gdn_request_capacity() {
        let topology = gdn_topology();
        let one_req = Qwen35MainReplayKey::from_shapes(
            single_q_token_gqa_shape(),
            single_gqa_topology(),
            GDNReplayShape::new(1, 2, 4, 4),
            topology,
        );
        let two_reqs = Qwen35MainReplayKey::from_shapes(
            single_q_token_gqa_shape(),
            single_gqa_topology(),
            GDNReplayShape::new(2, 2, 4, 4),
            topology,
        );
        let larger_capacity = Qwen35MainReplayKey::from_shapes(
            single_q_token_gqa_shape(),
            single_gqa_topology(),
            GDNReplayShape::new(2, 4, 4, 4),
            topology,
        );

        assert_eq!(one_req, two_reqs);
        assert_ne!(one_req, larger_capacity);
    }

    #[test]
    fn test_main_key_shares_partial_output_reduce_topology() {
        let one_task_template_per_token = single_q_token_gqa_shape();
        let multiple_task_templates_per_token = GQAReplayShape {
            reduce_sdpa_partial_outputs: true,
            ..one_task_template_per_token
        };

        assert_eq!(
            Qwen35MainReplayKey::from_shapes(
                one_task_template_per_token,
                single_gqa_topology(),
                gdn_shape(1),
                gdn_topology(),
            ),
            Qwen35MainReplayKey::from_shapes(
                multiple_task_templates_per_token,
                single_gqa_topology(),
                gdn_shape(1),
                gdn_topology(),
            )
        );
    }

    #[test]
    fn test_gather_unembed_key_separates_main_output_rows() {
        let one_main_output = one_req_batch(4, 0);
        let three_main_outputs = one_req_batch(4, 2);

        assert_ne!(
            Qwen35GatherUnembedReplayKey::from_microbatch(&one_main_output),
            Qwen35GatherUnembedReplayKey::from_microbatch(&three_main_outputs)
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
        let num_sample_rows = num_spec_tokens + 1;
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
                .map(|token_offset| token_offset + num_sample_rows >= num_tokens)
                .collect(),
        )
    }

    fn single_q_token_gqa_shape() -> GQAReplayShape {
        GQAReplayShape {
            num_tokens: 4,
            total_tokens: 4,
            num_q_token_tiles: 4,
            total_q_token_tiles: 4,
            num_sdpa_map_task_templates: 4,
            total_sdpa_map_task_templates: 4,
            reduce_sdpa_partial_outputs: false,
        }
    }

    fn gdn_shape(num_reqs: u32) -> GDNReplayShape {
        GDNReplayShape::new(num_reqs, num_reqs, 4, 4)
    }

    fn gdn_topology() -> GDNReplayTopology {
        GDNReplayTopology {
            materialize_candidate_states: true,
            qkvabz_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }

    fn tiled_gqa_shape() -> GQAReplayShape {
        GQAReplayShape {
            num_tokens: 4,
            total_tokens: 4,
            num_q_token_tiles: 1,
            total_q_token_tiles: 1,
            num_sdpa_map_task_templates: 1,
            total_sdpa_map_task_templates: 1,
            reduce_sdpa_partial_outputs: true,
        }
    }

    fn single_gqa_topology() -> GQAReplayTopology {
        GQAReplayTopology {
            compute_path: GQAComputePath::SingleQueryToken {
                kv_token_tile_size: 256,
                num_threads_per_threadblock: 256,
                q_head_tile_size: 6,
            },
            qgkv_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }

    fn tiled_gqa_topology() -> GQAReplayTopology {
        GQAReplayTopology {
            compute_path: GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            },
            qgkv_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }
}
