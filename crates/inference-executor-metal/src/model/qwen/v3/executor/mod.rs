use std::collections::VecDeque;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::ModelOutputTiming;
use inference_executor_core::model::ReplayableModel;
use inference_executor_core::model::qwen::v3::Qwen3DecodeDecision;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::Qwen3SampledTokens;
use inference_executor_core::model::qwen::v3::gather_flat_indices;
use inference_executor_core::model::qwen::v3::num_main_output_rows;
use inference_executor_core::model::qwen::v3::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3::sample_sampler_configs;
use inference_executor_core::model::qwen::v3::sample_token_positions;
use inference_executor_core::model::qwen::v3::to_core_batch_resp;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3ModelWeightBindings;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::SparseRejectionSamplingReqParams;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::build_spec_prefill_selection;
use inference_runtime_core::compute::BatchDevReq;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::runtime::RawComputeSlotSeq;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::model::embedding::Embed;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::page_arena::PageArena;
use crate::model::qwen::split_main_lane_page_ids;
use crate::model::qwen::v3::main::Qwen3Main;
use crate::model::qwen::v3::main::Qwen3MainArgs;
use crate::model::qwen::v3::main::Qwen3MainReplayKey;
use crate::model::qwen::v3::main::embed::Qwen3MainEmbed;
use crate::model::qwen::v3::main::embed::Qwen3MainEmbedArgs;
use crate::model::qwen::v3::main::embed::Qwen3MainEmbedReplayKey;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembed;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembedArgs;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembedReplayKey;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkDecodeRecording;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkExecution;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkPrefillRecording;
use crate::model::qwen::v3_x::dspark::execution::Qwen3xDSparkProposalInput;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;
use crate::model::unembedding::Unembed;
use crate::replay::Replay;
use crate::sampling::rejection_replay::PreparedRejection;
use crate::sampling::rejection_replay::RejectionReplayKey;
use crate::sampling::rejection_replay::RejectionSamplerInput;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::rejection_replay::RejectionSamplingInput;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;

mod load;

pub use load::Qwen3ExecutorConfig;
use load::Qwen3ModelLayout;
pub use load::init_qwen_3_model;
pub use load::init_qwen_3_model_with_dspark;

include!("batch.rs");
include!("dspark.rs");
include!("main.rs");
include!("recording.rs");
include!("sampling.rs");

const PAGE_ARENA_STATE_FILES: PageArenaStateSnapshotFiles =
    PageArenaStateSnapshotFiles::new(StateSnapshotFile::PageArena);
const MAIN_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);
const DSPARK_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::DSparkGQARequestPageTable);
const VANILLA_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
];
const DSPARK_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
    DSPARK_GQA_STATE_FILES.request_page_table(),
];

enum Qwen3Speculator {
    Vanilla,
    DSpark(Box<Qwen3DSparkSpeculator>),
}

enum Qwen3WeightSource {
    Vanilla,
    DSpark {
        model_dir: PathBuf,
        config: Box<Qwen3xDSparkConfig>,
    },
}

struct Qwen3DSparkSpeculator {
    execution: Qwen3xDSparkExecution,
    rejection_sampling: Replay<RejectionSampling>,
    spec_probs: SpecProbsStore,
    target_distribution_indices: Buffer,
}

impl Qwen3Speculator {
    fn is_dspark(&self) -> bool {
        matches!(self, Self::DSpark(_))
    }

    fn num_spec_tokens(&self) -> usize {
        match self {
            Self::Vanilla => 0,
            Self::DSpark(dspark) => dspark.execution.num_spec_tokens(),
        }
    }

    fn num_gqa_page_ids_per_main_lane_block(&self) -> usize {
        match self {
            Self::Vanilla => 0,
            Self::DSpark(dspark) => dspark.execution.num_gqa_page_ids_per_block(),
        }
    }

    fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        match self {
            Self::Vanilla => {
                assert!(
                    page_ids.is_empty(),
                    "Qwen3 Vanilla Main cache block must not contain speculator page IDs"
                )
            },
            Self::DSpark(dspark) => dspark.execution.write_page_ids(req_slot, block_index, page_ids),
        }
    }

    fn dspark(&self) -> &Qwen3DSparkSpeculator {
        match self {
            Self::Vanilla => panic!("Qwen3 Vanilla executor has no DSpark resources"),
            Self::DSpark(dspark) => dspark,
        }
    }

    fn dspark_mut(&mut self) -> &mut Qwen3DSparkSpeculator {
        match self {
            Self::Vanilla => panic!("Qwen3 Vanilla executor has no DSpark resources"),
            Self::DSpark(dspark) => dspark,
        }
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.reset_req_slots(request_slots);
            dspark.spec_probs.reset_req_slots(request_slots);
        }
    }

    fn clear_replay_cache(&mut self) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.clear_replay_cache();
            dspark.rejection_sampling.clear();
        }
    }

    fn write_full_state(&self, writer: &mut StateSnapshotWriter) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::DSpark(dspark) => dspark.execution.write_full_state(writer, DSPARK_GQA_STATE_FILES),
        }
    }

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::DSpark(dspark) => {
                dspark
                    .execution
                    .write_selected_state(writer, DSPARK_GQA_STATE_FILES, request_slot_ranges)
            },
        }
    }

    fn unload_state(&mut self) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.unload_state();
        }
    }

    fn allocate_resources(&mut self, device: &inference_backend_metal::metal::Device) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.allocate_resources(device);
        }
    }

    fn release_resources(&mut self) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.release_resources();
        }
    }

    fn attach_state(&mut self) {
        if let Self::DSpark(dspark) = self {
            dspark.execution.attach_state();
        }
    }

    fn read_full_state(&mut self, reader: &mut StateSnapshotReader) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::DSpark(dspark) => dspark.execution.read_full_state(reader, DSPARK_GQA_STATE_FILES),
        }
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::DSpark(dspark) => {
                dspark
                    .execution
                    .read_selected_state(reader, DSPARK_GQA_STATE_FILES, request_slot_ranges)
            },
        }
    }
}

pub struct Qwen3Executor {
    model_name: String,
    model_dir: PathBuf,
    weight_source: Qwen3WeightSource,
    model_config: Qwen3ModelConfig,
    default_stop_sequences: Vec<Vec<Token>>,
    config: Qwen3ExecutorConfig,
    runtime: MetalRuntime,
    layout: Qwen3ModelLayout,
    token_ids: Buffer,
    token_hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    main_embed: Replay<Qwen3MainEmbed>,
    main: Replay<Qwen3Main>,
    gather_unembed: Replay<Qwen3GatherUnembed>,
    sampling: Replay<Sampling>,
    sampler: Rc<TopKSampling>,
    sampler_bounds: TopKSamplingBounds,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    main_gqa_state: Qwen3MainGQAState,
    speculator: Qwen3Speculator,
    pages: PageArena,
    pending_transactions: Qwen3PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
    num_gqa_page_ids_per_main_lane_block: usize,
    unloaded_embed: Option<Embed>,
    unloaded_unembed: Option<Unembed>,
}

impl Qwen3Executor {
    fn state_snapshot_files(&self) -> &'static [StateSnapshotFile] {
        match &self.speculator {
            Qwen3Speculator::Vanilla => VANILLA_STATE_SNAPSHOT_FILES,
            Qwen3Speculator::DSpark(_) => DSPARK_STATE_SNAPSHOT_FILES,
        }
    }

    pub fn clear_replay_cache(&mut self) {
        self.main_embed.clear();
        self.main.clear();
        self.gather_unembed.clear();
        self.sampling.clear();
        self.speculator.clear_replay_cache();
    }

    pub fn unload_weights(&mut self) {
        let embed = self.main_embed.component_mut().unload_weights();
        let unembed = self.gather_unembed.component_mut().unload_weights();

        self.main.component_mut().unset_residual_capture();
        if let Qwen3Speculator::DSpark(dspark) = &mut self.speculator {
            dspark.execution.unload_weights();
        }
        self.main.component_mut().unload_weights();

        let mut unembed = Rc::try_unwrap(unembed)
            .unwrap_or_else(|_| panic!("qwen3 Main unembed must be uniquely owned during weight unloading"));
        unembed.unload_weights();
        self.unloaded_unembed = Some(unembed);

        let mut embed = Rc::try_unwrap(embed)
            .unwrap_or_else(|_| panic!("qwen3 Main embed must be uniquely owned during weight unloading"));
        embed.unload_weights();
        self.unloaded_embed = Some(embed);
    }

    pub fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        let device = self.runtime.device().clone();
        let mut store = SafeTensorStore::from_model_dir(&self.model_dir)?;
        let Qwen3ModelWeightBindings { embed, main, unembed } =
            resolve_qwen3_model_weight_bindings(&self.model_config, store.index().tensor_names())?;

        let mut loaded_embed = self
            .unloaded_embed
            .take()
            .expect("qwen3 Main embed shell must exist during weight loading");
        loaded_embed.load_weights(&device, &mut store, embed)?;
        let loaded_embed = Rc::new(loaded_embed);
        let mut loaded_unembed = self
            .unloaded_unembed
            .take()
            .expect("qwen3 Main unembed shell must exist during weight loading");
        loaded_unembed.load_weights(&device, &mut store, unembed)?;
        let loaded_unembed = Rc::new(loaded_unembed);
        self.main
            .component_mut()
            .load_weights(&device, &mut store, &self.model_config, main)?;

        let residual_capture = match (&mut self.speculator, &self.weight_source) {
            (Qwen3Speculator::Vanilla, Qwen3WeightSource::Vanilla) => None,
            (Qwen3Speculator::DSpark(dspark), Qwen3WeightSource::DSpark { model_dir, config }) => {
                dspark
                    .execution
                    .load_weights(&device, model_dir, config, &loaded_embed, &loaded_unembed)?;
                let capture: Rc<dyn MainResidualCapture> = dspark.execution.main_feature_projector();
                Some(capture)
            },
            _ => panic!("qwen3 speculator and weight source must have matching variants"),
        };
        self.main.component_mut().set_residual_capture(residual_capture);
        self.main_embed.component_mut().load_weights(loaded_embed);
        self.gather_unembed.component_mut().load_weights(loaded_unembed);
        Ok(())
    }

    fn write_state(&self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        assert!(
            self.pending_transactions.transactions.is_empty(),
            "qwen3 state snapshots require all pending model transactions to complete"
        );
        let mut writer = StateSnapshotWriter::new(
            snapshot_path,
            self.state_snapshot_files(),
            plan,
            self.runtime.buffer_io(),
        )?;
        match plan {
            ExecutorHibernationPlan::All => {
                self.main_gqa_state
                    .write_full_state(&mut writer, MAIN_GQA_STATE_FILES)?;
                self.speculator.write_full_state(&mut writer)?;
                self.pages.write_full_state(&mut writer, PAGE_ARENA_STATE_FILES)?;
            },
            ExecutorHibernationPlan::Selected {
                request_slot_ranges,
                page_id_ranges,
            } => {
                self.main_gqa_state
                    .write_selected_state(&mut writer, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                self.speculator.write_selected_state(&mut writer, request_slot_ranges)?;
                self.pages
                    .write_selected_state(&mut writer, PAGE_ARENA_STATE_FILES, page_id_ranges)?;
            },
        }
        writer.commit()
    }

    pub fn unload_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError> {
        self.write_state(snapshot_path, plan)?;

        self.speculator.unload_state();
        self.main.component_mut().unload_state();
        self.main_gqa_state.release_resources();
        self.pages.release_resources();
        Ok(())
    }

    pub fn load_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError> {
        let mut reader = StateSnapshotReader::open(
            snapshot_path,
            self.state_snapshot_files(),
            plan,
            self.runtime.buffer_io(),
        )?;
        let device = self.runtime.device().clone();
        self.main_gqa_state.allocate_resources(&device);
        self.speculator.allocate_resources(&device);
        self.pages.allocate_resources(&device);

        let result = (|| {
            match plan {
                ExecutorHibernationPlan::All => {
                    self.main_gqa_state.read_full_state(&mut reader, MAIN_GQA_STATE_FILES)?;
                    self.speculator.read_full_state(&mut reader)?;
                    self.pages.read_full_state(&mut reader, PAGE_ARENA_STATE_FILES)?;
                },
                ExecutorHibernationPlan::Selected {
                    request_slot_ranges,
                    page_id_ranges,
                } => {
                    self.main_gqa_state
                        .read_selected_state(&mut reader, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                    self.speculator.read_selected_state(&mut reader, request_slot_ranges)?;
                    self.pages
                        .read_selected_state(&mut reader, PAGE_ARENA_STATE_FILES, page_id_ranges)?;
                },
            }
            reader.finish()
        })();
        if let Err(error) = result {
            self.pages.release_resources();
            self.speculator.release_resources();
            self.main_gqa_state.release_resources();
            return Err(error);
        }

        self.main.component_mut().load_state(&self.main_gqa_state);
        self.speculator.attach_state();
        Ok(())
    }
}

pub struct Qwen3ModelOpsRecorder {
    main_embed_key: Qwen3MainEmbedReplayKey,
    main_embed_arguments: ReplayArguments,
    main_key: Qwen3MainReplayKey,
    main_arguments: ReplayArguments,
    gather_unembed_key: Option<Qwen3GatherUnembedReplayKey>,
    gather_unembed_arguments: ReplayArguments,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    rejection_key: Option<RejectionReplayKey>,
    rejection_arguments: ReplayArguments,
    rejection_prepared: Option<PreparedRejection>,
    dspark_prefill: Option<Qwen3xDSparkPrefillRecording>,
    dspark_decode: Option<Qwen3xDSparkDecodeRecording>,
    num_main_sample_rows: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3ModelBatchResponse;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen3SampledOutput {
    decisions: Vec<Qwen3DecodeDecision>,
    timing: ModelOutputTiming,
}

struct Qwen3PendingTransactions {
    transactions: VecDeque<Qwen3PendingTransaction>,
}

struct Qwen3PendingTransaction {
    compute_seq: RawComputeSlotSeq,
}

impl Qwen3PendingTransactions {
    fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
        }
    }

    fn push(&mut self, compute_seq: RawComputeSlotSeq) {
        if let Some(last) = self.transactions.back() {
            assert!(
                last.compute_seq < compute_seq,
                "qwen3 pending transaction sequences must increase"
            );
        }
        self.transactions.push_back(Qwen3PendingTransaction { compute_seq });
    }

    fn commit(&mut self, compute_seq: RawComputeSlotSeq) {
        let transaction = self
            .transactions
            .pop_front()
            .expect("qwen3 commit requires a pending batch");
        assert_eq!(
            transaction.compute_seq, compute_seq,
            "qwen3 commit sequence must match the oldest pending transaction"
        );
    }
}

impl ReplayableModel for Qwen3Executor {
    type ModelBatchRequest = Qwen3ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = Qwen3ModelBatchResponse;
    type SampledOutput = Qwen3SampledOutput;
    type ModelOpsRecorder = Qwen3ModelOpsRecorder;
    type Submission = MetalReplaySubmission;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_mode(&self) -> &'static str {
        match &self.speculator {
            Qwen3Speculator::Vanilla => "vanilla",
            Qwen3Speculator::DSpark(_) => "dspark",
        }
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
        self.speculator.reset_req_slots(request_slots);
    }

    fn clear_replay_cache(&mut self) {
        Qwen3Executor::clear_replay_cache(self);
    }

    fn unload_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen3Executor::unload_state(self, snapshot_path, plan)
    }

    fn unload_weights(&mut self) {
        Qwen3Executor::unload_weights(self);
    }

    fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        Qwen3Executor::load_weights(self)
    }

    fn load_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen3Executor::load_state(self, snapshot_path, plan)
    }

    fn prepare_batch(&mut self, core_batch_req: &BatchDeviceRequest) -> Self::ModelBatchRequest {
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
        let model_batch_request = Qwen3ModelBatchRequest::from_core_batch(core_batch_req, sampler_configs);
        let microbatch = model_batch_request.microbatch();
        self.write_token_ids(microbatch.flat_token_ids());
        let num_main_active_tokens = microbatch
            .total_tokens()
            .try_into()
            .expect("qwen3 Main token count must fit u32");
        let num_main_total_tokens = num_main_active_tokens;
        self.prepare_gqa_page_ids(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            num_main_total_tokens,
        );
        debug_assert_eq!(gqa_shape.num_tokens as usize, microbatch.total_tokens());
        model_batch_request
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse {
        self.pending_transactions.commit(core_batch_req.seq);
        to_core_batch_resp(core_batch_req, sampled_output.decisions)
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
        let num_main_active_tokens = model_batch_request
            .microbatch()
            .total_tokens()
            .try_into()
            .expect("qwen3 Main token count must fit u32");
        let (main_embed_key, main_embed_arguments) = self.main_embed.component().prepare_replay(num_main_active_tokens);
        let (main_key, mut main_arguments) = self.main.component().prepare_replay(
            num_main_active_tokens,
            self.main_gqa_state.metadata().replay_shape(),
            self.main_gqa_state.replay_topology(),
        );
        self.main_gqa_state.add_private_replay_arguments(&mut main_arguments);
        Qwen3ModelOpsRecorder {
            main_embed_key,
            main_embed_arguments,
            main_key,
            main_arguments,
            dspark_prefill: None,
            dspark_decode: None,
            gather_unembed_key: None,
            gather_unembed_arguments: ReplayArguments::new(),
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            rejection_key: None,
            rejection_arguments: ReplayArguments::new(),
            rejection_prepared: None,
            num_main_sample_rows: num_main_output_rows(model_batch_request.microbatch()),
        }
    }

    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_request: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden {
        let input = Qwen3MainEmbedArgs {
            num_tokens: model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3 MainEmbed token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &self.token_hidden_input,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, _) = self.main_embed.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_embed_key,
            "qwen3 MainEmbed replay input must match the prepared replay key"
        );
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
            "qwen3 Main must consume the MainEmbed hidden workspace"
        );
        let input = Qwen3MainArgs {
            num_tokens: microbatch
                .total_tokens()
                .try_into()
                .expect("qwen3 Main token count must fit u32"),
            hidden_input: &model_batch_hidden,
            hidden_output: &self.hidden_output,
            gqa: self.main_gqa_state.metadata(),
            gqa_replay_topology: self.main_gqa_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, _) = self.main.record(&runtime, &input);
        assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3 Main replay input must match the prepared replay key"
        );
        self.pending_transactions.push(model_batch_req.compute_seq());
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
            "qwen3 Output must consume the executor final-norm hidden workspace"
        );
        if num_main_output_rows(model_batch_req.microbatch()) == 0 {
            return Qwen3ModelBatchResponse;
        }
        let (key, arguments) = self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden);
        recorder.gather_unembed_key = Some(key);
        recorder.gather_unembed_arguments = arguments;
        Qwen3ModelBatchResponse
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
            "qwen3 sampling rows must match the recording"
        );
        if num_main_sample_rows == 0 {
            return;
        }
        if self.speculator.is_dspark() {
            self.record_rejection_sampling(recorder, microbatch);
        } else {
            assert!(
                !microbatch.has_spec_tokens(),
                "Qwen3 without DSpark does not accept speculative input tokens"
            );
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
            return Qwen3SampledOutput {
                decisions: Vec::new(),
                timing,
            };
        }
        let mut timing = ModelOutputTiming {
            main_sample_replay_elapsed: replay_elapsed,
            ..ModelOutputTiming::default()
        };
        let sample_read_start = Instant::now();
        let decisions = if self.speculator.is_dspark() {
            self.read_rejection_decisions(recorder, model_batch_req.microbatch())
        } else {
            self.read_sample_decisions(recorder.num_main_sample_rows)
        };
        timing.sample_read_elapsed = sample_read_start.elapsed();
        Qwen3SampledOutput { decisions, timing }
    }

    fn run_spec_prefill(&self, model_batch_req: &Self::ModelBatchRequest) -> bool {
        self.speculator.is_dspark() && model_batch_req.microbatch().total_tokens() > 0
    }

    fn prefill_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        sampled_output: &Self::SampledOutput,
    ) {
        let microbatch = model_batch_req.microbatch();
        let accepted_prefix_lengths = sampled_output
            .decisions
            .iter()
            .map(|decision| decision.validated_tokens.len())
            .collect::<Vec<_>>();
        let selection = build_spec_prefill_selection(microbatch, &accepted_prefix_lengths);
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        assert!(
            recorder.dspark_prefill.is_none(),
            "Qwen3 DSpark Prefill is already recorded"
        );
        recorder.dspark_prefill = Some(self.speculator.dspark_mut().execution.record_prefill(
            &runtime,
            &selection,
            self.pages.buffer(),
        ));
    }

    fn run_spec_decode(
        &self,
        _model_batch_req: &Self::ModelBatchRequest,
        sampled_output: &Self::SampledOutput,
    ) -> bool {
        self.speculator.is_dspark() && !sampled_output.decisions.is_empty()
    }

    fn decode_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        sampled_output: &Self::SampledOutput,
    ) {
        assert!(
            recorder.dspark_decode.is_none(),
            "Qwen3 DSpark Decode is already recorded"
        );
        recorder.dspark_decode =
            Some(self.record_dspark_decode(model_batch_req.microbatch(), &sampled_output.decisions));
    }

    fn submit_spec(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        let runtime = self.replay_runtime();
        self.speculator.dspark().execution.submit(
            &runtime,
            recorder.dspark_prefill.as_ref(),
            recorder.dspark_decode.as_ref(),
        )
    }

    fn read_spec(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        sampled_output: Self::SampledOutput,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        let mut timing = sampled_output.timing;
        timing.spec_replay_elapsed += replay_elapsed;
        timing.spec_passes += 1;
        let read_start = Instant::now();
        let decisions = self.read_dspark_proposal(recorder, sampled_output.decisions);
        timing.spec_read_elapsed += read_start.elapsed();
        Qwen3SampledOutput { decisions, timing }
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen3SampledOutput::default()
    }

    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize {
        sampled_output.decisions.len()
    }

    fn sampled_output_timing(&self, sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        (!sampled_output.timing.is_zero()).then_some(sampled_output.timing)
    }
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(num_tokens_per_block > 0, "qwen3 GQA requires positive tokens per block");
    assert!(num_tokens_per_page > 0, "qwen3 GQA requires positive tokens per page");
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "qwen3 GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}

#[cfg(test)]
mod tests {
    use inference_executor_core::model::ReplayableModel;
    use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;

    use super::Qwen3Executor;
    use super::Qwen3ExecutorConfig;

    #[test]
    fn test_executor_config_supports_qwen3_batch_contract() {
        Qwen3ExecutorConfig {
            max_requests: 1,
            max_tokens: 4,
            max_tokens_per_request: 4,
            num_cache_pages: 1,
            num_tokens_per_block: 1024,
        }
        .validate();
        fn assert_compact_qwen3_batch<T: ReplayableModel<ModelBatchRequest = Qwen3ModelBatchRequest>>() {}
        assert_compact_qwen3_batch::<Qwen3Executor>();
    }
}
