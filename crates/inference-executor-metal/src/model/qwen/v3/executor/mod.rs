use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::model::qwen::v3::Qwen3DecodeDecision;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::Qwen3SampledTokens;
use inference_executor_core::model::qwen::v3::gather_flat_indices;
use inference_executor_core::model::qwen::v3::num_target_hidden_states;
use inference_executor_core::model::qwen::v3::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3::sample_sampler_configs;
use inference_executor_core::model::qwen::v3::sample_token_positions;
use inference_executor_core::model::qwen::v3::to_core_batch_resp;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
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
use crate::model::page_arena::PageArena;
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
use crate::replay::Replay;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;

mod load;

pub use load::Qwen3ExecutorConfig;
use load::Qwen3ModelLayout;
pub use load::init_qwen_3_model;

include!("batch.rs");
include!("main.rs");
include!("recording.rs");
include!("sampling.rs");

pub struct Qwen3Executor {
    model_name: String,
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
    pages: PageArena,
    pending_transactions: Qwen3PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
}

pub struct Qwen3ModelOpsRecorder {
    main_embed_key: Qwen3MainEmbedReplayKey,
    main_key: Qwen3MainReplayKey,
    gather_unembed_key: Option<Qwen3GatherUnembedReplayKey>,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    num_sample_tokens: usize,
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

impl ReplayableModelBatchExecutor for Qwen3Executor {
    type ModelBatchRequest = Qwen3ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = Qwen3ModelBatchResponse;
    type SampledOutput = Qwen3SampledOutput;
    type ModelOpsRecorder = Qwen3ModelOpsRecorder;
    type Submission = MetalReplaySubmission;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
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
        self.main_gqa_state.prepare_pages(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
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
        let main_embed_key = Qwen3MainEmbedReplayKey::new(
            model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3 MainEmbed token count must fit u32"),
        );
        let main_key = Qwen3MainReplayKey::from_shape(self.main_gqa_state.metadata().replay_shape());
        Qwen3ModelOpsRecorder {
            main_embed_key,
            main_key,
            gather_unembed_key: None,
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            num_sample_tokens: num_target_hidden_states(model_batch_request.microbatch()),
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
        if num_target_hidden_states(model_batch_req.microbatch()) == 0 {
            return Qwen3ModelBatchResponse;
        }
        recorder.gather_unembed_key =
            Some(self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden));
        Qwen3ModelBatchResponse
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
            "qwen3 sampling rows must match the recording"
        );
        if num_sample_tokens == 0 {
            return;
        }
        let (sampling_key, sampling_arguments) = self.record_sampling(microbatch);
        recorder.sampling_key = Some(sampling_key);
        recorder.sampling_arguments = sampling_arguments;
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
        _model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
    ) -> Self::SampledOutput {
        if recorder.num_sample_tokens == 0 {
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
        let decisions = self.read_sample_decisions(recorder.num_sample_tokens);
        timing.sample_read_elapsed = sample_read_start.elapsed();
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

fn replay_bucket_capacity(active: u32, max_capacity: u32) -> u32 {
    assert!(active > 0, "qwen3 replay bucket requires active work");
    assert!(active <= max_capacity, "qwen3 replay active work exceeds capacity");
    active
        .checked_next_power_of_two()
        .unwrap_or(max_capacity)
        .min(max_capacity)
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::GQAReplayShape;
    use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;
    use inference_runtime_core::compute::ReplayableModelBatchExecutor;

    use super::Qwen3Executor;
    use super::Qwen3ExecutorConfig;
    use super::Qwen3MainReplayKey;
    use super::replay_bucket_capacity;

    #[test]
    fn test_executor_config_is_target_only() {
        Qwen3ExecutorConfig {
            max_requests: 1,
            max_tokens: 4,
            max_tokens_per_request: 4,
            num_cache_pages: 1,
            num_tokens_per_block: 1024,
        }
        .validate();
        fn assert_compact_qwen3_batch<T: ReplayableModelBatchExecutor<ModelBatchRequest = Qwen3ModelBatchRequest>>() {}
        assert_compact_qwen3_batch::<Qwen3Executor>();
    }

    #[test]
    fn test_main_key_contains_only_token_and_gqa_topology() {
        let shape = GQAReplayShape {
            num_tokens: 4,
            num_q_token_tiles: 2,
            total_sdpa_map_task_templates: 3,
            reduce_sdpa_partial_outputs: true,
        };
        let key = Qwen3MainReplayKey::from_shape(shape);

        assert_eq!(key.debug_parts(), (4, 2, 3));
        assert_eq!(
            key,
            Qwen3MainReplayKey::from_shape(GQAReplayShape {
                reduce_sdpa_partial_outputs: false,
                ..shape
            })
        );
        assert_ne!(
            key,
            Qwen3MainReplayKey::from_shape(GQAReplayShape {
                total_sdpa_map_task_templates: 4,
                ..shape
            })
        );
    }

    #[test]
    fn test_bucket_capacity() {
        assert_eq!(replay_bucket_capacity(1, 48), 1);
        assert_eq!(replay_bucket_capacity(3, 48), 4);
        assert_eq!(replay_bucket_capacity(33, 48), 48);
    }
}
