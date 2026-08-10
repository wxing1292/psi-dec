use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::attn::DSparkBlockMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::sampling::SamplerConfig;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbed;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbedArgs;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbedReplayKey;
use crate::model::qwen::v3_x::dspark::load::Qwen3xDSparkLoaded;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBody;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBodyArgs;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBodyReplayKey;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkContext;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkContextArgs;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkContextReplayKey;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembed;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembedArgs;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembedReplayKey;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSampling;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSamplingArgs;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSamplingReplayKey;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::sampling::dspark_markov::DSparkMarkovReplayShape;
use crate::sampling::dspark_markov::DSparkProposal;
use crate::sampling::spec_probs::SpecProbsStore;

pub struct Qwen3xDSparkExecution {
    context: Replay<Qwen3xDSparkContext>,
    gqa_state: UngatedDSparkGQAState,
    embed: Replay<Qwen3xDSparkEmbed>,
    body: Replay<Qwen3xDSparkBody>,
    gather_unembed: Replay<Qwen3xDSparkGatherUnembed>,
    sampling: Replay<Qwen3xDSparkSampling>,
    hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    unembed_hidden: Buffer,
    logits: Buffer,
    page_table_layout: GQAPageTableLayout,
    num_spec_tokens: usize,
    mask_token_id: i32,
}

pub struct Qwen3xDSparkProposalInput<'a> {
    req_slots: Vec<u32>,
    anchor_token_ids: &'a [u32],
    anchor_positions: &'a [u32],
    sampler_configs: &'a [SamplerConfig],
}

impl<'a> Qwen3xDSparkProposalInput<'a> {
    pub fn new(
        req_slots: Vec<u32>,
        anchor_token_ids: &'a [u32],
        anchor_positions: &'a [u32],
        sampler_configs: &'a [SamplerConfig],
    ) -> Self {
        debug_assert_eq!(req_slots.len(), anchor_token_ids.len());
        debug_assert_eq!(req_slots.len(), anchor_positions.len());
        debug_assert_eq!(req_slots.len(), sampler_configs.len());
        Self {
            req_slots,
            anchor_token_ids,
            anchor_positions,
            sampler_configs,
        }
    }
}

impl Qwen3xDSparkExecution {
    pub fn new(
        device: &Device,
        loaded: Qwen3xDSparkLoaded,
        max_requests: usize,
        unembed_config: UnembedConfig,
    ) -> Self {
        assert!(max_requests > 0, "Qwen3x DSpark execution requires requests");
        assert!(
            loaded.num_spec_tokens > 0,
            "Qwen3x DSpark execution requires speculative tokens"
        );
        let max_block_tokens = max_requests
            .checked_mul(loaded.num_spec_tokens)
            .expect("Qwen3x DSpark block token capacity must fit usize");
        let hidden_bytes = max_block_tokens
            .checked_mul(unembed_config.hidden_dim as usize)
            .and_then(|elements| elements.checked_mul(Dtype::Bfloat16.item_size()))
            .expect("Qwen3x DSpark hidden byte capacity must fit usize");
        let dspark_unembed_config = UnembedConfig {
            max_tokens: max_block_tokens
                .try_into()
                .expect("Qwen3x DSpark unembed rows must fit u32"),
            ..unembed_config
        };
        Self {
            context: Replay::new(
                "Qwen3x DSparkContext",
                Qwen3xDSparkContext::new(Rc::clone(&loaded.model)),
            ),
            gqa_state: loaded.gqa_state,
            embed: Replay::new("Qwen3x DSparkEmbed", Qwen3xDSparkEmbed::new(loaded.embed)),
            body: Replay::new("Qwen3x DSpark", Qwen3xDSparkBody::new(Rc::clone(&loaded.model))),
            gather_unembed: Replay::new(
                "Qwen3x DSpark GatherUnembed",
                Qwen3xDSparkGatherUnembed::new(
                    device,
                    loaded.num_spec_tokens,
                    max_requests,
                    unembed_config.hidden_dim,
                    loaded.unembed,
                ),
            ),
            sampling: Replay::new("Qwen3x DSpark Sampling", Qwen3xDSparkSampling::new(loaded.markov)),
            hidden_input: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            hidden_output: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            unembed_hidden: Buffer::new_zeroed(device, hidden_bytes),
            logits: Buffer::new_zeroed(device, dspark_unembed_config.logits_bytes()),
            page_table_layout: loaded.page_table_layout,
            num_spec_tokens: loaded.num_spec_tokens,
            mask_token_id: loaded.mask_token_id,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.num_spec_tokens
    }

    pub fn num_runtime_page_ids_per_block(&self) -> usize {
        usize::try_from(self.page_table_layout.num_gqa_layers)
            .expect("Qwen3x DSpark GQA layer count must fit usize")
            .checked_mul(
                usize::try_from(self.page_table_layout.num_page_ids_per_block)
                    .expect("Qwen3x DSpark GQA page count must fit usize"),
            )
            .expect("Qwen3x DSpark page IDs per block must fit usize")
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.gqa_state.num_tokens_per_page()
    }

    pub fn num_physical_pages_per_request(&self) -> usize {
        self.page_table_layout.num_physical_pages_per_request()
    }

    pub fn reset_req_slots(&self, request_slots: &[RawRequestSlot]) {
        self.gqa_state.reset_req_slots(request_slots);
    }

    pub fn clear_replay_cache(&mut self) {
        self.context.clear();
        self.embed.clear();
        self.body.clear();
        self.gather_unembed.clear();
        self.sampling.clear();
    }

    pub fn prepare_page_span(
        &self,
        core_batch: &BatchDeviceRequest,
        num_runtime_page_ids_per_block: usize,
        page_id_offset: usize,
    ) {
        self.gqa_state
            .prepare_page_span(core_batch, num_runtime_page_ids_per_block, page_id_offset);
    }

    pub fn record_context(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        input: &Qwen3xDSparkContextArgs<'_>,
        recording: &mut Qwen3xDSparkRecording,
    ) {
        let (key, _) = self.context.record(runtime, input);
        recording.context_key = Some(key);
    }

    pub fn context_replay<'a>(&'a self, recording: &'a Qwen3xDSparkRecording) -> &'a ReplayProgram {
        self.context.replay(
            recording
                .context_key
                .as_ref()
                .expect("Qwen3x DSpark Main submission requires a context key"),
        )
    }

    pub fn submit(&self, runtime: &MetalReplayRuntime<'_>, recording: &Qwen3xDSparkRecording) -> MetalReplaySubmission {
        let empty_arguments = ReplayArguments::new();
        let embed = self.embed.replay(
            recording
                .embed_key
                .as_ref()
                .expect("Qwen3x DSpark submission requires an embed key"),
        );
        let body = self.body.replay(
            recording
                .body_key
                .as_ref()
                .expect("Qwen3x DSpark submission requires a body key"),
        );
        let gather_unembed = self.gather_unembed.replay(
            recording
                .gather_unembed_key
                .as_ref()
                .expect("Qwen3x DSpark submission requires a GatherUnembed key"),
        );
        let sampling = self.sampling.replay(
            recording
                .sampling_key
                .as_ref()
                .expect("Qwen3x DSpark submission requires a Sampling key"),
        );
        runtime.submit_replay_sequence(&[
            ReplayExecution::new(embed, &empty_arguments),
            ReplayExecution::new(body, &empty_arguments),
            ReplayExecution::new(gather_unembed, &empty_arguments),
            ReplayExecution::new(sampling, &recording.sampling_arguments),
        ])
    }

    pub fn record_embed(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        token_ids: &Buffer,
        proposal: Qwen3xDSparkProposalInput<'_>,
        distribution_store: &SpecProbsStore,
        recording: &mut Qwen3xDSparkRecording,
    ) -> Rc<Buffer> {
        let block = DSparkBlockMetadata::new(&proposal.req_slots, proposal.anchor_positions, self.num_spec_tokens);
        self.gqa_state.prepare_block(&block);
        let mut block_token_ids = Vec::with_capacity(block.num_tokens());
        for &anchor_token_id in proposal.anchor_token_ids {
            block_token_ids.push(
                anchor_token_id
                    .try_into()
                    .expect("Qwen3x DSpark anchor token ID must fit i32"),
            );
            block_token_ids.extend(std::iter::repeat_n(self.mask_token_id, self.num_spec_tokens - 1));
        }
        token_ids.write_typed(0, &block_token_ids);
        let markov_replay_shape = self.sampling.component().prepare(
            &proposal.req_slots,
            proposal.anchor_token_ids,
            proposal.anchor_positions,
            proposal.sampler_configs,
            distribution_store,
        );
        self.gather_unembed.component().prepare(proposal.req_slots.len());
        let hidden = Rc::clone(&self.hidden_input);
        let input = Qwen3xDSparkEmbedArgs {
            num_tokens: block
                .num_tokens()
                .try_into()
                .expect("Qwen3x DSpark block token count must fit u32"),
            token_ids,
            hidden_output: &hidden,
        };
        let (key, _) = self.embed.record(runtime, &input);
        recording.embed_key = Some(key);
        recording.markov_replay_shape = Some(markov_replay_shape);
        recording.req_slots = proposal.req_slots;
        hidden
    }

    pub fn record_body(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        pages: &Buffer,
        recording: &mut Qwen3xDSparkRecording,
        hidden_input: Rc<Buffer>,
    ) -> Rc<Buffer> {
        assert!(
            Rc::ptr_eq(&hidden_input, &self.hidden_input),
            "Qwen3x DSpark must consume the DSparkEmbed workspace"
        );
        let hidden_output = Rc::clone(&self.hidden_output);
        let metadata = self.gqa_state.metadata();
        let input = Qwen3xDSparkBodyArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            metadata,
            hidden_input: &hidden_input,
            hidden_output: &hidden_output,
            pages,
        };
        let (key, _) = self.body.record(runtime, &input);
        recording.body_key = Some(key);
        hidden_output
    }

    pub fn record_gather_unembed(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        recording: &mut Qwen3xDSparkRecording,
        hidden_input: &Rc<Buffer>,
    ) {
        assert!(
            Rc::ptr_eq(hidden_input, &self.hidden_output),
            "Qwen3x DSpark GatherUnembed must consume the body output"
        );
        let input = Qwen3xDSparkGatherUnembedArgs {
            num_requests: recording
                .req_slots
                .len()
                .try_into()
                .expect("Qwen3x DSpark request count must fit u32"),
            hidden_input,
            hidden_output: &self.unembed_hidden,
            logits: &self.logits,
        };
        let (key, _) = self.gather_unembed.record(runtime, &input);
        recording.gather_unembed_key = Some(key);
    }

    pub fn record_sampling(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        distribution_store: &SpecProbsStore,
        recording: &mut Qwen3xDSparkRecording,
    ) {
        let shape = recording
            .markov_replay_shape
            .expect("Qwen3x DSpark sampling requires a prepared Markov shape");
        let input = Qwen3xDSparkSamplingArgs {
            shape,
            logits: &self.logits,
            hidden: &self.unembed_hidden,
            distribution_store,
        };
        let (key, _) = self.sampling.record(runtime, &input);
        let mut arguments = ReplayArguments::new();
        self.sampling.component().add_replay_arguments(shape, &mut arguments);
        recording.sampling_key = Some(key);
        recording.sampling_arguments = arguments;
    }

    pub fn read_proposal(
        &self,
        recording: &Qwen3xDSparkRecording,
        distribution_store: &mut SpecProbsStore,
    ) -> DSparkProposal {
        self.sampling
            .component()
            .read_proposal(&recording.req_slots, distribution_store)
    }
}

pub struct Qwen3xDSparkRecording {
    context_key: Option<Qwen3xDSparkContextReplayKey>,
    embed_key: Option<Qwen3xDSparkEmbedReplayKey>,
    body_key: Option<Qwen3xDSparkBodyReplayKey>,
    gather_unembed_key: Option<Qwen3xDSparkGatherUnembedReplayKey>,
    sampling_key: Option<Qwen3xDSparkSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    markov_replay_shape: Option<DSparkMarkovReplayShape>,
    req_slots: Vec<u32>,
}

impl Qwen3xDSparkRecording {
    pub fn new() -> Self {
        Self {
            context_key: None,
            embed_key: None,
            body_key: None,
            gather_unembed_key: None,
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            markov_replay_shape: None,
            req_slots: Vec::new(),
        }
    }
}
