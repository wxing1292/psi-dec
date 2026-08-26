use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::MetalReplayRuntime;
use crate::model::embedding::Embed;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbed;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbedArgs;
use crate::model::qwen::v3_x::dspark::embed::Qwen3xDSparkEmbedReplayKey;
use crate::model::qwen::v3_x::dspark::load::Qwen3xDSparkLoaded;
use crate::model::qwen::v3_x::dspark::main_feature::Qwen3xDSparkMainFeatureProjector;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBody;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBodyArgs;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkBodyReplayKey;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkModel;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkPrefill;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkPrefillArgs;
use crate::model::qwen::v3_x::dspark::model::Qwen3xDSparkPrefillReplayKey;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembed;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembedArgs;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembedReplayKey;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSampling;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSamplingArgs;
use crate::model::qwen::v3_x::dspark::output::Qwen3xDSparkSamplingReplayKey;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInput;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputArgs;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputConfig;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputRecording;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::sampling::dspark_markov::DSparkMarkovReplayShape;
use crate::sampling::dspark_markov::DSparkProposal;
use crate::sampling::rejection_sampling::SparseRejectionSamplingOutput;
use crate::sampling::spec_probs::SpecProbsStore;

mod file_io;

pub struct Qwen3xDSparkExecution {
    prefill: Replay<Qwen3xDSparkPrefill>,
    gqa_state: BiDiBlockGQAState,
    embed: Replay<Qwen3xDSparkEmbed>,
    body: Replay<Qwen3xDSparkBody>,
    gather_unembed: Replay<Qwen3xDSparkGatherUnembed>,
    sampling: Replay<Qwen3xDSparkSampling>,
    decode_input: Replay<SpecDecodeInput>,
    unloaded_model: Option<Qwen3xDSparkModel>,
    unloaded_embed: Option<Embed>,
    unloaded_unembed: Option<Unembed>,
    embed_uses_main: bool,
    unembed_uses_main: bool,
    hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    unembed_hidden: Buffer,
    logits: Buffer,
    page_table_layout: GQAPageTableLayout,
    num_spec_tokens: usize,
    page_bytes: usize,
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
        let sdpa = loaded.gqa_state.metadata().sdpa_execution().map.thread_block;
        let decode_input = SpecDecodeInput::new(
            device,
            SpecDecodeInputConfig::new(
                loaded.page_table_layout.num_req_slots,
                loaded.num_spec_tokens as u32,
                sdpa,
                u32::MAX,
                loaded.max_anchor_position,
                loaded.gqa_state.max_sdpa_map_task_templates(),
                loaded.mask_token_id,
            ),
        );
        Self {
            prefill: Replay::new(
                "Qwen3x DSpark Prefill",
                Qwen3xDSparkPrefill::new(Rc::clone(&loaded.model)),
            ),
            gqa_state: loaded.gqa_state,
            embed: Replay::new("Qwen3x DSpark Embed", Qwen3xDSparkEmbed::new(loaded.embed)),
            body: Replay::new("Qwen3x DSpark Body", Qwen3xDSparkBody::new(Rc::clone(&loaded.model))),
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
            decode_input: Replay::new("Qwen3x DSpark Spec Decode Prepare", decode_input),
            unloaded_model: None,
            unloaded_embed: None,
            unloaded_unembed: None,
            embed_uses_main: loaded.embed_uses_main,
            unembed_uses_main: loaded.unembed_uses_main,
            hidden_input: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            hidden_output: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            unembed_hidden: Buffer::new_zeroed(device, hidden_bytes),
            logits: Buffer::new_zeroed(device, dspark_unembed_config.logits_bytes()),
            page_table_layout: loaded.page_table_layout,
            num_spec_tokens: loaded.num_spec_tokens,
            page_bytes: loaded.page_bytes,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.num_spec_tokens
    }

    pub fn num_gqa_page_ids_per_block(&self) -> usize {
        (self.page_table_layout.num_gqa_layers as usize)
            .checked_mul(self.page_table_layout.num_page_ids_per_block as usize)
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
        self.prefill.clear();
        self.embed.clear();
        self.body.clear();
        self.gather_unembed.clear();
        self.sampling.clear();
        self.decode_input.clear();
    }

    pub fn unload_weights(&mut self) {
        self.unloaded_model
            .as_mut()
            .expect("Qwen3.x DSpark state must be unloaded before weights")
            .unload_weights();
        self.sampling.component_mut().unload_weights();

        let unembed = self.gather_unembed.component_mut().unload_weights();
        let mut unembed = Rc::try_unwrap(unembed)
            .unwrap_or_else(|_| panic!("Qwen3.x DSpark unembed must be uniquely owned during weight unloading"));
        if self.unembed_uses_main {
            drop(unembed);
        } else {
            unembed.unload_weights();
            self.unloaded_unembed = Some(unembed);
        }

        let embed = self.embed.component_mut().unload_weights();
        let mut embed = Rc::try_unwrap(embed)
            .unwrap_or_else(|_| panic!("Qwen3.x DSpark embed must be uniquely owned during weight unloading"));
        if self.embed_uses_main {
            drop(embed);
        } else {
            embed.unload_weights();
            self.unloaded_embed = Some(embed);
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        model_dir: &Path,
        config: &Qwen3xDSparkConfig,
        main_embed: &Rc<Embed>,
        main_unembed: &Rc<Unembed>,
    ) -> Result<(), ModelExecutorError> {
        let mut store = SafeTensorStore::from_model_dir(model_dir)?;
        let Qwen3xDSparkWeightBindings {
            embed,
            main_feature,
            layers,
            final_norm_weight,
            unembed,
            markov,
            confidence,
        } = resolve_qwen3x_dspark_weight_bindings(config, store.index().tensor_names())?;
        let max_block_tokens = self
            .num_spec_tokens
            .checked_mul(self.page_table_layout.num_req_slots as usize)
            .expect("Qwen3.x DSpark block token capacity must fit usize");

        let embed = match embed {
            Some(bindings) => {
                let mut embed = self
                    .unloaded_embed
                    .take()
                    .expect("Qwen3.x DSpark dedicated embed shell must exist during weight loading");
                embed.load_weights(device, &mut store, bindings)?;
                Rc::new(embed)
            },
            None => {
                Rc::new(
                    main_embed.with_max_tokens(
                        max_block_tokens
                            .try_into()
                            .expect("Qwen3.x DSpark embed row capacity must fit u32"),
                    ),
                )
            },
        };
        let unembed = match unembed {
            Some(bindings) => {
                let mut unembed = self
                    .unloaded_unembed
                    .take()
                    .expect("Qwen3.x DSpark dedicated unembed shell must exist during weight loading");
                unembed.load_weights(device, &mut store, bindings)?;
                Rc::new(unembed)
            },
            None => {
                Rc::new(
                    main_unembed.with_max_tokens(
                        max_block_tokens
                            .try_into()
                            .expect("Qwen3.x DSpark unembed row capacity must fit u32"),
                    ),
                )
            },
        };

        self.unloaded_model
            .as_mut()
            .expect("Qwen3.x DSpark state must remain unloaded during weight loading")
            .load_weights(device, &mut store, config, &main_feature, layers, final_norm_weight)?;
        self.sampling
            .component_mut()
            .load_weights(device, &mut store, &markov, &confidence)?;
        self.embed.component_mut().load_weights(embed);
        self.gather_unembed.component_mut().load_weights(unembed);
        Ok(())
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDSparkMainFeatureProjector> {
        self.unloaded_model
            .as_ref()
            .expect("Qwen3.x DSpark state must be unloaded while restoring Main shared weights")
            .main_feature_projector()
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.unloaded_model.is_none(),
            "Qwen3.x DSpark model state is already unloaded"
        );
        let prefill_model = self.prefill.component_mut().take_model();
        let body_model = self.body.component_mut().take_model();
        assert!(
            Rc::ptr_eq(&prefill_model, &body_model),
            "Qwen3.x DSpark Prefill and body must share one model"
        );
        drop(body_model);
        let mut model = Rc::try_unwrap(prefill_model)
            .unwrap_or_else(|_| panic!("Qwen3.x DSpark model must be uniquely owned during state unloading"));
        model.unload_state();
        self.unloaded_model = Some(model);
        self.gqa_state.release_resources();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        self.gqa_state.allocate_resources(device);
    }

    pub fn release_resources(&mut self) {
        self.gqa_state.release_resources();
    }

    pub fn attach_state(&mut self) {
        let mut model = self
            .unloaded_model
            .take()
            .expect("Qwen3.x DSpark model state is not unloaded");
        model.load_state(&self.gqa_state);
        let model = Rc::new(model);
        self.prefill.component_mut().set_model(Rc::clone(&model));
        self.body.component_mut().set_model(model);
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        self.gqa_state.write_page_ids(req_slot, block_index, page_ids);
    }

    pub fn read_page_ids(&self, req_slot: u32, block_index: usize) -> Vec<u32> {
        self.gqa_state.read_page_ids(req_slot, block_index)
    }

    pub fn record_spec_prefill(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        spec_prefill: &GQAMetadataBuffers,
        pages: &Buffer,
    ) -> Qwen3xDSparkPrefillRecording {
        let num_tokens = spec_prefill.replay_shape().num_tokens;
        let input = Qwen3xDSparkPrefillArgs {
            num_tokens,
            req_slots: spec_prefill.req_slots(),
            flat_token_indices: spec_prefill.flat_token_indices(),
            pages,
        };
        let (prepared_key, arguments) = self.prefill.component().prepare_replay(num_tokens);
        let (key, _) = self.prefill.record(runtime, &input);
        assert_eq!(key, prepared_key);
        Qwen3xDSparkPrefillRecording { key, arguments }
    }

    pub fn append_spec_replays<'a>(
        &'a self,
        sequence: &mut Vec<ReplayExecution<'a>>,
        prefill: &'a Qwen3xDSparkPrefillRecording,
        decode: Option<&'a Qwen3xDSparkDecodeRecording>,
    ) {
        if let Some(decode) = decode {
            sequence.push(ReplayExecution::new(
                self.decode_input.replay(&decode.decode_prepare.key),
                &decode.decode_prepare.arguments,
            ));
        }
        sequence.push(ReplayExecution::new(
            self.prefill.replay(&prefill.key),
            &prefill.arguments,
        ));
        if let Some(decode) = decode {
            sequence.push(ReplayExecution::new(
                self.embed.replay(&decode.embed_key),
                &decode.embed_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.body.replay(&decode.body_key),
                &decode.body_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.gather_unembed.replay(&decode.gather_unembed_key),
                &decode.gather_unembed_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.sampling.replay(&decode.sampling_key),
                &decode.sampling_arguments,
            ));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_decode_prepare(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        token_ids: &Buffer,
        rejection_sampling: SparseRejectionSamplingOutput<'_>,
        req_slots: &[u32],
        anchor_indices: &[u32],
        num_spec_tokens: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> (SpecDecodeInputRecording, DSparkMarkovReplayShape) {
        let block = self
            .decode_input
            .component()
            .prepare(req_slots, anchor_indices, num_spec_tokens);
        let num_active_requests = req_slots.len();
        let num_total_requests = block.num_requests();
        self.gqa_state
            .prepare_bidi_block_with_active_requests(&block, num_active_requests);
        let markov_replay_shape =
            self.sampling
                .component()
                .prepare_static(req_slots, sampler_configs, distribution_store);
        self.gather_unembed.component().prepare(num_active_requests);
        let input = SpecDecodeInputArgs {
            num_active_requests: num_active_requests as u32,
            num_total_requests: num_total_requests as u32,
            rejection_sampling,
            metadata: self.gqa_state.metadata(),
            block_token_ids: token_ids,
            anchor_token_ids: self.sampling.component().anchor_token_ids(),
        };
        let (prepared_key, arguments) = self.decode_input.component().prepare_replay_arguments(&input);
        let (key, _) = self.decode_input.record(runtime, &input);
        assert_eq!(key, prepared_key);
        (SpecDecodeInputRecording { key, arguments }, markov_replay_shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_spec_decode(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        token_ids: &Buffer,
        decode_prepare: SpecDecodeInputRecording,
        markov_replay_shape: DSparkMarkovReplayShape,
        req_slots: Vec<u32>,
        pages: &Buffer,
        distribution_store: &SpecProbsStore,
    ) -> Qwen3xDSparkDecodeRecording {
        let metadata = self.gqa_state.metadata();
        let embed_input = Qwen3xDSparkEmbedArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            token_ids,
            hidden_output: &self.hidden_input,
        };
        let (prepared_embed_key, embed_arguments) = self.embed.component().prepare_replay(embed_input.num_tokens);
        let (embed_key, _) = self.embed.record(runtime, &embed_input);
        assert_eq!(embed_key, prepared_embed_key);
        let body_input = Qwen3xDSparkBodyArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            metadata,
            hidden_input: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages,
        };
        let (body_key, _) = self.body.record(runtime, &body_input);
        let mut body_arguments = ReplayArguments::new();
        self.gqa_state.add_replay_arguments(&mut body_arguments);
        self.body
            .component()
            .add_replay_arguments(metadata.replay_shape(), &mut body_arguments);
        let gather_unembed_input = Qwen3xDSparkGatherUnembedArgs {
            num_requests: req_slots
                .len()
                .try_into()
                .expect("Qwen3x DSpark request count must fit u32"),
            hidden_input: &self.hidden_output,
            hidden_output: &self.unembed_hidden,
            logits: &self.logits,
        };
        let (prepared_gather_unembed_key, gather_unembed_arguments) = self
            .gather_unembed
            .component()
            .prepare_replay(gather_unembed_input.num_requests);
        let (gather_unembed_key, _) = self.gather_unembed.record(runtime, &gather_unembed_input);
        assert_eq!(gather_unembed_key, prepared_gather_unembed_key);
        let sampling_input = Qwen3xDSparkSamplingArgs {
            shape: markov_replay_shape,
            logits: &self.logits,
            sample_positions: self.decode_input.component().sample_positions(),
            hidden: &self.unembed_hidden,
            distribution_store,
        };
        let (sampling_key, _) = self.sampling.record(runtime, &sampling_input);
        let mut sampling_arguments = ReplayArguments::new();
        self.sampling
            .component()
            .add_replay_arguments(markov_replay_shape, &mut sampling_arguments);
        Qwen3xDSparkDecodeRecording {
            decode_prepare,
            embed_key,
            embed_arguments,
            body_key,
            body_arguments,
            gather_unembed_key,
            gather_unembed_arguments,
            sampling_key,
            sampling_arguments,
            req_slots,
        }
    }

    pub fn read_proposal(
        &self,
        recording: &Qwen3xDSparkDecodeRecording,
        distribution_store: &mut SpecProbsStore,
    ) -> DSparkProposal {
        self.sampling
            .component()
            .read_proposal(&recording.req_slots, distribution_store)
    }
}

pub struct Qwen3xDSparkPrefillRecording {
    key: Qwen3xDSparkPrefillReplayKey,
    arguments: ReplayArguments,
}

pub struct Qwen3xDSparkDecodeRecording {
    decode_prepare: SpecDecodeInputRecording,
    embed_key: Qwen3xDSparkEmbedReplayKey,
    embed_arguments: ReplayArguments,
    body_key: Qwen3xDSparkBodyReplayKey,
    body_arguments: ReplayArguments,
    gather_unembed_key: Qwen3xDSparkGatherUnembedReplayKey,
    gather_unembed_arguments: ReplayArguments,
    sampling_key: Qwen3xDSparkSamplingReplayKey,
    sampling_arguments: ReplayArguments,
    req_slots: Vec<u32>,
}
