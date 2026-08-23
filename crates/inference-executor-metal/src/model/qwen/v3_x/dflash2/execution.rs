use std::ops::Range;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::attn::BlockSpecMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2WeightBindings;
use inference_executor_core::model::qwen::v3_x::dflash2::resolve_qwen3x_dflash2_weight_bindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SpecPrefillSelection;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::block_spec::state::BlockSpecGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::model::embedding::Embed;
use crate::model::main_residual_capture::MainResidualRows;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2Embed;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2EmbedArgs;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2EmbedReplayKey;
use crate::model::qwen::v3_x::dflash2::load::Qwen3xDFlash2Loaded;
use crate::model::qwen::v3_x::dflash2::main_feature::Qwen3xDFlash2MainFeatureProjector;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Body;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2BodyArgs;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2BodyReplayKey;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Model;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Prefill;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2PrefillArgs;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2PrefillReplayKey;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2Output;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2OutputArgs;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2OutputPrepare;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2OutputReplayKey;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2Proposal;
use crate::model::unembedding::Unembed;
use crate::replay::Replay;
use crate::sampling::spec_probs::SpecProbsStore;

mod file_io;

pub struct Qwen3xDFlash2Execution {
    prefill: Replay<Qwen3xDFlash2Prefill>,
    gqa_state: BlockSpecGQAState,
    embed: Replay<Qwen3xDFlash2Embed>,
    body: Replay<Qwen3xDFlash2Body>,
    output: Replay<Qwen3xDFlash2Output>,
    unloaded_model: Option<Qwen3xDFlash2Model>,
    prefill_main_row_indices: Buffer,
    prefill_req_slots: Buffer,
    prefill_flat_token_indices: Buffer,
    hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    page_table_layout: GQAPageTableLayout,
    num_spec_tokens: usize,
    mask_token_id: i32,
    sliding_window: u32,
    page_bytes: usize,
}

pub struct Qwen3xDFlash2ProposalInput<'a> {
    req_slots: Vec<u32>,
    anchor_token_ids: &'a [u32],
    anchor_positions: &'a [u32],
    sampler_configs: &'a [SamplerConfig],
}

impl<'a> Qwen3xDFlash2ProposalInput<'a> {
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

impl Qwen3xDFlash2Execution {
    pub fn new(device: &Device, loaded: Qwen3xDFlash2Loaded) -> Self {
        assert!(loaded.num_spec_tokens > 0);
        let query_block_size = loaded.num_spec_tokens + 1;
        let max_query_tokens = (loaded.page_table_layout.num_req_slots as usize)
            .checked_mul(query_block_size)
            .expect("Qwen3x DFlash2 query token capacity must fit usize");
        let hidden_bytes = max_query_tokens
            .checked_mul(loaded.output.hidden_dim() as usize)
            .and_then(|elements| elements.checked_mul(Dtype::Bfloat16.item_size()))
            .expect("Qwen3x DFlash2 hidden byte capacity must fit usize");
        Self {
            prefill: Replay::new(
                "Qwen3x DFlash2 Prefill",
                Qwen3xDFlash2Prefill::new(Rc::clone(&loaded.model)),
            ),
            gqa_state: loaded.gqa_state,
            embed: Replay::new("Qwen3x DFlash2 Embed", Qwen3xDFlash2Embed::new(loaded.embed)),
            body: Replay::new("Qwen3x DFlash2 Body", Qwen3xDFlash2Body::new(Rc::clone(&loaded.model))),
            output: Replay::new("Qwen3x DFlash2 Output", loaded.output),
            prefill_main_row_indices: Buffer::new_zeroed_elements(device, loaded.max_main_tokens, Dtype::Uint32),
            prefill_req_slots: Buffer::new_zeroed_elements(device, loaded.max_main_tokens, Dtype::Uint32),
            prefill_flat_token_indices: Buffer::new_zeroed_elements(device, loaded.max_main_tokens, Dtype::Uint32),
            unloaded_model: None,
            hidden_input: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            hidden_output: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            page_table_layout: loaded.page_table_layout,
            num_spec_tokens: loaded.num_spec_tokens,
            mask_token_id: loaded.mask_token_id,
            sliding_window: loaded
                .sliding_window
                .try_into()
                .expect("Qwen3x DFlash2 sliding window must fit u32"),
            page_bytes: loaded.page_bytes,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.num_spec_tokens
    }

    pub fn num_gqa_page_ids_per_block(&self) -> usize {
        (self.page_table_layout.num_gqa_layers as usize)
            .checked_mul(self.page_table_layout.num_page_ids_per_block as usize)
            .expect("Qwen3x DFlash2 page IDs per block must fit usize")
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
        self.output.clear();
    }

    pub fn unload_weights(&mut self) {
        self.unloaded_model
            .as_mut()
            .expect("Qwen3x DFlash2 state must be unloaded before weights")
            .unload_weights();
        drop(self.output.component_mut().unload_weights());
        drop(self.embed.component_mut().unload_weights());
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        model_dir: &Path,
        config: &Qwen3xDFlash2Config,
        main_embed: &Rc<Embed>,
        main_unembed: &Rc<Unembed>,
    ) -> Result<(), ModelExecutorError> {
        let mut store = SafeTensorStore::from_model_dir(model_dir)?;
        let Qwen3xDFlash2WeightBindings {
            main_feature,
            layers,
            final_norm_weight,
            selector,
        } = resolve_qwen3x_dflash2_weight_bindings(config, store.index().tensor_names())?;
        let query_block_size = self.num_spec_tokens + 1;
        let max_requests = self.page_table_layout.num_req_slots as usize;
        let max_query_tokens = max_requests
            .checked_mul(query_block_size)
            .expect("Qwen3x DFlash2 query token capacity must fit usize");
        let max_proposal_tokens = max_requests
            .checked_mul(self.num_spec_tokens)
            .expect("Qwen3x DFlash2 proposal token capacity must fit usize");
        let embed = Rc::new(
            main_embed.with_max_tokens(
                max_query_tokens
                    .try_into()
                    .expect("Qwen3x DFlash2 embed row capacity must fit u32"),
            ),
        );
        let unembed = Rc::new(
            main_unembed.with_max_tokens(
                max_proposal_tokens
                    .try_into()
                    .expect("Qwen3x DFlash2 unembed row capacity must fit u32"),
            ),
        );
        self.unloaded_model
            .as_mut()
            .expect("Qwen3x DFlash2 state must remain unloaded during weight loading")
            .load_weights(device, &mut store, config, &main_feature, layers, final_norm_weight)?;
        self.output.component_mut().load_unembed(unembed);
        self.output.component_mut().load_weights(device, &mut store, selector)?;
        self.embed.component_mut().load_weights(embed);
        Ok(())
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDFlash2MainFeatureProjector> {
        self.unloaded_model
            .as_ref()
            .expect("Qwen3x DFlash2 state must be unloaded while restoring Main shared weights")
            .main_feature_projector()
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.unloaded_model.is_none(),
            "Qwen3x DFlash2 model state is already unloaded"
        );
        let prefill_model = self.prefill.component_mut().take_model();
        let body_model = self.body.component_mut().take_model();
        assert!(Rc::ptr_eq(&prefill_model, &body_model));
        drop(body_model);
        let mut model = Rc::try_unwrap(prefill_model)
            .unwrap_or_else(|_| panic!("Qwen3x DFlash2 model must be uniquely owned during state unloading"));
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
            .expect("Qwen3x DFlash2 model state is not unloaded");
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

    pub fn record_prefill(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        selection: &SpecPrefillSelection,
        pages: &Buffer,
    ) -> Qwen3xDFlash2PrefillRecording {
        let num_tokens: u32 = selection
            .main_row_indices
            .len()
            .try_into()
            .expect("Qwen3x DFlash2 Prefill token count must fit u32");
        assert_eq!(selection.req_slots.len(), selection.main_row_indices.len());
        assert_eq!(selection.flat_token_indices.len(), selection.main_row_indices.len());
        self.prefill_main_row_indices
            .write_typed(0, &selection.main_row_indices);
        self.prefill_req_slots.write_typed(0, &selection.req_slots);
        self.prefill_flat_token_indices
            .write_typed(0, &selection.flat_token_indices);
        let main_rows = if selection.main_rows_are_prefix() {
            MainResidualRows::Prefix
        } else {
            MainResidualRows::Indices(&self.prefill_main_row_indices)
        };
        let input = Qwen3xDFlash2PrefillArgs {
            num_tokens,
            main_rows,
            req_slots: &self.prefill_req_slots,
            flat_token_indices: &self.prefill_flat_token_indices,
            pages,
        };
        let (key, _) = self.prefill.record(runtime, &input);
        Qwen3xDFlash2PrefillRecording { key }
    }

    pub fn record_decode(
        &mut self,
        runtime: &MetalReplayRuntime<'_>,
        token_ids: &Buffer,
        proposal: Qwen3xDFlash2ProposalInput<'_>,
        pages: &Buffer,
        distribution_store: &SpecProbsStore,
    ) -> Qwen3xDFlash2DecodeRecording {
        let query_block_size = self.num_spec_tokens + 1;
        let (flat_query_token_indices, visible_history_token_ranges) =
            dflash2_query_layout(proposal.anchor_positions, query_block_size, self.sliding_window);
        let block = BlockSpecMetadata::new(
            &proposal.req_slots,
            &flat_query_token_indices,
            &visible_history_token_ranges,
            query_block_size,
        );
        self.gqa_state.prepare_block(&block);
        let mut block_token_ids = Vec::with_capacity(block.num_tokens());
        for &anchor_token_id in proposal.anchor_token_ids {
            block_token_ids.push(i32::try_from(anchor_token_id).expect("Qwen3x DFlash2 anchor token ID must fit i32"));
            block_token_ids.extend(std::iter::repeat_n(self.mask_token_id, self.num_spec_tokens));
        }
        token_ids.write_typed(0, &block_token_ids);
        let num_requests = self.output.component().prepare(Qwen3xDFlash2OutputPrepare {
            req_slots: &proposal.req_slots,
            anchor_token_ids: proposal.anchor_token_ids,
            anchor_positions: proposal.anchor_positions,
            sampler_configs: proposal.sampler_configs,
            distribution_store,
        });
        let embed_input = Qwen3xDFlash2EmbedArgs {
            num_tokens: block
                .num_tokens()
                .try_into()
                .expect("Qwen3x DFlash2 query token count must fit u32"),
            token_ids,
            hidden_output: &self.hidden_input,
        };
        let (embed_key, _) = self.embed.record(runtime, &embed_input);
        let metadata = self.gqa_state.metadata();
        let body_input = Qwen3xDFlash2BodyArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            metadata,
            hidden_input: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages,
        };
        let (body_key, _) = self.body.record(runtime, &body_input);
        let mut body_arguments = ReplayArguments::new();
        self.gqa_state.add_replay_arguments(&mut body_arguments);
        let output_input = Qwen3xDFlash2OutputArgs {
            num_requests,
            hidden: &self.hidden_output,
            distribution_store,
        };
        let (output_key, _) = self.output.record(runtime, &output_input);
        let mut output_arguments = ReplayArguments::new();
        self.output
            .component()
            .add_replay_arguments(num_requests, &mut output_arguments);
        Qwen3xDFlash2DecodeRecording {
            embed_key,
            body_key,
            body_arguments,
            output_key,
            output_arguments,
            req_slots: proposal.req_slots,
        }
    }

    pub fn submit(
        &self,
        runtime: &MetalReplayRuntime<'_>,
        prefill: Option<&Qwen3xDFlash2PrefillRecording>,
        decode: Option<&Qwen3xDFlash2DecodeRecording>,
    ) -> MetalReplaySubmission {
        let empty_arguments = ReplayArguments::new();
        let mut sequence = Vec::with_capacity(4);
        if let Some(prefill) = prefill {
            sequence.push(ReplayExecution::new(
                self.prefill.replay(&prefill.key),
                &empty_arguments,
            ));
        }
        if let Some(decode) = decode {
            sequence.push(ReplayExecution::new(
                self.embed.replay(&decode.embed_key),
                &empty_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.body.replay(&decode.body_key),
                &decode.body_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.output.replay(&decode.output_key),
                &decode.output_arguments,
            ));
        }
        assert!(!sequence.is_empty(), "Qwen3x DFlash2 submission requires Spec work");
        runtime.submit_replay_sequence(&sequence)
    }

    pub fn read_proposal(
        &self,
        recording: &Qwen3xDFlash2DecodeRecording,
        distribution_store: &mut SpecProbsStore,
    ) -> Qwen3xDFlash2Proposal {
        self.output
            .component()
            .read_proposal(&recording.req_slots, distribution_store)
    }
}

fn dflash2_query_layout(
    anchor_positions: &[u32],
    query_block_size: usize,
    sliding_window: u32,
) -> (Vec<u32>, Vec<Range<u32>>) {
    assert!(query_block_size > 0, "Qwen3x DFlash2 query block must contain rows");
    assert!(sliding_window > 0, "Qwen3x DFlash2 sliding window must contain tokens");
    let query_block_size_u32 = u32::try_from(query_block_size).expect("Qwen3x DFlash2 query block size must fit u32");
    let capacity = anchor_positions
        .len()
        .checked_mul(query_block_size)
        .expect("Qwen3x DFlash2 query row count must fit usize");
    let mut flat_query_token_indices = Vec::with_capacity(capacity);
    let mut visible_history_token_ranges = Vec::with_capacity(capacity);
    for &anchor_position in anchor_positions {
        assert!(anchor_position > 0, "Qwen3x DFlash2 Decode requires a nonempty history");
        assert!(
            anchor_position <= u32::MAX - query_block_size_u32,
            "Qwen3x DFlash2 query positions must fit u32"
        );
        for block_offset in 0..query_block_size {
            let query_position = anchor_position + block_offset as u32;
            let history_begin = (query_position + 1).saturating_sub(sliding_window);
            assert!(
                history_begin < anchor_position,
                "Qwen3x DFlash2 sliding history must contain a token for every query row"
            );
            flat_query_token_indices.push(query_position);
            visible_history_token_ranges.push(history_begin..anchor_position);
        }
    }
    (flat_query_token_indices, visible_history_token_ranges)
}

pub struct Qwen3xDFlash2PrefillRecording {
    key: Qwen3xDFlash2PrefillReplayKey,
}

pub struct Qwen3xDFlash2DecodeRecording {
    embed_key: Qwen3xDFlash2EmbedReplayKey,
    body_key: Qwen3xDFlash2BodyReplayKey,
    body_arguments: ReplayArguments,
    output_key: Qwen3xDFlash2OutputReplayKey,
    output_arguments: ReplayArguments,
    req_slots: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::dflash2_query_layout;

    #[test]
    fn test_query_layout_applies_sliding_window_per_query_row() {
        let (positions, ranges) = dflash2_query_layout(&[10, 20], 4, 8);

        assert_eq!(positions, [10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!(ranges, [3..10, 4..10, 5..10, 6..10, 13..20, 14..20, 15..20, 16..20]);
    }

    #[test]
    fn test_query_layout_clamps_history_begin_to_zero() {
        let (_, ranges) = dflash2_query_layout(&[1], 4, 8);

        assert_eq!(ranges, [0..1, 0..1, 0..1, 0..1]);
    }
}
