use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::dspark::backend::UngatedDSparkGQA;
use crate::attn::dspark::backend::UngatedDSparkGQAInput;
use crate::attn::dspark::context::DSparkGQAContextScratch;
use crate::attn::dspark::context::UngatedDSparkGQAContextAppender;
use crate::attn::dspark::context::UngatedDSparkGQAContextInput;
use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratch;
use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkLayerPlan;
use crate::model::qwen::v3_x::layer::Qwen3xUngatedGQAWeightBuffers;

pub struct Qwen3xDSparkAttention {
    dspark_layer_index: u32,
    weights: Qwen3xUngatedGQAWeightBuffers,
    backend: Rc<UngatedDSparkGQA>,
    context_appender: Rc<UngatedDSparkGQAContextAppender>,
    block_scratch: Rc<DSparkBlockScratch>,
    context_scratch: Rc<DSparkGQAContextScratch>,
    request_page_table: Rc<GQARequestPageTable>,
}

impl Qwen3xDSparkAttention {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkLayerPlan,
        bindings: Qwen3xGQAWeightBindings,
        state: &UngatedDSparkGQAState,
    ) -> Result<Self, ModelExecutorError> {
        Ok(Self {
            dspark_layer_index: plan
                .dspark_layer_index
                .try_into()
                .expect("Qwen3 DSpark layer index must fit u32"),
            weights: Qwen3xUngatedGQAWeightBuffers::load(
                device,
                store,
                &bindings,
                &plan.attention_core.attention,
                plan.attention_metal,
            )?,
            backend: state.backend(),
            context_appender: state.context_appender(),
            block_scratch: state.block_scratch(),
            context_scratch: state.context_scratch(),
            request_page_table: state.request_page_table(),
        })
    }

    pub fn record_context<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        main_feature: &'a Buffer,
        req_slots: &'a Buffer,
        flat_token_indices: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.context_appender.record(
            recorder,
            UngatedDSparkGQAContextInput {
                num_tokens,
                page_table_layout: self.request_page_table.layout(),
                gqa_layer_index: self.dspark_layer_index,
                main_feature,
                req_slots,
                flat_token_indices,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table.page_ids_buffer(),
                },
                weights: self.weights.as_borrowed(),
                scratch: self.context_scratch.bindings(),
            },
        );
    }

    pub fn record_block<'a, R>(
        &'a self,
        recorder: &mut R,
        metadata: &'a DSparkGQAMetadataBuffers,
        hidden_input: &'a Buffer,
        hidden_output: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <UngatedDSparkGQA as ReplayLayer>::record(
            &self.backend,
            recorder,
            UngatedDSparkGQAInput {
                page_table_layout: self.request_page_table.layout(),
                gqa_layer_index: self.dspark_layer_index,
                metadata,
                hidden_state: hidden_input,
                next_hidden_state: hidden_output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table.page_ids_buffer(),
                },
                weights: self.weights.as_borrowed(),
                scratch: self.block_scratch.bindings(),
            },
        );
    }
}
