use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayer;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerInput;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod layer;

pub struct Qwen35MTP {
    layer: Qwen35MTPLayer,
    output_norm: RMSNorm,
    request_page_table: Rc<crate::attn::gqa::request_page_table::GQARequestPageTable>,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub pages: &'a Buffer,
}

impl Qwen35MTP {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        main_config: &Qwen35ModelConfig,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35LayerWeightBindings,
        final_norm_weight: String,
        gqa_state: &Qwen3xGQAState,
        layer_scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let layer = Qwen35MTPLayer::load(
            device,
            store,
            config,
            defaults,
            main_config.text_config.num_hidden_layers,
            bindings,
            gqa_state,
            layer_scratch,
            dense_scratch,
            moe_scratch,
        )?;
        let mut tensors = store.load_tensors([final_norm_weight.as_str()])?;
        let final_norm_weight = remove_qwen3x_norm_weight(device, &mut tensors, &final_norm_weight, &[hidden_dim])?;
        assert!(tensors.is_empty(), "qwen3.5 MTP must consume its final norm tensor map");
        Ok(Self {
            layer,
            output_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps, final_norm_weight),
            request_page_table: Rc::clone(gqa_state.request_page_table()),
        })
    }

    pub fn validate_batch(&self, microbatch: &Qwen35Microbatch) {
        let max_context_tokens = (0..microbatch.num_reqs())
            .map(|req_index| {
                microbatch.token_indices()[req_index]
                    .checked_add(microbatch.q_len(req_index))
                    .expect("qwen3.5 MTP GQA request context length overflow")
            })
            .max()
            .expect("qwen3.5 MTP batch requires requests") as usize;
        let page_capacity = self
            .request_page_table
            .num_blocks()
            .checked_mul(self.request_page_table.num_page_ids_per_block())
            .expect("qwen3.5 MTP GQA page capacity must fit usize");
        let tokens_per_page = self.layer.gqa_tokens_per_page();
        assert!(
            max_context_tokens.div_ceil(tokens_per_page.max(1)) <= page_capacity,
            "qwen3.5 MTP GQA request context exceeds page-table capacity"
        );
    }
}

impl Qwen35MTP {
    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MTPArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let hidden = self.layer.record(
            recorder,
            Qwen35MTPLayerInput {
                gqa: args.gqa,
                num_tokens,
                pages: args.pages,
                residual_input: args.hidden_input,
            },
        );
        self.output_norm
            .record(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPReplayKey {
    mtp_module_index: usize,
    num_tokens: usize,
    num_q_token_tiles: u32,
    total_sdpa_map_task_templates: u32,
}

impl Qwen35MTPReplayKey {
    pub fn new(
        mtp_module_index: usize,
        num_tokens: usize,
        gqa_shape: inference_executor_core::attn::GQAReplayShape,
    ) -> Self {
        gqa_shape.validate();
        Self {
            mtp_module_index,
            num_tokens,
            num_q_token_tiles: gqa_shape.num_q_token_tiles,
            total_sdpa_map_task_templates: gqa_shape.total_sdpa_map_task_templates,
        }
    }
}

impl ReplayComponent for Qwen35MTP {
    type Key = Qwen35MTPReplayKey;
    type Input<'a> = Qwen35MTPArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::Key::new(0, input.num_tokens as usize, input.gqa.replay_shape())
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35MTP::record(self, recorder, *input);
    }
}
