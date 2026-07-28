use std::rc::Rc;

use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35AttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MLPWeightBindings;

use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_gdn_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_moe_core_and_metal;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::layer::Qwen3xGDN;
use crate::model::qwen::v3_x::layer::Qwen3xGQA;
use crate::model::qwen::v3_x::layer::Qwen3xMoE;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;

pub struct Qwen35MainLayer {
    layer_index: usize,
    input_norm: RmsNorm,
    attention: Qwen35MainAttention,
    residual: Residual,
    post_attention_norm: RmsNorm,
    mlp: Qwen35MainMLP,
    scratch: Rc<Qwen35MainLayerScratch>,
}

pub struct Qwen35MainLayerScratch {
    hidden_dim: usize,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

enum Qwen35MainAttention {
    Gqa(Qwen3xGQA),
    Gdn(Qwen3xGDN),
}

#[allow(clippy::large_enum_variant)]
enum Qwen35MainMLP {
    Dense(Qwen3xDenseMLP),
    MoE(Qwen3xMoE),
}

#[derive(Clone, Copy)]
pub struct Qwen35MainLayerInput<'a> {
    pub gdn: &'a GDNMetadataBuffers,
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
    pub residual_capture_target: Option<ResidualCaptureTarget<'a>>,
}

enum Qwen35MainAttentionInput<'a> {
    Gqa {
        metadata: &'a GQAMetadataBuffers,
        pages: &'a Buffer,
    },
    Gdn {
        metadata: &'a GDNMetadataBuffers,
    },
}

impl Qwen35MainLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        compact_gqa_layer_index: usize,
        compact_gdn_layer_index: usize,
        bindings: Qwen35LayerWeightBindings,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
        scratch: Rc<Qwen35MainLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
        } = bindings;
        let attention = Qwen35MainAttention::load(
            device,
            store,
            config,
            defaults,
            model_layer_index,
            compact_gqa_layer_index,
            compact_gdn_layer_index,
            attention,
            gqa_state,
            gdn_state,
        )?;
        let mlp = Qwen35MainMLP::load(
            device,
            store,
            config,
            defaults,
            model_layer_index,
            mlp,
            dense_scratch,
            moe_scratch,
        )?;
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        Ok(Self {
            layer_index: model_layer_index,
            input_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(device, store, &input_norm_weight, &[hidden_dim])?,
            ),
            attention,
            residual: Residual::new(device),
            post_attention_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(device, store, &post_attention_norm_weight, &[hidden_dim])?,
            ),
            mlp,
            scratch,
        })
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.layer_index)
    }
}

impl ReplayLayer for Qwen35MainLayer {
    type Input<'a> = Qwen35MainLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_values = residual_values(input.num_tokens, self.scratch.hidden_dim());
        let attention_input = match &self.attention {
            Qwen35MainAttention::Gqa(_) => {
                Qwen35MainAttentionInput::Gqa {
                    metadata: input.gqa,
                    pages: input.pages,
                }
            },
            Qwen35MainAttention::Gdn(_) => Qwen35MainAttentionInput::Gdn { metadata: input.gdn },
        };
        self.input_norm.record_with_barrier(
            recorder,
            input.num_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            attention_input,
        );
        self.residual.record(
            recorder,
            num_values,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
            None,
        );
        self.post_attention_norm.record(
            recorder,
            input.num_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.num_tokens,
        );
        self.residual.record(
            recorder,
            num_values,
            &self.scratch.post_attention_hidden,
            &self.scratch.branch_output,
            input.residual_output,
            input.residual_capture_target,
        );
        input.residual_output
    }
}

impl Qwen35MainAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        compact_gqa_layer_index: usize,
        compact_gdn_layer_index: usize,
        bindings: Qwen35AttentionWeightBindings,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
    ) -> Result<Self, ModelExecutorError> {
        match (config.layer_type_at(model_layer_index)?, bindings) {
            (LayerType::FullAttention, Qwen35AttentionWeightBindings::GQA(bindings)) => {
                let (core, metal) = qwen35_gqa_core_and_metal(model_layer_index, &config.text_config, defaults)?;
                Ok(Self::Gqa(Qwen3xGQA::load(
                    device,
                    store,
                    &core,
                    metal,
                    compact_gqa_layer_index,
                    bindings,
                    Rc::clone(gqa_state.backend()),
                    Rc::clone(gqa_state.scratch()),
                    Rc::clone(gqa_state.request_page_table()),
                )?))
            },
            (LayerType::GDN, Qwen35AttentionWeightBindings::GDN(bindings)) => {
                let (core, metal) = qwen35_gdn_core_and_metal(model_layer_index, &config.text_config, defaults)?;
                Ok(Self::Gdn(Qwen3xGDN::load(
                    device,
                    store,
                    &core,
                    metal,
                    compact_gdn_layer_index,
                    bindings,
                    Rc::clone(gdn_state.backend()),
                    Rc::clone(gdn_state.scratch()),
                    Rc::clone(gdn_state.request_state_table()),
                )?))
            },
            _ => panic!("qwen3.5 Main layer attention config and checkpoint bindings must have the same kind"),
        }
    }

    fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        metadata: Qwen35MainAttentionInput<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match (self, metadata) {
            (Self::Gqa(component), Qwen35MainAttentionInput::Gqa { metadata, pages }) => {
                component.record(recorder, input, output, pages, metadata)
            },
            (Self::Gdn(component), Qwen35MainAttentionInput::Gdn { metadata }) => {
                component.record(recorder, input, output, metadata)
            },
            _ => panic!("qwen3.5 attention component and metadata must have the same kind"),
        }
    }
}

impl Qwen35MainMLP {
    #[allow(clippy::too_many_arguments)]
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        bindings: Qwen35MLPWeightBindings,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        match (config.layer_uses_moe(model_layer_index), bindings) {
            (false, Qwen35MLPWeightBindings::Dense(bindings)) => {
                let (core, metal) = qwen35_dense_mlp_core_and_metal(model_layer_index, &config.text_config, defaults)?;
                Ok(Self::Dense(Qwen3xDenseMLP::load(
                    device,
                    store,
                    &core,
                    metal,
                    *bindings,
                    Rc::clone(dense_scratch.expect("qwen3.5 dense layer requires shared dense scratch")),
                )?))
            },
            (true, Qwen35MLPWeightBindings::MoE(bindings)) => {
                let layer_prefix = format!("layers.{model_layer_index}");
                let (core, metal) = qwen35_moe_core_and_metal(&layer_prefix, model_layer_index, config, defaults)?;
                Ok(Self::MoE(Qwen3xMoE::load(
                    device,
                    store,
                    &core,
                    metal,
                    *bindings,
                    Rc::clone(moe_scratch.expect("qwen3.5 MoE layer requires shared MoE scratch")),
                )?))
            },
            _ => panic!("qwen3.5 Main layer MLP config and checkpoint bindings must have the same kind"),
        }
    }

    fn record<'a, R>(&'a self, recorder: &mut R, input: &'a Buffer, output: &'a Buffer, num_tokens: u32)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match self {
            Self::Dense(component) => component.record(recorder, input, output, num_tokens),
            Self::MoE(component) => component.record(recorder, input, output, num_tokens),
        }
    }
}

impl Qwen35MainLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 Main layer scratch element count must fit usize");
        Self {
            hidden_dim,
            residual_stream: [
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            ],
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn residual_stream(&self, model_layer_index: usize) -> &Buffer {
        &self.residual_stream[model_layer_index % self.residual_stream.len()]
    }
}

fn residual_values(num_tokens: u32, hidden_dim: usize) -> u32 {
    num_tokens
        .checked_mul(hidden_dim.try_into().expect("qwen3.5 hidden dimension must fit u32"))
        .expect("qwen3.5 residual element index must fit u32")
}
