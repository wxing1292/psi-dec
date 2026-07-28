use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35AttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MLPWeightBindings;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;
use crate::model::qwen::v3_5::plan::qwen35_moe_core_and_metal;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::layer::Qwen3xGQA;
use crate::model::qwen::v3_x::layer::Qwen3xMoE;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;

pub struct Qwen35MTPLayer {
    input_norm: RmsNorm,
    attention: Qwen3xGQA,
    residual: Residual,
    post_attention_norm: RmsNorm,
    mlp: Qwen35MTPMLP,
    scratch: Rc<Qwen35MTPLayerScratch>,
}

pub struct Qwen35MTPLayerScratch {
    hidden_dim: usize,
    residual_output: Buffer,
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[allow(clippy::large_enum_variant)]
enum Qwen35MTPMLP {
    Dense(Qwen3xDenseMLP),
    MoE(Qwen3xMoE),
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPLayerInput<'a> {
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
}

impl Qwen35MTPLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        bindings: Qwen35LayerWeightBindings,
        gqa_state: &Qwen3xGQAState,
        scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
        } = bindings;
        let attention = match attention {
            Qwen35AttentionWeightBindings::GQA(bindings) => {
                let (core, metal) = qwen35_gqa_core_and_metal(model_layer_index, &config.text_config, defaults)?;
                Qwen3xGQA::load(
                    device,
                    store,
                    &core,
                    metal,
                    config.quantization.is_some(),
                    0,
                    bindings,
                    Rc::clone(gqa_state.backend()),
                    Rc::clone(gqa_state.scratch()),
                    Rc::clone(gqa_state.request_page_table()),
                )?
            },
            Qwen35AttentionWeightBindings::GDN(_) => {
                panic!("qwen3.5 MTP layer bindings must contain GQA attention")
            },
        };
        let mlp = Qwen35MTPMLP::load(device, store, config, defaults, mlp, dense_scratch, moe_scratch)?;
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        let stores_actual_scale = config.quantization.is_some();
        Ok(Self {
            input_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(device, store, &input_norm_weight, &[hidden_dim], stores_actual_scale)?,
            ),
            attention,
            residual: Residual::new(device),
            post_attention_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(
                    device,
                    store,
                    &post_attention_norm_weight,
                    &[hidden_dim],
                    stores_actual_scale,
                )?,
            ),
            mlp,
            scratch,
        })
    }

    pub fn gqa_tokens_per_page(&self) -> usize {
        self.attention.num_tokens_per_page()
    }
}

impl ReplayLayer for Qwen35MTPLayer {
    type Input<'a> = Qwen35MTPLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_values = residual_values(input.num_tokens, self.scratch.hidden_dim);
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
            input.pages,
            input.gqa,
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
            &self.scratch.residual_output,
            None,
        );
        &self.scratch.residual_output
    }
}

impl Qwen35MTPMLP {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35MLPWeightBindings,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        match (config.layer_uses_moe(0), bindings) {
            (false, Qwen35MLPWeightBindings::Dense(bindings)) => {
                let (core, metal) = qwen35_dense_mlp_core_and_metal(0, &config.text_config, defaults)?;
                Ok(Self::Dense(Qwen3xDenseMLP::load(
                    device,
                    store,
                    &core,
                    metal,
                    *bindings,
                    Rc::clone(dense_scratch.expect("qwen3.5 MTP dense layer requires shared dense scratch")),
                )?))
            },
            (true, Qwen35MLPWeightBindings::MoE(bindings)) => {
                let (core, metal) = qwen35_moe_core_and_metal("layers.0", 0, config, defaults)?;
                Ok(Self::MoE(Qwen3xMoE::load(
                    device,
                    store,
                    &core,
                    metal,
                    *bindings,
                    Rc::clone(moe_scratch.expect("qwen3.5 MTP MoE layer requires shared MoE scratch")),
                )?))
            },
            _ => panic!("qwen3.5 MTP layer MLP config and checkpoint bindings must have the same kind"),
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

impl Qwen35MTPLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 MTP layer scratch element count must fit usize");
        Self {
            hidden_dim,
            residual_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }
}

fn residual_values(num_tokens: u32, hidden_dim: usize) -> u32 {
    num_tokens
        .checked_mul(
            hidden_dim
                .try_into()
                .expect("qwen3.5 MTP hidden dimension must fit u32"),
        )
        .expect("qwen3.5 MTP residual element index must fit u32")
}
