mod dense_mlp;
mod gdn;
mod gqa;
mod moe;
pub mod scratch;

use std::rc::Rc;

pub use dense_mlp::Qwen35DenseMLP;
pub use gdn::Qwen35GDN;
pub use gqa::Qwen35GQA;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35AttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MLPWeightBindings;
pub use moe::Qwen35MoE;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::layer::scratch::Qwen35LayerScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::weight::load_qwen35_norm_weight;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;

pub struct Qwen35Layer {
    layer_index: usize,
    input_norm: RmsNorm,
    attention: Qwen35Attention,
    residual: Residual,
    post_attention_norm: RmsNorm,
    mlp: Qwen35MLP,
    scratch: Rc<Qwen35LayerScratch>,
}

pub enum Qwen35Attention {
    GQA(Qwen35GQA),
    GDN(Qwen35GDN),
}

#[allow(clippy::large_enum_variant)]
pub enum Qwen35MLP {
    Dense(Qwen35DenseMLP),
    MoE(Qwen35MoE),
}

#[derive(Clone, Copy)]
pub struct Qwen35LayerInput<'a> {
    pub gdn: Option<&'a GDNMetadataBuffers>,
    pub gqa: &'a GQAMetadataBuffers,
    pub input: &'a Buffer,
    pub output: &'a Buffer,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35LayerResidualInput<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
}

impl Qwen35Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        layer_index: usize,
        input_norm_weight: String,
        post_attention_norm_weight: String,
        attention: Qwen35Attention,
        mlp: Qwen35MLP,
        scratch: Rc<Qwen35LayerScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        let stores_actual_scale = config.quantization.is_some();
        let norm_op = RmsNorm::kernel(device);
        Ok(Self {
            layer_index,
            input_norm: RmsNorm::new(
                hidden_dim,
                eps,
                load_qwen35_norm_weight(device, store, &input_norm_weight, &[hidden_dim], stores_actual_scale)?,
                Rc::clone(&norm_op),
            ),
            attention,
            residual: Residual::new(device),
            post_attention_norm: RmsNorm::new(
                hidden_dim,
                eps,
                load_qwen35_norm_weight(
                    device,
                    store,
                    &post_attention_norm_weight,
                    &[hidden_dim],
                    stores_actual_scale,
                )?,
                norm_op,
            ),
            mlp,
            scratch,
        })
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn output(&self) -> &Buffer {
        self.scratch.residual_stream(self.layer_index)
    }

    pub fn gqa_tokens_per_page(&self) -> Option<usize> {
        match &self.attention {
            Qwen35Attention::GQA(gqa) => Some(gqa.num_tokens_per_page()),
            Qwen35Attention::GDN(_) => None,
        }
    }

    pub fn record_body<'a, R>(&'a self, recorder: &mut R, input: Qwen35LayerInput<'a>) -> Qwen35LayerResidualInput<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = input.num_tokens;
        let num_values = residual_values(num_tokens, self.scratch.hidden_dim());
        self.input_norm
            .record_with_barrier(recorder, num_tokens, input.input, &self.scratch.normalized_hidden);
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.pages,
            input.gqa,
            input.gdn,
        );
        self.residual.record(
            recorder,
            num_values,
            input.input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
            None,
        );
        self.post_attention_norm.record(
            recorder,
            num_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            num_tokens,
        );
        Qwen35LayerResidualInput {
            lhs: &self.scratch.post_attention_hidden,
            rhs: &self.scratch.branch_output,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_attention(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        gqa_layer_index: &mut usize,
        gdn_layer_index: &mut usize,
        bindings: Qwen35AttentionWeightBindings,
        gqa_backend: &Rc<GQA>,
        gqa_scratch: &Rc<GQAScratch>,
        gqa_page_table: &Rc<GQARequestPageTable>,
        gdn_backend: &Rc<GDN>,
        gdn_scratch: &Rc<GDNScratch>,
        gdn_state_table: &Rc<GDNRequestStateTable>,
    ) -> Result<Qwen35Attention, ModelExecutorError> {
        match (config.layer_type_at(model_layer_index)?, bindings) {
            (LayerType::FullAttention, Qwen35AttentionWeightBindings::GQA(bindings)) => {
                let layer_index = *gqa_layer_index;
                *gqa_layer_index += 1;
                Ok(Qwen35Attention::GQA(Qwen35GQA::load(
                    device,
                    store,
                    config,
                    defaults,
                    model_layer_index,
                    layer_index,
                    bindings,
                    Rc::clone(gqa_backend),
                    Rc::clone(gqa_scratch),
                    Rc::clone(gqa_page_table),
                )?))
            },
            (LayerType::GDN, Qwen35AttentionWeightBindings::GDN(bindings)) => {
                let layer_index = *gdn_layer_index;
                *gdn_layer_index += 1;
                Ok(Qwen35Attention::GDN(Qwen35GDN::load(
                    device,
                    store,
                    config,
                    defaults,
                    model_layer_index,
                    layer_index,
                    bindings,
                    Rc::clone(gdn_backend),
                    Rc::clone(gdn_scratch),
                    Rc::clone(gdn_state_table),
                )?))
            },
            _ => panic!("qwen3.5 layer attention config and checkpoint bindings must have the same kind"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_mlp(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        bindings: Qwen35MLPWeightBindings,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Qwen35MLP, ModelExecutorError> {
        match (config.layer_uses_moe(model_layer_index), bindings) {
            (false, Qwen35MLPWeightBindings::Dense(bindings)) => {
                Ok(Qwen35MLP::Dense(Qwen35DenseMLP::load(
                    device,
                    store,
                    config,
                    defaults,
                    model_layer_index,
                    *bindings,
                    Rc::clone(dense_scratch.expect("qwen3.5 dense layer requires shared dense scratch")),
                )?))
            },
            (true, Qwen35MLPWeightBindings::MoE(bindings)) => {
                Ok(Qwen35MLP::MoE(Qwen35MoE::load(
                    device,
                    store,
                    config,
                    defaults,
                    model_layer_index,
                    *bindings,
                    Rc::clone(moe_scratch.expect("qwen3.5 MoE layer requires shared MoE scratch")),
                )?))
            },
            _ => panic!("qwen3.5 layer MLP config and checkpoint bindings must have the same kind"),
        }
    }
}

impl Layer for Qwen35Layer {
    type Input<'a> = Qwen35LayerInput<'a>;
    type Output<'a> = &'a Buffer;
    type InputShape = ();
    type OutputShape = ();

    fn input_shape(&self) -> Self::InputShape {}
    fn output_shape(&self) -> Self::OutputShape {}
}

impl ReplayLayer for Qwen35Layer {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = input.num_tokens;
        let num_values = residual_values(num_tokens, self.scratch.hidden_dim());
        let residual = self.record_body(recorder, input);
        self.residual
            .record(recorder, num_values, residual.lhs, residual.rhs, input.output, None);
        input.output
    }
}

impl Qwen35Attention {
    fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        pages: &'a Buffer,
        gqa: &'a GQAMetadataBuffers,
        gdn: Option<&'a GDNMetadataBuffers>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match self {
            Self::GQA(component) => component.record(recorder, input, output, pages, gqa),
            Self::GDN(component) => {
                component.record(
                    recorder,
                    input,
                    output,
                    gdn.expect("qwen3.5 GDN layer requires prepared metadata"),
                )
            },
        }
    }
}

impl Qwen35MLP {
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

fn residual_values(num_tokens: u32, hidden_dim: usize) -> u32 {
    num_tokens
        .checked_mul(hidden_dim.try_into().expect("qwen3.5 hidden dimension must fit u32"))
        .expect("qwen3.5 residual element index must fit u32")
}
