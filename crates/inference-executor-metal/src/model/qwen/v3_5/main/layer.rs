use std::rc::Rc;

use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::components::residual_add;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
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
use crate::mlp::moe::backend::GatedMoEReplayTopology;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::component_config::derive_qwen35_dense_mlp_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_gdn_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_gqa_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_moe_configs;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::layer::Qwen3xGDN;
use crate::model::qwen::v3_x::layer::Qwen3xGQA;
use crate::model::qwen::v3_x::layer::Qwen3xMoE;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual_add::ResidualAdd;
use crate::model::rms_norm::RMSNorm;

pub struct Qwen35MainLayer {
    layer_index: usize,
    input_norm: RMSNorm,
    attention: Qwen35MainAttention,
    residual_add: ResidualAdd,
    post_attention_norm: RMSNorm,
    mlp: Qwen35MainMLP,
    scratch: Rc<Qwen35MainLayerScratch>,
}

pub struct Qwen35MainLayerScratch {
    max_tokens: u32,
    hidden_dim: u32,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35MainAttention {
    GQA(Qwen3xGQA),
    GDN(Qwen3xGDN),
}

#[allow(clippy::large_enum_variant)]
enum Qwen35MainMLP {
    Dense(Qwen3xDenseMLP),
    MoE(Qwen3xMoE),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Qwen35MainMLPReplayTopology {
    Dense(dense_mlp::ReplayTopology),
    MoE(GatedMoEReplayTopology),
}

#[derive(Clone, Copy)]
pub struct Qwen35MainLayerInput<'a> {
    pub gdn: &'a GDNMetadataBuffers,
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
    pub residual_capture_dest: Option<residual_add::CaptureTarget<'a>>,
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35MainAttentionInput<'a> {
    GQA {
        metadata: &'a GQAMetadataBuffers,
        pages: &'a Buffer,
    },
    GDN {
        metadata: &'a GDNMetadataBuffers,
    },
}

impl Qwen35MainLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        attn_layer_index: usize,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
        scratch: Rc<Qwen35MainLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let attention = Qwen35MainAttention::new(
            config,
            defaults,
            model_layer_index,
            attn_layer_index,
            gqa_state,
            gdn_state,
        )?;
        let mlp = Qwen35MainMLP::new(device, config, defaults, model_layer_index, dense_scratch, moe_scratch)?;
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        Ok(Self {
            layer_index: model_layer_index,
            input_norm: RMSNorm::new(device, hidden_dim, eps),
            attention,
            residual_add: ResidualAdd::new(device),
            post_attention_norm: RMSNorm::new(device, hidden_dim, eps),
            mlp,
            scratch,
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35LayerWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen35LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
        } = bindings;
        self.attention.load_weights(device, store, attention)?;
        self.mlp.load_weights(device, store, mlp)?;
        let hidden_dim = config.text_config.hidden_size;
        self.input_norm.load_weights(load_qwen3x_norm_weight(
            device,
            store,
            &input_norm_weight,
            &[hidden_dim],
        )?);
        self.post_attention_norm.load_weights(load_qwen3x_norm_weight(
            device,
            store,
            &post_attention_norm_weight,
            &[hidden_dim],
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.post_attention_norm.unload_weights();
        self.input_norm.unload_weights();
        self.mlp.unload_weights();
        self.attention.unload_weights();
    }

    pub fn unload_state(&mut self) {
        self.attention.unload_state();
    }

    pub fn load_state(&mut self, gqa_state: &Qwen3xGQAState, gdn_state: &Qwen3xGDNState) {
        self.attention.load_state(gqa_state, gdn_state);
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.layer_index)
    }

    pub fn mlp_replay_topology(&self, num_total_tokens: u32) -> Qwen35MainMLPReplayTopology {
        self.mlp.replay_topology(num_total_tokens)
    }

    pub fn mlp_replay_topology_boundaries(&self) -> Box<[u32]> {
        self.mlp.replay_topology_boundaries()
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: Qwen35MainLayerInput<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(input.num_tokens > 0, "qwen3.5 Main replay requires active tokens");
        assert!(
            input.num_tokens <= num_total_tokens,
            "qwen3.5 Main active tokens must not exceed the replay capacity"
        );
        let hidden_dim = self.scratch.hidden_dim();
        let attention_input = match &self.attention {
            Qwen35MainAttention::GQA(_) => {
                Qwen35MainAttentionInput::GQA {
                    metadata: input.gqa,
                    pages: input.pages,
                }
            },
            Qwen35MainAttention::GDN(_) => Qwen35MainAttentionInput::GDN { metadata: input.gdn },
        };
        self.input_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            attention_input,
            num_active_tokens,
        );
        self.residual_add.record(
            recorder,
            num_total_tokens,
            hidden_dim,
            num_active_tokens,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
        );
        self.post_attention_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            num_total_tokens,
            num_active_tokens,
        );
        match input.residual_capture_dest {
            Some(capture) => {
                self.residual_add.record_with_capture(
                    recorder,
                    num_total_tokens,
                    hidden_dim,
                    num_active_tokens,
                    &self.scratch.post_attention_hidden,
                    &self.scratch.branch_output,
                    input.residual_output,
                    capture,
                )
            },
            None => {
                self.residual_add.record(
                    recorder,
                    num_total_tokens,
                    hidden_dim,
                    num_active_tokens,
                    &self.scratch.post_attention_hidden,
                    &self.scratch.branch_output,
                    input.residual_output,
                )
            },
        }
        input.residual_output
    }
}

impl ReplayLayer for Qwen35MainLayer {
    type Input<'a> = Qwen35MainLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        Qwen35MainLayer::record(
            self,
            recorder,
            input.num_tokens,
            ReplayU32::Fixed(input.num_tokens),
            input,
        )
    }
}

impl Qwen35MainAttention {
    fn new(
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        attn_layer_index: usize,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
    ) -> Result<Self, ModelExecutorError> {
        match config.layer_type_at(model_layer_index)? {
            LayerType::FullAttention => {
                let (core, metal) = derive_qwen35_gqa_configs(model_layer_index, &config.text_config, defaults)?;
                Ok(Self::GQA(Qwen3xGQA::new(
                    inference_backend_metal::metal::ReplayU32::Fixed(
                        attn_layer_index
                            .try_into()
                            .expect("qwen3.5 GQA layer index must fit u32"),
                    ),
                    core,
                    metal,
                    Rc::clone(gqa_state.backend()),
                    Rc::clone(gqa_state.scratch()),
                    Rc::clone(gqa_state.request_page_table()),
                )))
            },
            LayerType::GDN => {
                let (core, metal) = derive_qwen35_gdn_configs(model_layer_index, &config.text_config, defaults)?;
                Ok(Self::GDN(Qwen3xGDN::new(
                    attn_layer_index,
                    core,
                    metal,
                    Rc::clone(gdn_state.backend()),
                    Rc::clone(gdn_state.scratch()),
                    Rc::clone(gdn_state.request_state_resources()),
                )))
            },
        }
    }

    fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen35AttentionWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        match (self, bindings) {
            (Self::GQA(component), Qwen35AttentionWeightBindings::GQA(bindings)) => {
                component.load_weights(device, store, bindings)
            },
            (Self::GDN(component), Qwen35AttentionWeightBindings::GDN(bindings)) => {
                component.load_weights(device, store, bindings)
            },
            _ => panic!("qwen3.5 Main layer attention config and checkpoint bindings must have the same kind"),
        }
    }

    fn unload_weights(&mut self) {
        match self {
            Self::GQA(component) => component.unload_weights(),
            Self::GDN(component) => component.unload_weights(),
        }
    }

    fn unload_state(&mut self) {
        match self {
            Self::GQA(component) => component.unload_state(),
            Self::GDN(component) => component.unload_state(),
        }
    }

    fn load_state(&mut self, gqa_state: &Qwen3xGQAState, gdn_state: &Qwen3xGDNState) {
        match self {
            Self::GQA(component) => component.load_state(gqa_state),
            Self::GDN(component) => component.load_state(gdn_state),
        }
    }

    fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        metadata: Qwen35MainAttentionInput<'a>,
        num_active_tokens: ReplayU32,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match (self, metadata) {
            (Self::GQA(component), Qwen35MainAttentionInput::GQA { metadata, pages }) => {
                component.record(recorder, input, output, pages, metadata, num_active_tokens)
            },
            (Self::GDN(component), Qwen35MainAttentionInput::GDN { metadata }) => {
                component.record(recorder, input, output, metadata, num_active_tokens)
            },
            _ => panic!("qwen3.5 attention component and metadata must have the same kind"),
        }
    }
}

impl Qwen35MainMLP {
    fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        if config.layer_uses_moe(model_layer_index) {
            let layer_prefix = format!("layers.{model_layer_index}");
            let (core, metal) = derive_qwen35_moe_configs(&layer_prefix, model_layer_index, config, defaults)?;
            Ok(Self::MoE(Qwen3xMoE::new(
                device,
                core,
                metal,
                Rc::clone(moe_scratch.expect("qwen3.5 MoE layer requires shared MoE scratch")),
            )))
        } else {
            let (core, metal) = derive_qwen35_dense_mlp_configs(model_layer_index, &config.text_config, defaults)?;
            Ok(Self::Dense(Qwen3xDenseMLP::new(
                device,
                core,
                metal,
                Rc::clone(dense_scratch.expect("qwen3.5 dense layer requires shared dense scratch")),
            )))
        }
    }

    fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen35MLPWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        match (self, bindings) {
            (Self::Dense(component), Qwen35MLPWeightBindings::Dense(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            (Self::MoE(component), Qwen35MLPWeightBindings::MoE(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            _ => panic!("qwen3.5 Main layer MLP config and checkpoint bindings must have the same kind"),
        }
    }

    fn unload_weights(&mut self) {
        match self {
            Self::Dense(component) => component.unload_weights(),
            Self::MoE(component) => component.unload_weights(),
        }
    }

    fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match self {
            Self::Dense(component) => component.record(recorder, input, output, num_total_tokens, num_active_tokens),
            Self::MoE(component) => component.record(recorder, input, output, num_total_tokens, num_active_tokens),
        }
    }

    fn replay_topology(&self, num_total_tokens: u32) -> Qwen35MainMLPReplayTopology {
        match self {
            Self::Dense(component) => Qwen35MainMLPReplayTopology::Dense(component.replay_topology(num_total_tokens)),
            Self::MoE(component) => Qwen35MainMLPReplayTopology::MoE(component.replay_topology(num_total_tokens)),
        }
    }

    fn replay_topology_boundaries(&self) -> Box<[u32]> {
        match self {
            Self::Dense(component) => component.replay_topology_boundaries(),
            Self::MoE(component) => component.replay_topology_boundaries(),
        }
    }
}

impl Qwen35MainLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let max_tokens_u32 = u32::try_from(max_tokens).expect("qwen3.5 Main layer token capacity must fit u32");
        let hidden_dim_u32 = u32::try_from(hidden_dim).expect("qwen3.5 Main hidden dimension must fit u32");
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 Main layer scratch element count must fit usize");
        u32::try_from(hidden_elements)
            .expect("qwen3.5 Main layer scratch must fit the shader u32 element-count domain");
        Self {
            max_tokens: max_tokens_u32,
            hidden_dim: hidden_dim_u32,
            residual_stream: [
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            ],
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }

    fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    fn residual_values(&self, num_tokens: u32) -> u32 {
        debug_assert!(num_tokens > 0 && num_tokens <= self.max_tokens);
        num_tokens * self.hidden_dim
    }

    fn residual_stream(&self, model_layer_index: usize) -> &Buffer {
        &self.residual_stream[model_layer_index % self.residual_stream.len()]
    }
}
