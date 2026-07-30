use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayer;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayerInput;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayerScratch;
use crate::model::qwen::v3_x::dspark::main_feature::Qwen3xDSparkMainFeatureProjector;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkPlan;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RmsNorm;
use crate::replay::ReplayComponent;

pub struct Qwen3xDSparkModel {
    main_feature_projector: Rc<Qwen3xDSparkMainFeatureProjector>,
    layers: Vec<Qwen3xDSparkLayer>,
    final_norm: RmsNorm,
}

pub struct Qwen3xDSparkContext {
    model: Rc<Qwen3xDSparkModel>,
}

pub struct Qwen3xDSparkBody {
    model: Rc<Qwen3xDSparkModel>,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkContextArgs<'a> {
    pub num_tokens: u32,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub pages: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkBodyArgs<'a> {
    pub num_tokens: u32,
    pub metadata: &'a DSparkGQAMetadataBuffers,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub pages: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkContextReplayKey {
    num_tokens: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDSparkBodyReplayKey {
    num_tokens: u32,
    total_sdpa_map_task_templates: u32,
}

impl Qwen3xDSparkModel {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkPlan,
        bindings: Qwen3xDSparkWeightBindings,
        gqa_state: &UngatedDSparkGQAState,
        max_main_tokens: usize,
        max_block_tokens: usize,
    ) -> Result<Rc<Self>, ModelExecutorError> {
        let Qwen3xDSparkWeightBindings {
            embed: _,
            main_feature,
            layers: layer_bindings,
            final_norm_weight,
            unembed: _,
            markov: _,
            confidence: _,
        } = bindings;
        assert_eq!(
            plan.layers.len(),
            layer_bindings.len(),
            "Qwen3 DSpark plan and checkpoint binding layer counts must match"
        );
        let first_layer = plan
            .layers
            .first()
            .expect("Qwen3 DSpark model requires transformer layers");
        let hidden_dim = first_layer.attention_core.attention.hidden_dim;
        let layer_scratch = Rc::new(Qwen3xDSparkLayerScratch::new(device, max_block_tokens, hidden_dim));
        let dense_scratch = Rc::new(DenseMLPScratch::new(
            device,
            &first_layer.mlp_core,
            first_layer.mlp_metal.io_dtype,
            max_block_tokens,
        ));
        let main_feature_projector = Rc::new(Qwen3xDSparkMainFeatureProjector::load(
            device,
            store,
            plan,
            &main_feature,
            max_main_tokens,
        )?);
        let mut layers = Vec::with_capacity(plan.layers.len());
        for (layer_plan, bindings) in plan.layers.iter().zip(layer_bindings) {
            assert_eq!(
                layer_plan.attention_core.attention.hidden_dim, hidden_dim,
                "Qwen3 DSpark layers must use one hidden dimension"
            );
            assert_eq!(
                layer_plan.mlp_core.hidden_dim, hidden_dim,
                "Qwen3 DSpark layer MLP hidden dimension must match attention"
            );
            assert_eq!(
                layer_plan.mlp_core.intermediate_dim, first_layer.mlp_core.intermediate_dim,
                "Qwen3 DSpark layers must share one MLP scratch geometry"
            );
            assert_eq!(
                layer_plan.mlp_metal, first_layer.mlp_metal,
                "Qwen3 DSpark layers must share one MLP Metal layout"
            );
            layers.push(Qwen3xDSparkLayer::load(
                device,
                store,
                layer_plan,
                bindings,
                gqa_state,
                Rc::clone(&layer_scratch),
                Rc::clone(&dense_scratch),
            )?);
            store.unload_all();
        }
        let final_norm_weight = load_qwen3x_norm_weight(device, store, &final_norm_weight, &[hidden_dim])?;
        Ok(Rc::new(Self {
            main_feature_projector,
            layers,
            final_norm: RmsNorm::new(device, hidden_dim, plan.norm_eps, final_norm_weight),
        }))
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDSparkMainFeatureProjector> {
        Rc::clone(&self.main_feature_projector)
    }

    fn record_context<'a, R>(&'a self, recorder: &mut R, args: Qwen3xDSparkContextArgs<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let main_feature = self.main_feature_projector.record(recorder, args.num_tokens);
        for layer in &self.layers {
            layer.record_context(
                recorder,
                args.num_tokens,
                main_feature,
                args.req_slots,
                args.flat_token_indices,
                args.pages,
            );
        }
    }

    fn record_body<'a, R>(&'a self, recorder: &mut R, args: Qwen3xDSparkBodyArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = layer.record_block(
                recorder,
                Qwen3xDSparkLayerInput {
                    num_tokens: args.num_tokens,
                    metadata: args.metadata,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                },
            );
        }
        self.final_norm
            .record(recorder, args.num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

impl Qwen3xDSparkContext {
    pub fn new(model: Rc<Qwen3xDSparkModel>) -> Self {
        Self { model }
    }
}

impl ReplayComponent for Qwen3xDSparkContext {
    type Key = Qwen3xDSparkContextReplayKey;
    type Input<'a> = Qwen3xDSparkContextArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDSparkContextReplayKey {
            num_tokens: input.num_tokens,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.model.record_context(recorder, *input);
    }
}

impl Qwen3xDSparkBody {
    pub fn new(model: Rc<Qwen3xDSparkModel>) -> Self {
        Self { model }
    }
}

impl ReplayComponent for Qwen3xDSparkBody {
    type Key = Qwen3xDSparkBodyReplayKey;
    type Input<'a> = Qwen3xDSparkBodyArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let shape = input.metadata.replay_shape();
        Qwen3xDSparkBodyReplayKey {
            num_tokens: shape.num_tokens,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.model.record_body(recorder, *input);
    }
}
