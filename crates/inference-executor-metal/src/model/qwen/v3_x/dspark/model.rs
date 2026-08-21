use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkLayerWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMainFeatureWeightBindings;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::state::DSparkGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayer;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayerInput;
use crate::model::qwen::v3_x::dspark::layer::Qwen3xDSparkLayerScratch;
use crate::model::qwen::v3_x::dspark::main_feature::Qwen3xDSparkMainFeatureProjector;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub struct Qwen3xDSparkModel {
    main_feature_projector: Option<Rc<Qwen3xDSparkMainFeatureProjector>>,
    layers: Vec<Qwen3xDSparkLayer>,
    final_norm: RMSNorm,
}

pub struct Qwen3xDSparkContext {
    model: Option<Rc<Qwen3xDSparkModel>>,
}

pub struct Qwen3xDSparkBody {
    model: Option<Rc<Qwen3xDSparkModel>>,
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
    num_total_sdpa_map_task_templates: u32,
}

impl Qwen3xDSparkModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen3xDSparkConfig,
        num_spec_tokens: usize,
        page_bytes: usize,
        main_feature_bindings: &Qwen3xDSparkMainFeatureWeightBindings,
        layer_bindings: &[Qwen3xDSparkLayerWeightBindings],
        gqa_state: &DSparkGQAState,
        max_main_tokens: usize,
        max_block_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        assert_eq!(
            layer_bindings.len(),
            config.num_hidden_layers,
            "Qwen3x DSpark config and checkpoint binding layer counts must match"
        );
        let hidden_dim = config.hidden_size;
        let layer_scratch = Rc::new(Qwen3xDSparkLayerScratch::new(device, max_block_tokens, hidden_dim));
        let dense_scratch_core = DenseMLPCore {
            model_layer_index: 0,
            hidden_dim,
            intermediate_dim: config.intermediate_size,
        };
        let dense_scratch = Rc::new(DenseMLPScratch::new(
            device,
            &dense_scratch_core,
            Dtype::Bfloat16,
            max_block_tokens,
        ));
        let main_feature_projector = Rc::new(Qwen3xDSparkMainFeatureProjector::new(
            device,
            config,
            main_feature_bindings,
            max_main_tokens,
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (dspark_layer_index, bindings) in layer_bindings.iter().enumerate() {
            layers.push(Qwen3xDSparkLayer::new(
                device,
                config,
                num_spec_tokens,
                dspark_layer_index,
                page_bytes,
                bindings,
                gqa_state,
                Rc::clone(&layer_scratch),
                Rc::clone(&dense_scratch),
            )?);
        }
        Ok(Self {
            main_feature_projector: Some(main_feature_projector),
            layers,
            final_norm: RMSNorm::new(device, hidden_dim, config.rms_norm_eps),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3xDSparkConfig,
        main_feature_bindings: &Qwen3xDSparkMainFeatureWeightBindings,
        layer_bindings: Vec<Qwen3xDSparkLayerWeightBindings>,
        final_norm_weight: String,
    ) -> Result<(), ModelExecutorError> {
        let projector = self
            .main_feature_projector
            .take()
            .expect("DSpark Main-feature projector shell must exist during weight loading");
        let mut projector = Rc::try_unwrap(projector)
            .unwrap_or_else(|_| panic!("DSpark Main-feature projector must be uniquely owned during weight loading"));
        let load_result = projector.load_weights(device, store, config, main_feature_bindings);
        self.main_feature_projector = Some(Rc::new(projector));
        load_result?;
        assert_eq!(
            self.layers.len(),
            layer_bindings.len(),
            "Qwen3x DSpark component and checkpoint binding layer counts must match"
        );
        for (layer, bindings) in self.layers.iter_mut().zip(layer_bindings) {
            layer.load_weights(device, store, config, bindings)?;
        }
        let mut tensors = store.load_tensors([final_norm_weight.as_str()])?;
        self.final_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &final_norm_weight,
            &[config.hidden_size],
        )?);
        assert!(
            tensors.is_empty(),
            "Qwen3x DSpark model must consume its final norm tensor map"
        );
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.final_norm.unload_weights();
        for layer in self.layers.iter_mut().rev() {
            layer.unload_weights();
        }
        let projector = self
            .main_feature_projector
            .take()
            .expect("DSpark Main-feature projector must exist during weight unloading");
        let mut projector = Rc::try_unwrap(projector)
            .unwrap_or_else(|_| panic!("DSpark Main-feature projector must be uniquely owned during weight unloading"));
        projector.unload_weights();
        self.main_feature_projector = Some(Rc::new(projector));
    }

    pub fn unload_state(&mut self) {
        for layer in self.layers.iter_mut().rev() {
            layer.unload_state();
        }
    }

    pub fn load_state(&mut self, state: &DSparkGQAState) {
        for layer in &mut self.layers {
            layer.load_state(state);
        }
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDSparkMainFeatureProjector> {
        Rc::clone(
            self.main_feature_projector
                .as_ref()
                .expect("DSpark Main-feature projector shell must exist"),
        )
    }

    fn record_context<'a, R>(&'a self, recorder: &mut R, args: Qwen3xDSparkContextArgs<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let main_feature = self
            .main_feature_projector
            .as_ref()
            .expect("DSpark Main-feature projector shell must exist")
            .record(recorder, args.num_tokens);
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
        self.final_norm.record_with_barrier(
            recorder,
            args.num_tokens,
            ReplayU32::Fixed(args.num_tokens),
            hidden,
            args.hidden_output,
        );
        args.hidden_output
    }
}

impl Qwen3xDSparkContext {
    pub fn new(model: Rc<Qwen3xDSparkModel>) -> Self {
        Self { model: Some(model) }
    }

    pub fn take_model(&mut self) -> Rc<Qwen3xDSparkModel> {
        self.model
            .take()
            .expect("Qwen3.x DSpark context model state must be loaded")
    }

    pub fn set_model(&mut self, model: Rc<Qwen3xDSparkModel>) {
        assert!(
            self.model.is_none(),
            "Qwen3.x DSpark context model state is already loaded"
        );
        self.model = Some(model);
    }

    fn model(&self) -> &Qwen3xDSparkModel {
        self.model
            .as_deref()
            .expect("Qwen3.x DSpark context model state must be loaded before execution")
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
        self.model().record_context(recorder, *input);
    }
}

impl Qwen3xDSparkBody {
    pub fn new(model: Rc<Qwen3xDSparkModel>) -> Self {
        Self { model: Some(model) }
    }

    pub fn take_model(&mut self) -> Rc<Qwen3xDSparkModel> {
        self.model
            .take()
            .expect("Qwen3.x DSpark body model state must be loaded")
    }

    pub fn set_model(&mut self, model: Rc<Qwen3xDSparkModel>) {
        assert!(
            self.model.is_none(),
            "Qwen3.x DSpark body model state is already loaded"
        );
        self.model = Some(model);
    }

    fn model(&self) -> &Qwen3xDSparkModel {
        self.model
            .as_deref()
            .expect("Qwen3.x DSpark body model state must be loaded before execution")
    }
}

impl ReplayComponent for Qwen3xDSparkBody {
    type Key = Qwen3xDSparkBodyReplayKey;
    type Input<'a> = Qwen3xDSparkBodyArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let shape = input.metadata.replay_shape();
        Qwen3xDSparkBodyReplayKey {
            num_tokens: shape.num_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.model().record_body(recorder, *input);
    }
}
