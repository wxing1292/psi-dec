use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2LayerWeightBindings;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2MainFeatureWeightBindings;

use crate::attn::block_spec::metadata::BlockSpecGQAMetadataBuffers;
use crate::attn::block_spec::state::BlockSpecGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::main_residual_capture::MainResidualRows;
use crate::model::qwen::v3_x::dflash2::layer::Qwen3xDFlash2Layer;
use crate::model::qwen::v3_x::dflash2::layer::Qwen3xDFlash2LayerInput;
use crate::model::qwen::v3_x::dflash2::layer::Qwen3xDFlash2LayerScratch;
use crate::model::qwen::v3_x::dflash2::main_feature::Qwen3xDFlash2MainFeatureProjector;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub struct Qwen3xDFlash2Model {
    main_feature_projector: Option<Rc<Qwen3xDFlash2MainFeatureProjector>>,
    layers: Vec<Qwen3xDFlash2Layer>,
    final_norm: RMSNorm,
}

pub struct Qwen3xDFlash2Prefill {
    model: Option<Rc<Qwen3xDFlash2Model>>,
}

pub struct Qwen3xDFlash2Body {
    model: Option<Rc<Qwen3xDFlash2Model>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDFlash2PrefillArgs<'a> {
    pub num_tokens: u32,
    pub main_rows: MainResidualRows<'a>,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub pages: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDFlash2BodyArgs<'a> {
    pub num_tokens: u32,
    pub metadata: &'a BlockSpecGQAMetadataBuffers,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub pages: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDFlash2PrefillReplayKey {
    num_tokens: u32,
    gathers_main_residual_rows: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDFlash2BodyReplayKey {
    num_tokens: u32,
    num_total_sdpa_map_task_templates: u32,
}

impl Qwen3xDFlash2Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen3xDFlash2Config,
        num_spec_tokens: usize,
        page_bytes: usize,
        main_feature_bindings: &Qwen3xDFlash2MainFeatureWeightBindings,
        layer_bindings: &[Qwen3xDFlash2LayerWeightBindings],
        gqa_state: &BlockSpecGQAState,
        max_main_tokens: usize,
        max_requests: usize,
        max_block_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        assert_eq!(
            layer_bindings.len(),
            config.num_hidden_layers,
            "Qwen3x DFlash2 config and checkpoint binding layer counts must match"
        );
        let hidden_dim = config.hidden_size;
        let layer_scratch = Rc::new(Qwen3xDFlash2LayerScratch::new(device, max_block_tokens, hidden_dim));
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
        let main_feature_projector = Rc::new(Qwen3xDFlash2MainFeatureProjector::new(
            device,
            config,
            main_feature_bindings,
            max_main_tokens,
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (dflash2_layer_index, bindings) in layer_bindings.iter().enumerate() {
            layers.push(Qwen3xDFlash2Layer::new(
                device,
                config,
                num_spec_tokens,
                max_requests,
                dflash2_layer_index,
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
        config: &Qwen3xDFlash2Config,
        main_feature_bindings: &Qwen3xDFlash2MainFeatureWeightBindings,
        layer_bindings: Vec<Qwen3xDFlash2LayerWeightBindings>,
        final_norm_weight: String,
    ) -> Result<(), ModelExecutorError> {
        let projector = self
            .main_feature_projector
            .take()
            .expect("DFlash2 Main-feature projector shell must exist during weight loading");
        let mut projector = Rc::try_unwrap(projector)
            .unwrap_or_else(|_| panic!("DFlash2 Main-feature projector must be uniquely owned during weight loading"));
        let load_result = projector.load_weights(device, store, config, main_feature_bindings);
        self.main_feature_projector = Some(Rc::new(projector));
        load_result?;
        assert_eq!(
            self.layers.len(),
            layer_bindings.len(),
            "Qwen3x DFlash2 component and checkpoint binding layer counts must match"
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
            "Qwen3x DFlash2 model must consume its final norm tensor map"
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
            .expect("DFlash2 Main-feature projector must exist during weight unloading");
        let mut projector = Rc::try_unwrap(projector).unwrap_or_else(|_| {
            panic!("DFlash2 Main-feature projector must be uniquely owned during weight unloading")
        });
        projector.unload_weights();
        self.main_feature_projector = Some(Rc::new(projector));
    }

    pub fn unload_state(&mut self) {
        for layer in self.layers.iter_mut().rev() {
            layer.unload_state();
        }
    }

    pub fn load_state(&mut self, state: &BlockSpecGQAState) {
        for layer in &mut self.layers {
            layer.load_state(state);
        }
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDFlash2MainFeatureProjector> {
        Rc::clone(
            self.main_feature_projector
                .as_ref()
                .expect("DFlash2 Main-feature projector shell must exist"),
        )
    }

    fn record_prefill<'a, R>(&'a self, recorder: &mut R, args: Qwen3xDFlash2PrefillArgs<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let main_feature = self
            .main_feature_projector
            .as_ref()
            .expect("DFlash2 Main-feature projector shell must exist")
            .record(recorder, args.num_tokens, args.main_rows);
        for layer in &self.layers {
            layer.record_prefill(
                recorder,
                args.num_tokens,
                main_feature,
                args.req_slots,
                args.flat_token_indices,
                args.pages,
            );
        }
    }

    fn record_body<'a, R>(&'a self, recorder: &mut R, args: Qwen3xDFlash2BodyArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = layer.record_block(
                recorder,
                Qwen3xDFlash2LayerInput {
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

impl Qwen3xDFlash2Prefill {
    pub fn new(model: Rc<Qwen3xDFlash2Model>) -> Self {
        Self { model: Some(model) }
    }

    pub fn take_model(&mut self) -> Rc<Qwen3xDFlash2Model> {
        self.model
            .take()
            .expect("Qwen3.x DFlash2 Prefill model state must be loaded")
    }

    pub fn set_model(&mut self, model: Rc<Qwen3xDFlash2Model>) {
        assert!(
            self.model.is_none(),
            "Qwen3.x DFlash2 Prefill model state is already loaded"
        );
        self.model = Some(model);
    }

    fn model(&self) -> &Qwen3xDFlash2Model {
        self.model
            .as_deref()
            .expect("Qwen3.x DFlash2 Prefill model state must be loaded before execution")
    }
}

impl ReplayComponent for Qwen3xDFlash2Prefill {
    type Key = Qwen3xDFlash2PrefillReplayKey;
    type Input<'a> = Qwen3xDFlash2PrefillArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDFlash2PrefillReplayKey {
            num_tokens: input.num_tokens,
            gathers_main_residual_rows: input.main_rows.gathers(),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.model().record_prefill(recorder, *input);
    }
}

impl Qwen3xDFlash2Body {
    pub fn new(model: Rc<Qwen3xDFlash2Model>) -> Self {
        Self { model: Some(model) }
    }

    pub fn take_model(&mut self) -> Rc<Qwen3xDFlash2Model> {
        self.model
            .take()
            .expect("Qwen3.x DFlash2 body model state must be loaded")
    }

    pub fn set_model(&mut self, model: Rc<Qwen3xDFlash2Model>) {
        assert!(
            self.model.is_none(),
            "Qwen3.x DFlash2 body model state is already loaded"
        );
        self.model = Some(model);
    }

    fn model(&self) -> &Qwen3xDFlash2Model {
        self.model
            .as_deref()
            .expect("Qwen3.x DFlash2 body model state must be loaded before execution")
    }
}

impl ReplayComponent for Qwen3xDFlash2Body {
    type Key = Qwen3xDFlash2BodyReplayKey;
    type Input<'a> = Qwen3xDFlash2BodyArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        dflash2_body_replay_key(input.metadata.replay_shape())
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.model().record_body(recorder, *input);
    }
}

fn dflash2_body_replay_key(shape: GQAReplayShape) -> Qwen3xDFlash2BodyReplayKey {
    Qwen3xDFlash2BodyReplayKey {
        num_tokens: shape.num_tokens,
        num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
    }
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::GQAReplayShape;

    use super::dflash2_body_replay_key;

    #[test]
    fn body_replay_key_reuses_one_capacity_for_different_active_history_task_counts() {
        let shape = GQAReplayShape {
            num_tokens: 8,
            num_total_tokens: 8,
            num_q_token_tiles: 1,
            num_total_q_token_tiles: 1,
            num_sdpa_map_task_templates: 7,
            num_total_sdpa_map_task_templates: 8,
            reduce_sdpa_partial_outputs: true,
        };
        let mut longer_history = shape;
        longer_history.num_sdpa_map_task_templates = 8;

        assert_eq!(dflash2_body_replay_key(shape), dflash2_body_replay_key(longer_history));
    }
}
