use std::rc::Rc;

use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3MainWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::layer::Qwen3MainLayer;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerInput;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerScratch;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RmsNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod gqa;
pub mod layer;
pub mod output;
pub mod plan;

pub struct Qwen3Main {
    layers: Vec<Qwen3MainLayer>,
    final_norm: RmsNorm,
    residual_capture: Option<Rc<dyn Qwen3MainResidualCapture>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub pages: &'a Buffer,
}

/// Selects capture targets for Qwen3 Main layer residual outputs.
///
/// Capture selection and destinations are fixed replay topology. The owner
/// must keep returned buffers and their column ranges stable for the lifetime
/// of Main, and targets must not alias Main workspaces.
pub trait Qwen3MainResidualCapture {
    fn capture_target_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualCaptureTarget<'_>>;
}

impl Qwen3Main {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3ModelConfig,
        bindings: Qwen3MainWeightBindings,
        gqa_state: &Qwen3MainGQAState,
        residual_capture: Option<Rc<dyn Qwen3MainResidualCapture>>,
        layer_scratch: Rc<Qwen3MainLayerScratch>,
        dense_scratch: &Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen3MainWeightBindings {
            final_norm_weight,
            layers: layer_bindings,
        } = bindings;
        assert_eq!(
            layer_bindings.len(),
            config.text_config.num_hidden_layers,
            "qwen3 Main config and checkpoint binding layer counts must match"
        );
        let text = &config.text_config;
        let mut layers = Vec::with_capacity(layer_bindings.len());
        for (model_layer_index, bindings) in layer_bindings.into_iter().enumerate() {
            layers.push(Qwen3MainLayer::load(
                device,
                store,
                config,
                model_layer_index,
                bindings,
                gqa_state,
                Rc::clone(&layer_scratch),
                Rc::clone(dense_scratch),
            )?);
            store.unload_all();
        }

        let final_norm_weight = load_qwen3x_norm_weight(
            device,
            store,
            &final_norm_weight,
            &[text.hidden_size],
            config.quantization.is_some(),
        )?;
        Ok(Self {
            layers,
            final_norm: RmsNorm::new(device, text.hidden_size, text.rms_norm_eps, final_norm_weight),
            residual_capture,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen3MainArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = <Qwen3MainLayer as ReplayLayer>::record(
                layer,
                recorder,
                Qwen3MainLayerInput {
                    gqa: args.gqa,
                    num_tokens: args.num_tokens,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                    residual_capture_target: self
                        .residual_capture
                        .as_ref()
                        .and_then(|capture| capture.capture_target_for_model_layer(layer.layer_index())),
                },
            );
        }
        self.final_norm
            .record(recorder, args.num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen3MainGQAReplayKey {
    num_q_token_tiles: u32,
    total_sdpa_map_task_templates: u32,
}

impl Qwen3MainGQAReplayKey {
    fn from_shape(gqa_shape: inference_executor_core::attn::GQAReplayShape) -> Self {
        gqa_shape.validate();
        Self {
            num_q_token_tiles: gqa_shape.num_q_token_tiles,
            total_sdpa_map_task_templates: gqa_shape.total_sdpa_map_task_templates,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3MainReplayKey {
    num_tokens: u32,
    gqa: Qwen3MainGQAReplayKey,
}

impl Qwen3MainReplayKey {
    pub fn from_shape(gqa_shape: inference_executor_core::attn::GQAReplayShape) -> Self {
        gqa_shape.validate();
        Self {
            num_tokens: gqa_shape.num_tokens,
            gqa: Qwen3MainGQAReplayKey::from_shape(gqa_shape),
        }
    }

    #[cfg(test)]
    pub fn debug_parts(&self) -> (u32, u32, u32) {
        (
            self.num_tokens,
            self.gqa.num_q_token_tiles,
            self.gqa.total_sdpa_map_task_templates,
        )
    }
}

impl ReplayComponent for Qwen3Main {
    type Key = Qwen3MainReplayKey;
    type Input<'a> = Qwen3MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3MainReplayKey::from_shape(input.gqa.replay_shape())
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen3Main::record(self, recorder, *input);
    }
}
