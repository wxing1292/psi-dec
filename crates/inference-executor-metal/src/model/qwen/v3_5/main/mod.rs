use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MainWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayer;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerInput;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod layer;
pub mod output;

pub struct Qwen35Main {
    layers: Vec<Qwen35MainLayer>,
    final_norm: RMSNorm,
    residual_capture: Option<Rc<dyn MainResidualCapture>>,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gdn: &'a crate::attn::gdn::batch_metadata::GDNMetadataBuffers,
    pub pages: &'a Buffer,
}

impl Qwen35Main {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35MainWeightBindings,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
        residual_capture: Option<Rc<dyn MainResidualCapture>>,
        layer_scratch: Rc<Qwen35MainLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35MainWeightBindings {
            final_norm_weight,
            layers: layer_bindings,
        } = bindings;
        assert_eq!(
            layer_bindings.len(),
            config.text_config.num_hidden_layers,
            "qwen3.5 Main config and checkpoint binding layer counts must match"
        );
        let mut num_loaded_gqa_layers = 0;
        let mut num_loaded_gdn_layers = 0;
        let mut layers = Vec::with_capacity(layer_bindings.len());
        for (layer_index, bindings) in layer_bindings.into_iter().enumerate() {
            let layer_type = config.layer_type_at(layer_index)?;
            let attn_layer_index = match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers,
                LayerType::GDN => num_loaded_gdn_layers,
            };
            layers.push(Qwen35MainLayer::load(
                device,
                store,
                config,
                defaults,
                layer_index,
                attn_layer_index,
                bindings,
                gqa_state,
                gdn_state,
                Rc::clone(&layer_scratch),
                dense_scratch,
                moe_scratch,
            )?);
            match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers += 1,
                LayerType::GDN => num_loaded_gdn_layers += 1,
            }
            store.unload_all();
        }

        let final_norm_weight =
            load_qwen3x_norm_weight(device, store, &final_norm_weight, &[config.text_config.hidden_size])?;
        Ok(Self {
            layers,
            final_norm: RMSNorm::new(
                device,
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
                final_norm_weight,
            ),
            residual_capture,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = <Qwen35MainLayer as ReplayLayer>::record(
                layer,
                recorder,
                Qwen35MainLayerInput {
                    gdn: args.gdn,
                    gqa: args.gqa,
                    num_tokens,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                    residual_capture_dest: self
                        .residual_capture
                        .as_ref()
                        .and_then(|capture| capture.capture_for_model_layer(layer.layer_index())),
                },
            );
        }
        self.final_norm.record(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35MainGQAReplayKey {
    num_q_token_tiles: u32,
    total_sdpa_map_task_templates: u32,
}

impl Qwen35MainGQAReplayKey {
    fn from_shape(gqa_shape: inference_executor_core::attn::GQAReplayShape) -> Self {
        gqa_shape.validate();
        Self {
            num_q_token_tiles: gqa_shape.num_q_token_tiles,
            total_sdpa_map_task_templates: gqa_shape.total_sdpa_map_task_templates,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainReplayKey {
    num_tokens: u32,
    gqa: Qwen35MainGQAReplayKey,
    gdn: Qwen35MainGDNReplayKey,
}

impl Qwen35MainReplayKey {
    pub fn from_shapes(gqa_shape: inference_executor_core::attn::GQAReplayShape, gdn_shape: GDNReplayShape) -> Self {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, gdn_shape.num_tokens,
            "qwen3.5 main GQA and GDN replay token counts must match"
        );
        Self {
            num_tokens: gqa_shape.num_tokens,
            gqa: Qwen35MainGQAReplayKey::from_shape(gqa_shape),
            gdn: Qwen35MainGDNReplayKey::from_shape(gdn_shape),
        }
    }

    #[cfg(test)]
    pub fn debug_parts(&self) -> (u32, u32, u32, u32) {
        (
            self.num_tokens,
            self.gqa.num_q_token_tiles,
            self.gqa.total_sdpa_map_task_templates,
            self.gdn.num_reqs,
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35MainGDNReplayKey {
    num_reqs: u32,
}

impl Qwen35MainGDNReplayKey {
    fn from_shape(gdn_shape: GDNReplayShape) -> Self {
        gdn_shape.validate();
        Self {
            num_reqs: gdn_shape.num_reqs,
        }
    }
}

impl ReplayComponent for Qwen35Main {
    type Key = Qwen35MainReplayKey;
    type Input<'a> = Qwen35MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen35MainReplayKey::from_shapes(input.gqa.replay_shape(), input.gdn.replay_shape())
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35Main::record(self, recorder, *input);
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::ResidualAddCaptureTarget;
    use inference_backend_metal::metal::Dtype;

    use super::*;

    struct SelectedLayerCapture {
        model_layer_index: usize,
        capture_output: Buffer,
    }

    impl MainResidualCapture for SelectedLayerCapture {
        fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualAddCaptureTarget<'_>> {
            (model_layer_index == self.model_layer_index)
                .then(|| ResidualAddCaptureTarget::columns(&self.capture_output, 16, 4..12))
        }
    }

    fn assert_replay_component<T: ReplayComponent>() {}

    #[test]
    fn test_residual_capture_selects_only_the_configured_model_layer() {
        let device = Device::system_default();
        let capture = Rc::new(SelectedLayerCapture {
            model_layer_index: 10,
            capture_output: Buffer::new_zeroed_elements(&device, 64, Dtype::Bfloat16),
        });
        let erased_capture: Rc<dyn MainResidualCapture> = capture.clone();

        assert!(erased_capture.capture_for_model_layer(9).is_none());
        assert!(erased_capture.capture_for_model_layer(10).is_some());
        assert!(erased_capture.capture_for_model_layer(11).is_none());

        assert_replay_component::<Qwen35Main>();
    }

    #[test]
    fn test_absent_main_capture_never_captures() {
        let capture: Option<Rc<dyn MainResidualCapture>> = None;
        for model_layer_index in [0, 1, 10, usize::MAX] {
            assert!(
                capture
                    .as_ref()
                    .and_then(|capture| capture.capture_for_model_layer(model_layer_index))
                    .is_none()
            );
        }

        assert_replay_component::<Qwen35Main>();
    }
}
