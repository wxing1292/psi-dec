use std::ops::Range;

use inference_backend_metal::components::ResidualAddCaptureTarget;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMainFeatureWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::remove_typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::rms_norm::RMSNorm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen3xDSparkMainFeatureLayout {
    max_tokens: u32,
    num_selected_residuals: u32,
    hidden_dim: u32,
    selected_hidden_dim: u32,
}

impl Qwen3xDSparkMainFeatureLayout {
    fn new(config: &Qwen3xDSparkConfig, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "Qwen3 DSpark Main-feature workspace requires tokens");
        assert!(
            !config.target_layer_ids.is_empty(),
            "Qwen3 DSpark Main-feature workspace requires selected decoder outputs"
        );
        let max_tokens = max_tokens
            .try_into()
            .expect("Qwen3 DSpark Main-feature max_tokens must fit u32");
        let num_selected_residuals = config
            .target_layer_ids
            .len()
            .try_into()
            .expect("Qwen3 DSpark selected decoder-output count must fit u32");
        let hidden_dim: u32 = config
            .hidden_size
            .try_into()
            .expect("Qwen3 DSpark Main hidden dimension must fit u32");
        let selected_hidden_dim = hidden_dim
            .checked_mul(num_selected_residuals)
            .expect("Qwen3 DSpark selected Main width must fit u32");
        Self {
            max_tokens,
            num_selected_residuals,
            hidden_dim,
            selected_hidden_dim,
        }
    }

    fn capture_columns(self, residual_slice_index: usize) -> Range<u32> {
        assert!(
            residual_slice_index < self.num_selected_residuals as usize,
            "Qwen3 DSpark Main residual slice is outside the workspace"
        );
        let start: u32 = residual_slice_index
            .checked_mul(self.hidden_dim as usize)
            .and_then(|offset| offset.try_into().ok())
            .expect("Qwen3 DSpark Main residual column start must fit u32");
        let end = start
            .checked_add(self.hidden_dim)
            .expect("Qwen3 DSpark Main residual column end must fit u32");
        start..end
    }

    fn main_residual_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.selected_hidden_dim as usize)
            .expect("Qwen3 DSpark Main residual workspace must fit usize")
    }

    fn main_feature_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("Qwen3 DSpark Main-feature workspace must fit usize")
    }
}

struct Qwen3xDSparkMainResidualBindings {
    by_model_layer: Vec<Option<usize>>,
}

impl Qwen3xDSparkMainResidualBindings {
    fn new(target_layer_ids: &[usize]) -> Self {
        let num_main_layers = target_layer_ids
            .iter()
            .copied()
            .max()
            .and_then(|last_layer| last_layer.checked_add(1))
            .expect("Qwen3 DSpark Main residual bindings require layers");
        let mut by_model_layer = vec![None; num_main_layers];
        for (residual_slice_index, &model_layer_index) in target_layer_ids.iter().enumerate() {
            let slot = &mut by_model_layer[model_layer_index];
            assert!(
                slot.is_none(),
                "Qwen3 DSpark selected Main decoder layers must be unique"
            );
            *slot = Some(residual_slice_index);
        }
        Self { by_model_layer }
    }

    fn get(&self, model_layer_index: usize) -> Option<usize> {
        self.by_model_layer.get(model_layer_index).copied().flatten()
    }
}

struct Qwen3xDSparkMainFeatureWeights {
    fc_weight: Buffer,
    fc_scales: Buffer,
    fc_biases: Buffer,
}

pub struct Qwen3xDSparkMainFeatureProjector {
    layout: Qwen3xDSparkMainFeatureLayout,
    residual_bindings: Qwen3xDSparkMainResidualBindings,
    fc: AffineQuantizedMatmul,
    hidden_norm: RMSNorm,
    weights: Option<Qwen3xDSparkMainFeatureWeights>,
    main_residuals: Buffer,
    main_feature: Buffer,
}

impl MainResidualCapture for Qwen3xDSparkMainFeatureProjector {
    fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualAddCaptureTarget<'_>> {
        Qwen3xDSparkMainFeatureProjector::capture_for_model_layer(self, model_layer_index)
    }
}

impl Qwen3xDSparkMainFeatureProjector {
    pub fn new(
        device: &Device,
        config: &Qwen3xDSparkConfig,
        bindings: &Qwen3xDSparkMainFeatureWeightBindings,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        let layout = Qwen3xDSparkMainFeatureLayout::new(config, max_tokens);
        let quantization = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Main feature requires quantization config"))?
            .resolve_for_tensor(&bindings.fc.weight);
        let fc_config = AffineQuantizedMatmulConfig::same_dtype(
            layout
                .hidden_dim
                .try_into()
                .expect("Qwen3 DSpark Main hidden_dim must fit i32"),
            layout
                .selected_hidden_dim
                .try_into()
                .expect("Qwen3 DSpark selected Main width must fit i32"),
            quantization
                .group_size
                .try_into()
                .expect("Qwen3 DSpark Main FC group_size must fit i32"),
            quantization
                .bits
                .try_into()
                .expect("Qwen3 DSpark Main FC bits must fit i32"),
            Dtype::Bfloat16,
        );
        Ok(Self {
            layout,
            residual_bindings: Qwen3xDSparkMainResidualBindings::new(&config.target_layer_ids),
            fc: AffineQuantizedMatmul::new(device, fc_config),
            hidden_norm: RMSNorm::new(device, layout.hidden_dim as usize, config.rms_norm_eps),
            weights: None,
            main_residuals: Buffer::new_zeroed_elements(device, layout.main_residual_elements(), Dtype::Bfloat16),
            main_feature: Buffer::new_zeroed_elements(device, layout.main_feature_elements(), Dtype::Bfloat16),
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3xDSparkConfig,
        bindings: &Qwen3xDSparkMainFeatureWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3.x DSpark Main-feature weights are already loaded"
        );
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let quantization = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Main feature requires quantization config"))?
            .resolve_for_tensor(&bindings.fc.weight);
        let fc_config = AffineQuantizedMatmulConfig::same_dtype(
            self.layout
                .hidden_dim
                .try_into()
                .expect("Qwen3 DSpark Main hidden_dim must fit i32"),
            self.layout
                .selected_hidden_dim
                .try_into()
                .expect("Qwen3 DSpark selected Main width must fit i32"),
            quantization
                .group_size
                .try_into()
                .expect("Qwen3 DSpark Main FC group_size must fit i32"),
            quantization
                .bits
                .try_into()
                .expect("Qwen3 DSpark Main FC bits must fit i32"),
            Dtype::Bfloat16,
        );
        let weight = remove_quant_weight(&mut tensors, &bindings.fc.weight)?;
        let scales = remove_typed_tensor(&mut tensors, &bindings.fc.scales, safetensors::Dtype::BF16)?.into_data();
        let biases = remove_typed_tensor(&mut tensors, &bindings.fc.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("Qwen3 DSpark Main FC weight", weight.len(), fc_config.weight_bytes())?;
        validate_len(
            "Qwen3 DSpark Main FC scales",
            scales.len(),
            fc_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3 DSpark Main FC biases",
            biases.len(),
            fc_config.scale_or_bias_bytes(),
        )?;
        self.hidden_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &bindings.hidden_norm_weight,
            &[self.layout.hidden_dim as usize],
        )?);
        self.weights = Some(Qwen3xDSparkMainFeatureWeights {
            fc_weight: Buffer::from_slice(device, &weight),
            fc_scales: Buffer::from_slice(device, &scales),
            fc_biases: Buffer::from_slice(device, &biases),
        });
        assert!(
            tensors.is_empty(),
            "Qwen3x DSpark Main-feature projector must consume its tensor map"
        );
        Ok(())
    }

    pub fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualAddCaptureTarget<'_>> {
        self.residual_bindings
            .get(model_layer_index)
            .map(|residual_slice_index| {
                ResidualAddCaptureTarget::columns(
                    &self.main_residuals,
                    self.layout.selected_hidden_dim,
                    self.layout.capture_columns(residual_slice_index),
                )
            })
    }

    pub fn main_feature(&self) -> &Buffer {
        &self.main_feature
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, num_tokens: u32) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(num_tokens > 0, "Qwen3 DSpark Main projection requires tokens");
        assert!(
            num_tokens <= self.layout.max_tokens,
            "Qwen3 DSpark Main projection exceeds capacity"
        );
        let weights = self
            .weights
            .as_ref()
            .expect("Qwen3.x DSpark Main-feature weights must be loaded before execution");
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.fc.invoke(
                num_tokens
                    .try_into()
                    .expect("Qwen3 DSpark Main token count must fit i32"),
                &self.main_feature,
                0,
                &self.main_residuals,
                0,
                &weights.fc_weight,
                0,
                &weights.fc_scales,
                0,
                &weights.fc_biases,
                0,
            ),
        ));
        self.hidden_norm
            .record_with_barrier(recorder, num_tokens, &self.main_feature, &self.main_feature);
        &self.main_feature
    }
}

#[cfg(test)]
#[path = "main_feature_test.rs"]
mod tests;
