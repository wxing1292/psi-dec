use std::ops::Range;

use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMainFeatureWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3::main::Qwen3MainResidualCapture;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkMainResidualPlan;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkPlan;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::quant_weight;
use crate::model::qwen::v3_x::weight::typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::rms_norm::RmsNorm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen3xDSparkMainFeatureLayout {
    max_tokens: u32,
    num_selected_residuals: u32,
    hidden_dim: u32,
    selected_hidden_dim: u32,
}

impl Qwen3xDSparkMainFeatureLayout {
    fn new(plan: &Qwen3xDSparkPlan, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "Qwen3 DSpark Main-feature workspace requires tokens");
        assert!(
            !plan.main_residuals.is_empty(),
            "Qwen3 DSpark Main-feature workspace requires selected decoder outputs"
        );
        let max_tokens = max_tokens
            .try_into()
            .expect("Qwen3 DSpark Main-feature max_tokens must fit u32");
        let num_selected_residuals = plan
            .main_residuals
            .len()
            .try_into()
            .expect("Qwen3 DSpark selected decoder-output count must fit u32");
        let hidden_dim: u32 = plan
            .fc
            .output_dim
            .try_into()
            .expect("Qwen3 DSpark Main hidden dimension must fit u32");
        let selected_hidden_dim = hidden_dim
            .checked_mul(num_selected_residuals)
            .expect("Qwen3 DSpark selected Main width must fit u32");
        assert_eq!(
            plan.fc.input_dim, selected_hidden_dim as usize,
            "Qwen3 DSpark FC input must equal the selected Main width"
        );
        for (expected_slice, residual) in plan.main_residuals.iter().enumerate() {
            assert_eq!(
                residual.residual_slice_index, expected_slice,
                "Qwen3 DSpark Main residual slices must preserve config order"
            );
        }
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
    by_model_layer: Vec<Option<Qwen3xDSparkMainResidualPlan>>,
}

impl Qwen3xDSparkMainResidualBindings {
    fn new(plan: &Qwen3xDSparkPlan) -> Self {
        let num_main_layers = plan
            .main_residuals
            .iter()
            .map(|residual| residual.model_layer_index)
            .max()
            .and_then(|last_layer| last_layer.checked_add(1))
            .expect("Qwen3 DSpark Main residual bindings require layers");
        let mut by_model_layer = vec![None; num_main_layers];
        for &residual in &plan.main_residuals {
            let slot = &mut by_model_layer[residual.model_layer_index];
            assert!(
                slot.is_none(),
                "Qwen3 DSpark selected Main decoder layers must be unique"
            );
            *slot = Some(residual);
        }
        Self { by_model_layer }
    }

    fn get(&self, model_layer_index: usize) -> Option<Qwen3xDSparkMainResidualPlan> {
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
    hidden_norm: RmsNorm,
    weights: Qwen3xDSparkMainFeatureWeights,
    main_residuals: Buffer,
    main_feature: Buffer,
}

impl Qwen3MainResidualCapture for Qwen3xDSparkMainFeatureProjector {
    fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualCaptureTarget<'_>> {
        Qwen3xDSparkMainFeatureProjector::capture_for_model_layer(self, model_layer_index)
    }
}

impl Qwen3xDSparkMainFeatureProjector {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkPlan,
        bindings: &Qwen3xDSparkMainFeatureWeightBindings,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        let layout = Qwen3xDSparkMainFeatureLayout::new(plan, max_tokens);
        let fc_config = AffineQuantizedMatmulConfig::same_dtype(
            layout
                .hidden_dim
                .try_into()
                .expect("Qwen3 DSpark Main hidden_dim must fit i32"),
            layout
                .selected_hidden_dim
                .try_into()
                .expect("Qwen3 DSpark selected Main width must fit i32"),
            plan.fc
                .group_size
                .try_into()
                .expect("Qwen3 DSpark Main FC group_size must fit i32"),
            plan.fc.bits.try_into().expect("Qwen3 DSpark Main FC bits must fit i32"),
            Dtype::Bfloat16,
        );
        let weight = quant_weight(store, &bindings.fc.weight)?;
        let scales = typed_tensor(store, &bindings.fc.scales, safetensors::Dtype::BF16)?.into_data();
        let biases = typed_tensor(store, &bindings.fc.biases, safetensors::Dtype::BF16)?.into_data();
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
        let norm_weight = load_qwen3x_norm_weight(
            device,
            store,
            &bindings.hidden_norm_weight,
            &[layout.hidden_dim as usize],
        )?;
        Ok(Self {
            layout,
            residual_bindings: Qwen3xDSparkMainResidualBindings::new(plan),
            fc: AffineQuantizedMatmul::new(device, fc_config),
            hidden_norm: RmsNorm::new(device, layout.hidden_dim as usize, plan.hidden_norm_eps, norm_weight),
            weights: Qwen3xDSparkMainFeatureWeights {
                fc_weight: Buffer::from_slice(device, &weight),
                fc_scales: Buffer::from_slice(device, &scales),
                fc_biases: Buffer::from_slice(device, &biases),
            },
            main_residuals: Buffer::new_zeroed_elements(device, layout.main_residual_elements(), Dtype::Bfloat16),
            main_feature: Buffer::new_zeroed_elements(device, layout.main_feature_elements(), Dtype::Bfloat16),
        })
    }

    pub fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualCaptureTarget<'_>> {
        self.residual_bindings.get(model_layer_index).map(|residual| {
            ResidualCaptureTarget::columns(
                &self.main_residuals,
                self.layout.selected_hidden_dim,
                self.layout.capture_columns(residual.residual_slice_index),
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
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.fc.invoke(
                num_tokens
                    .try_into()
                    .expect("Qwen3 DSpark Main token count must fit i32"),
                &self.main_feature,
                0,
                &self.main_residuals,
                0,
                &self.weights.fc_weight,
                0,
                &self.weights.fc_scales,
                0,
                &self.weights.fc_biases,
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
