use inference_backend_metal::components::DuplicateResidualOutput;
use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkTargetWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::dspark::weights::Qwen35DSparkTargetWeights;
use crate::model::qwen::v3_5::plan::Qwen35DSparkPlan;
use crate::model::qwen::v3_5::plan::Qwen35DSparkTargetResidualPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35DSparkTargetLayout {
    max_tokens: u32,
    num_selected_residuals: u32,
    hidden_dim: u32,
    selected_hidden_dim: u32,
}

impl Qwen35DSparkTargetLayout {
    fn new(plan: &Qwen35DSparkPlan, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "DSpark target workspace requires token capacity");
        assert!(
            !plan.target_residuals.is_empty(),
            "DSpark target workspace requires selected residuals"
        );
        let max_tokens = max_tokens.try_into().expect("DSpark target max_tokens must fit u32");
        let num_selected_residuals = plan
            .target_residuals
            .len()
            .try_into()
            .expect("DSpark selected residual count must fit u32");
        let hidden_dim: u32 = plan
            .fc
            .output_dim
            .try_into()
            .expect("DSpark target hidden dimension must fit u32");
        let selected_hidden_dim = hidden_dim
            .checked_mul(num_selected_residuals)
            .expect("DSpark selected hidden dimension must fit u32");
        assert_eq!(
            plan.fc.input_dim, selected_hidden_dim as usize,
            "DSpark target FC input must equal selected residual width"
        );
        for (expected_slice, residual) in plan.target_residuals.iter().enumerate() {
            assert_eq!(
                residual.residual_slice_index, expected_slice,
                "DSpark target residual slices must be dense and ordered"
            );
        }
        Self {
            max_tokens,
            num_selected_residuals,
            hidden_dim,
            selected_hidden_dim,
        }
    }

    fn target_residual_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.selected_hidden_dim as usize)
            .expect("DSpark target residual workspace size must fit usize")
    }

    fn projected_target_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("DSpark projected target workspace size must fit usize")
    }

    fn duplicate_column_offset(self, residual_slice_index: usize) -> u32 {
        assert!(
            residual_slice_index < self.num_selected_residuals as usize,
            "DSpark target residual slice is outside the workspace"
        );
        residual_slice_index
            .checked_mul(self.hidden_dim as usize)
            .and_then(|offset| offset.try_into().ok())
            .expect("DSpark target residual column offset must fit u32")
    }

    fn duplicate_output<'a>(
        self,
        target_residuals: &'a Buffer,
        residual: Qwen35DSparkTargetResidualPlan,
    ) -> DuplicateResidualOutput<'a> {
        DuplicateResidualOutput {
            buffer: target_residuals,
            row_stride: self.selected_hidden_dim,
            column_offset: self.duplicate_column_offset(residual.residual_slice_index),
        }
    }
}

struct Qwen35DSparkTargetWorkspace {
    target_residuals: Buffer,
    projected_target: Buffer,
}

struct Qwen35DSparkTargetResidualBindings {
    by_model_layer: Vec<Option<Qwen35DSparkTargetResidualPlan>>,
}

impl Qwen35DSparkTargetResidualBindings {
    fn new(plan: &Qwen35DSparkPlan) -> Self {
        let num_target_layers = plan
            .target_residuals
            .iter()
            .map(|residual| residual.model_layer_index)
            .max()
            .and_then(|last_layer| last_layer.checked_add(1))
            .expect("DSpark target residual bindings require selected target layers");
        let mut by_model_layer = vec![None; num_target_layers];
        for &residual in &plan.target_residuals {
            let slot = &mut by_model_layer[residual.model_layer_index];
            assert!(slot.is_none(), "DSpark target residual model layers must be unique");
            *slot = Some(residual);
        }
        Self { by_model_layer }
    }

    fn get(&self, model_layer_index: usize) -> Option<Qwen35DSparkTargetResidualPlan> {
        self.by_model_layer.get(model_layer_index).copied().flatten()
    }
}

impl Qwen35DSparkTargetWorkspace {
    fn new(device: &Device, layout: Qwen35DSparkTargetLayout) -> Self {
        Self {
            target_residuals: Buffer::new_zeroed_elements(device, layout.target_residual_elements(), Dtype::Bfloat16),
            projected_target: Buffer::new_zeroed_elements(device, layout.projected_target_elements(), Dtype::Bfloat16),
        }
    }
}

pub struct Qwen35DSparkTargetProjector {
    layout: Qwen35DSparkTargetLayout,
    residual_bindings: Qwen35DSparkTargetResidualBindings,
    fc_shape: AffineQuantizedMatmulShape,
    hidden_norm_eps: f32,
    single_token_fc: AffineQuantizedMatmulKernel,
    multi_token_fc: AffineQuantizedMatmulKernel,
    hidden_norm: RMSNormKernel,
    weights: Qwen35DSparkTargetWeights,
    workspace: Qwen35DSparkTargetWorkspace,
}

impl Qwen35DSparkTargetProjector {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkPlan,
        weight_bindings: &Qwen35DSparkTargetWeightBindings,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        let layout = Qwen35DSparkTargetLayout::new(plan, max_tokens);
        assert!(plan.hidden_norm_eps.is_finite() && plan.hidden_norm_eps > 0.0);
        let fc_shape = AffineQuantizedMatmulShape::same_dtype(
            layout
                .max_tokens
                .try_into()
                .expect("DSpark target max_tokens must fit i32"),
            layout
                .hidden_dim
                .try_into()
                .expect("DSpark target hidden_dim must fit i32"),
            layout
                .selected_hidden_dim
                .try_into()
                .expect("DSpark selected hidden_dim must fit i32"),
            plan.fc
                .group_size
                .try_into()
                .expect("DSpark target FC group_size must fit i32"),
            plan.fc.bits.try_into().expect("DSpark target FC bits must fit i32"),
            Dtype::Bfloat16,
        );
        fc_shape.validate();
        let single_token_shape = AffineQuantizedMatmulShape { m: 1, ..fc_shape };
        let weights = Qwen35DSparkTargetWeights::load(device, store, plan, weight_bindings)?;
        Ok(Self {
            layout,
            residual_bindings: Qwen35DSparkTargetResidualBindings::new(plan),
            fc_shape,
            hidden_norm_eps: plan.hidden_norm_eps,
            single_token_fc: AffineQuantizedMatmulKernel::new(device, single_token_shape),
            multi_token_fc: AffineQuantizedMatmulKernel::new(device, fc_shape),
            hidden_norm: RMSNormKernel::new(device),
            weights,
            workspace: Qwen35DSparkTargetWorkspace::new(device, layout),
        })
    }

    pub fn duplicate_residual_output(&self, residual: Qwen35DSparkTargetResidualPlan) -> DuplicateResidualOutput<'_> {
        self.layout.duplicate_output(&self.workspace.target_residuals, residual)
    }

    pub fn duplicate_residual_output_for_model_layer(
        &self,
        model_layer_index: usize,
    ) -> Option<DuplicateResidualOutput<'_>> {
        self.residual_bindings
            .get(model_layer_index)
            .map(|residual| self.duplicate_residual_output(residual))
    }

    pub fn projected_target(&self) -> &Buffer {
        &self.workspace.projected_target
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, num_tokens: u32) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(num_tokens > 0, "DSpark target projection requires tokens");
        assert!(
            num_tokens <= self.layout.max_tokens,
            "DSpark target projection num_tokens={num_tokens} exceed max_tokens={}",
            self.layout.max_tokens
        );
        let active_fc_shape = AffineQuantizedMatmulShape {
            m: num_tokens.try_into().expect("DSpark target token count must fit i32"),
            ..self.fc_shape
        };
        let fc = if num_tokens == 1 {
            &self.single_token_fc
        } else {
            &self.multi_token_fc
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(fc.invoke_with_shape(
            active_fc_shape,
            &self.workspace.projected_target,
            0,
            &self.workspace.target_residuals,
            0,
            &self.weights.fc_weight,
            0,
            &self.weights.fc_scales,
            0,
            &self.weights.fc_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.hidden_norm.invoke(
            RMSNormShape::bf16(num_tokens, self.layout.hidden_dim),
            RMSNormBuffers {
                input: &self.workspace.projected_target,
                weight: &self.weights.hidden_norm_weight,
                output: &self.workspace.projected_target,
            },
            self.hidden_norm_eps,
        )));
        &self.workspace.projected_target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan;
    use crate::model::qwen::v3_5::plan::Qwen35QuantizedEmbeddingPlan;
    use crate::model::qwen::v3_5::plan::Qwen35QuantizedLinearPlan;

    #[test]
    fn target_layout_matches_token_major_selected_residual_slices() {
        let plan = test_plan();
        let layout = Qwen35DSparkTargetLayout::new(&plan, 128);
        assert_eq!(layout.max_tokens, 128);
        assert_eq!(layout.num_selected_residuals, 5);
        assert_eq!(layout.hidden_dim, 5120);
        assert_eq!(layout.selected_hidden_dim, 25_600);
        assert_eq!(layout.target_residual_elements(), 3_276_800);
        assert_eq!(layout.projected_target_elements(), 655_360);
        assert_eq!(layout.duplicate_column_offset(0), 0);
        assert_eq!(layout.duplicate_column_offset(1), 5120);
        assert_eq!(layout.duplicate_column_offset(4), 20_480);

        let bindings = Qwen35DSparkTargetResidualBindings::new(&plan);
        assert_eq!(bindings.get(0), None);
        assert_eq!(bindings.get(1), Some(plan.target_residuals[0]));
        assert_eq!(bindings.get(16), Some(plan.target_residuals[1]));
        assert_eq!(bindings.get(61), Some(plan.target_residuals[4]));
        assert_eq!(bindings.get(62), None);
    }

    fn test_plan() -> Qwen35DSparkPlan {
        Qwen35DSparkPlan {
            block_size: 8,
            mask_token_id: 1,
            target_residuals: [1, 16, 31, 46, 61]
                .into_iter()
                .enumerate()
                .map(|(residual_slice_index, model_layer_index)| {
                    Qwen35DSparkTargetResidualPlan {
                        model_layer_index,
                        residual_slice_index,
                    }
                })
                .collect(),
            fc: Qwen35QuantizedLinearPlan {
                input_dim: 25_600,
                output_dim: 5120,
                group_size: 64,
                bits: 4,
            },
            hidden_norm_eps: 1e-6,
            layers: Vec::<Qwen35DSparkLayerPlan>::new(),
            norm_eps: 1e-6,
            markov_w1: Qwen35QuantizedEmbeddingPlan {
                num_embeddings: 248_320,
                embedding_dim: 1024,
                group_size: 64,
                bits: 4,
            },
            markov_w2: Qwen35QuantizedLinearPlan {
                input_dim: 1024,
                output_dim: 248_320,
                group_size: 64,
                bits: 8,
            },
        }
    }
}
