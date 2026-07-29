use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkLayerWeightBindings;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::dspark::attention::Qwen3xDSparkAttention;
use crate::model::qwen::v3_x::dspark::plan::Qwen3xDSparkLayerPlan;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;

pub struct Qwen3xDSparkLayer {
    dspark_layer_index: usize,
    input_norm: RmsNorm,
    attention: Qwen3xDSparkAttention,
    residual: Residual,
    post_attention_norm: RmsNorm,
    mlp: Qwen3xDenseMLP,
    scratch: Rc<Qwen3xDSparkLayerScratch>,
}

pub struct Qwen3xDSparkLayerScratch {
    hidden_dim: usize,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDSparkLayerInput<'a> {
    pub num_tokens: u32,
    pub metadata: &'a DSparkGQAMetadataBuffers,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
}

impl Qwen3xDSparkLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen3xDSparkLayerPlan,
        bindings: Qwen3xDSparkLayerWeightBindings,
        gqa_state: &UngatedDSparkGQAState,
        scratch: Rc<Qwen3xDSparkLayerScratch>,
        dense_scratch: Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen3xDSparkLayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            gqa,
            mlp,
        } = bindings;
        let hidden_dim = plan.attention_core.attention.hidden_dim;
        assert_eq!(
            hidden_dim, plan.mlp_core.hidden_dim,
            "Qwen3 DSpark attention and MLP hidden dimensions must match"
        );
        Ok(Self {
            dspark_layer_index: plan.dspark_layer_index,
            input_norm: RmsNorm::new(
                device,
                hidden_dim,
                plan.input_norm_eps,
                load_qwen3x_norm_weight(device, store, &input_norm_weight, &[hidden_dim])?,
            ),
            attention: Qwen3xDSparkAttention::load(device, store, plan, gqa, gqa_state)?,
            residual: Residual::new(device),
            post_attention_norm: RmsNorm::new(
                device,
                hidden_dim,
                plan.post_attention_norm_eps,
                load_qwen3x_norm_weight(device, store, &post_attention_norm_weight, &[hidden_dim])?,
            ),
            mlp: Qwen3xDenseMLP::load(device, store, &plan.mlp_core, plan.mlp_metal, mlp, dense_scratch)?,
            scratch,
        })
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.dspark_layer_index)
    }

    pub fn record_context<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        main_feature: &'a Buffer,
        req_slots: &'a Buffer,
        flat_token_indices: &'a Buffer,
        pages: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.attention
            .record_context(recorder, num_tokens, main_feature, req_slots, flat_token_indices, pages);
    }

    pub fn record_block<'a, R>(&'a self, recorder: &mut R, input: Qwen3xDSparkLayerInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_values = residual_values(input.num_tokens, self.scratch.hidden_dim);
        self.input_norm.record_with_barrier(
            recorder,
            input.num_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention.record_block(
            recorder,
            input.metadata,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.pages,
        );
        self.residual.record(
            recorder,
            num_values,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
            None,
        );
        self.post_attention_norm.record(
            recorder,
            input.num_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.num_tokens,
        );
        self.residual.record(
            recorder,
            num_values,
            &self.scratch.post_attention_hidden,
            &self.scratch.branch_output,
            input.residual_output,
            None,
        );
        input.residual_output
    }
}

impl Qwen3xDSparkLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0, "Qwen3 DSpark layer scratch requires tokens");
        assert!(hidden_dim > 0, "Qwen3 DSpark layer scratch requires hidden values");
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("Qwen3 DSpark layer scratch element count must fit usize");
        Self {
            hidden_dim,
            residual_stream: [
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            ],
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }

    fn residual_stream(&self, dspark_layer_index: usize) -> &Buffer {
        &self.residual_stream[dspark_layer_index % self.residual_stream.len()]
    }
}

fn residual_values(num_tokens: u32, hidden_dim: usize) -> u32 {
    num_tokens
        .checked_mul(
            hidden_dim
                .try_into()
                .expect("Qwen3 DSpark hidden dimension must fit u32"),
        )
        .expect("Qwen3 DSpark residual element index must fit u32")
}
