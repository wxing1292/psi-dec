use std::rc::Rc;

use inference_backend_metal::components::ResidualCaptureTarget;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3LayerWeightBindings;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQA;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::plan::qwen3_dense_mlp_core_and_metal;
use crate::model::qwen::v3::main::plan::qwen3_gqa_core_and_metal;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;

pub struct Qwen3MainLayer {
    layer_index: usize,
    input_norm: RmsNorm,
    attention: Qwen3MainGQA,
    residual: Residual,
    post_attention_norm: RmsNorm,
    mlp: Qwen3xDenseMLP,
    scratch: Rc<Qwen3MainLayerScratch>,
}

pub struct Qwen3MainLayerScratch {
    hidden_dim: usize,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainLayerInput<'a> {
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
    pub residual_capture_dest: Option<ResidualCaptureTarget<'a>>,
}

impl Qwen3MainLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3ModelConfig,
        layer_index: usize,
        bindings: Qwen3LayerWeightBindings,
        gqa_state: &Qwen3MainGQAState,
        scratch: Rc<Qwen3MainLayerScratch>,
        dense_scratch: Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen3LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            gqa: attention_bindings,
            mlp: mlp_bindings,
        } = bindings;
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        let (gqa_core, gqa_metal) = qwen3_gqa_core_and_metal(layer_index, config)?;
        let attention = Qwen3MainGQA::load(device, store, &gqa_core, gqa_metal, attention_bindings, gqa_state)?;
        let (mlp_core, mlp_metal) = qwen3_dense_mlp_core_and_metal(layer_index, config)?;
        let mlp = Qwen3xDenseMLP::load(device, store, &mlp_core, mlp_metal, mlp_bindings, dense_scratch)?;
        Ok(Self {
            layer_index,
            input_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(device, store, &input_norm_weight, &[hidden_dim])?,
            ),
            attention,
            residual: Residual::new(device),
            post_attention_norm: RmsNorm::new(
                device,
                hidden_dim,
                eps,
                load_qwen3x_norm_weight(device, store, &post_attention_norm_weight, &[hidden_dim])?,
            ),
            mlp,
            scratch,
        })
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.layer_index)
    }
}

impl ReplayLayer for Qwen3MainLayer {
    type Input<'a> = Qwen3MainLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_values = residual_values(input.num_tokens, self.scratch.hidden_dim());
        self.input_norm.record_with_barrier(
            recorder,
            input.num_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.pages,
            input.gqa,
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
            input.residual_capture_dest,
        );
        input.residual_output
    }
}

impl Qwen3MainLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3 layer scratch element count must fit usize");
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

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn residual_stream(&self, model_layer_index: usize) -> &Buffer {
        &self.residual_stream[model_layer_index % self.residual_stream.len()]
    }
}

fn residual_values(num_tokens: u32, hidden_dim: usize) -> u32 {
    num_tokens
        .checked_mul(hidden_dim.try_into().expect("qwen3 hidden dimension must fit u32"))
        .expect("qwen3 residual element index must fit u32")
}
