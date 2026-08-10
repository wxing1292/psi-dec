use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkLayerWeightBindings;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::state::UngatedDSparkGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::dspark::attention::Qwen3xDSparkAttention;
use crate::model::qwen::v3_x::dspark::attention::qwen3x_dspark_gqa_core_and_metal;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::resolve_uniform_quantization;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::residual_add::ResidualAdd;
use crate::model::rms_norm::RMSNorm;

pub struct Qwen3xDSparkLayer {
    dspark_layer_index: usize,
    input_norm: RMSNorm,
    attention: Qwen3xDSparkAttention,
    residual_add: ResidualAdd,
    post_attention_norm: RMSNorm,
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
    pub fn new(
        device: &Device,
        config: &Qwen3xDSparkConfig,
        num_spec_tokens: usize,
        dspark_layer_index: usize,
        page_bytes: usize,
        bindings: &Qwen3xDSparkLayerWeightBindings,
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
        let (attention_core, attention_metal) =
            qwen3x_dspark_gqa_core_and_metal(config, num_spec_tokens, dspark_layer_index, gqa, page_bytes)?;
        let (mlp_core, mlp_metal) = qwen3x_dspark_dense_mlp_core_and_metal(config, dspark_layer_index, mlp)?;
        let hidden_dim = attention_core.attention.hidden_dim;
        assert_eq!(
            hidden_dim, mlp_core.hidden_dim,
            "Qwen3 DSpark attention and MLP hidden dimensions must match"
        );
        assert_eq!(
            mlp_core.model_layer_index, dspark_layer_index,
            "Qwen3 DSpark MLP core must match the layer index"
        );
        Ok(Self {
            dspark_layer_index,
            input_norm: RMSNorm::new(device, hidden_dim, config.rms_norm_eps),
            attention: Qwen3xDSparkAttention::new(
                device,
                &attention_core,
                attention_metal,
                dspark_layer_index,
                gqa_state,
            ),
            residual_add: ResidualAdd::new(device),
            post_attention_norm: RMSNorm::new(device, hidden_dim, config.rms_norm_eps),
            mlp: Qwen3xDenseMLP::new(device, &mlp_core, mlp_metal, dense_scratch),
            scratch,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3xDSparkConfig,
        num_spec_tokens: usize,
        page_bytes: usize,
        bindings: Qwen3xDSparkLayerWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen3xDSparkLayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            gqa,
            mlp,
        } = bindings;
        let (attention_core, attention_metal) =
            qwen3x_dspark_gqa_core_and_metal(config, num_spec_tokens, self.dspark_layer_index, &gqa, page_bytes)?;
        let (mlp_core, mlp_metal) = qwen3x_dspark_dense_mlp_core_and_metal(config, self.dspark_layer_index, &mlp)?;
        self.attention
            .load_weights(device, store, &attention_core, attention_metal, gqa)?;
        self.mlp.load_weights(device, store, &mlp_core, mlp_metal, mlp)?;
        let hidden_dim = attention_core.attention.hidden_dim;
        let mut tensors = store.load_tensors([input_norm_weight.as_str(), post_attention_norm_weight.as_str()])?;
        self.input_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &input_norm_weight,
            &[hidden_dim],
        )?);
        self.post_attention_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &post_attention_norm_weight,
            &[hidden_dim],
        )?);
        assert!(
            tensors.is_empty(),
            "Qwen3x DSpark layer must consume its norm tensor map"
        );
        Ok(())
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
        self.residual_add.record(
            recorder,
            num_values,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
        );
        self.post_attention_norm.record_with_barrier(
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
        self.residual_add.record(
            recorder,
            num_values,
            &self.scratch.post_attention_hidden,
            &self.scratch.branch_output,
            input.residual_output,
        );
        input.residual_output
    }
}

fn qwen3x_dspark_dense_mlp_core_and_metal(
    config: &Qwen3xDSparkConfig,
    dspark_layer_index: usize,
    bindings: &Qwen3xDenseMLPWeightBindings,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark dense MLP requires quantization config"))?;
    let resolved = resolve_uniform_quantization(
        quantization,
        &[
            bindings.gate.weight.as_str(),
            bindings.up.weight.as_str(),
            bindings.down.weight.as_str(),
        ],
        "Qwen3x DSpark dense MLP",
    )?;
    let core = DenseMLPCore {
        model_layer_index: dspark_layer_index,
        hidden_dim: config.hidden_size,
        intermediate_dim: config.intermediate_size,
    };
    core.validate();
    let metal = DenseMLPMetalConfig {
        group_size: to_u32("Qwen3x DSpark dense MLP group_size", resolved.group_size)?,
        bits: to_u32("Qwen3x DSpark dense MLP bits", resolved.bits)?,
        io_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
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

#[cfg(test)]
mod tests {
    use inference_executor_core::model::qwen::v3_x::QuantizationConfig;
    use inference_executor_core::model::qwen::v3_x::TensorQuantizationOverride;
    use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkWeightBindings;

    use super::*;

    #[test]
    fn test_dense_mlp_resolves_each_layer_affine_layout() {
        let mut config = config();
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            config.quantization.as_mut().unwrap().tensor_overrides.insert(
                format!("layers.1.mlp.{projection}.weight"),
                TensorQuantizationOverride {
                    group_size: Some(32),
                    bits: Some(8),
                    mode: None,
                },
            );
        }
        let bindings = Qwen3xDSparkWeightBindings::from_config(&config);

        let (first_core, first_metal) =
            qwen3x_dspark_dense_mlp_core_and_metal(&config, 0, &bindings.layers[0].mlp).unwrap();
        let (second_core, second_metal) =
            qwen3x_dspark_dense_mlp_core_and_metal(&config, 1, &bindings.layers[1].mlp).unwrap();

        assert_eq!(first_core.model_layer_index, 0);
        assert_eq!(second_core.model_layer_index, 1);
        assert_eq!(first_metal.bits, 4);
        assert_eq!(second_metal.bits, 8);
    }

    #[test]
    fn test_dense_mlp_rejects_mixed_projection_affine_layouts() {
        let mut config = config();
        config.quantization.as_mut().unwrap().tensor_overrides.insert(
            "layers.0.mlp.down_proj.weight".to_string(),
            TensorQuantizationOverride {
                group_size: Some(32),
                bits: Some(8),
                mode: None,
            },
        );
        let bindings = Qwen3xDSparkWeightBindings::from_config(&config);

        let error = qwen3x_dspark_dense_mlp_core_and_metal(&config, 0, &bindings.layers[0].mlp).unwrap_err();

        assert!(error.to_string().contains("dense MLP requires one affine layout"));
    }

    fn config() -> Qwen3xDSparkConfig {
        Qwen3xDSparkConfig {
            block_size: 7,
            mask_token_id: 15,
            target_layer_ids: vec![1, 4],
            num_target_layers: 8,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            vocab_size: 64,
            markov_rank: 8,
            num_anchors: 8,
            quantization: Some(QuantizationConfig {
                group_size: 32,
                bits: 4,
                mode: None,
                tensor_overrides: Default::default(),
            }),
        }
    }
}
