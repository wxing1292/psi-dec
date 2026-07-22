use std::rc::Rc;

use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::DenseMLPReplayShape;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35DenseMLPWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::backend::DenseMLP;
use crate::mlp::dense::backend::DenseMLPReplayInput;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
use crate::model::qwen::v3_5::weight::concat_bytes;
use crate::model::qwen::v3_5::weight::quant_weight;
use crate::model::qwen::v3_5::weight::to_u32;
use crate::model::qwen::v3_5::weight::typed_tensor;
use crate::model::qwen::v3_5::weight::validate_len;

pub struct Qwen35DenseMLP {
    backend: DenseMLP,
    weights: DenseMLPWeightBuffers,
    scratch: Rc<DenseMLPScratch>,
}

impl Qwen35DenseMLP {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        layer_index: usize,
        bindings: Qwen35DenseMLPWeightBindings,
        scratch: Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let (core, metal) = qwen35_dense_mlp_core_and_metal(layer_index, &config.text_config, defaults)?;
        Ok(Self {
            backend: DenseMLP::new(device, core.clone(), metal),
            weights: DenseMLPWeightBuffers::load(device, store, &bindings, &core, metal)?,
            scratch,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: &'a Buffer, output: &'a Buffer, num_tokens: u32)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <DenseMLP as ReplayLayer>::record(
            &self.backend,
            recorder,
            DenseMLPReplayInput {
                shape: DenseMLPReplayShape { num_tokens },
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch.bindings(),
                weights: self.weights.as_borrowed(),
            },
        );
    }
}

// Public only inside the private `dense_mlp` module path so the sibling MoE
// owner can reuse the identical common-expert tensor layout.
pub struct DenseMLPWeightBuffers {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl DenseMLPWeightBuffers {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35DenseMLPWeightBindings,
        core: &DenseMLPCore,
        metal: crate::mlp::dense::backend::DenseMLPMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        Self::load_with_intermediate(device, store, bindings, core.hidden_dim, core.intermediate_dim, metal)
    }

    pub fn load_with_intermediate(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35DenseMLPWeightBindings,
        hidden_dim: usize,
        intermediate_dim: usize,
        metal: crate::mlp::dense::backend::DenseMLPMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let gate_weight = quant_weight(store, &bindings.gate.weight)?;
        let up_weight = quant_weight(store, &bindings.up.weight)?;
        let gate_scales = typed_tensor(store, &bindings.gate.scales, safetensors::Dtype::BF16)?.into_data();
        let up_scales = typed_tensor(store, &bindings.up.scales, safetensors::Dtype::BF16)?.into_data();
        let gate_biases = typed_tensor(store, &bindings.gate.biases, safetensors::Dtype::BF16)?.into_data();
        let up_biases = typed_tensor(store, &bindings.up.biases, safetensors::Dtype::BF16)?.into_data();
        let down_weight = quant_weight(store, &bindings.down.weight)?;
        let down_scales = typed_tensor(store, &bindings.down.scales, safetensors::Dtype::BF16)?.into_data();
        let down_biases = typed_tensor(store, &bindings.down.biases, safetensors::Dtype::BF16)?.into_data();

        let shape = QuantizedDenseMLPShape { num_tokens: 1 };
        let config = inference_backend_metal::components::QuantizedDenseMLPConfig {
            hidden_dim: to_u32("dense hidden_dim", hidden_dim)?,
            intermediate_dim: to_u32("dense intermediate_dim", intermediate_dim)?,
            group_size: metal.group_size,
            bits: metal.bits,
            dtype: metal.dtype,
        };
        let gate_up_shape = config.gate_up_shape(shape);
        let down_shape = config.down_shape(shape);
        let gate_up_weight = concat_bytes(&[&gate_weight, &up_weight]);
        let gate_up_scales = concat_bytes(&[&gate_scales, &up_scales]);
        let gate_up_biases = concat_bytes(&[&gate_biases, &up_biases]);
        validate_len(
            "dense gate_up weight",
            gate_up_weight.len(),
            gate_up_shape.weight_bytes(),
        )?;
        validate_len(
            "dense gate_up scales",
            gate_up_scales.len(),
            gate_up_shape.affine_param_bytes(),
        )?;
        validate_len(
            "dense gate_up biases",
            gate_up_biases.len(),
            gate_up_shape.affine_param_bytes(),
        )?;
        validate_len("dense down weight", down_weight.len(), down_shape.weight_bytes())?;
        validate_len("dense down scales", down_scales.len(), down_shape.affine_param_bytes())?;
        validate_len("dense down biases", down_biases.len(), down_shape.affine_param_bytes())?;
        Ok(Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight),
            gate_up_scales: Buffer::from_slice(device, &gate_up_scales),
            gate_up_biases: Buffer::from_slice(device, &gate_up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        })
    }

    pub fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}
