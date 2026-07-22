use std::rc::Rc;

use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::mlp::moe::GatedMoEReplayShape;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MoEWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35SparseExpertWeightBindings;

use super::dense_mlp::DenseMLPWeightBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::moe::backend::GatedMoE;
use crate::mlp::moe::backend::GatedMoECommonExpertReplayInput;
use crate::mlp::moe::backend::GatedMoECommonExpertWeights;
use crate::mlp::moe::backend::GatedMoEReplayInput;
use crate::mlp::moe::backend::GatedMoEWeights;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_moe_core_and_metal;
use crate::model::qwen::v3_5::weight::affine_shape;
use crate::model::qwen::v3_5::weight::quant_weight;
use crate::model::qwen::v3_5::weight::sparse_affine_layout;
use crate::model::qwen::v3_5::weight::typed_tensor;
use crate::model::qwen::v3_5::weight::validate_len;

pub struct Qwen35MoE {
    backend: GatedMoE,
    weights: Box<Qwen35MoEWeights>,
    scratch: Rc<MoEScratch>,
}

impl Qwen35MoE {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        layer_index: usize,
        bindings: Qwen35MoEWeightBindings,
        scratch: Rc<MoEScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let layer_prefix = format!("layers.{layer_index}");
        let (core, metal) = qwen35_moe_core_and_metal(&layer_prefix, layer_index, config, defaults)?;
        Ok(Self {
            backend: GatedMoE::new(device, core.clone(), metal),
            weights: Box::new(Qwen35MoEWeights::load(device, store, &bindings, &core, metal)?),
            scratch,
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: &'a Buffer, output: &'a Buffer, num_tokens: u32)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let common_expert = self.weights.as_common_expert_weights().map(|weights| {
            GatedMoECommonExpertReplayInput {
                scratch: self
                    .scratch
                    .common_expert_bindings()
                    .expect("qwen3.5 common-expert weights require common-expert scratch"),
                weights,
            }
        });
        if common_expert.is_none() {
            assert!(
                !self.backend.input_shape().has_common_expert(),
                "qwen3.5 MoE layer expected common-expert weights"
            );
        }
        let _ = <GatedMoE as ReplayLayer>::record(
            &self.backend,
            recorder,
            GatedMoEReplayInput {
                shape: GatedMoEReplayShape { num_tokens },
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch.bindings(),
                weights: self.weights.as_moe_weights(),
                common_expert,
            },
        );
    }
}

struct Qwen35MoEWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    experts: Qwen35SparseExpertWeights,
    common_gate_weight: Option<Buffer>,
    common_gate_scales: Option<Buffer>,
    common_gate_biases: Option<Buffer>,
    common_expert: Option<DenseMLPWeightBuffers>,
}

struct Qwen35SparseExpertWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl Qwen35MoEWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35MoEWeightBindings,
        core: &GatedMoECore,
        metal: crate::mlp::moe::backend::GatedMoEMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let router_shape = affine_shape(
            core.num_experts,
            core.hidden_dim,
            metal.group_size,
            metal.router_bits,
            metal.dtype,
            metal.dtype,
            metal.dtype,
        );
        let router_weight = quant_weight(store, &bindings.router.weight)?;
        let router_scales = typed_tensor(store, &bindings.router.scales, safetensors::Dtype::BF16)?.into_data();
        let router_biases = typed_tensor(store, &bindings.router.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("sparse router weight", router_weight.len(), router_shape.weight_bytes())?;
        validate_len(
            "sparse router scales",
            router_scales.len(),
            router_shape.affine_param_bytes(),
        )?;
        validate_len(
            "sparse router biases",
            router_biases.len(),
            router_shape.affine_param_bytes(),
        )?;

        let experts = Qwen35SparseExpertWeights::load(device, store, &bindings.experts, core, metal)?;
        let (common_gate_weight, common_gate_scales, common_gate_biases, common_expert) =
            if let Some(common_expert_intermediate_dim) = core.common_expert_intermediate_dim {
                let gate_bindings = bindings
                    .shared_expert_gate
                    .as_ref()
                    .expect("qwen3.5 common expert geometry requires shared expert gate bindings");
                let expert_bindings = bindings
                    .shared_expert
                    .as_ref()
                    .expect("qwen3.5 common expert geometry requires shared expert bindings");
                let gate_shape = affine_shape(
                    1,
                    core.hidden_dim,
                    metal.group_size,
                    metal.common_gate_bits,
                    metal.dtype,
                    metal.dtype,
                    metal.dtype,
                );
                let gate_weight = quant_weight(store, &gate_bindings.weight)?;
                let gate_scales = typed_tensor(store, &gate_bindings.scales, safetensors::Dtype::BF16)?.into_data();
                let gate_biases = typed_tensor(store, &gate_bindings.biases, safetensors::Dtype::BF16)?.into_data();
                validate_len(
                    "sparse shared gate weight",
                    gate_weight.len(),
                    gate_shape.weight_bytes(),
                )?;
                validate_len(
                    "sparse shared gate scales",
                    gate_scales.len(),
                    gate_shape.affine_param_bytes(),
                )?;
                validate_len(
                    "sparse shared gate biases",
                    gate_biases.len(),
                    gate_shape.affine_param_bytes(),
                )?;
                let common_metal = crate::mlp::dense::backend::DenseMLPMetalConfig {
                    group_size: metal.group_size,
                    bits: metal.bits,
                    dtype: metal.dtype,
                };
                (
                    Some(Buffer::from_slice(device, &gate_weight)),
                    Some(Buffer::from_slice(device, &gate_scales)),
                    Some(Buffer::from_slice(device, &gate_biases)),
                    Some(DenseMLPWeightBuffers::load_with_intermediate(
                        device,
                        store,
                        expert_bindings,
                        core.hidden_dim,
                        common_expert_intermediate_dim,
                        common_metal,
                    )?),
                )
            } else {
                assert!(
                    bindings.shared_expert_gate.is_none() && bindings.shared_expert.is_none(),
                    "qwen3.5 MoE without common expert geometry must not bind common expert tensors"
                );
                (None, None, None, None)
            };
        Ok(Self {
            router_weight: Buffer::from_slice(device, &router_weight),
            router_scales: Buffer::from_slice(device, &router_scales),
            router_biases: Buffer::from_slice(device, &router_biases),
            experts,
            common_gate_weight,
            common_gate_scales,
            common_gate_biases,
            common_expert,
        })
    }

    fn as_moe_weights(&self) -> GatedMoEWeights<'_> {
        GatedMoEWeights {
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            topk_experts: self.experts.as_borrowed(),
        }
    }

    fn as_common_expert_weights(&self) -> Option<GatedMoECommonExpertWeights<'_>> {
        Some(GatedMoECommonExpertWeights {
            common_gate_weight: self.common_gate_weight.as_ref()?,
            common_gate_scales: self.common_gate_scales.as_ref()?,
            common_gate_biases: self.common_gate_biases.as_ref()?,
            common_expert: self.common_expert.as_ref()?.as_borrowed(),
        })
    }
}

impl Qwen35SparseExpertWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35SparseExpertWeightBindings,
        core: &GatedMoECore,
        metal: crate::mlp::moe::backend::GatedMoEMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let expert_gate_layout = sparse_affine_layout(core.num_experts, core.intermediate_dim, core.hidden_dim, metal);
        let expert_down_layout = sparse_affine_layout(core.num_experts, core.hidden_dim, core.intermediate_dim, metal);
        let gate_weight = quant_weight(store, &bindings.gate.weight)?;
        let gate_scales = typed_tensor(store, &bindings.gate.scales, safetensors::Dtype::BF16)?.into_data();
        let gate_biases = typed_tensor(store, &bindings.gate.biases, safetensors::Dtype::BF16)?.into_data();
        let up_weight = quant_weight(store, &bindings.up.weight)?;
        let up_scales = typed_tensor(store, &bindings.up.scales, safetensors::Dtype::BF16)?.into_data();
        let up_biases = typed_tensor(store, &bindings.up.biases, safetensors::Dtype::BF16)?.into_data();
        let down_weight = quant_weight(store, &bindings.down.weight)?;
        let down_scales = typed_tensor(store, &bindings.down.scales, safetensors::Dtype::BF16)?.into_data();
        let down_biases = typed_tensor(store, &bindings.down.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "sparse expert gate weight",
            gate_weight.len(),
            expert_gate_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert gate scales",
            gate_scales.len(),
            expert_gate_layout.affine_param_bytes(),
        )?;
        validate_len(
            "sparse expert gate biases",
            gate_biases.len(),
            expert_gate_layout.affine_param_bytes(),
        )?;
        validate_len(
            "sparse expert up weight",
            up_weight.len(),
            expert_gate_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert up scales",
            up_scales.len(),
            expert_gate_layout.affine_param_bytes(),
        )?;
        validate_len(
            "sparse expert up biases",
            up_biases.len(),
            expert_gate_layout.affine_param_bytes(),
        )?;
        validate_len(
            "sparse expert down weight",
            down_weight.len(),
            expert_down_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert down scales",
            down_scales.len(),
            expert_down_layout.affine_param_bytes(),
        )?;
        validate_len(
            "sparse expert down biases",
            down_biases.len(),
            expert_down_layout.affine_param_bytes(),
        )?;
        Ok(Self {
            gate_weight: Buffer::from_slice(device, &gate_weight),
            gate_scales: Buffer::from_slice(device, &gate_scales),
            gate_biases: Buffer::from_slice(device, &gate_biases),
            up_weight: Buffer::from_slice(device, &up_weight),
            up_scales: Buffer::from_slice(device, &up_scales),
            up_biases: Buffer::from_slice(device, &up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        })
    }

    fn as_borrowed(&self) -> QuantizedSparseMLPWeights<'_> {
        QuantizedSparseMLPWeights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}
