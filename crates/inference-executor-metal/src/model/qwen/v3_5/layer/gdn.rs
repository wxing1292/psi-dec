use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35GDNWeightBindings;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::backend::GDNInput;
use crate::attn::gdn::backend::GDNLayerStateBindings;
use crate::attn::gdn::backend::GDNWeights;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::plan::qwen35_gdn_core_and_metal;
use crate::model::qwen::v3_5::weight::affine_shape;
use crate::model::qwen::v3_5::weight::bf16_tensor_as_f32;
use crate::model::qwen::v3_5::weight::concat_bytes;
use crate::model::qwen::v3_5::weight::concat_f32;
use crate::model::qwen::v3_5::weight::quant_weight;
use crate::model::qwen::v3_5::weight::typed_tensor;
use crate::model::qwen::v3_5::weight::validate_len;

pub struct Qwen35GDN {
    layer_index: usize,
    weights: Qwen35GDNWeights,
    backend: Rc<GDN>,
    scratch: Rc<GDNScratch>,
    request_state_table: Rc<GDNRequestStateTable>,
}

impl Qwen35GDN {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        layer_index: usize,
        bindings: Qwen35GDNWeightBindings,
        backend: Rc<GDN>,
        scratch: Rc<GDNScratch>,
        request_state_table: Rc<GDNRequestStateTable>,
    ) -> Result<Self, ModelExecutorError> {
        let (core, metal) = qwen35_gdn_core_and_metal(model_layer_index, &config.text_config, defaults)?;
        Ok(Self {
            layer_index,
            weights: Qwen35GDNWeights::load(device, store, &bindings, &core, metal)?,
            backend,
            scratch,
            request_state_table,
        })
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        metadata: &'a GDNMetadataBuffers,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let state = self.request_state_table.layer_bindings(self.layer_index);
        let _ = <GDN as ReplayLayer>::record(
            &self.backend,
            recorder,
            GDNInput {
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch.bindings(),
                batch_metadata: metadata,
                state: GDNLayerStateBindings {
                    conv_state: state.conv_states,
                    conv_state_offset_bytes: state.conv_layer_offset_bytes,
                    next_conv_state: state.conv_states,
                    next_conv_state_offset_bytes: state.conv_layer_offset_bytes,
                    recurrent_state_arena: state.recurrent_states,
                    recurrent_state_arena_offset_bytes: state.recurrent_layer_offset_bytes,
                },
                materialize_candidate_states: true,
                weights: self.weights.as_borrowed(),
            },
        );
    }
}

struct Qwen35GDNWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log_decay: Buffer,
    dt_bias: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen35GDNWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35GDNWeightBindings,
        core: &GDNCore,
        metal: crate::attn::gdn::backend::GDNMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let qkv_weight = quant_weight(store, &bindings.qkv.weight)?;
        let a_weight = quant_weight(store, &bindings.a.weight)?;
        let b_weight = quant_weight(store, &bindings.b.weight)?;
        let z_weight = quant_weight(store, &bindings.z.weight)?;
        let qkv_scales = bf16_tensor_as_f32(store, &bindings.qkv.scales)?;
        let a_scales = bf16_tensor_as_f32(store, &bindings.a.scales)?;
        let b_scales = bf16_tensor_as_f32(store, &bindings.b.scales)?;
        let z_scales = bf16_tensor_as_f32(store, &bindings.z.scales)?;
        let qkv_biases = bf16_tensor_as_f32(store, &bindings.qkv.biases)?;
        let a_biases = bf16_tensor_as_f32(store, &bindings.a.biases)?;
        let b_biases = bf16_tensor_as_f32(store, &bindings.b.biases)?;
        let z_biases = bf16_tensor_as_f32(store, &bindings.z.biases)?;
        let qkvabz_weight = concat_bytes(&[&qkv_weight, &a_weight, &b_weight, &z_weight]);
        let qkvabz_scales = concat_f32(&[&qkv_scales, &a_scales, &b_scales, &z_scales]);
        let qkvabz_biases = concat_f32(&[&qkv_biases, &a_biases, &b_biases, &z_biases]);

        let qkvabz_shape = affine_shape(
            core.qkvabz_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.input_dtype,
            metal.internal_dtype(),
            metal.qkvabz_affine_dtype,
        );
        validate_len("GDN qkvabz weight", qkvabz_weight.len(), qkvabz_shape.weight_bytes())?;
        validate_len(
            "GDN qkvabz scales",
            qkvabz_scales.len() * Dtype::Float32.item_size(),
            qkvabz_shape.affine_param_bytes(),
        )?;
        validate_len(
            "GDN qkvabz biases",
            qkvabz_biases.len() * Dtype::Float32.item_size(),
            qkvabz_shape.affine_param_bytes(),
        )?;

        let output_shape = affine_shape(
            core.hidden_dim,
            core.v_dim(),
            metal.group_size,
            metal.bits,
            metal.internal_dtype(),
            metal.boundary_dtype(),
            metal.output_affine_dtype,
        );
        let output_weight = quant_weight(store, &bindings.output.weight)?;
        let output_scales = typed_tensor(store, &bindings.output.scales, safetensors::Dtype::BF16)?.into_data();
        let output_biases = typed_tensor(store, &bindings.output.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("GDN output weight", output_weight.len(), output_shape.weight_bytes())?;
        validate_len(
            "GDN output scales",
            output_scales.len(),
            output_shape.affine_param_bytes(),
        )?;
        validate_len(
            "GDN output biases",
            output_biases.len(),
            output_shape.affine_param_bytes(),
        )?;

        let conv_weight = bf16_tensor_as_f32(store, &bindings.conv_weight)?;
        validate_len(
            "GDN conv weight",
            conv_weight.len() * Dtype::Float32.item_size(),
            core.qkv_dim() * core.conv_kernel_size * Dtype::Float32.item_size(),
        )?;
        let norm_weight = bf16_tensor_as_f32(store, &bindings.norm_weight)?;
        validate_len(
            "GDN norm weight",
            norm_weight.len() * Dtype::Float32.item_size(),
            core.v_head_dim * Dtype::Float32.item_size(),
        )?;
        let a_log_decay = bf16_tensor_as_f32(store, &bindings.a_log)?
            .into_iter()
            .map(|value| -value.exp())
            .collect::<Vec<_>>();
        let dt_bias = bf16_tensor_as_f32(store, &bindings.dt_bias)?;
        validate_len(
            "GDN A_log",
            a_log_decay.len() * Dtype::Float32.item_size(),
            core.num_v_heads * Dtype::Float32.item_size(),
        )?;
        validate_len(
            "GDN dt_bias",
            dt_bias.len() * Dtype::Float32.item_size(),
            core.num_v_heads * Dtype::Float32.item_size(),
        )?;

        Ok(Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scales),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_biases),
            conv_weight: Buffer::from_slice(device, &conv_weight),
            norm_weight: Buffer::from_slice(device, &norm_weight),
            a_log_decay: Buffer::from_slice(device, &a_log_decay),
            dt_bias: Buffer::from_slice(device, &dt_bias),
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        })
    }

    fn as_borrowed(&self) -> GDNWeights<'_> {
        GDNWeights {
            qkvabz_weight: &self.qkvabz_weight,
            qkvabz_scales: &self.qkvabz_scales,
            qkvabz_biases: &self.qkvabz_biases,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            a_log_decay: &self.a_log_decay,
            dt_bias: &self.dt_bias,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}
