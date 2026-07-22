use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;

use crate::attn::gdn::backend::GDNMetalConfig;

pub struct GDNScratch {
    hidden_state_f32: Buffer,
    qkvabz: Buffer,
    projected_qkv: Buffer,
    a: Buffer,
    b: Buffer,
    z: Buffer,
    conv_qkv: Buffer,
    recurrent_output: Buffer,
    pre_output_hidden_states: Buffer,
    pre_output_hidden_states_bf16: Buffer,
}

#[derive(Clone, Copy)]
pub struct GDNScratchBindings<'a> {
    pub hidden_state_f32: &'a Buffer,
    pub qkvabz: &'a Buffer,
    pub projected_qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
    pub conv_qkv: &'a Buffer,
    pub recurrent_output: &'a Buffer,
    pub pre_output_hidden_states: &'a Buffer,
    pub pre_output_hidden_states_bf16: &'a Buffer,
}

impl GDNScratch {
    pub fn new(device: &Device, core: &GDNCore, config: GDNMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        config.validate();
        assert!(max_tokens > 0);
        let tensor_elements = |dim: usize| {
            max_tokens
                .checked_mul(dim)
                .expect("GDN scratch tensor element count must fit usize")
        };

        Self {
            hidden_state_f32: Buffer::new_zeroed_elements(device, tensor_elements(core.hidden_dim), Dtype::Float32),
            qkvabz: Buffer::new_zeroed_elements(device, tensor_elements(core.qkvabz_dim()), config.internal_dtype()),
            projected_qkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qkv_dim()), Dtype::Float32),
            a: Buffer::new_zeroed_elements(device, tensor_elements(core.num_v_heads), Dtype::Float32),
            b: Buffer::new_zeroed_elements(device, tensor_elements(core.num_v_heads), Dtype::Float32),
            z: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), Dtype::Float32),
            conv_qkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qkv_dim()), Dtype::Float32),
            recurrent_output: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), Dtype::Float32),
            pre_output_hidden_states: Buffer::new_zeroed_elements(
                device,
                tensor_elements(core.v_dim()),
                Dtype::Float32,
            ),
            pre_output_hidden_states_bf16: Buffer::new_zeroed_elements(
                device,
                tensor_elements(core.v_dim()),
                Dtype::Bfloat16,
            ),
        }
    }

    pub fn bindings(&self) -> GDNScratchBindings<'_> {
        GDNScratchBindings {
            hidden_state_f32: &self.hidden_state_f32,
            qkvabz: &self.qkvabz,
            projected_qkv: &self.projected_qkv,
            a: &self.a,
            b: &self.b,
            z: &self.z,
            conv_qkv: &self.conv_qkv,
            recurrent_output: &self.recurrent_output,
            pre_output_hidden_states: &self.pre_output_hidden_states,
            pre_output_hidden_states_bf16: &self.pre_output_hidden_states_bf16,
        }
    }
}
