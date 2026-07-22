use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::mlp::dense::DenseMLPCore;

use crate::mlp::dense::backend::DenseMLPMetalConfig;

pub struct DenseMLPScratch {
    gate_up_proj: Buffer,
    activation: Buffer,
}

#[derive(Clone, Copy)]
pub struct DenseMLPScratchBindings<'a> {
    pub gate_up_proj: &'a Buffer,
    pub activation: &'a Buffer,
}

impl DenseMLPScratch {
    pub fn new(device: &Device, core: &DenseMLPCore, config: DenseMLPMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        config.validate();
        assert!(max_tokens > 0);

        let backend_config = QuantizedDenseMLPConfig {
            hidden_dim: core.hidden_dim.try_into().expect("dense MLP hidden_dim must fit u32"),
            intermediate_dim: core
                .intermediate_dim
                .try_into()
                .expect("dense MLP intermediate_dim must fit u32"),
            group_size: config.group_size,
            bits: config.bits,
            dtype: config.dtype,
        };
        let shape = QuantizedDenseMLPShape {
            num_tokens: max_tokens
                .try_into()
                .expect("dense MLP scratch token capacity must fit u32"),
        };
        Self {
            gate_up_proj: Buffer::new_zeroed(device, backend_config.gate_up_shape(shape).output_bytes()),
            activation: Buffer::new_zeroed(device, backend_config.activation_shape(shape).bytes()),
        }
    }

    pub fn bindings(&self) -> DenseMLPScratchBindings<'_> {
        DenseMLPScratchBindings {
            gate_up_proj: &self.gate_up_proj,
            activation: &self.activation,
        }
    }
}
