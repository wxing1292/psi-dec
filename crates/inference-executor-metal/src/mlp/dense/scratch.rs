use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::mlp::dense::DenseMLPCore;

pub struct DenseMLPScratch {
    swiglu: Buffer,
}

#[derive(Clone, Copy)]
pub struct DenseMLPScratchBindings<'a> {
    pub swiglu: &'a Buffer,
}

impl DenseMLPScratch {
    pub fn new(device: &Device, core: &DenseMLPCore, io_dtype: Dtype, max_tokens: usize) -> Self {
        core.validate();
        assert!(max_tokens > 0);
        match io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 dense MLP model boundary is not supported"),
            dtype => panic!("unsupported dense MLP model boundary dtype {dtype:?}"),
        }
        let swiglu_elements = max_tokens
            .checked_mul(core.intermediate_dim)
            .expect("dense MLP SwiGLU scratch element capacity must fit usize");
        Self {
            swiglu: Buffer::new_zeroed_elements(device, swiglu_elements, io_dtype),
        }
    }

    pub fn bindings(&self) -> DenseMLPScratchBindings<'_> {
        DenseMLPScratchBindings { swiglu: &self.swiglu }
    }
}
