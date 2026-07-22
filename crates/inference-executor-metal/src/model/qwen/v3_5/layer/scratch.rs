use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;

pub struct Qwen35LayerScratch {
    hidden_dim: usize,
    residual_stream: [Buffer; 2],
    pub normalized_hidden: Buffer,
    pub branch_output: Buffer,
    pub post_attention_hidden: Buffer,
}

impl Qwen35LayerScratch {
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 layer scratch element count must fit usize");
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

    pub fn residual_stream(&self, model_layer_index: usize) -> &Buffer {
        &self.residual_stream[model_layer_index % self.residual_stream.len()]
    }
}
