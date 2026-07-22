use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQACore;

use crate::attn::gqa::backend::GQAMetalConfig;

fn assert_u32_element_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

pub struct GQAScratch {
    qgkv_proj: Buffer,
    q: Buffer,
    g: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    sdpa_partial_exp_sums: Buffer,
    sdpa_partial_max_logits: Buffer,
    sdpa_partial_output: Buffer,
    attention_output: Buffer,
    gated_attention_output: Buffer,
}

#[derive(Clone, Copy)]
pub struct GQAScratchBindings<'a> {
    pub qgkv_proj: &'a Buffer,
    pub q: &'a Buffer,
    pub g: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_norm_rope: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
    pub sdpa_partial_exp_sums: &'a Buffer,
    pub sdpa_partial_max_logits: &'a Buffer,
    pub sdpa_partial_output: &'a Buffer,
    pub attention_output: &'a Buffer,
    pub gated_attention_output: &'a Buffer,
}

impl GQAScratch {
    pub fn new(device: &Device, core: &GQACore, config: GQAMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        config.validate();
        assert!(max_tokens > 0);
        let max_tokens_per_partial_output = if config.supports_tiled(core) {
            config.q_token_tile_size as usize
        } else {
            1
        };
        let num_sdpa_partial_output_tokens = max_tokens
            .checked_mul(max_tokens_per_partial_output)
            .expect("GQA scratch partial-token capacity must fit usize");
        let num_sdpa_partial_output_stats = num_sdpa_partial_output_tokens
            .checked_mul(core.num_q_heads)
            .expect("GQA scratch partial-output statistic count must fit usize");
        let tensor_elements = |dim: usize| {
            let elements = max_tokens
                .checked_mul(dim)
                .expect("GQA scratch tensor element count must fit usize");
            u32::try_from(elements).expect("GQA scratch tensor element count must fit the shader u32 count domain");
            elements
        };
        assert_u32_element_index_domain(num_sdpa_partial_output_stats, "GQA SDPA partial-output statistics");
        let num_sdpa_partial_output_values = num_sdpa_partial_output_stats
            .checked_mul(core.head_dim)
            .expect("GQA SDPA partial output element count must fit usize");
        assert_u32_element_index_domain(num_sdpa_partial_output_values, "GQA SDPA partial output");
        Self {
            qgkv_proj: Buffer::new_zeroed_elements(device, tensor_elements(core.qgkv_dim()), config.dtype),
            q: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.dtype),
            g: Buffer::new_zeroed_elements(device, tensor_elements(core.g_dim()), config.dtype),
            k: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.dtype),
            v: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), config.dtype),
            q_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.dtype),
            sdpa_partial_exp_sums: Buffer::new_zeroed_elements(device, num_sdpa_partial_output_stats, Dtype::Float32),
            sdpa_partial_max_logits: Buffer::new_zeroed_elements(device, num_sdpa_partial_output_stats, Dtype::Float32),
            sdpa_partial_output: Buffer::new_zeroed_elements(device, num_sdpa_partial_output_values, config.dtype),
            attention_output: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.dtype),
            gated_attention_output: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.dtype),
        }
    }

    pub fn bindings(&self) -> GQAScratchBindings<'_> {
        GQAScratchBindings {
            qgkv_proj: &self.qgkv_proj,
            q: &self.q,
            g: &self.g,
            k: &self.k,
            v: &self.v,
            q_norm_rope: &self.q_norm_rope,
            k_norm_rope: &self.k_norm_rope,
            sdpa_partial_exp_sums: &self.sdpa_partial_exp_sums,
            sdpa_partial_max_logits: &self.sdpa_partial_max_logits,
            sdpa_partial_output: &self.sdpa_partial_output,
            attention_output: &self.attention_output,
            gated_attention_output: &self.gated_attention_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::assert_u32_element_index_domain;

    #[test]
    fn test_u32_element_index_domain_accepts_two_to_32_elements() {
        assert_u32_element_index_domain(u32::MAX as usize + 1, "test GQA scratch");
    }
}
