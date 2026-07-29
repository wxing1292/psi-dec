use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::DSparkBlockCapacity;
use inference_executor_core::attn::UngatedDSparkGQACore;

use crate::attn::gqa::backend::GQAMetalConfig;

pub struct DSparkBlockScratch {
    capacity: DSparkBlockCapacity,
    qkv_proj: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    partial_exp_sums: Buffer,
    partial_max_logits: Buffer,
    partial_output: Buffer,
    attention_output: Buffer,
}

#[derive(Clone, Copy)]
pub struct DSparkBlockScratchBindings<'a> {
    pub capacity: DSparkBlockCapacity,
    pub qkv_proj: &'a Buffer,
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_norm_rope: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub attention_output: &'a Buffer,
}

impl DSparkBlockScratch {
    pub fn new(
        device: &Device,
        core: &UngatedDSparkGQACore,
        metal: GQAMetalConfig,
        capacity: DSparkBlockCapacity,
    ) -> Self {
        core.validate();
        metal.validate();
        assert_eq!(
            core.block_size, capacity.block_size,
            "DSpark GQA core and scratch block sizes must match"
        );
        let attention = &core.attention;
        let tensor_elements = |dim: usize| {
            capacity
                .max_tokens
                .checked_mul(dim)
                .expect("DSpark block scratch tensor element count must fit usize")
        };
        let partial_stats = capacity
            .max_sdpa_map_task_templates
            .checked_mul(attention.num_q_heads)
            .expect("DSpark block scratch partial statistic count must fit usize");
        let partial_values = partial_stats
            .checked_mul(attention.head_dim)
            .expect("DSpark block scratch partial output count must fit usize");
        assert_u32_index_domain(partial_stats, "DSpark block partial statistics");
        assert_u32_index_domain(partial_values, "DSpark block partial output");

        Self {
            capacity,
            qkv_proj: Buffer::new_zeroed_elements(device, tensor_elements(attention.qkv_dim()), metal.dtype),
            q: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), metal.dtype),
            k: Buffer::new_zeroed_elements(device, tensor_elements(attention.k_dim()), metal.dtype),
            v: Buffer::new_zeroed_elements(device, tensor_elements(attention.v_dim()), metal.dtype),
            q_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), metal.dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(attention.k_dim()), metal.dtype),
            partial_exp_sums: Buffer::new_zeroed_elements(device, partial_stats, Dtype::Float32),
            partial_max_logits: Buffer::new_zeroed_elements(device, partial_stats, Dtype::Float32),
            partial_output: Buffer::new_zeroed_elements(device, partial_values, metal.dtype),
            attention_output: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), metal.dtype),
        }
    }

    pub fn capacity(&self) -> DSparkBlockCapacity {
        self.capacity
    }

    pub fn bindings(&self) -> DSparkBlockScratchBindings<'_> {
        DSparkBlockScratchBindings {
            capacity: self.capacity,
            qkv_proj: &self.qkv_proj,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            q_norm_rope: &self.q_norm_rope,
            k_norm_rope: &self.k_norm_rope,
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
            attention_output: &self.attention_output,
        }
    }
}

fn assert_u32_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}
