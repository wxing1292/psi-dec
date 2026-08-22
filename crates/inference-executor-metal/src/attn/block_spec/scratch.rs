//! Shared scratch buffers for block-spec GQA.

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BlockSpecGQACore;

use crate::attn::block_spec::capacity::BlockSpecGQACapacity;

pub struct BlockSpecScratch {
    capacity: BlockSpecGQACapacity,
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
pub struct BlockSpecScratchBindings<'a> {
    pub capacity: BlockSpecGQACapacity,
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

impl BlockSpecScratch {
    pub fn new(device: &Device, core: &BlockSpecGQACore, io_dtype: Dtype, capacity: BlockSpecGQACapacity) -> Self {
        core.validate();
        match io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 block-spec GQA model boundary is not supported"),
            dtype => panic!("unsupported block-spec GQA model boundary dtype {dtype:?}"),
        }
        assert_eq!(
            core.block_size, capacity.block.block_size,
            "block-spec GQA core and scratch block sizes must match"
        );
        let attention = &core.attention;
        let tensor_elements = |dim: usize| {
            capacity
                .block
                .max_tokens
                .checked_mul(dim)
                .expect("block-spec block scratch tensor element count must fit usize")
        };
        let partial_stats = capacity
            .max_sdpa_partial_state_groups
            .checked_mul(attention.num_q_heads)
            .expect("block-spec block scratch partial statistic count must fit usize");
        let partial_values = partial_stats
            .checked_mul(attention.head_dim)
            .expect("block-spec block scratch partial output count must fit usize");
        assert_u32_index_domain(partial_stats, "block-spec block partial statistics");
        assert_u32_index_domain(partial_values, "block-spec block partial output");

        Self {
            capacity,
            q: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), io_dtype),
            k: Buffer::new_zeroed_elements(device, tensor_elements(attention.k_dim()), io_dtype),
            v: Buffer::new_zeroed_elements(device, tensor_elements(attention.v_dim()), io_dtype),
            q_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), io_dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(attention.k_dim()), io_dtype),
            partial_exp_sums: Buffer::new_zeroed_elements(device, partial_stats, Dtype::Float32),
            partial_max_logits: Buffer::new_zeroed_elements(device, partial_stats, Dtype::Float32),
            partial_output: Buffer::new_zeroed_elements(device, partial_values, io_dtype),
            attention_output: Buffer::new_zeroed_elements(device, tensor_elements(attention.q_dim()), io_dtype),
        }
    }

    pub fn bindings(&self) -> BlockSpecScratchBindings<'_> {
        BlockSpecScratchBindings {
            capacity: self.capacity,
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
