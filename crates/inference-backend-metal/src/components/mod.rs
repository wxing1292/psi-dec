//! Reusable buffer-first Metal model execution components.
//!
//! These operators are implemented with the low-level Metal API, but their
//! contracts are component semantics rather than generic Metal primitives.
//! They should not own model layer order, scheduler policy, or runtime page
//! allocation.
//!
//! Each public component module uses short, scoped names. `Config` owns fixed
//! workload facts. These facts include dtype, model dimensions, and fixed
//! scalar parameters. `Shape` owns invocation-time extents and runtime
//! metadata. `Buffers`, `Weights`, and `Scratch` bind storage. `Compute` owns
//! reusable compiled Metal execution state. `Invocation` records one
//! execution into a stream batch. Private `KernelConstants` values describe
//! compile-time kernel constants.

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

fn assert_u32_count_domain(count: usize, name: &str) {
    assert!(count > 0, "{name} must be positive");
    assert!(
        u32::try_from(count).is_ok(),
        "{name} exceeds the shader u32 count domain: count={count}"
    );
}

fn assert_u32_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

pub mod dense_mlp;
pub mod sparse_mlp;

pub mod embedding;

pub mod gdn;

mod gdn_qkvabz_split;
pub use gdn_qkvabz_split::GDNQKVABZSplitBuffers;
pub use gdn_qkvabz_split::GDNQKVABZSplitConfig;
pub use gdn_qkvabz_split::GDNQKVABZSplitKernel;
pub use gdn_qkvabz_split::GDNQKVABZSplitShape;

mod gqa_split_kv_single_q;
pub use gqa_split_kv_single_q::GQAActivationGateBuffers;
pub use gqa_split_kv_single_q::GQAActivationGateConfig;
pub use gqa_split_kv_single_q::GQAActivationGateKernel;
pub use gqa_split_kv_single_q::GQAActivationGateShape;
pub use gqa_split_kv_single_q::GQASplitKVSingleQConfig;
pub use gqa_split_kv_single_q::GQASplitKVSingleQKernels;
pub use gqa_split_kv_single_q::GQASplitKVSingleQMapBuffers;
pub use gqa_split_kv_single_q::GQASplitKVSingleQReduceBuffers;
pub use gqa_split_kv_single_q::GQASplitKVSingleQScratch;
pub use gqa_split_kv_single_q::GQASplitKVSingleQShape;

mod gqa_split_kv_tiled_q;
pub use gqa_split_kv_tiled_q::GQASplitKVTiledQConfig;
pub use gqa_split_kv_tiled_q::GQASplitKVTiledQKernels;
pub use gqa_split_kv_tiled_q::GQASplitKVTiledQMapBuffers;
pub use gqa_split_kv_tiled_q::GQASplitKVTiledQReduceBuffers;
pub use gqa_split_kv_tiled_q::GQASplitKVTiledQShape;

pub mod gqa;

mod gqa_qgkv_split;
pub use gqa_qgkv_split::GQAQGKVSplitBuffers;
pub use gqa_qgkv_split::GQAQGKVSplitConfig;
pub use gqa_qgkv_split::GQAQGKVSplitKernel;
pub use gqa_qgkv_split::GQAQGKVSplitShape;

mod gqa_qkv_split;
pub use gqa_qkv_split::GQAQKVSplitBuffers;
pub use gqa_qkv_split::GQAQKVSplitConfig;
pub use gqa_qkv_split::GQAQKVSplitKernel;
pub use gqa_qkv_split::GQAQKVSplitShape;

mod gqa_kv_page_write;
pub use gqa_kv_page_write::GQAKVPageWrite;
pub use gqa_kv_page_write::GQAKVPageWriteBuffers;
pub use gqa_kv_page_write::GQAKVPageWriteConfig;
pub use gqa_kv_page_write::GQAKVPageWriteShape;
pub use gqa_kv_page_write::GQAPageTableLayout;

mod gqa_block_attention;
pub use gqa_block_attention::GQABlockSDPABuffers;
pub use gqa_block_attention::GQABlockSDPAConfig;
pub use gqa_block_attention::GQABlockSDPAKernel;
pub use gqa_block_attention::GQABlockSDPAShape;

pub mod moe;

pub mod residual_add;
pub mod residual_add_rms_norm;

mod replay;
pub use replay::ReplayOp;
pub use replay::ReplayRecorder;

pub mod rms_norm;
pub mod rms_norm_rope;

pub mod sampling;

#[cfg(test)]
mod tests {
    use super::assert_u32_count_domain;
    use super::assert_u32_index_domain;

    #[test]
    #[should_panic(expected = "exceeds the shader u32 count domain")]
    fn test_u32_count_domain_rejects_two_to_32() {
        assert_u32_count_domain(u32::MAX as usize + 1, "test count");
    }

    #[test]
    fn test_u32_index_domain_accepts_two_to_32_elements() {
        assert_u32_index_domain(u32::MAX as usize + 1, "test elements");
    }

    #[test]
    #[should_panic(expected = "exceeds the shader u32 element-index domain")]
    fn test_u32_index_domain_rejects_more_than_two_to_32_elements() {
        assert_u32_index_domain(u32::MAX as usize + 2, "test elements");
    }
}
