//! Reusable buffer-first Metal model execution components.
//!
//! These operators are implemented with the low-level Metal API, but their
//! contracts are component semantics rather than generic Metal primitives.
//! They should not own model layer order, scheduler policy, or runtime page
//! allocation.
//!
//! `FooConfig` owns fixed workload facts. These facts include dtype,
//! specialization constants, model dimensions, and fixed scalar parameters.
//! `FooShape` owns invocation-time extents and runtime metadata.
//! `FooBuffers` / `FooWeights` / `FooScratch` bind storage. `FooKernel` /
//! `FooKernels` owns reusable compiled Metal execution state. `FooInvocation`
//! records one execution into a stream batch.

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

mod quantized_dense_mlp;
pub use quantized_dense_mlp::QuantizedDenseMLP;
pub use quantized_dense_mlp::QuantizedDenseMLPBuffers;
pub use quantized_dense_mlp::QuantizedDenseMLPConfig;
pub use quantized_dense_mlp::QuantizedDenseMLPScratch;
pub use quantized_dense_mlp::QuantizedDenseMLPShape;
pub use quantized_dense_mlp::QuantizedDenseMLPWeights;

mod quantized_sparse_mlp;
pub use quantized_sparse_mlp::QuantizedSparseMLP;
pub use quantized_sparse_mlp::QuantizedSparseMLPConfig;
pub use quantized_sparse_mlp::QuantizedSparseMLPExpertMajorBuffers;
pub use quantized_sparse_mlp::QuantizedSparseMLPExpertMajorShape;
pub use quantized_sparse_mlp::QuantizedSparseMLPScratch;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorBuffers;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorKernels;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorShape;
pub use quantized_sparse_mlp::QuantizedSparseMLPWeights;

mod quantized_embedding;
pub use quantized_embedding::QuantizedEmbeddingBuffers;
pub use quantized_embedding::QuantizedEmbeddingConfig;
pub use quantized_embedding::QuantizedEmbeddingKernel;
pub use quantized_embedding::QuantizedEmbeddingShape;

mod gdn_compute;
pub use gdn_compute::GDNCompute;
pub use gdn_compute::GDNComputeBuffers;
pub use gdn_compute::GDNComputeConfig;
pub use gdn_compute::GDNComputeShape;
pub use gdn_compute::GDNComputeWithCandidateStateUpdateBuffers;

mod gdn_qkvabz_split;
pub use gdn_qkvabz_split::GDNQKVABZSplitBuffers;
pub use gdn_qkvabz_split::GDNQKVABZSplitConfig;
pub use gdn_qkvabz_split::GDNQKVABZSplitKernel;
pub use gdn_qkvabz_split::GDNQKVABZSplitShape;

mod gdn_state_pages;
pub use gdn_state_pages::GDNStatePageBatchConfig;
pub use gdn_state_pages::GDNStatePageBatchRead;
pub use gdn_state_pages::GDNStatePageBatchReadBuffers;
pub use gdn_state_pages::GDNStatePageBatchShape;
pub use gdn_state_pages::GDNStatePageBatchWrite;
pub use gdn_state_pages::GDNStatePageBatchWriteBuffers;

mod gqa_attention;
pub use gqa_attention::GQAActivationGateBuffers;
pub use gqa_attention::GQAActivationGateConfig;
pub use gqa_attention::GQAActivationGateKernel;
pub use gqa_attention::GQAActivationGateShape;
pub use gqa_attention::GQAPagedSDPAConfig;
pub use gqa_attention::GQAPagedSDPAKernels;
pub use gqa_attention::GQAPagedSDPAMapBuffers;
pub use gqa_attention::GQAPagedSDPAReduceBuffers;
pub use gqa_attention::GQAPagedSDPAScratch;
pub use gqa_attention::GQAPagedSDPAShape;

mod gqa_tiled_attention;
pub use gqa_tiled_attention::GQATiledSDPAConfig;
pub use gqa_tiled_attention::GQATiledSDPAKernels;
pub use gqa_tiled_attention::GQATiledSDPAMapBuffers;
pub use gqa_tiled_attention::GQATiledSDPAReduceBuffers;
pub use gqa_tiled_attention::GQATiledSDPAShape;

mod gqa_compute;
pub use gqa_compute::GQACompute;
pub use gqa_compute::GQAComputeConfig;
pub use gqa_compute::GQAComputePath;

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

mod moe_combine;
pub use moe_combine::MoECombineConfig;
pub use moe_combine::MoECombineKernels;
pub use moe_combine::MoECombineShape;
pub use moe_combine::MoECombineWithSharedExpertsBuffers;
pub use moe_combine::MoECombineWithoutSharedExpertsBuffers;

mod moe_expert_major;
pub use moe_expert_major::MoEExpertMajorConfig;
pub use moe_expert_major::MoEExpertMajorKernels;
pub use moe_expert_major::MoEExpertMajorLayoutBuffers;
pub use moe_expert_major::MoEExpertMajorPackInputBuffers;
pub use moe_expert_major::MoEExpertMajorScatterWithSharedExpertsBuffers;
pub use moe_expert_major::MoEExpertMajorScatterWithoutSharedExpertsBuffers;
pub use moe_expert_major::MoEExpertMajorShape;

mod moe_routing;
pub use moe_routing::MoERoutingBuffers;
pub use moe_routing::MoERoutingConfig;
pub use moe_routing::MoERoutingKernel;
pub use moe_routing::MoERoutingShape;

mod gqa_norm_rope;
pub use gqa_norm_rope::GQANormRopeBuffers;
pub use gqa_norm_rope::GQANormRopeConfig;
pub use gqa_norm_rope::GQANormRopeKernel;
pub use gqa_norm_rope::GQANormRopeShape;

mod residual_add;
pub use residual_add::ResidualAddBuffers;
pub use residual_add::ResidualAddCaptureTarget;
pub use residual_add::ResidualAddConfig;
pub use residual_add::ResidualAddInvocation;
pub use residual_add::ResidualAddKernel;
pub use residual_add::ResidualAddShape;

mod residual_add_rms_norm;
pub use residual_add_rms_norm::ResidualAddRMSNormBuffers;
pub use residual_add_rms_norm::ResidualAddRMSNormConfig;
pub use residual_add_rms_norm::ResidualAddRMSNormInvocation;
pub use residual_add_rms_norm::ResidualAddRMSNormKernel;
pub use residual_add_rms_norm::ResidualAddRMSNormKernelKind;
pub use residual_add_rms_norm::ResidualAddRMSNormShape;

mod replay;
pub use replay::ReplayOp;
pub use replay::ReplayRecorder;

mod rms_norm;
pub use rms_norm::RMSNormBuffers;
pub use rms_norm::RMSNormConfig;
pub use rms_norm::RMSNormInvocation;
pub use rms_norm::RMSNormKernel;
pub use rms_norm::RMSNormShape;

mod sampling;
pub use sampling::DSparkConfidenceBuffers;
pub use sampling::DSparkConfidenceConfig;
pub use sampling::DSparkMarkovTopKMapBuffers;
pub use sampling::DSparkMarkovTopKMapConfig;
pub use sampling::DSparkMarkovTopKMapKernel;
pub use sampling::DSparkMarkovTopKMapShape;
pub use sampling::SparseRejectionSampleBuffers;
pub use sampling::SparseRejectionSampleKernel;
pub use sampling::SparseRejectionSampleShape;
pub use sampling::TopKMergeKernels;
pub use sampling::TopKSampleAndWriteDistributionBuffers;
pub use sampling::TopKSampleBuffers;
pub use sampling::TopKSampleShape;
pub use sampling::TopKSamplingOperation;
pub use sampling::TopKTileBuffers;
pub use sampling::TopKTileKernels;
pub use sampling::TopKWriteDistributionBuffers;

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
