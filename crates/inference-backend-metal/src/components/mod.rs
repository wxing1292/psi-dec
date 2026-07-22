//! Reusable buffer-first Metal model execution components.
//!
//! These operators are implemented with the low-level Metal API, but their
//! contracts are component semantics rather than generic Metal primitives.
//! They should not own model layer order, scheduler policy, or runtime page
//! allocation.
//!
//! Components use `FooConfig` only when model/kernel configuration affects
//! reusable kernel construction, operator specialization, weight layout, or
//! object compatibility. `FooShape` owns invocation-time extents, scalar
//! parameters, and runtime metadata. `FooBuffers` / `FooWeights` / `FooScratch`
//! bind storage, `FooKernel` / `FooKernels` owns reusable compiled Metal
//! execution state, and `FooInvocation` records one execution into a stream
//! batch.

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

mod buffer_cast;
pub use buffer_cast::BufferCastBuffers;
pub use buffer_cast::BufferCastKernel;
pub use buffer_cast::BufferCastShape;

mod buffer_copy;
pub use buffer_copy::BufferCopy32Buffers;
pub use buffer_copy::BufferCopy32Shape;
pub use buffer_copy::F32BufferCopyKernel;
pub use buffer_copy::U32BufferCopyKernel;

mod bf16_concat_rows;
pub use bf16_concat_rows::Bf16ConcatRowsBuffers;
pub use bf16_concat_rows::Bf16ConcatRowsKernel;
pub use bf16_concat_rows::Bf16ConcatRowsShape;

mod quantized_dense_mlp;
pub use quantized_dense_mlp::QuantizedDenseMLPBuffers;
pub use quantized_dense_mlp::QuantizedDenseMLPConfig;
pub use quantized_dense_mlp::QuantizedDenseMLPKernels;
pub use quantized_dense_mlp::QuantizedDenseMLPScratch;
pub use quantized_dense_mlp::QuantizedDenseMLPShape;
pub use quantized_dense_mlp::QuantizedDenseMLPWeights;

mod quantized_sparse_mlp;
pub use quantized_sparse_mlp::QuantizedSparseMLP;
pub use quantized_sparse_mlp::QuantizedSparseMLPConfig;
pub use quantized_sparse_mlp::QuantizedSparseMLPExpertMajorBuffers;
pub use quantized_sparse_mlp::QuantizedSparseMLPExpertMajorScratch;
pub use quantized_sparse_mlp::QuantizedSparseMLPExpertMajorShape;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorBuffers;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorKernels;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorScratch;
pub use quantized_sparse_mlp::QuantizedSparseMLPTokenMajorShape;
pub use quantized_sparse_mlp::QuantizedSparseMLPWeights;

mod rowwise_add;
pub use rowwise_add::RowwiseAddBuffers;
pub use rowwise_add::RowwiseAddConfig;
pub use rowwise_add::RowwiseAddKernel;
pub use rowwise_add::RowwiseAddShape;

mod quantized_embedding;
pub use quantized_embedding::QuantizedEmbeddingBuffers;
pub use quantized_embedding::QuantizedEmbeddingConfig;
pub use quantized_embedding::QuantizedEmbeddingKernel;
pub use quantized_embedding::QuantizedEmbeddingShape;

mod gdn_attention;
pub use gdn_attention::GDNCoreBuffers;
pub use gdn_attention::GDNCoreConfig;
pub use gdn_attention::GDNCoreForwardCandidateStateUpdateBuffers;
pub use gdn_attention::GDNCoreKernels;
pub use gdn_attention::GDNCoreShape;

mod gdn_projection;
pub use gdn_projection::GDNProjectionSplitBuffers;
pub use gdn_projection::GDNProjectionSplitKernel;
pub use gdn_projection::GDNProjectionSplitShape;

mod gdn_state_pages;
pub use gdn_state_pages::GDNStatePageBatchRead;
pub use gdn_state_pages::GDNStatePageBatchReadBuffers;
pub use gdn_state_pages::GDNStatePageBatchShape;
pub use gdn_state_pages::GDNStatePageBatchWrite;
pub use gdn_state_pages::GDNStatePageBatchWriteBuffers;
pub use gdn_state_pages::GDNStatePageRead;
pub use gdn_state_pages::GDNStatePageReadBuffers;
pub use gdn_state_pages::GDNStatePageShape;
pub use gdn_state_pages::GDNStatePageWrite;
pub use gdn_state_pages::GDNStatePageWriteBuffers;

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
pub use gqa_tiled_attention::GQATiledSDPAKernels;
pub use gqa_tiled_attention::GQATiledSDPAMapBuffers;
pub use gqa_tiled_attention::GQATiledSDPAReduceBuffers;
pub use gqa_tiled_attention::GQATiledSDPAShape;

mod gqa_projection;
pub use gqa_projection::GQAProjectionSplitBuffers;
pub use gqa_projection::GQAProjectionSplitConfig;
pub use gqa_projection::GQAProjectionSplitKernel;
pub use gqa_projection::GQAProjectionSplitShape;

mod gqa_kv_pages;
pub use gqa_kv_pages::GQAKVPageUpdate;
pub use gqa_kv_pages::GQAKVPageUpdateBuffers;
pub use gqa_kv_pages::GQAKVPageUpdateConfig;
pub use gqa_kv_pages::GQAKVPageUpdateShape;
pub use gqa_kv_pages::GQAPageTableLayout;

mod gqa_local_attention;
pub use gqa_local_attention::GQALocalSDPABuffers;
pub use gqa_local_attention::GQALocalSDPAConfig;
pub use gqa_local_attention::GQALocalSDPAKernel;
pub use gqa_local_attention::GQALocalSDPAShape;

mod moe_combine;
pub use moe_combine::MoECombineKernel;
pub use moe_combine::MoECombineWithCommonBuffers;
pub use moe_combine::MoECombineWithCommonShape;
pub use moe_combine::MoECombineWithoutCommonBuffers;
pub use moe_combine::MoECombineWithoutCommonShape;

mod moe_expert_major;
pub use moe_expert_major::MoEExpertMajorKernels;
pub use moe_expert_major::MoEExpertMajorLayoutBuffers;
pub use moe_expert_major::MoEExpertMajorPackInputBuffers;
pub use moe_expert_major::MoEExpertMajorScatterWithCommonBuffers;
pub use moe_expert_major::MoEExpertMajorScatterWithoutCommonBuffers;
pub use moe_expert_major::MoEExpertMajorShape;

mod moe_routing;
pub use moe_routing::MoERoutingBuffers;
pub use moe_routing::MoERoutingKernel;
pub use moe_routing::MoERoutingShape;

mod gqa_norm_rope;
pub use gqa_norm_rope::GQANormRopeBuffers;
pub use gqa_norm_rope::GQANormRopeConfig;
pub use gqa_norm_rope::GQANormRopeKernel;
pub use gqa_norm_rope::GQANormRopeShape;

mod residual;
pub use residual::DuplicateResidualOutput;
pub use residual::ResidualBuffers;
pub use residual::ResidualInvocation;
pub use residual::ResidualKernel;
pub use residual::ResidualShape;

mod residual_rms_norm;
pub use residual_rms_norm::ResidualRMSNormBuffers;
pub use residual_rms_norm::ResidualRMSNormInvocation;
pub use residual_rms_norm::ResidualRMSNormKernel;
pub use residual_rms_norm::ResidualRMSNormShape;

mod row_gather;
pub use row_gather::RowGatherBuffers;
pub use row_gather::RowGatherKernel;
pub use row_gather::RowGatherShape;

mod replay;
pub use replay::ReplayOp;
pub use replay::ReplayRecorder;

mod rms_norm;
pub use rms_norm::RMSNormBuffers;
pub use rms_norm::RMSNormInvocation;
pub use rms_norm::RMSNormKernel;
pub use rms_norm::RMSNormShape;

mod sampling;
pub use sampling::MAX_TOP_K;
pub use sampling::REJECTION_NUM_ACTIVE_THREADS_KEY;
pub use sampling::REJECTION_NUM_DRAFT_DISTRIBUTIONS_KEY;
pub use sampling::REJECTION_NUM_TARGET_DISTRIBUTIONS_KEY;
pub use sampling::SAMPLING_NUM_THREADS_PER_THREADBLOCK;
pub use sampling::SparseRejectionSampleBuffers;
pub use sampling::SparseRejectionSampleKernel;
pub use sampling::SparseRejectionSampleShape;
pub use sampling::TOP_K_MERGE_NUM_ACTIVE_THREADS_KEY;
pub use sampling::TOP_K_REDUCTION_LIMIT;
pub use sampling::TOP_K_TILE_NUM_ACTIVE_THREADS_KEY;
pub use sampling::TOP_K_VOCAB_TILE_SIZE;
pub use sampling::TopKSampleAndSparseDistributionBuffers;
pub use sampling::TopKSampleAndSparseDistributionKernel;
pub use sampling::TopKSampleBuffers;
pub use sampling::TopKSampleKernel;
pub use sampling::TopKSampleShape;
pub use sampling::TopKSparseDistributionBuffers;
pub use sampling::TopKSparseDistributionKernel;
pub use sampling::TopKTileBf16BitonicKernel;
pub use sampling::TopKTileBf16Kernel;
pub use sampling::TopKTileBitonicKernel;
pub use sampling::TopKTileBuffers;
pub use sampling::TopKTileKernel;

mod silu;
pub use silu::SiluBufferOffsets;
pub use silu::SiluBuffers;
pub use silu::SiluKernel;
pub use silu::SiluShape;

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
