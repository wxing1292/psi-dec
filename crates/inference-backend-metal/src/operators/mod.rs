//! Reusable Metal-backed operators without model component semantics.
//!
//! Each operator separates fixed workload facts from invocation-time extents:
//!
//! ```text
//! Config + Shape + Buffers
//!            |
//!            v
//!          Kernel
//!            |
//!            v
//!        Invocation
//!            |
//!            v
//!     CommandRecorder
//! ```

mod affine_quantized;
mod bf16_concat_rows;
mod mlx_headers;
mod row_gather;
mod softmax;

pub use affine_quantized::AffineQuantizedMatmul;
pub use affine_quantized::AffineQuantizedMatmulConfig;
pub use affine_quantized::AffineQuantizedMatmulKernel;
pub use affine_quantized::AffineQuantizedMatmulKernelKind;
pub use affine_quantized::ExpertAffineQuantizedConfig;
pub use affine_quantized::GatherAffineQuantizedGateUpSwiGLUKernel;
pub use affine_quantized::GatherAffineQuantizedMatmulKernel;
pub use affine_quantized::GatherAffineQuantizedShape;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedGateUpSwiGLUKernel;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedMatmulKernel;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedShape;
pub use bf16_concat_rows::Bf16ConcatRowsBuffers;
pub use bf16_concat_rows::Bf16ConcatRowsConfig;
pub use bf16_concat_rows::Bf16ConcatRowsKernel;
pub use bf16_concat_rows::Bf16ConcatRowsShape;
pub use row_gather::RowGatherBuffers;
pub use row_gather::RowGatherConfig;
pub use row_gather::RowGatherKernel;
pub use row_gather::RowGatherShape;
pub use softmax::SoftmaxBuffers;
pub use softmax::SoftmaxConfig;
pub use softmax::SoftmaxKernel;
pub use softmax::SoftmaxShape;
