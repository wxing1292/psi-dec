//! Reusable Metal-backed operators without model component semantics.

pub mod affine_quantized;
pub mod elementwise;
mod mlx_headers;
pub mod softmax;

pub use affine_quantized::AffineQuantizedMatmulKernel;
pub use affine_quantized::AffineQuantizedMatmulShape;
pub use affine_quantized::GatherAffineQuantizedMatmulKernel;
pub use affine_quantized::GatherAffineQuantizedMatmulShape;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedGateUpSiluKernel;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedGateUpSiluShape;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedMatmulKernel;
pub use affine_quantized::RaggedExpertMajorAffineQuantizedMatmulShape;
pub use elementwise::MLXElementwiseShape;
pub use elementwise::MLXMultiplyKernel;
pub use elementwise::MLXSigmoidKernel;
pub use softmax::SoftmaxKernel;
pub use softmax::SoftmaxShape;
