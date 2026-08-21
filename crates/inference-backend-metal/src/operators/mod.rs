//! Reusable Metal-backed operators without model component semantics.
//!
//! Each public operator has a scoped module. The module path identifies the
//! operation. Short leaf names identify API roles.
//!
//! A fixed operator separates fixed workload facts from invocation-time extents:
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
//!
//! `affine_quantized::Matmul` is an adaptive owner. It selects one
//! `affine_quantized::Kernel` from its private registry for the runtime row
//! count.

pub mod affine_quantized;
pub mod bf16_concat_rows;
mod mlx_headers;
pub mod row_gather;
pub mod softmax;
