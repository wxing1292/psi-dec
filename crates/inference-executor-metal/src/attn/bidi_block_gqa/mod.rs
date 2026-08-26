//! Paged-history GQA with a bidirectional local query block.

pub mod backend;
pub mod capacity;
pub mod kv_cache_write;
pub mod metadata;
pub mod scratch;
mod sdpa;
pub mod state;
