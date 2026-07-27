pub mod api;
pub mod codec;
pub mod consts;
pub mod executor;
pub mod observability;
pub mod perf_metrics;
pub mod profiling;
pub mod runtime;
pub mod rpc;
pub mod tool;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
