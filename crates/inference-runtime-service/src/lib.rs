pub mod consts;
pub mod executor;
pub mod observability;
pub mod perf_metrics;
pub mod profiling;
pub mod service;
pub mod runtime;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
