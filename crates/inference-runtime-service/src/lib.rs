pub mod api;
pub mod codec;
pub mod consts;
pub mod executor;
pub mod perf_metrics;
pub mod profiling;
#[path = "bin/qwen_server/mod.rs"]
pub mod qwen_server;
pub mod runtime;
pub mod rpc;
pub mod specialization;
pub mod telemetry;
pub mod tool;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
