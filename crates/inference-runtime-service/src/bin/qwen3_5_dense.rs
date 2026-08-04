include!(concat!(env!("OUT_DIR"), "/qwen35_cache_lanes.rs"));

fn main() {
    inference_runtime_service::qwen_server::qwen35::run_dense::<QWEN35_CACHE_LANES>();
}
