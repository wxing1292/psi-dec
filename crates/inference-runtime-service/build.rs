use std::env;
use std::fs;
use std::path::PathBuf;

const QWEN35_BUILD_CACHE_LANES_ENV: &str = "PSI_QWEN35_CACHE_LANES";

fn main() {
    println!("cargo:rerun-if-env-changed={QWEN35_BUILD_CACHE_LANES_ENV}");
    let num_cache_lanes = env::var(QWEN35_BUILD_CACHE_LANES_ENV)
        .unwrap_or_else(|_| "1".to_owned())
        .parse::<usize>()
        .expect("qwen3.5 cache lane count must be a positive usize");
    assert!(
        num_cache_lanes > 0,
        "qwen3.5 cache lane count must include the Main lane"
    );
    let generated = format!("pub const QWEN35_CACHE_LANES: usize = {num_cache_lanes};\n");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR for runtime service"));
    fs::write(out_dir.join("qwen35_cache_lanes.rs"), generated)
        .expect("runtime service must write the generated qwen3.5 cache lane constant");
}
