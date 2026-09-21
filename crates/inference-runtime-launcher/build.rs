use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // OUT_DIR is <target>/<profile>/build/<package>/out, with an optional target triple.
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Cargo build output must have a profile directory");
    println!(
        "cargo:rustc-env=INFERENCE_LAUNCHER_PROFILE_DIR={}",
        profile_dir.display()
    );
    println!(
        "cargo:rustc-env=INFERENCE_LAUNCHER_TARGET={}",
        env::var("TARGET").unwrap()
    );
}
