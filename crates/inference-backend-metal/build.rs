use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const MLX_VERSION: &str = "v0.28.0";
const MLX_COMMIT: &str = "56be7736103569e661f2c894e3e972dc525ebdf1";
const MLX_ARCHIVE_URL: &str =
    "https://github.com/ml-explore/mlx/archive/56be7736103569e661f2c894e3e972dc525ebdf1.tar.gz";
const MLX_ARCHIVE_SHA256: &str = "4cffd6c19ce371f3216035982c55377784e1c21aafd478ae3685d0a5ed30b1aa";
const BUNDLED_MLX_ENV: &str = "INFERENCE_BACKEND_METAL_BUNDLED_MLX_SOURCE_DIR";

fn main() {
    println!("cargo:rerun-if-env-changed=INFERENCE_BACKEND_METAL_MLX_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=INFERENCE_BACKEND_METAL_MLX_PREFIX");

    let source_dir = ensure_mlx_source();
    println!("cargo:rustc-env={BUNDLED_MLX_ENV}={}", source_dir.display());
}

fn ensure_mlx_source() -> PathBuf {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set for build scripts"));
    let source_dir = out_dir.join(format!("mlx-{MLX_COMMIT}"));
    if is_compatible_mlx_source(&source_dir) {
        return source_dir;
    }

    let archive = out_dir.join(format!("mlx-{MLX_COMMIT}.tar.gz"));
    download_archive(&archive);

    let staging = out_dir.join(format!("mlx-{MLX_COMMIT}.staging-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging).unwrap_or_else(|err| panic!("cannot create {}: {err}", staging.display()));
    extract_archive(&archive, &staging);

    if !is_compatible_mlx_source(&staging) {
        panic!("downloaded MLX {MLX_VERSION} from {MLX_ARCHIVE_URL}, but it does not contain compatible Metal headers");
    }

    let _ = fs::remove_dir_all(&source_dir);
    fs::rename(&staging, &source_dir).unwrap_or_else(|err| {
        panic!(
            "cannot move downloaded MLX source from {} to {}: {err}",
            staging.display(),
            source_dir.display()
        )
    });
    source_dir
}

fn download_archive(archive: &Path) {
    if archive.exists() {
        if archive_sha256(archive) == MLX_ARCHIVE_SHA256 {
            return;
        }
        println!("cargo:warning=discarding cached MLX {MLX_VERSION} archive with an unexpected SHA-256");
        fs::remove_file(archive)
            .unwrap_or_else(|err| panic!("cannot remove invalid MLX archive {}: {err}", archive.display()));
    }

    let tmp_archive = archive.with_extension("tar.gz.download");
    let _ = fs::remove_file(&tmp_archive);
    let status = Command::new("curl")
        .arg("-fL")
        .arg("--retry")
        .arg("3")
        .arg("--retry-delay")
        .arg("2")
        .arg("-o")
        .arg(&tmp_archive)
        .arg(MLX_ARCHIVE_URL)
        .status()
        .unwrap_or_else(|err| panic!("cannot run curl to download MLX {MLX_VERSION}: {err}"));
    if !status.success() {
        panic!("curl failed while downloading MLX {MLX_VERSION} from {MLX_ARCHIVE_URL}");
    }
    assert_eq!(
        archive_sha256(&tmp_archive),
        MLX_ARCHIVE_SHA256,
        "downloaded MLX {MLX_VERSION} commit {MLX_COMMIT} has an unexpected SHA-256"
    );
    fs::rename(&tmp_archive, archive)
        .unwrap_or_else(|err| panic!("cannot save MLX archive to {}: {err}", archive.display()));
}

fn archive_sha256(archive: &Path) -> String {
    let output = Command::new("shasum")
        .arg("-a")
        .arg("256")
        .arg(archive)
        .output()
        .unwrap_or_else(|err| panic!("cannot run shasum for {}: {err}", archive.display()));
    assert!(output.status.success(), "shasum failed for {}", archive.display());
    String::from_utf8(output.stdout)
        .expect("shasum output must be UTF-8")
        .split_whitespace()
        .next()
        .expect("shasum output must contain a digest")
        .to_owned()
}

fn extract_archive(archive: &Path, destination: &Path) {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(destination)
        .arg("--strip-components=1")
        .status()
        .unwrap_or_else(|err| panic!("cannot run tar to extract {}: {err}", archive.display()));
    if !status.success() {
        panic!("tar failed while extracting {}", archive.display());
    }
}

fn is_compatible_mlx_source(root: &Path) -> bool {
    has_file(root, "mlx/backend/metal/kernels/unary_ops.h")
        && has_file(root, "mlx/backend/metal/kernels/binary_ops.h")
        && has_file(root, "mlx/backend/metal/kernels/softmax.h")
        && has_file(root, "mlx/backend/metal/kernels/metal_3_1/bf16.h")
        && has_compatible_quantized_headers(root)
}

fn has_file(root: &Path, rel_path: &str) -> bool {
    root.join(rel_path).exists()
}

fn has_compatible_quantized_headers(root: &Path) -> bool {
    let quantized = root.join("mlx/backend/metal/kernels/quantized.h");
    let Ok(content) = fs::read_to_string(quantized) else {
        return false;
    };
    content.contains("[[kernel]] void qmv_quad(") && content.contains("[[kernel]] void qmm_t(")
}
