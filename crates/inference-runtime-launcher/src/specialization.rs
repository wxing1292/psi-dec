use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use inference_error::Result;
use inference_error::log_err_unavailable;

/// Owns one generated worker package. Cargo owns dependency and artifact freshness.
pub struct SpecializedWorker {
    binary: String,
    source: String,
}

impl SpecializedWorker {
    pub fn new(binary: impl Into<String>, source: impl Into<String>) -> Self {
        let binary = binary.into();
        assert!(
            !binary.is_empty()
                && binary
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "worker name must be a Cargo package name and one path component"
        );
        Self {
            binary,
            source: source.into(),
        }
    }

    pub fn exec<I, S>(self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let worker = self.build()?;
        let mut command = Command::new(&worker);
        command.args(args);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;

            let error = command.exec();
            Err(log_err_unavailable!(
                "unable to exec specialized worker at {worker:?}: {error}"
            ))
        }
        #[cfg(not(unix))]
        {
            let status = command
                .status()
                .map_err(|error| log_err_unavailable!("unable to run specialized worker at {worker:?}: {error}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(log_err_unavailable!(
                    "specialized worker at {worker:?} exited with {status}"
                ))
            }
        }
    }

    fn build(&self) -> Result<PathBuf> {
        let workspace_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let profile_dir = Path::new(env!("INFERENCE_LAUNCHER_PROFILE_DIR"));
        let target = env!("INFERENCE_LAUNCHER_TARGET");
        let build_dir = profile_dir.parent().unwrap();
        let explicit_target = build_dir.file_name() == Some(OsStr::new(target));
        let target_dir = if explicit_target {
            build_dir.parent().unwrap()
        } else {
            build_dir
        };
        let project_dir = target_dir.join("specialized").join(&self.binary);
        fs::create_dir_all(&project_dir)
            .map_err(|error| log_err_unavailable!("unable to create worker package at {project_dir:?}: {error}"))?;
        // Serialize generation and compilation of the same package. Different L values share Cargo's build lock.
        let lock = fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(project_dir.join("build.lock"))
            .and_then(|file| {
                file.lock()?;
                Ok(file)
            })
            .map_err(|error| log_err_unavailable!("unable to lock worker package at {project_dir:?}: {error}"))?;
        self.prepare(&project_dir, workspace_dir)?;
        let profile = profile_dir.file_name().unwrap();
        let profile = if profile == "debug" { OsStr::new("dev") } else { profile };
        let mut cargo = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        cargo
            .current_dir(workspace_dir)
            .arg("build")
            .arg("--manifest-path")
            .arg(project_dir.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(target_dir)
            .arg("--profile")
            .arg(profile)
            .arg("--bin")
            .arg(&self.binary);
        if explicit_target {
            cargo.arg("--target").arg(target);
        }
        let status = cargo
            .status()
            .map_err(|error| log_err_unavailable!("unable to build specialized worker {}: {error}", self.binary))?;
        if !status.success() {
            return Err(log_err_unavailable!(
                "specialized worker {} build exited with {status}",
                self.binary
            ));
        }
        drop(lock);
        Ok(profile_dir.join(format!("{}{}", self.binary, std::env::consts::EXE_SUFFIX)))
    }

    fn prepare(&self, project_dir: &Path, workspace_dir: &Path) -> Result<()> {
        let prepare = || -> std::result::Result<(), Box<dyn std::error::Error>> {
            let workspace: toml::Value = toml::from_str(&fs::read_to_string(workspace_dir.join("Cargo.toml"))?)?;
            let service_dir = workspace_dir.join("crates/inference-runtime-service");
            let mut manifest = toml::toml! {
                [package]
                name = (self.binary.clone())
                version = "0.0.0"
                edition = "2024"
                publish = false
                [workspace]
                resolver = "2"
                [dependencies.inference-runtime-service]
                path = (service_dir.to_str().unwrap())
            };
            // Profiles belong to the source workspace, including package-specific overrides.
            if let Some(profile) = workspace.get("profile") {
                manifest.insert("profile".to_owned(), profile.clone());
            }
            // Cargo reads overrides only from the root manifest. The worker is a separate workspace.
            for key in ["patch", "replace"] {
                if let Some(overrides) = workspace.get(key) {
                    let mut overrides = overrides.clone();
                    if key == "patch" {
                        if let Some(sources) = overrides.as_table_mut() {
                            for (_, dependencies) in sources.iter_mut() {
                                resolve_override_paths(dependencies, workspace_dir);
                            }
                        }
                    } else {
                        resolve_override_paths(&mut overrides, workspace_dir);
                    }
                    manifest.insert(key.to_owned(), overrides);
                }
            }
            fs::create_dir_all(project_dir.join("src"))?;
            write_if_changed(&project_dir.join("Cargo.toml"), toml::to_string(&manifest)?.as_bytes())?;
            write_if_changed(&project_dir.join("src/main.rs"), self.source.as_bytes())?;
            // Cargo prunes unrelated workspace packages from the worker lockfile. Seed it again only when the
            // source lockfile changes, so switching workers retains the exact workspace dependency versions.
            let workspace_lock = fs::read(workspace_dir.join("Cargo.lock"))?;
            let previous_workspace_lock = match fs::read(project_dir.join("workspace.lock")) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.into()),
            };
            if !project_dir.join("Cargo.lock").try_exists()?
                || previous_workspace_lock.as_deref() != Some(workspace_lock.as_slice())
            {
                fs::write(project_dir.join("Cargo.lock"), &workspace_lock)?;
                fs::write(project_dir.join("workspace.lock"), &workspace_lock)?;
            }
            Ok(())
        };
        prepare().map_err(|error| log_err_unavailable!("unable to prepare worker package at {project_dir:?}: {error}"))
    }
}

fn resolve_override_paths(dependencies: &mut toml::Value, workspace_dir: &Path) {
    if let Some(dependencies) = dependencies.as_table_mut() {
        for (_, dependency) in dependencies.iter_mut() {
            if let Some(toml::Value::String(path)) = dependency.get_mut("path") {
                *path = workspace_dir.join(&*path).to_str().unwrap().to_owned();
            }
        }
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> io::Result<()> {
    match fs::read(path) {
        Ok(current) if current == bytes => Ok(()),
        Ok(_) => fs::write(path, bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => fs::write(path, bytes),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_reuses_package_and_refreshes_inputs() {
        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace with spaces");
        let project = root.path().join("worker");
        fs::create_dir_all(&workspace).unwrap();
        fs::write(
            workspace.join("Cargo.toml"),
            "[profile.release]\nlto = 'thin'\ncodegen-units = 1\n[replace]\n'local-dependency:0.1.0' = { path = \
             'local dependency' }\n",
        )
        .unwrap();
        fs::write(workspace.join("Cargo.lock"), "version = 4\n").unwrap();
        let worker = SpecializedWorker::new("worker_lanes_3", "fn main() {}\n");
        worker.prepare(&project, &workspace).unwrap();
        let manifest: toml::Value = toml::from_str(&fs::read_to_string(project.join("Cargo.toml")).unwrap()).unwrap();
        assert_eq!(manifest["profile"]["release"]["lto"].as_str(), Some("thin"));
        assert_eq!(
            manifest["replace"]["local-dependency:0.1.0"]["path"].as_str(),
            workspace.join("local dependency").to_str()
        );
        assert_eq!(
            manifest["dependencies"]["inference-runtime-service"]["path"].as_str(),
            workspace.join("crates/inference-runtime-service").to_str()
        );
        let source_time = fs::metadata(project.join("src/main.rs")).unwrap().modified().unwrap();
        let manifest_time = fs::metadata(project.join("Cargo.toml")).unwrap().modified().unwrap();
        fs::write(project.join("Cargo.lock"), "pruned lockfile").unwrap();
        worker.prepare(&project, &workspace).unwrap();
        assert_eq!(
            fs::metadata(project.join("src/main.rs")).unwrap().modified().unwrap(),
            source_time
        );
        assert_eq!(
            fs::metadata(project.join("Cargo.toml")).unwrap().modified().unwrap(),
            manifest_time
        );
        assert_eq!(
            fs::read_to_string(project.join("Cargo.lock")).unwrap(),
            "pruned lockfile"
        );
        fs::remove_file(project.join("Cargo.lock")).unwrap();
        worker.prepare(&project, &workspace).unwrap();
        assert_eq!(
            fs::read(project.join("Cargo.lock")).unwrap(),
            fs::read(workspace.join("Cargo.lock")).unwrap()
        );
        fs::write(workspace.join("Cargo.lock"), "version = 4\n# updated\n").unwrap();
        let updated = SpecializedWorker::new("worker_lanes_3", "fn main() { println!(\"updated\"); }\n");
        updated.prepare(&project, &workspace).unwrap();
        assert_eq!(fs::read_to_string(project.join("src/main.rs")).unwrap(), updated.source);
        assert_eq!(
            fs::read(project.join("Cargo.lock")).unwrap(),
            fs::read(workspace.join("Cargo.lock")).unwrap()
        );
    }

    #[test]
    fn test_worker_uses_workspace_dependency_override() {
        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace with spaces");
        let project = root.path().join("worker");
        let service = workspace.join("crates/inference-runtime-service");
        let dependency = workspace.join("patched dependency");
        fs::create_dir_all(service.join("src")).unwrap();
        fs::create_dir_all(dependency.join("src")).unwrap();
        fs::write(
            workspace.join("Cargo.toml"),
            "[workspace]\nmembers = ['crates/inference-runtime-service']\nresolver = \
             '2'\n[patch.crates-io]\nstartup-audit-dependency = { path = 'patched dependency' }\n",
        )
        .unwrap();
        fs::write(workspace.join("Cargo.lock"), "version = 4\n").unwrap();
        fs::write(
            service.join("Cargo.toml"),
            "[package]\nname = 'inference-runtime-service'\nversion = '0.1.0'\nedition = \
             '2024'\n[dependencies]\nstartup-audit-dependency = '=0.1.0'\n",
        )
        .unwrap();
        fs::write(service.join("src/lib.rs"), "pub use startup_audit_dependency::value;\n").unwrap();
        fs::write(
            dependency.join("Cargo.toml"),
            "[package]\nname = 'startup-audit-dependency'\nversion = '0.1.0'\nedition = '2024'\n",
        )
        .unwrap();
        fs::write(dependency.join("src/lib.rs"), "pub fn value() -> u8 { 42 }\n").unwrap();
        let worker = SpecializedWorker::new(
            "worker_lanes_3",
            "fn main() { assert_eq!(inference_runtime_service::value(), 42); }\n",
        );
        worker.prepare(&project, &workspace).unwrap();
        let output = Command::new(env!("CARGO"))
            .current_dir(&workspace)
            .arg("run")
            .arg("--offline")
            .arg("--manifest-path")
            .arg(project.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(root.path().join("target"))
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
}
