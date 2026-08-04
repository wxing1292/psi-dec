use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use escargot::CargoBuild;
use inference_runtime_core::Result;
use inference_runtime_core::log_err_unavailable;

/// Builds and replaces the launcher with one const-specialized worker.
pub struct SpecializedWorker {
    manifest_path: PathBuf,
    binary: OsString,
    target_dir: PathBuf,
    build_env: Vec<(OsString, OsString)>,
    run_env: Vec<(OsString, OsString)>,
}

impl SpecializedWorker {
    pub fn new(manifest_path: impl Into<PathBuf>, binary: impl Into<OsString>, target_dir: impl Into<PathBuf>) -> Self {
        Self {
            manifest_path: manifest_path.into(),
            binary: binary.into(),
            target_dir: target_dir.into(),
            build_env: Vec::new(),
            run_env: Vec::new(),
        }
    }

    pub fn build_env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.build_env
            .push((key.as_ref().to_owned(), value.as_ref().to_owned()));
        self
    }

    pub fn run_env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.run_env.push((key.as_ref().to_owned(), value.as_ref().to_owned()));
        self
    }

    pub fn exec<I, S>(self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let Self {
            manifest_path,
            binary,
            target_dir,
            build_env,
            run_env,
        } = self;
        let binary_label = binary.to_string_lossy();
        let mut build = CargoBuild::new()
            .manifest_path(manifest_path)
            .bin(&binary)
            .current_release()
            .current_target()
            .target_dir(target_dir);
        for (key, value) in build_env {
            build = build.env(key, value);
        }
        let worker = build
            .run()
            .map_err(|error| log_err_unavailable!("unable to build specialized worker {binary_label}: {error}"))?;
        let mut command = Command::new(worker.path());
        command.args(args);
        for (key, value) in run_env {
            command.env(key, value);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;

            let error = command.exec();
            Err(log_err_unavailable!(
                "unable to exec specialized worker {binary_label} at {:?}: {error}",
                worker.path()
            ))
        }
        #[cfg(not(unix))]
        {
            let status = command.status().map_err(|error| {
                log_err_unavailable!(
                    "unable to run specialized worker {binary_label} at {:?}: {error}",
                    worker.path()
                )
            })?;
            if status.success() {
                Ok(())
            } else {
                Err(log_err_unavailable!(
                    "specialized worker {binary_label} exited with {status}"
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::Path;

    use super::SpecializedWorker;

    #[test]
    fn test_configuration_owns_model_specific_build_inputs() {
        let worker = SpecializedWorker::new("worker/Cargo.toml", "worker", "target/specialized/key")
            .build_env("CONST_A", "3")
            .build_env("CONST_B", "8")
            .run_env("WORKER_MODE", "1");

        assert_eq!(worker.manifest_path, Path::new("worker/Cargo.toml"));
        assert_eq!(worker.binary, OsStr::new("worker"));
        assert_eq!(worker.target_dir, Path::new("target/specialized/key"));
        assert_eq!(
            worker.build_env,
            [("CONST_A".into(), "3".into()), ("CONST_B".into(), "8".into())]
        );
        assert_eq!(worker.run_env, [("WORKER_MODE".into(), "1".into())]);
    }
}
