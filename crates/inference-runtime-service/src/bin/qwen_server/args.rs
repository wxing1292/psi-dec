use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::Args;
use clap::Parser;
use clap::ValueEnum;
use inference_runtime_core::config::ExecutorHibernationMode;

#[derive(Args, Debug)]
pub struct QwenSpecArgs {
    #[arg(
        long,
        value_name = "DIR",
        help = "Speculative checkpoint directory (requires --spec-type)"
    )]
    pub hf_spec_model_dir: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        help = "Speculative checkpoint type (requires --hf-spec-model-dir)"
    )]
    pub spec_type: Option<QwenSpecType>,
}

#[derive(Debug, Parser)]
pub struct Qwen3Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[command(flatten)]
    pub spec: QwenSpecArgs,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds without model execution before state and weights unload"
    )]
    pub executor_hibernation_timeout_secs: NonZeroU64,

    #[arg(
        long,
        default_value = "selected",
        value_name = "MODE",
        help = "Executor hibernation state scope: all or selected"
    )]
    pub executor_hibernation_mode: ExecutorHibernationMode,

    #[arg(
        long,
        default_value = "393216",
        help = "Total shared cache pages used by GQA KV cache and GDN state cache"
    )]
    pub num_cache_pages: NonZeroUsize,

    #[arg(
        long,
        default_value = "4",
        help = "Maximum running requests and requests scheduled per batch"
    )]
    pub max_requests: NonZeroUsize,

    #[arg(long, default_value = "128", help = "Maximum flattened tokens scheduled per batch")]
    pub max_tokens: NonZeroUsize,

    #[arg(
        long,
        default_value = "64",
        help = "Maximum tokens from one request in one forward transaction"
    )]
    pub max_tokens_per_request: NonZeroUsize,
}

#[derive(Debug, Parser)]
pub struct Qwen35Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[command(flatten)]
    pub spec: QwenSpecArgs,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds without model execution before state and weights unload"
    )]
    pub executor_hibernation_timeout_secs: NonZeroU64,

    #[arg(
        long,
        default_value = "selected",
        value_name = "MODE",
        help = "Executor hibernation state scope: all or selected"
    )]
    pub executor_hibernation_mode: ExecutorHibernationMode,

    #[arg(
        long,
        help = "Number of speculative tokens per MTP proposal; defaults to 1 and requires --spec-type mtp"
    )]
    pub num_spec_tokens: Option<NonZeroUsize>,

    #[arg(
        long,
        default_value = "393216",
        help = "Total shared cache pages used by GQA KV cache and GDN state cache"
    )]
    pub num_cache_pages: NonZeroUsize,

    #[arg(
        long,
        default_value = "4",
        help = "Maximum running requests and requests scheduled per batch"
    )]
    pub max_requests: NonZeroUsize,

    #[arg(long, default_value = "128", help = "Maximum flattened tokens scheduled per batch")]
    pub max_tokens: NonZeroUsize,

    #[arg(
        long,
        default_value = "64",
        help = "Maximum tokens from one request in one forward transaction"
    )]
    pub max_tokens_per_request: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QwenProfileMode {
    Component,
    Operation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[allow(clippy::upper_case_acronyms)]
pub enum QwenSpecType {
    #[value(name = "mtp")]
    MTP,
    #[value(name = "dspark")]
    DSpark,
    #[value(name = "dflash2")]
    DFlash2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QwenLogLevel {
    Info,
    Debug,
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use inference_runtime_core::config::ExecutorHibernationMode;

    use super::Qwen3Args;
    use super::Qwen35Args;

    #[test]
    fn test_scheduler_defaults() {
        let args = Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model"]).unwrap();

        assert_eq!(args.max_requests.get(), 4);
        assert_eq!(args.max_tokens.get(), 128);
        assert_eq!(args.max_tokens_per_request.get(), 64);
        assert_eq!(args.num_cache_pages.get(), 384 * 1024);
        assert_eq!(args.executor_hibernation_timeout_secs.get(), 300);
        assert_eq!(args.executor_hibernation_mode, ExecutorHibernationMode::Selected);
    }

    #[test]
    fn test_qwen3_defaults() {
        let args = Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model"]).unwrap();

        assert_eq!(args.grpc_listen_addr, "127.0.0.1:50051".parse().unwrap());
        assert_eq!(args.http_listen_addr, "127.0.0.1:8000".parse().unwrap());
        assert_eq!(args.max_requests.get(), 4);
        assert_eq!(args.max_tokens.get(), 128);
        assert_eq!(args.max_tokens_per_request.get(), 64);
        assert_eq!(args.num_cache_pages.get(), 384 * 1024);
        assert_eq!(args.spec.hf_spec_model_dir, None);
        assert_eq!(args.spec.spec_type, None);
        assert_eq!(args.executor_hibernation_timeout_secs.get(), 300);
        assert_eq!(args.executor_hibernation_mode, ExecutorHibernationMode::Selected);
    }

    #[test]
    fn test_executor_hibernation_mode_accepts_all() {
        let qwen3 =
            Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", "--executor-hibernation-mode", "all"])
                .unwrap();
        let qwen35 = Qwen35Args::try_parse_from([
            "qwen3.5",
            "--hf-model-dir",
            "model",
            "--executor-hibernation-mode",
            "all",
        ])
        .unwrap();

        assert_eq!(qwen3.executor_hibernation_mode, ExecutorHibernationMode::All);
        assert_eq!(qwen35.executor_hibernation_mode, ExecutorHibernationMode::All);
    }

    #[test]
    fn test_positive_capacities_reject_zero() {
        for flag in [
            "--executor-hibernation-timeout-secs",
            "--num-cache-pages",
            "--max-requests",
            "--max-tokens",
            "--max-tokens-per-request",
        ] {
            assert!(
                Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model", flag, "0"]).is_err(),
                "{flag} must reject zero"
            );
            assert!(
                Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", flag, "0"]).is_err(),
                "{flag} must reject zero for Qwen3"
            );
        }
        assert!(Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model", "--num-spec-tokens", "0"]).is_err());
    }
}
