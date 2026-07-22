use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::Parser;
use clap::ValueEnum;

#[derive(Debug, Parser)]
pub struct Qwen3Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[arg(long, help = "Total shared cache pages used by the KV/state cache")]
    pub num_cache_pages: Option<usize>,
}

#[derive(Debug, Parser)]
pub struct Qwen35Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[arg(long, value_name = "DIR", help = "Optional Qwen3.5 HF MTP model directory")]
    pub hf_mtp_model_dir: Option<PathBuf>,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        help = "Number of Qwen3.5 MTP modules to enable; defaults to 1 when --hf-mtp-model-dir is set, otherwise 0"
    )]
    pub mtp_module: Option<usize>,

    #[arg(long, help = "Total shared cache pages used by GQA KV cache and GDN state cache")]
    pub num_cache_pages: Option<NonZeroUsize>,

    #[arg(
        long,
        default_value = "4",
        help = "Maximum resident request slots and requests scheduled per batch"
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
pub enum QwenLogLevel {
    Info,
    Debug,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::super::config::Qwen35ServerConfig;
    use super::Qwen35Args;

    #[test]
    fn test_scheduler_defaults() {
        let args = Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model"]).unwrap();

        assert_eq!(args.max_requests.get(), 4);
        assert_eq!(args.max_tokens.get(), 128);
        assert_eq!(args.max_tokens_per_request.get(), 64);
    }

    #[test]
    fn test_scheduler_overrides() {
        let config = Qwen35ServerConfig::from_args(
            Qwen35Args::try_parse_from([
                "qwen3.5",
                "--hf-model-dir",
                "model",
                "--max-requests",
                "8",
                "--max-tokens",
                "256",
                "--max-tokens-per-request",
                "32",
            ])
            .unwrap(),
        );
        let scheduler = config.scheduler_config();

        assert_eq!(scheduler.max_requests, 8);
        assert_eq!(scheduler.max_tokens, 256);
        assert_eq!(scheduler.max_tokens_per_request, 32);
    }

    #[test]
    fn test_positive_capacities_reject_zero() {
        for flag in [
            "--num-cache-pages",
            "--max-requests",
            "--max-tokens",
            "--max-tokens-per-request",
        ] {
            assert!(
                Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model", flag, "0"]).is_err(),
                "{flag} must reject zero"
            );
        }
    }
}
