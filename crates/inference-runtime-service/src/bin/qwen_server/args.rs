use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::Parser;
use clap::ValueEnum;

#[derive(Debug, Parser)]
pub struct Qwen3Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[arg(long, value_name = "DIR", help = "Optional official Qwen3 DSpark model directory")]
    pub hf_dspark_model_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Number of speculative tokens per DSpark proposal; defaults to the checkpoint block_size and must not \
                exceed it"
    )]
    pub num_spec_tokens: Option<NonZeroUsize>,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "262144",
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

    #[arg(
        long,
        value_name = "DIR",
        help = "Optional matching Qwen3.5/Qwen3.6 MTP checkpoint directory"
    )]
    pub hf_mtp_model_dir: Option<PathBuf>,

    #[arg(
        long,
        value_name = "DIR",
        help = "Optional matching Qwen3x DSpark checkpoint directory"
    )]
    pub hf_dspark_model_dir: Option<PathBuf>,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        help = "Number of speculative tokens per proposal; defaults to 1 for MTP or the checkpoint block_size for \
                DSpark; a DSpark value must not exceed that block_size"
    )]
    pub num_spec_tokens: Option<NonZeroUsize>,

    #[arg(
        long,
        default_value = "262144",
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
pub enum QwenLogLevel {
    Info,
    Debug,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Qwen3Args;
    use super::Qwen35Args;

    #[test]
    fn test_scheduler_defaults() {
        let args = Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model"]).unwrap();

        assert_eq!(args.max_requests.get(), 4);
        assert_eq!(args.max_tokens.get(), 128);
        assert_eq!(args.max_tokens_per_request.get(), 64);
        assert_eq!(args.num_cache_pages.get(), 262_144);
    }

    #[test]
    fn test_qwen3_defaults() {
        let args = Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model"]).unwrap();

        assert_eq!(args.grpc_listen_addr, "127.0.0.1:50051".parse().unwrap());
        assert_eq!(args.http_listen_addr, "127.0.0.1:8000".parse().unwrap());
        assert_eq!(args.max_requests.get(), 4);
        assert_eq!(args.max_tokens.get(), 128);
        assert_eq!(args.max_tokens_per_request.get(), 64);
        assert_eq!(args.num_cache_pages.get(), 262_144);
        assert_eq!(args.hf_dspark_model_dir, None);
        assert_eq!(args.num_spec_tokens, None);
    }

    #[test]
    fn test_qwen3_accepts_dspark_checkpoint() {
        let args =
            Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", "--hf-dspark-model-dir", "dspark"]).unwrap();

        assert_eq!(args.hf_dspark_model_dir, Some("dspark".into()));
    }

    #[test]
    fn test_qwen35_accepts_dspark_checkpoint() {
        let args =
            Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model", "--hf-dspark-model-dir", "dspark"])
                .unwrap();

        assert_eq!(args.hf_dspark_model_dir, Some("dspark".into()));
    }

    #[test]
    fn test_spec_token_flag_is_shared_by_mtp_and_dspark() {
        let qwen3_dspark = Qwen3Args::try_parse_from([
            "qwen3",
            "--hf-model-dir",
            "model",
            "--hf-dspark-model-dir",
            "dspark",
            "--num-spec-tokens",
            "3",
        ])
        .unwrap();
        let qwen35_mtp = Qwen35Args::try_parse_from([
            "qwen3.5",
            "--hf-model-dir",
            "model",
            "--hf-mtp-model-dir",
            "mtp",
            "--num-spec-tokens",
            "3",
        ])
        .unwrap();
        let qwen35_dspark = Qwen35Args::try_parse_from([
            "qwen3.5",
            "--hf-model-dir",
            "model",
            "--hf-dspark-model-dir",
            "dspark",
            "--num-spec-tokens",
            "3",
        ])
        .unwrap();

        assert_eq!(qwen3_dspark.num_spec_tokens.unwrap().get(), 3);
        assert_eq!(qwen35_mtp.num_spec_tokens.unwrap().get(), 3);
        assert_eq!(qwen35_dspark.num_spec_tokens.unwrap().get(), 3);
    }

    #[test]
    fn test_qwen35_mtp_and_dspark_commands_share_common_capacity_args() {
        let common = [
            "--hf-model-dir",
            "main",
            "--max-requests",
            "3",
            "--max-tokens",
            "96",
            "--max-tokens-per-request",
            "48",
        ];
        let mut mtp_argv = vec!["qwen3.5"];
        mtp_argv.extend(common);
        mtp_argv.extend(["--hf-mtp-model-dir", "mtp"]);
        let mut dspark_argv = vec!["qwen3.5"];
        dspark_argv.extend(common);
        dspark_argv.extend(["--hf-dspark-model-dir", "dspark"]);

        let mtp = Qwen35Args::try_parse_from(mtp_argv).unwrap();
        let dspark = Qwen35Args::try_parse_from(dspark_argv).unwrap();

        assert_eq!(mtp.max_requests, dspark.max_requests);
        assert_eq!(mtp.max_tokens, dspark.max_tokens);
        assert_eq!(mtp.max_tokens_per_request, dspark.max_tokens_per_request);
        assert_eq!(mtp.hf_mtp_model_dir, Some("mtp".into()));
        assert_eq!(mtp.hf_dspark_model_dir, None);
        assert_eq!(dspark.hf_mtp_model_dir, None);
        assert_eq!(dspark.hf_dspark_model_dir, Some("dspark".into()));
        assert_eq!(mtp.num_spec_tokens, None);
        assert_eq!(dspark.num_spec_tokens, None);
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
            assert!(
                Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", flag, "0"]).is_err(),
                "{flag} must reject zero for Qwen3"
            );
        }
        assert!(Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", "--num-spec-tokens", "0"]).is_err());
        assert!(Qwen35Args::try_parse_from(["qwen3.5", "--hf-model-dir", "model", "--num-spec-tokens", "0"]).is_err());
    }
}
