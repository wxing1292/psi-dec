use clap::Parser;
use inference_runtime_core::Error;

use super::Qwen35Config;
use crate::qwen_server::args::Qwen35Args;

fn parse(extra: &[&str]) -> Qwen35Args {
    let mut args = vec!["qwen3.5", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen35Args::try_parse_from(args).unwrap()
}

#[test]
fn test_scheduler_overrides() {
    let config = Qwen35Config::from_args(parse(&[
        "--max-requests",
        "8",
        "--max-tokens",
        "256",
        "--max-tokens-per-request",
        "32",
    ]))
    .unwrap();
    let scheduler = config.scheduler_config();

    assert_eq!(scheduler.max_requests, 8);
    assert_eq!(scheduler.max_tokens, 256);
    assert_eq!(scheduler.max_tokens_per_request, 32);
}

#[test]
fn test_listener_overrides() {
    let config = Qwen35Config::from_args(parse(&[
        "--grpc-listen-addr",
        "127.0.0.1:50061",
        "--http-listen-addr",
        "127.0.0.1:8001",
    ]))
    .unwrap();

    assert_eq!(config.grpc_listen_addr(), "127.0.0.1:50061".parse().unwrap());
    assert_eq!(config.http_listen_addr(), "127.0.0.1:8001".parse().unwrap());
}

#[test]
fn test_mtp_defaults_from_checkpoint_presence() {
    let target_only = Qwen35Config::from_args(parse(&[])).unwrap();
    assert_eq!(target_only.num_mtp_modules(), 0);

    let mtp = Qwen35Config::from_args(parse(&["--hf-mtp-model-dir", "mtp-model"])).unwrap();
    assert_eq!(mtp.num_mtp_modules(), 1);

    let disabled = Qwen35Config::from_args(parse(&["--hf-mtp-model-dir", "mtp-model", "--mtp-module", "0"])).unwrap();
    assert_eq!(disabled.num_mtp_modules(), 0);
}

#[test]
fn test_mtp_validation() {
    assert!(matches!(
        Qwen35Config::from_args(parse(&["--mtp-module", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("--hf-mtp-model-dir")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse(&["--mtp-module", "2"])),
        Err(Error::InvalidArgument(message)) if message.contains("must be 0 or 1")
    ));
}
