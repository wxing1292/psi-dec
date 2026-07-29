use clap::Parser;
use inference_runtime_core::Error;

use super::QWEN3_MAX_RUNNING_REQUESTS;
use super::Qwen3Config;
use super::Qwen35Config;
use crate::qwen_server::args::Qwen3Args;
use crate::qwen_server::args::Qwen35Args;

fn parse_qwen35(extra: &[&str]) -> Qwen35Args {
    let mut args = vec!["qwen3.5", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen35Args::try_parse_from(args).unwrap()
}

fn parse_qwen3(extra: &[&str]) -> Qwen3Args {
    let mut args = vec!["qwen3", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen3Args::try_parse_from(args).unwrap()
}

#[test]
fn test_scheduler_overrides() {
    let config = Qwen35Config::from_args(parse_qwen35(&[
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
    let config = Qwen35Config::from_args(parse_qwen35(&[
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
    let main_only = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    assert_eq!(main_only.num_mtp_modules(), 0);

    let mtp = Qwen35Config::from_args(parse_qwen35(&["--hf-mtp-model-dir", "mtp-model"])).unwrap();
    assert_eq!(mtp.num_mtp_modules(), 1);

    let disabled =
        Qwen35Config::from_args(parse_qwen35(&["--hf-mtp-model-dir", "mtp-model", "--mtp-module", "0"])).unwrap();
    assert_eq!(disabled.num_mtp_modules(), 0);
}

#[test]
fn test_mtp_validation() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--mtp-module", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("--hf-mtp-model-dir")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--mtp-module", "2"])),
        Err(Error::InvalidArgument(message)) if message.contains("must be 0 or 1")
    ));
}

#[test]
fn test_qwen3_scheduler_and_listener_overrides() {
    let config = Qwen3Config::from_args(parse_qwen3(&[
        "--grpc-listen-addr",
        "127.0.0.1:50062",
        "--http-listen-addr",
        "127.0.0.1:8002",
        "--num-cache-pages",
        "40960",
        "--max-requests",
        "8",
        "--max-tokens",
        "256",
        "--max-tokens-per-request",
        "32",
    ]))
    .unwrap();
    let scheduler = config.scheduler_config();

    assert_eq!(config.grpc_listen_addr(), "127.0.0.1:50062".parse().unwrap());
    assert_eq!(config.http_listen_addr(), "127.0.0.1:8002".parse().unwrap());
    assert_eq!(config.num_cache_pages(), 40_960);
    assert_eq!(scheduler.max_requests, 8);
    assert_eq!(scheduler.max_tokens, 256);
    assert_eq!(scheduler.max_tokens_per_request, 32);
}

#[test]
fn test_qwen3_default_cache_pages() {
    assert_eq!(
        Qwen3Config::from_args(parse_qwen3(&[])).unwrap().num_cache_pages(),
        384 * 1024
    );
}

#[test]
fn test_qwen3_scheduler_cannot_exceed_executor_slots() {
    let requested_slots = (QWEN3_MAX_RUNNING_REQUESTS + 1).to_string();
    assert!(matches!(
        Qwen3Config::from_args(parse_qwen3(&["--max-requests", &requested_slots])),
        Err(Error::InvalidArgument(message)) if message.contains("exceeds the Qwen3 runtime capacity")
    ));
}
