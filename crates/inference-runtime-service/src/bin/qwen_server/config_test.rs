use clap::Parser;
use inference_runtime_core::Error;

use super::Qwen3Config;
use super::Qwen3ModelMode;
use super::Qwen35Config;
use super::Qwen35ModelMode;
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
    assert_eq!(config.max_queued_requests(), 32);
    assert_eq!(config.max_running_requests(), 8);
}

#[test]
fn test_scheduler_rejects_per_request_token_capacity_above_batch_capacity() {
    assert!(matches!(
        Qwen3Config::from_args(parse_qwen3(&["--max-tokens", "3", "--max-tokens-per-request", "4"])),
        Err(Error::InvalidArgument(message))
            if message.contains("--max-tokens-per-request=4 must not exceed --max-tokens=3")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--max-tokens", "3", "--max-tokens-per-request", "4"])),
        Err(Error::InvalidArgument(message))
            if message.contains("--max-tokens-per-request=4 must not exceed --max-tokens=3")
    ));
}

#[test]
fn test_qwen35_request_slots_follow_max_requests_for_all_spec_modes() {
    let main = Qwen35Config::from_args(parse_qwen35(&["--max-requests", "3"])).unwrap();
    let mtp = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-mtp-model-dir",
        "mtp-model",
        "--max-requests",
        "3",
    ]))
    .unwrap();
    let dspark = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-dspark-model-dir",
        "dspark-model",
        "--max-requests",
        "3",
    ]))
    .unwrap();

    for config in [&main, &mtp, &dspark] {
        assert_eq!(config.scheduler_config().max_requests, 3);
        assert_eq!(config.max_queued_requests(), 32);
        assert_eq!(config.max_running_requests(), 3);
    }
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
fn test_mtp_defaults_to_one_spec_token_and_two_cache_lanes() {
    let main_only = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    assert_eq!(main_only.num_cache_lanes(), 1);
    assert_eq!(main_only.model_mode(), &Qwen35ModelMode::Vanilla);

    let mtp = Qwen35Config::from_args(parse_qwen35(&["--hf-mtp-model-dir", "mtp-model"])).unwrap();
    assert_eq!(mtp.num_cache_lanes(), 2);
    assert_eq!(
        mtp.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: std::num::NonZeroUsize::MIN,
        }
    );

    let four_steps = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-mtp-model-dir",
        "mtp-model",
        "--num-spec-tokens",
        "4",
    ]))
    .unwrap();
    assert_eq!(four_steps.num_cache_lanes(), 5);
    assert_eq!(
        four_steps.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: std::num::NonZeroUsize::new(4).unwrap(),
        }
    );
}

#[test]
fn test_dspark_inputs_normalize_to_one_model_mode() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&["--hf-dspark-model-dir", "qwen3-dspark"])).unwrap();
    assert_eq!(
        qwen3.model_mode(),
        &Qwen3ModelMode::DSpark {
            model_dir: "qwen3-dspark".into(),
            num_spec_tokens: None,
        }
    );

    let qwen35 = Qwen35Config::from_args(parse_qwen35(&["--hf-dspark-model-dir", "qwen35-dspark"])).unwrap();
    assert_eq!(
        qwen35.model_mode(),
        &Qwen35ModelMode::DSpark {
            model_dir: "qwen35-dspark".into(),
            num_spec_tokens: None,
        }
    );
    assert_eq!(qwen35.num_cache_lanes(), 1);
}

#[test]
fn test_spec_token_validation() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--num-spec-tokens", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("--hf-mtp-model-dir or --hf-dspark-model-dir")
    ));
    assert!(matches!(
        Qwen3Config::from_args(parse_qwen3(&["--num-spec-tokens", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("--hf-dspark-model-dir")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-mtp-model-dir",
            "mtp-model",
            "--num-spec-tokens",
            "4",
            "--max-tokens-per-request",
            "3",
        ])),
        Err(Error::InvalidArgument(message)) if message.contains("--max-tokens-per-request=3")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-mtp-model-dir",
            "mtp-model",
            "--num-spec-tokens",
            "4",
            "--max-tokens",
            "4",
            "--max-tokens-per-request",
            "3",
        ])),
        Err(Error::InvalidArgument(message)) if message.contains("--max-tokens-per-request=3")
    ));
    let exact = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-mtp-model-dir",
        "mtp-model",
        "--num-spec-tokens",
        "4",
        "--max-tokens",
        "4",
        "--max-tokens-per-request",
        "4",
    ]))
    .unwrap();
    assert_eq!(exact.scheduler_config().max_tokens, 4);
    assert_eq!(exact.scheduler_config().max_tokens_per_request, 4);
}

#[test]
fn test_dspark_accepts_configured_spec_tokens_without_adding_cache_lanes() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&[
        "--hf-dspark-model-dir",
        "qwen3-dspark",
        "--num-spec-tokens",
        "4",
    ]))
    .unwrap();
    let qwen35 = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-dspark-model-dir",
        "qwen35-dspark",
        "--num-spec-tokens",
        "4",
    ]))
    .unwrap();
    let configured = Some(std::num::NonZeroUsize::new(4).unwrap());

    assert_eq!(
        qwen3.model_mode(),
        &Qwen3ModelMode::DSpark {
            model_dir: "qwen3-dspark".into(),
            num_spec_tokens: configured,
        }
    );
    assert_eq!(
        qwen35.model_mode(),
        &Qwen35ModelMode::DSpark {
            model_dir: "qwen35-dspark".into(),
            num_spec_tokens: configured,
        }
    );
    assert_eq!(qwen35.num_cache_lanes(), 1);
}

#[test]
fn test_dspark_proposal_length_is_independent_of_main_request_budget() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&[
        "--hf-dspark-model-dir",
        "qwen3-dspark",
        "--num-spec-tokens",
        "7",
        "--max-tokens-per-request",
        "2",
    ]))
    .unwrap();
    let qwen35 = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-dspark-model-dir",
        "qwen35-dspark",
        "--num-spec-tokens",
        "7",
        "--max-tokens-per-request",
        "2",
    ]))
    .unwrap();

    assert!(matches!(qwen3.model_mode(), Qwen3ModelMode::DSpark { .. }));
    assert!(matches!(qwen35.model_mode(), Qwen35ModelMode::DSpark { .. }));
}

#[test]
fn test_qwen35_dspark_and_mtp_are_mutually_exclusive() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-mtp-model-dir",
            "mtp-model",
            "--hf-dspark-model-dir",
            "dspark-model",
        ])),
        Err(Error::InvalidArgument(message)) if message.contains("mutually exclusive")
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
    assert_eq!(config.max_queued_requests(), 32);
    assert_eq!(config.max_running_requests(), 8);
}

#[test]
fn test_qwen3_request_slots_follow_max_requests_without_a_hidden_limit() {
    let config = Qwen3Config::from_args(parse_qwen3(&["--max-requests", "9"])).unwrap();

    assert_eq!(config.scheduler_config().max_requests, 9);
    assert_eq!(config.max_running_requests(), 9);
}

#[test]
fn test_qwen35_request_slots_follow_max_requests_without_a_hidden_limit() {
    let config = Qwen35Config::from_args(parse_qwen35(&["--max-requests", "9"])).unwrap();

    assert_eq!(config.scheduler_config().max_requests, 9);
    assert_eq!(config.max_running_requests(), 9);
}
