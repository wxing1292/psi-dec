use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Parser;
use inference_runtime_core::Error;
use inference_runtime_core::config::ExecutorHibernationMode;

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
fn test_executor_hibernation_timeout() {
    let qwen3_default = Qwen3Config::from_args(parse_qwen3(&[])).unwrap();
    let qwen35_default = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    let qwen3_override = Qwen3Config::from_args(parse_qwen3(&["--executor-hibernation-timeout-secs", "17"])).unwrap();
    let qwen35_override =
        Qwen35Config::from_args(parse_qwen35(&["--executor-hibernation-timeout-secs", "17"])).unwrap();

    assert_eq!(qwen3_default.executor_hibernation_timeout(), Duration::from_secs(300));
    assert_eq!(qwen35_default.executor_hibernation_timeout(), Duration::from_secs(300));
    assert_eq!(qwen3_override.executor_hibernation_timeout(), Duration::from_secs(17));
    assert_eq!(qwen35_override.executor_hibernation_timeout(), Duration::from_secs(17));
}

#[test]
fn test_executor_hibernation_mode() {
    let qwen3_default = Qwen3Config::from_args(parse_qwen3(&[])).unwrap();
    let qwen35_default = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    let qwen3_all = Qwen3Config::from_args(parse_qwen3(&["--executor-hibernation-mode", "all"])).unwrap();
    let qwen35_all = Qwen35Config::from_args(parse_qwen35(&["--executor-hibernation-mode", "all"])).unwrap();

    assert_eq!(
        qwen3_default.executor_hibernation_mode(),
        ExecutorHibernationMode::Selected
    );
    assert_eq!(
        qwen35_default.executor_hibernation_mode(),
        ExecutorHibernationMode::Selected
    );
    assert_eq!(qwen3_all.executor_hibernation_mode(), ExecutorHibernationMode::All);
    assert_eq!(qwen35_all.executor_hibernation_mode(), ExecutorHibernationMode::All);
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
fn test_request_capacity_follows_max_requests() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&["--max-requests", "9"])).unwrap();
    assert_eq!(qwen3.scheduler_config().max_requests, 9);
    assert_eq!(qwen3.max_running_requests(), 9);
    assert_eq!(qwen3.max_queued_requests(), 32);

    let qwen35_configs = [
        Qwen35Config::from_args(parse_qwen35(&["--max-requests", "9"])).unwrap(),
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "mtp-model",
            "--spec-type",
            "mtp",
            "--max-requests",
            "9",
        ]))
        .unwrap(),
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "dspark-model",
            "--spec-type",
            "dspark",
            "--max-requests",
            "9",
        ]))
        .unwrap(),
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "dflash2-model",
            "--spec-type",
            "dflash2",
            "--max-requests",
            "9",
        ]))
        .unwrap(),
    ];
    for config in qwen35_configs {
        assert_eq!(config.scheduler_config().max_requests, 9);
        assert_eq!(config.max_running_requests(), 9);
        assert_eq!(config.max_queued_requests(), 32);
    }
}

#[test]
fn test_mtp_inputs_normalize_model_mode_and_cache_lanes() {
    let main = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    assert_eq!(main.model_mode(), &Qwen35ModelMode::Vanilla);
    assert_eq!(main.num_cache_lanes(), 1);

    let default_mtp = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
    ]))
    .unwrap();
    assert_eq!(
        default_mtp.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: NonZeroUsize::MIN,
        }
    );
    assert_eq!(default_mtp.num_cache_lanes(), 2);

    let four_steps = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
        "--num-spec-tokens",
        "4",
    ]))
    .unwrap();
    assert_eq!(
        four_steps.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: NonZeroUsize::new(4).unwrap(),
        }
    );
    assert_eq!(four_steps.num_cache_lanes(), 5);
}

#[test]
fn test_dspark_inputs_normalize_model_mode_and_cache_lanes() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&[
        "--hf-spec-model-dir",
        "qwen3-dspark",
        "--spec-type",
        "dspark",
    ]))
    .unwrap();
    assert_eq!(
        qwen3.model_mode(),
        &Qwen3ModelMode::DSpark {
            model_dir: "qwen3-dspark".into(),
        }
    );

    let qwen35 = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "qwen35-dspark",
        "--spec-type",
        "dspark",
    ]))
    .unwrap();
    assert_eq!(
        qwen35.model_mode(),
        &Qwen35ModelMode::DSpark {
            model_dir: "qwen35-dspark".into(),
        }
    );
    assert_eq!(qwen35.num_cache_lanes(), 1);
}

#[test]
fn test_dflash2_inputs_normalize_model_mode_and_cache_lanes() {
    let qwen35 = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "qwen35-dflash2",
        "--spec-type",
        "dflash2",
    ]))
    .unwrap();
    assert_eq!(
        qwen35.model_mode(),
        &Qwen35ModelMode::DFlash2 {
            model_dir: "qwen35-dflash2".into(),
        }
    );
    assert_eq!(qwen35.num_cache_lanes(), 1);
}

#[test]
fn test_spec_tokens_require_a_speculator() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--num-spec-tokens", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("requires --spec-type mtp")
    ));
}

#[test]
fn test_mtp_spec_tokens_fit_per_request_budget() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "mtp-model",
            "--spec-type",
            "mtp",
            "--num-spec-tokens",
            "4",
            "--max-tokens-per-request",
            "3",
        ])),
        Err(Error::InvalidArgument(message)) if message.contains("--max-tokens-per-request=3")
    ));
    Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
        "--num-spec-tokens",
        "4",
        "--max-tokens-per-request",
        "4",
    ]))
    .unwrap();
}

#[test]
fn test_num_spec_tokens_rejects_checkpoint_defined_block_spec_modes() {
    for spec_type in ["dspark", "dflash2"] {
        assert!(matches!(
            Qwen35Config::from_args(parse_qwen35(&[
                "--hf-spec-model-dir",
                "spec-model",
                "--spec-type",
                spec_type,
                "--num-spec-tokens",
                "7",
            ])),
            Err(Error::InvalidArgument(message)) if message.contains("controls MTP proposal length")
        ));
    }
}

#[test]
fn test_spec_checkpoint_arguments_must_be_specified_together() {
    for incomplete_args in [["--hf-spec-model-dir", "spec-model"], ["--spec-type", "mtp"]] {
        assert!(matches!(
            Qwen35Config::from_args(parse_qwen35(&incomplete_args)),
            Err(Error::InvalidArgument(message)) if message.contains("must be specified together")
        ));
    }
    for incomplete_args in [["--hf-spec-model-dir", "spec-model"], ["--spec-type", "dspark"]] {
        assert!(matches!(
            Qwen3Config::from_args(parse_qwen3(&incomplete_args)),
            Err(Error::InvalidArgument(message)) if message.contains("must be specified together")
        ));
    }
}

#[test]
fn test_qwen3_rejects_unsupported_spec_types() {
    for spec_type in ["mtp", "dflash2"] {
        assert!(matches!(
            Qwen3Config::from_args(parse_qwen3(&[
                "--hf-spec-model-dir",
                "spec-model",
                "--spec-type",
                spec_type,
            ])),
            Err(Error::InvalidArgument(message)) if message.contains("supports only --spec-type dspark")
        ));
    }
}
