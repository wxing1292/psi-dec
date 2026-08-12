use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_dspark;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_mtp;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::log_info_invalid_argument;

use crate::codec::qwen::QwenCodec;
use crate::qwen_server::args::Qwen35Args;
use crate::qwen_server::config::Qwen35Config;
use crate::qwen_server::config::Qwen35ModelMode;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::qwen_server::sizing::context_window;
use crate::qwen_server::sizing::kv_dtype_bytes;
use crate::runtime::serve_replay_model;
use crate::specialization::SpecializedWorker;
use crate::telemetry::CacheLaneLogSummary;
use crate::telemetry::StartupLogger;

const TOKENS_PER_CACHE_BLOCK: usize = 2048;
const QWEN35_DENSE_BIN: &str = "qwen3_5_dense";
const QWEN35_SPARSE_BIN: &str = "qwen3_5_sparse";
const QWEN35_BUILD_CACHE_LANES_ENV: &str = "PSI_QWEN35_CACHE_LANES";
const QWEN35_SPECIALIZED_RUN_ENV: &str = "PSI_QWEN35_SPECIALIZED_WORKER";
const QWEN35_SPECIALIZED_TARGET_DIR_NAME: &str = "qwen3_5_specialized";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelKind {
    Dense,
    Sparse,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            ModelKind::Dense => "qwen3.5 dense 27B",
            ModelKind::Sparse => "qwen3.5 sparse 35B-A3B",
        }
    }

    fn binary(self) -> &'static str {
        match self {
            ModelKind::Dense => QWEN35_DENSE_BIN,
            ModelKind::Sparse => QWEN35_SPARSE_BIN,
        }
    }
}

pub fn run_dense<const L: usize>() {
    run_or_launch::<L>(ModelKind::Dense);
}

pub fn run_sparse<const L: usize>() {
    run_or_launch::<L>(ModelKind::Sparse);
}

fn run_or_launch<const L: usize>(kind: ModelKind) {
    if std::env::var_os(QWEN35_SPECIALIZED_RUN_ENV).is_some() {
        run_worker::<L>(kind, Qwen35Args::parse());
    } else {
        launch_or_exit(kind);
    }
}

fn launch_or_exit(kind: ModelKind) {
    if let Err(error) = launch_worker(kind) {
        eprintln!("unable to start {}: {error}", kind.label());
        std::process::exit(1);
    }
}

fn launch_worker(kind: ModelKind) -> Result<()> {
    let config = Qwen35Config::from_args(Qwen35Args::parse())?;
    let num_cache_lanes = config.num_cache_lanes();
    let service_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worker_manifest = service_manifest_dir.join("Cargo.toml");
    let worker_target_dir = service_manifest_dir
        .join("../../target")
        .join(QWEN35_SPECIALIZED_TARGET_DIR_NAME)
        .join(format!("cache_lanes_{num_cache_lanes}"));
    SpecializedWorker::new(worker_manifest, kind.binary(), worker_target_dir)
        .build_env(QWEN35_BUILD_CACHE_LANES_ENV, num_cache_lanes.to_string())
        .run_env(QWEN35_SPECIALIZED_RUN_ENV, "1")
        .exec(std::env::args_os().skip(1))
}

fn run_worker<const L: usize>(kind: ModelKind, args: Qwen35Args) {
    if let Err(error) = run_service::<L>(kind, args) {
        eprintln!("unable to start {}: {error}", kind.label());
        std::process::exit(1);
    }
}

fn run_service<const L: usize>(kind: ModelKind, args: Qwen35Args) -> Result<()> {
    let config = Qwen35Config::from_args(args)?;
    assert!(L > 0, "qwen3.5 worker requires at least the Main cache lane");
    assert_eq!(
        L,
        config.num_cache_lanes(),
        "qwen3.5 worker const L must equal the configured cache-lane count"
    );
    let telemetry = config.telemetry_config();
    telemetry.init();
    let startup = StartupLogger::new(kind.label());

    startup.event("reading model config");
    let model_config = load_model_config(config.hf_model_dir())?;
    tracing::info!(
        target: "inference-runtime-service::startup",
        grpc_listen_addr = %config.grpc_listen_addr(),
        http_listen_addr = %config.http_listen_addr(),
        "decode service listeners configured"
    );
    validate_checkpoint_kind(kind, &model_config)?;
    startup.event("loading Qwen codec");
    let qwen_codec = Arc::new(QwenCodec::load(config.hf_model_dir()).map_err(|error| {
        log_err_unavailable!("unable to load Qwen codec from {:?}: {error}", config.hf_model_dir())
    })?);
    startup.event("Qwen codec loaded");
    let scheduler_config = config.scheduler_config();
    let num_cache_pages = config.num_cache_pages();
    let max_queued_requests = config.max_queued_requests();
    let max_running_requests = config.max_running_requests();

    startup.event("initializing model executor");
    let model = build_model(
        kind,
        &config,
        Qwen35ExecutorConfig {
            max_requests: max_running_requests,
            max_tokens: scheduler_config.max_tokens,
            max_tokens_per_request: scheduler_config.max_tokens_per_request,
            num_cache_pages,
            num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
        },
    )?;
    startup.event("model executor initialized");

    let runtime_config = build_runtime_config(&startup, &config, &model_config, &model)?;

    tracing::info!(
        target: "inference-runtime-service::startup",
        component = kind.label(),
        checkpoint_mtp_layers = model_config.text_config.mtp_num_hidden_layers,
        num_spec_tokens = model.num_spec_tokens(),
        model_mode = ?config.model_mode(),
        num_cache_pages,
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        max_queued_requests,
        max_running_requests,
        context_window = runtime_config.context_window,
        max_batch_requests = scheduler_config.max_requests,
        max_tokens = scheduler_config.max_tokens,
        max_tokens_per_request = scheduler_config.max_tokens_per_request,
        model_idle_timeout_secs = config.model_idle_timeout().as_secs(),
        "qwen3.5 Spec/cache configuration"
    );

    startup.event("initializing runtime");
    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, L, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        qwen_codec,
        runtime_config,
        scheduler_config,
        model,
        telemetry.debug_logging,
    )
}

fn validate_checkpoint_kind(kind: ModelKind, model_config: &Qwen35ModelConfig) -> Result<()> {
    let num_experts = model_config.text_config.num_experts;
    match kind {
        ModelKind::Dense if num_experts != 0 => {
            Err(log_info_invalid_argument!(
                "qwen3_5_dense expects a dense checkpoint, got num_experts={num_experts}"
            ))
        },
        ModelKind::Sparse if num_experts == 0 => {
            Err(log_info_invalid_argument!(
                "qwen3_5_sparse expects a sparse checkpoint, got num_experts=0"
            ))
        },
        _ => Ok(()),
    }
}

fn all_lane_block_cache_capacity(
    num_cache_pages: usize,
    num_gqa_pages_per_main_block: usize,
    mtp_gqa_page_ids_per_block: &[usize],
    num_gdn_pages_per_main_block: usize,
) -> Result<usize> {
    let num_gqa_pages_per_cached_block = mtp_gqa_page_ids_per_block.iter().try_fold(
        u64::try_from(num_gqa_pages_per_main_block)
            .map_err(|_| log_err_internal!("qwen3.5 main GQA page count must fit u64"))?,
        |total, &num_pages| {
            total
                .checked_add(
                    u64::try_from(num_pages)
                        .map_err(|_| log_err_internal!("qwen3.5 MTP GQA page count must fit u64"))?,
                )
                .ok_or_else(|| log_err_internal!("qwen3.5 all-lane GQA page count overflow"))
        },
    )?;
    block_cache_capacity(
        num_cache_pages,
        usize::try_from(num_gqa_pages_per_cached_block)
            .map_err(|_| log_err_internal!("qwen3.5 all-lane GQA page count must fit usize"))?,
        num_gdn_pages_per_main_block,
    )
}

fn build_model(
    kind: ModelKind,
    config: &Qwen35Config,
    executor_config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor> {
    let hf_model_dir = config.hf_model_dir();
    let init_result = match config.model_mode() {
        Qwen35ModelMode::Vanilla => init_qwen_3_5_model(hf_model_dir, executor_config),
        Qwen35ModelMode::MTP {
            model_dir,
            num_spec_tokens,
        } => init_qwen_3_5_model_with_mtp(hf_model_dir, model_dir, *num_spec_tokens, executor_config),
        Qwen35ModelMode::DSpark {
            model_dir,
            num_spec_tokens,
        } => init_qwen_3_5_model_with_dspark(hf_model_dir, model_dir, *num_spec_tokens, executor_config),
    };
    init_result.map_err(|error| {
        log_err_internal!(
            "unable to initialize {} model from {hf_model_dir:?} in mode {:?}: {error}",
            kind.label(),
            config.model_mode(),
        )
    })
}

fn build_runtime_config(
    startup: &StartupLogger,
    service_config: &Qwen35Config,
    model_config: &Qwen35ModelConfig,
    model: &Qwen35Executor,
) -> Result<RuntimeConfig> {
    let num_cache_pages = service_config.num_cache_pages();
    let text = &model_config.text_config;
    let kv_dtype_bytes = kv_dtype_bytes(text.dtype.as_deref())?;
    let num_gqa_pages_per_main_block = model.num_main_lane_gqa_page_ids_per_block();
    let num_gdn_pages_per_main_block = model.num_gdn_state_page_ids_per_block();
    let mtp_gqa_page_ids_per_block = model.num_mtp_gqa_page_ids_per_block();
    let block_cache_capacity = all_lane_block_cache_capacity(
        num_cache_pages,
        num_gqa_pages_per_main_block,
        &mtp_gqa_page_ids_per_block,
        num_gdn_pages_per_main_block,
    )?;
    let mut cache_lanes = vec![CacheLaneRuntimeConfig {
        num_pages_per_kv_block: num_gqa_pages_per_main_block,
        num_pages_per_state_block: num_gdn_pages_per_main_block,
        block_cache_capacity,
    }];
    for num_pages_per_kv_block in mtp_gqa_page_ids_per_block {
        cache_lanes.push(CacheLaneRuntimeConfig {
            num_pages_per_kv_block,
            num_pages_per_state_block: 0,
            block_cache_capacity,
        });
    }
    let dspark_num_spec_tokens = match service_config.model_mode() {
        Qwen35ModelMode::DSpark { .. } => model.num_spec_tokens(),
        Qwen35ModelMode::Vanilla | Qwen35ModelMode::MTP { .. } => 0,
    };
    let runtime_config = RuntimeConfig {
        max_queued_requests: service_config.max_queued_requests(),
        max_running_requests: service_config.max_running_requests(),
        idle_timeout: service_config.model_idle_timeout(),
        context_window: context_window(text.max_position_embeddings, dspark_num_spec_tokens)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_kv_heads: text.num_key_value_heads,
        kv_head_dim: text.head_dim,
        kv_dtype_bytes,
        num_pages: num_cache_pages,
        page_bytes: QWEN35_PAGE_SIZE_BYTES,
        cache_lanes,
    };
    for cache_lane in 0..runtime_config.num_cache_lanes() {
        let lane = runtime_config.cache_lane(cache_lane);
        startup.cache_lane_config(CacheLaneLogSummary {
            cache_lane,
            mtp: cache_lane > 0,
            num_kv_pages_per_block: lane.num_pages_per_kv_block,
            num_state_pages_per_block: lane.num_pages_per_state_block,
            block_cache_capacity: lane.block_cache_capacity,
        });
    }
    Ok(runtime_config)
}

fn load_model_config(hf_model_dir: &std::path::Path) -> Result<Qwen35ModelConfig> {
    init_qwen35_model_config(hf_model_dir)
        .map_err(|error| log_err_unavailable!("unable to read qwen3.5 model config from {hf_model_dir:?}: {error}"))
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::all_lane_block_cache_capacity;

    #[test]
    fn test_all_lane_capacity_rejects_incomplete_block() {
        assert!(matches!(
            all_lane_block_cache_capacity(9151, 4096, &[256], 4800),
            Err(Error::InvalidArgument(message)) if message.contains("requiring 9152 pages")
        ));
    }

    #[test]
    fn test_all_lane_capacity_accepts_exact_block() {
        assert_eq!(all_lane_block_cache_capacity(9152, 4096, &[256], 4800).unwrap(), 1);
    }

    #[test]
    fn test_all_lane_capacity_counts_complete_blocks() {
        assert_eq!(all_lane_block_cache_capacity(29, 3, &[2, 1], 4).unwrap(), 2);
        assert_eq!(all_lane_block_cache_capacity(20, 10, &[], 0).unwrap(), 2);
    }
}
