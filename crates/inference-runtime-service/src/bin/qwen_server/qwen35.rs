use clap::Parser;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_model_config;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_hf_mtp;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::log_info_invalid_argument;
use inference_runtime_service::observability::CacheLaneLogSummary;
use inference_runtime_service::observability::StartupLogger;
use inference_runtime_service::runtime::serve_replay_model;

use super::args::Qwen35Args;
use super::config::QWEN35_MAX_RUNNING_REQUESTS;
use super::config::Qwen35Config;
use super::sizing::block_cache_capacity;
use super::sizing::kv_dtype_bytes;

const TOKENS_PER_CACHE_BLOCK: usize = 2048;
const MAX_QUEUED_REQUESTS: usize = 32;

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
}

pub fn run_dense() {
    run_or_exit(ModelKind::Dense);
}

pub fn run_sparse() {
    run_or_exit(ModelKind::Sparse);
}

fn run_or_exit(kind: ModelKind) {
    if let Err(error) = run(kind) {
        eprintln!("unable to start {}: {error}", kind.label());
        std::process::exit(1);
    }
}

fn run(kind: ModelKind) -> Result<()> {
    let args = Qwen35Args::parse();
    let config = Qwen35Config::from_args(args)?;
    let observability = config.observability_config();
    let _observability = observability.init();
    let startup = StartupLogger::new(kind.label());

    startup.event("reading model config");
    let model_config = load_model_config(config.hf_model_dir())?;
    tracing::info!(
        target: "inference-runtime-service::startup",
        listen_addr = %config.listen_addr(),
        "decode service listener configured"
    );
    validate_checkpoint_kind(kind, &model_config)?;
    let scheduler_config = config.scheduler_config();
    let num_cache_pages = config.num_cache_pages();

    startup.event("initializing model executor");
    let model = build_model(
        kind,
        &config,
        Qwen35ExecutorConfig {
            max_requests: QWEN35_MAX_RUNNING_REQUESTS,
            max_tokens: scheduler_config.max_tokens,
            max_tokens_per_request: scheduler_config.max_tokens_per_request,
            num_cache_pages,
            num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
            num_mtp_modules: config.num_mtp_modules(),
        },
    )?;
    startup.event("model executor initialized");

    tracing::info!(
        target: "inference-runtime-service::startup",
        component = kind.label(),
        detected_mtp_modules = model_config.text_config.mtp_num_hidden_layers,
        num_mtp_modules = config.num_mtp_modules(),
        num_cache_pages,
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        max_queued_requests = MAX_QUEUED_REQUESTS,
        max_running_requests = QWEN35_MAX_RUNNING_REQUESTS,
        max_batch_requests = scheduler_config.max_requests,
        max_tokens = scheduler_config.max_tokens,
        max_tokens_per_request = scheduler_config.max_tokens_per_request,
        "qwen3.5 MTP/cache configuration"
    );

    let runtime_config = build_runtime_config(&startup, &model_config, num_cache_pages, &model)?;

    startup.event("initializing runtime");
    serve_qwen35(
        config.listen_addr(),
        runtime_config,
        scheduler_config,
        model,
        config.num_mtp_modules(),
        observability.debug_logging,
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
    let hf_mtp_model_dir = config.hf_mtp_model_dir();
    let init_result = match config.num_mtp_modules() {
        0 => init_qwen_3_5_model(hf_model_dir, executor_config),
        1 => {
            init_qwen_3_5_model_with_hf_mtp(
                hf_model_dir,
                hf_mtp_model_dir.expect("validated Qwen MTP model directory"),
                executor_config,
            )
        },
        _ => unreachable!("validated Qwen config supports at most one MTP module"),
    };
    init_result.map_err(|error| {
        log_err_unavailable!(
            "unable to initialize {} model from {hf_model_dir:?} with hf_mtp_model_dir={hf_mtp_model_dir:?}: {error}",
            kind.label()
        )
    })
}

fn build_runtime_config(
    startup: &StartupLogger,
    model_config: &Qwen35ModelConfig,
    num_cache_pages: usize,
    model: &Qwen35Executor,
) -> Result<RuntimeConfig> {
    let text = &model_config.text_config;
    let kv_dtype_bytes = kv_dtype_bytes(text.dtype.as_deref())?;
    let num_gqa_pages_per_main_block = model.num_main_gqa_page_ids_per_block();
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
    let runtime_config = RuntimeConfig {
        max_queued_requests: MAX_QUEUED_REQUESTS,
        max_running_requests: QWEN35_MAX_RUNNING_REQUESTS,
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

fn serve_qwen35(
    listen_addr: std::net::SocketAddr,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    model: Qwen35Executor,
    num_mtp_modules: usize,
    debug_logging: bool,
) -> Result<()> {
    match num_mtp_modules {
        0 => {
            serve_replay_model::<TOKENS_PER_CACHE_BLOCK, 1, _>(
                listen_addr,
                runtime_config,
                scheduler_config,
                model,
                debug_logging,
            )
        },
        1 => {
            serve_replay_model::<TOKENS_PER_CACHE_BLOCK, 2, _>(
                listen_addr,
                runtime_config,
                scheduler_config,
                model,
                debug_logging,
            )
        },
        _ => unreachable!("validated Qwen config supports at most one MTP module"),
    }
}

fn load_model_config(hf_model_dir: &std::path::Path) -> Result<Qwen35ModelConfig> {
    init_model_config(hf_model_dir)
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
