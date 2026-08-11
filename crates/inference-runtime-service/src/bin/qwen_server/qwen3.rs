use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_metal::model::qwen::v3::executor::Qwen3Executor;
use inference_executor_metal::model::qwen::v3::executor::Qwen3ExecutorConfig;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model_with_dspark;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;

use crate::codec::qwen::QwenCodec;
use crate::qwen_server::args::Qwen3Args;
use crate::qwen_server::config::Qwen3Config;
use crate::qwen_server::config::Qwen3ModelMode;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::runtime::serve_replay_model;
use crate::telemetry::CacheLaneLogSummary;
use crate::telemetry::StartupLogger;

// Qwen3 has no GDN snapshot to amortize across a large logical block. Two
// eight-token physical KV pages per layer keep trie granularity small without
// making each physical page a separate runtime block.
const TOKENS_PER_CACHE_BLOCK: usize = 16;

pub fn run() {
    if let Err(error) = run_inner() {
        eprintln!("unable to start qwen3: {error}");
        std::process::exit(1);
    }
}

fn run_inner() -> Result<()> {
    let args = Qwen3Args::parse();
    let config = Qwen3Config::from_args(args)?;
    let telemetry = config.telemetry_config();
    telemetry.init();
    let startup = StartupLogger::new("qwen3");

    tracing::info!(
        target: "inference-runtime-service::startup",
        grpc_listen_addr = %config.grpc_listen_addr(),
        http_listen_addr = %config.http_listen_addr(),
        "decode service listeners configured"
    );
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
    let executor_config = Qwen3ExecutorConfig {
        max_requests: max_running_requests,
        max_tokens: scheduler_config.max_tokens,
        max_tokens_per_request: scheduler_config.max_tokens_per_request,
        num_cache_pages,
        num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
    };
    let model = match config.model_mode() {
        Qwen3ModelMode::DSpark {
            model_dir: dspark_model_dir,
            num_spec_tokens,
        } => {
            init_qwen_3_model_with_dspark(
                config.hf_model_dir(),
                dspark_model_dir,
                *num_spec_tokens,
                executor_config,
            )
            .map_err(|error| {
                log_err_internal!(
                    "unable to initialize qwen3 Main model from {:?} with DSpark model from {:?}: {error}",
                    config.hf_model_dir(),
                    dspark_model_dir,
                )
            })?
        },
        Qwen3ModelMode::Vanilla => {
            init_qwen_3_model(config.hf_model_dir(), executor_config).map_err(|error| {
                log_err_internal!(
                    "unable to initialize qwen3 Main model from {:?}: {error}",
                    config.hf_model_dir()
                )
            })?
        },
    };
    startup.event("model executor initialized");
    if let Qwen3ModelMode::DSpark {
        model_dir: dspark_model_dir,
        ..
    } = config.model_mode()
    {
        tracing::info!(
            target: "inference-runtime-service::startup",
            component = "qwen3",
            dspark_model_dir = ?dspark_model_dir,
            num_spec_tokens = model.num_spec_tokens(),
            proposal_mode = "fixed-block",
            "qwen3 DSpark configured"
        );
    }

    let runtime_config = build_runtime_config(&startup, &config, &model)?;

    tracing::info!(
        target: "inference-runtime-service::startup",
        component = "qwen3",
        num_cache_pages,
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        max_queued_requests,
        max_running_requests,
        max_batch_requests = scheduler_config.max_requests,
        max_tokens = scheduler_config.max_tokens,
        max_tokens_per_request = scheduler_config.max_tokens_per_request,
        model_idle_timeout_secs = config.model_idle_timeout().as_secs(),
        "qwen3 Main/cache configuration"
    );

    startup.event("initializing runtime");
    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, 1, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        qwen_codec,
        runtime_config,
        scheduler_config,
        model,
        telemetry.debug_logging,
    )
}

fn build_runtime_config(
    startup: &StartupLogger,
    service_config: &Qwen3Config,
    model: &Qwen3Executor,
) -> Result<RuntimeConfig> {
    let num_cache_pages = service_config.num_cache_pages();
    let model_config = model.model_config();
    let text = &model_config.text_config;
    let num_pages_per_kv_block = model.num_kv_page_ids_per_block();
    let cache_lane = main_cache_lane(num_cache_pages, num_pages_per_kv_block)?;
    startup.cache_lane_config(CacheLaneLogSummary {
        cache_lane: 0,
        mtp: false,
        num_kv_pages_per_block: cache_lane.num_pages_per_kv_block,
        num_state_pages_per_block: cache_lane.num_pages_per_state_block,
        block_cache_capacity: cache_lane.block_cache_capacity,
    });
    Ok(RuntimeConfig {
        max_queued_requests: service_config.max_queued_requests(),
        max_running_requests: service_config.max_running_requests(),
        model_idle_timeout: service_config.model_idle_timeout(),
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_kv_heads: text.num_key_value_heads,
        kv_head_dim: text.head_dim,
        kv_dtype_bytes: 2,
        num_pages: num_cache_pages,
        page_bytes: QWEN3_PAGE_SIZE_BYTES,
        cache_lanes: vec![cache_lane],
    })
}

fn main_cache_lane(num_cache_pages: usize, num_pages_per_kv_block: usize) -> Result<CacheLaneRuntimeConfig> {
    Ok(CacheLaneRuntimeConfig {
        num_pages_per_kv_block,
        num_pages_per_state_block: 0,
        block_cache_capacity: block_cache_capacity(num_cache_pages, num_pages_per_kv_block, 0)?,
    })
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::TOKENS_PER_CACHE_BLOCK;
    use super::main_cache_lane;

    #[test]
    fn test_qwen3_uses_small_gqa_logical_blocks() {
        assert_eq!(TOKENS_PER_CACHE_BLOCK, 16);
    }

    #[test]
    fn test_main_cache_lane_uses_no_state_pages() {
        let lane = main_cache_lane(40 * 1024, 80).unwrap();

        assert_eq!(lane.num_pages_per_kv_block, 80);
        assert_eq!(lane.num_pages_per_state_block, 0);
        assert_eq!(lane.block_cache_capacity, 512);
    }

    #[test]
    fn test_main_cache_lane_reports_dynamic_minimum_pages() {
        assert!(matches!(
            main_cache_lane(79, 80),
            Err(Error::InvalidArgument(message))
                if message.contains("--num-cache-pages=79")
                    && message.contains("requiring 80 pages")
        ));
    }
}
