use clap::Parser;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_model_config;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_hf_mtp;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::config::ServiceConfig;
use inference_runtime_service::observability::CacheLaneLogSummary;
use inference_runtime_service::observability::StartupLogger;

use super::args::Qwen35Args;
use super::config::ModelSource;
use super::config::QWEN35_TOKENS_PER_CACHE_BLOCK;
use super::config::Qwen35ModelKind;
use super::config::Qwen35ServerConfig;
use super::runtime::run_replay_model_runtime;
use super::sizing::block_cache_capacity;
use super::sizing::kv_dtype_bytes;

pub fn run_qwen35_dense() {
    let args = Qwen35Args::parse();
    let config = Qwen35ServerConfig::from_args(args);
    let _observability = config.service_observability_config().init();
    run_qwen35(Qwen35ModelKind::Qwen35Dense27B, config);
}

pub fn run_qwen35_sparse() {
    let args = Qwen35Args::parse();
    let config = Qwen35ServerConfig::from_args(args);
    let _observability = config.service_observability_config().init();
    run_qwen35(Qwen35ModelKind::Qwen35Sparse35B, config);
}

fn run_qwen35(kind: Qwen35ModelKind, config: Qwen35ServerConfig) {
    let startup = StartupLogger::new(kind.label());

    startup.event("reading model config");
    let model_config = init_model_config_for_source(config.model_source());
    assert_checkpoint_kind(kind, &model_config);
    let scheduler_config = config.scheduler_config();
    let service_config = ServiceConfig {
        user_req_queue_capacity: 1024,
        batch_req_queue_capacity: 16,
        batch_resp_queue_capacity: 16,
        token_prob_channel_capacity: 128,
    };
    let num_cache_pages = config.num_cache_pages();

    startup.event("initializing model executor");
    let model = build_metal_model(
        kind,
        config.model_source(),
        Qwen35ExecutorConfig {
            max_requests: scheduler_config.max_requests,
            max_tokens: scheduler_config.max_tokens,
            max_tokens_per_request: scheduler_config.max_tokens_per_request,
            num_cache_pages,
            num_tokens_per_block: QWEN35_TOKENS_PER_CACHE_BLOCK,
            num_mtp_modules: config.mtp_modules(),
        },
    );
    startup.event("model executor initialized");

    tracing::info!(
        target: "inference-runtime-service::startup",
        component = kind.label(),
        detected_mtp_modules = model_config.text_config.mtp_num_hidden_layers,
        requested_mtp_modules = config.mtp_modules(),
        num_mtp_modules = config.mtp_modules(),
        num_cache_pages,
        cache_block_tokens = QWEN35_TOKENS_PER_CACHE_BLOCK,
        max_requests = scheduler_config.max_requests,
        max_tokens = scheduler_config.max_tokens,
        max_tokens_per_request = scheduler_config.max_tokens_per_request,
        "qwen3.5 MTP/cache configuration"
    );

    let runtime_config = build_metal_runtime(&startup, &model_config, num_cache_pages, &model);

    startup.event("initializing runtime");
    run_qwen35_metal_runtime(
        config.listen_addr(),
        runtime_config,
        scheduler_config,
        service_config,
        model,
        config.mtp_modules(),
        config.debug_logging(),
    );
}

fn assert_checkpoint_kind(kind: Qwen35ModelKind, model_config: &Qwen35ModelConfig) {
    let num_experts = model_config.text_config.num_experts;
    match kind {
        Qwen35ModelKind::Qwen35Dense27B => {
            assert_eq!(
                0, num_experts,
                "qwen3_5_dense expects a dense checkpoint, got num_experts={num_experts}"
            )
        },
        Qwen35ModelKind::Qwen35Sparse35B => {
            assert!(
                num_experts > 0,
                "qwen3_5_sparse expects a sparse checkpoint, got num_experts={num_experts}"
            )
        },
    }
}

fn all_lane_block_cache_capacity(
    num_cache_pages: usize,
    num_gqa_pages_per_main_block: usize,
    mtp_gqa_page_ids_per_block: &[usize],
    num_gdn_pages_per_main_block: usize,
) -> usize {
    let num_gqa_pages_per_cached_block = mtp_gqa_page_ids_per_block
        .iter()
        .try_fold(
            u64::try_from(num_gqa_pages_per_main_block).expect("qwen3.5 main GQA page count must fit u64"),
            |total, &num_pages| {
                total.checked_add(u64::try_from(num_pages).expect("qwen3.5 MTP GQA page count must fit u64"))
            },
        )
        .expect("qwen3.5 all-lane GQA page count overflow");
    block_cache_capacity(
        num_cache_pages,
        usize::try_from(num_gqa_pages_per_cached_block).expect("qwen3.5 all-lane GQA page count must fit usize"),
        num_gdn_pages_per_main_block,
    )
}

fn build_metal_model(
    kind: Qwen35ModelKind,
    model_source: &ModelSource,
    executor_config: Qwen35ExecutorConfig,
) -> Qwen35Executor {
    let hf_model_dir = model_source.hf_model_dir();
    let hf_mtp_model_dir = model_source.hf_mtp_model_dir();
    let init_result = match hf_mtp_model_dir {
        Some(hf_mtp_model_dir) => init_qwen_3_5_model_with_hf_mtp(hf_model_dir, hf_mtp_model_dir, executor_config),
        None => init_qwen_3_5_model(hf_model_dir, executor_config),
    };
    init_result.unwrap_or_else(|err| {
        panic!(
            "unable to initialize {} replay model from {hf_model_dir:?} with hf_mtp_model_dir={hf_mtp_model_dir:?}: \
             {err:?}",
            kind.label()
        )
    })
}

fn build_metal_runtime(
    startup: &StartupLogger,
    model_config: &Qwen35ModelConfig,
    num_cache_pages: usize,
    model: &Qwen35Executor,
) -> RuntimeConfig {
    let text = &model_config.text_config;
    let kv_dtype_bytes = kv_dtype_bytes(text.dtype.as_deref());
    let num_gqa_pages_per_main_block = model.num_main_gqa_page_ids_per_block();
    let num_gdn_pages_per_main_block = model.num_gdn_state_page_ids_per_block();
    let mtp_gqa_page_ids_per_block = model.num_mtp_gqa_page_ids_per_block();
    let block_cache_capacity = all_lane_block_cache_capacity(
        num_cache_pages,
        num_gqa_pages_per_main_block,
        &mtp_gqa_page_ids_per_block,
        num_gdn_pages_per_main_block,
    );
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
        num_tokens_per_cache_block: QWEN35_TOKENS_PER_CACHE_BLOCK,
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
    runtime_config
}

#[allow(clippy::too_many_arguments)]
fn run_qwen35_metal_runtime(
    listen_addr: std::net::SocketAddr,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    service_config: ServiceConfig,
    model: Qwen35Executor,
    num_mtp_modules: usize,
    debug_logging: bool,
) {
    run_qwen35_metal_runtime_n::<QWEN35_TOKENS_PER_CACHE_BLOCK>(
        listen_addr,
        runtime_config,
        scheduler_config,
        service_config,
        model,
        num_mtp_modules,
        debug_logging,
    )
}

fn run_qwen35_metal_runtime_n<const N: usize>(
    listen_addr: std::net::SocketAddr,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    service_config: ServiceConfig,
    model: Qwen35Executor,
    num_mtp_modules: usize,
    debug_logging: bool,
) {
    match num_mtp_modules + 1 {
        1 => {
            run_replay_model_runtime::<N, 1, _>(
                listen_addr,
                runtime_config,
                scheduler_config,
                service_config,
                model,
                debug_logging,
            )
        },
        2 => {
            run_replay_model_runtime::<N, 2, _>(
                listen_addr,
                runtime_config,
                scheduler_config,
                service_config,
                model,
                debug_logging,
            )
        },
        lanes => panic!("qwen3.5 replay runtime supports up to one MTP module ({lanes} cache lanes requested)"),
    }
}

fn init_model_config_for_source(model_source: &ModelSource) -> Qwen35ModelConfig {
    let hf_model_dir = model_source.hf_model_dir();
    init_model_config(hf_model_dir)
        .unwrap_or_else(|err| panic!("unable to read qwen3.5 model config from {hf_model_dir:?}: {err:?}"))
}

#[cfg(test)]
mod tests {
    use super::all_lane_block_cache_capacity;

    #[test]
    #[should_panic(expected = "one cache block requiring 9152 pages")]
    fn test_all_lane_capacity_rejects_incomplete_block() {
        all_lane_block_cache_capacity(9151, 4096, &[256], 4800);
    }

    #[test]
    fn test_all_lane_capacity_accepts_exact_block() {
        assert_eq!(all_lane_block_cache_capacity(9152, 4096, &[256], 4800), 1);
    }

    #[test]
    fn test_all_lane_capacity_counts_complete_blocks() {
        assert_eq!(all_lane_block_cache_capacity(29, 3, &[2, 1], 4), 2);
        assert_eq!(all_lane_block_cache_capacity(20, 10, &[], 0), 2);
    }
}
