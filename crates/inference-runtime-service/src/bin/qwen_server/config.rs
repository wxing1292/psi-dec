use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;

use inference_runtime_core::Result;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_info_invalid_argument;
use inference_runtime_service::observability::ProfileMode;
use inference_runtime_service::observability::ProfilingConfig;
use inference_runtime_service::observability::ServiceObservabilityConfig;

use crate::qwen_server::args::Qwen3Args;
use crate::qwen_server::args::Qwen35Args;
use crate::qwen_server::args::QwenLogLevel;
use crate::qwen_server::args::QwenProfileMode;
use crate::qwen_server::sizing::QWEN3_DEFAULT_NUM_CACHE_PAGES;
use crate::qwen_server::sizing::QWEN35_DEFAULT_NUM_CACHE_PAGES;

pub const QWEN3_MAX_RUNNING_REQUESTS: usize = 8;
pub const QWEN35_MAX_RUNNING_REQUESTS: usize = 8;

#[derive(Debug)]
pub struct Qwen3Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    hf_dspark_model_dir: Option<PathBuf>,
    observability: ServiceObservabilityConfig,
    num_cache_pages: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen3Config {
    pub fn from_args(args: Qwen3Args) -> Result<Self> {
        let num_cache_pages = args
            .num_cache_pages
            .map_or(QWEN3_DEFAULT_NUM_CACHE_PAGES, NonZeroUsize::get);
        if args.max_requests.get() > QWEN3_MAX_RUNNING_REQUESTS {
            return Err(log_info_invalid_argument!(
                "--max-requests={} exceeds the Qwen3 runtime capacity {QWEN3_MAX_RUNNING_REQUESTS}",
                args.max_requests
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(num_cache_pages - 1).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            hf_dspark_model_dir: args.hf_dspark_model_dir,
            observability: ServiceObservabilityConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            num_cache_pages,
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                max_compute_slots: 1,
            },
        })
    }

    pub fn grpc_listen_addr(&self) -> SocketAddr {
        self.grpc_listen_addr
    }

    pub fn http_listen_addr(&self) -> SocketAddr {
        self.http_listen_addr
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn hf_dspark_model_dir(&self) -> Option<&Path> {
        self.hf_dspark_model_dir.as_deref()
    }

    pub fn observability_config(&self) -> ServiceObservabilityConfig {
        self.observability
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

#[derive(Debug)]
pub struct Qwen35Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    hf_mtp_model_dir: Option<PathBuf>,
    observability: ServiceObservabilityConfig,
    num_mtp_modules: usize,
    num_cache_pages: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen35Config {
    pub fn from_args(args: Qwen35Args) -> Result<Self> {
        let num_cache_pages = args
            .num_cache_pages
            .map_or(QWEN35_DEFAULT_NUM_CACHE_PAGES, NonZeroUsize::get);
        let num_mtp_modules = args
            .mtp_module
            .unwrap_or_else(|| usize::from(args.hf_mtp_model_dir.is_some()));
        if num_mtp_modules > 1 {
            return Err(log_info_invalid_argument!(
                "--mtp-module must be 0 or 1, got {num_mtp_modules}"
            ));
        }
        if num_mtp_modules == 1 && args.hf_mtp_model_dir.is_none() {
            return Err(log_info_invalid_argument!(
                "--hf-mtp-model-dir is required when --mtp-module is 1"
            ));
        }
        if args.max_requests.get() > QWEN35_MAX_RUNNING_REQUESTS {
            return Err(log_info_invalid_argument!(
                "--max-requests={} exceeds the Qwen runtime capacity {QWEN35_MAX_RUNNING_REQUESTS}",
                args.max_requests
            ));
        }
        if num_mtp_modules + 1 > args.max_tokens_per_request.get() {
            return Err(log_info_invalid_argument!(
                "--max-tokens-per-request={} cannot schedule {} target/MTP tokens",
                args.max_tokens_per_request,
                num_mtp_modules + 1
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(num_cache_pages - 1).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            hf_mtp_model_dir: args.hf_mtp_model_dir,
            observability: ServiceObservabilityConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            num_mtp_modules,
            num_cache_pages,
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                max_compute_slots: 1,
            },
        })
    }

    pub fn grpc_listen_addr(&self) -> SocketAddr {
        self.grpc_listen_addr
    }

    pub fn http_listen_addr(&self) -> SocketAddr {
        self.http_listen_addr
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn hf_mtp_model_dir(&self) -> Option<&Path> {
        self.hf_mtp_model_dir.as_deref()
    }

    pub fn observability_config(&self) -> ServiceObservabilityConfig {
        self.observability
    }

    pub fn num_mtp_modules(&self) -> usize {
        self.num_mtp_modules
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

impl From<QwenProfileMode> for ProfileMode {
    fn from(value: QwenProfileMode) -> Self {
        match value {
            QwenProfileMode::Component => Self::Component,
            QwenProfileMode::Operation => Self::Operation,
        }
    }
}

#[cfg(test)]
#[path = "./config_test.rs"]
mod config_test;
