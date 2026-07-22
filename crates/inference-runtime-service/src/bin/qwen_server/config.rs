use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;

use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_service::observability::ProfileMode;
use inference_runtime_service::observability::ProfilingConfig;
use inference_runtime_service::observability::ServiceObservabilityConfig;

use super::args::Qwen3Args;
use super::args::Qwen35Args;
use super::args::QwenLogLevel;
use super::args::QwenProfileMode;
use super::sizing::QWEN35_DEFAULT_NUM_CACHE_PAGES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35ModelKind {
    Qwen35Dense27B,
    Qwen35Sparse35B,
}

pub const QWEN35_TOKENS_PER_CACHE_BLOCK: usize = 2048;

impl Qwen35ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35ModelKind::Qwen35Dense27B => "qwen3.5 dense 27B",
            Qwen35ModelKind::Qwen35Sparse35B => "qwen3.5 sparse 35B-A3B",
        }
    }
}

#[derive(Debug)]
pub struct ModelSource {
    hf_model_dir: PathBuf,
    hf_mtp_model_dir: Option<PathBuf>,
}

impl ModelSource {
    pub fn new(hf_model_dir: PathBuf, hf_mtp_model_dir: Option<PathBuf>) -> Self {
        Self {
            hf_model_dir,
            hf_mtp_model_dir,
        }
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn hf_mtp_model_dir(&self) -> Option<&Path> {
        self.hf_mtp_model_dir.as_deref()
    }

    fn default_mtp_modules(&self) -> usize {
        usize::from(self.hf_mtp_model_dir.is_some())
    }
}

#[derive(Debug)]
pub struct Qwen3ServerConfig {
    listen_addr: SocketAddr,
    model_source: ModelSource,
    num_cache_pages: Option<usize>,
}

impl Qwen3ServerConfig {
    pub fn from_args(args: Qwen3Args) -> Self {
        Self {
            listen_addr: args.listen_addr,
            model_source: ModelSource::new(args.hf_model_dir, None),
            num_cache_pages: args.num_cache_pages,
        }
    }

    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    pub fn model_source(&self) -> &ModelSource {
        &self.model_source
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages.unwrap_or(DEFAULT_NUM_CACHE_PAGES)
    }
}

const DEFAULT_NUM_CACHE_PAGES: usize = 32 * 1024;

#[derive(Debug)]
pub struct Qwen35ServerConfig {
    listen_addr: SocketAddr,
    model_source: ModelSource,
    observability: Qwen35ObservabilityConfig,
    mtp_modules: usize,
    num_cache_pages: Option<usize>,
    scheduler_config: SchedulerConfig,
}

impl Qwen35ServerConfig {
    pub fn from_args(args: Qwen35Args) -> Self {
        let model_source = ModelSource::new(args.hf_model_dir, args.hf_mtp_model_dir);
        let mtp_modules = args.mtp_module.unwrap_or_else(|| model_source.default_mtp_modules());
        Self {
            listen_addr: args.listen_addr,
            model_source,
            observability: Qwen35ObservabilityConfig {
                profile: args.profile.map(ProfileMode::from),
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            mtp_modules,
            num_cache_pages: args.num_cache_pages.map(NonZeroUsize::get),
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                wait_duration: std::time::Duration::ZERO,
                max_compute_slots: 1,
            },
        }
    }

    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    pub fn model_source(&self) -> &ModelSource {
        &self.model_source
    }

    pub fn service_observability_config(&self) -> ServiceObservabilityConfig {
        ServiceObservabilityConfig {
            profiling: ProfilingConfig {
                mode: self.observability.profile,
                summary_every: self.observability.profile.map_or(0, |_| 32),
            },
            debug_logging: self.observability.debug_logging,
        }
    }

    pub fn debug_logging(&self) -> bool {
        self.observability.debug_logging
    }

    pub fn mtp_modules(&self) -> usize {
        self.mtp_modules
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages.unwrap_or(QWEN35_DEFAULT_NUM_CACHE_PAGES)
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

#[derive(Debug)]
struct Qwen35ObservabilityConfig {
    profile: Option<ProfileMode>,
    debug_logging: bool,
}

impl From<QwenProfileMode> for ProfileMode {
    fn from(value: QwenProfileMode) -> Self {
        match value {
            QwenProfileMode::Component => Self::Component,
            QwenProfileMode::Operation => Self::Operation,
        }
    }
}
