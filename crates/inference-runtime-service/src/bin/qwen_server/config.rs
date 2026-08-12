use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use inference_runtime_core::Result;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_info_invalid_argument;

use crate::qwen_server::args::Qwen3Args;
use crate::qwen_server::args::Qwen35Args;
use crate::qwen_server::args::QwenLogLevel;
use crate::qwen_server::args::QwenProfileMode;
use crate::telemetry::ProfileMode;
use crate::telemetry::ProfilingConfig;
use crate::telemetry::TelemetryConfig;

const MAX_QUEUED_REQUESTS: usize = 32;

#[derive(Debug, Eq, PartialEq)]
pub enum Qwen3ModelMode {
    Vanilla,
    DSpark {
        model_dir: PathBuf,
        num_spec_tokens: Option<NonZeroUsize>,
    },
}

#[derive(Debug)]
pub struct Qwen3Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    model_mode: Qwen3ModelMode,
    telemetry: TelemetryConfig,
    executor_hibernation_timeout: Duration,
    executor_hibernation_mode: ExecutorHibernationMode,
    num_cache_pages: usize,
    max_queued_requests: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen3Config {
    pub fn from_args(args: Qwen3Args) -> Result<Self> {
        if args.num_spec_tokens.is_some() && args.hf_dspark_model_dir.is_none() {
            return Err(log_info_invalid_argument!(
                "--hf-dspark-model-dir is required when --num-spec-tokens is set"
            ));
        }
        validate_scheduler_token_capacity(args.max_tokens, args.max_tokens_per_request)?;
        if u32::try_from(args.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(args.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        let model_mode = match args.hf_dspark_model_dir {
            Some(model_dir) => {
                Qwen3ModelMode::DSpark {
                    model_dir,
                    num_spec_tokens: args.num_spec_tokens,
                }
            },
            None => Qwen3ModelMode::Vanilla,
        };
        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            model_mode,
            telemetry: TelemetryConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            executor_hibernation_timeout: Duration::from_secs(args.executor_hibernation_timeout_secs.get()),
            executor_hibernation_mode: args.executor_hibernation_mode,
            num_cache_pages: args.num_cache_pages.get(),
            max_queued_requests: MAX_QUEUED_REQUESTS,
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

    pub fn model_mode(&self) -> &Qwen3ModelMode {
        &self.model_mode
    }

    pub fn telemetry_config(&self) -> TelemetryConfig {
        self.telemetry
    }

    pub fn executor_hibernation_timeout(&self) -> Duration {
        self.executor_hibernation_timeout
    }

    pub fn executor_hibernation_mode(&self) -> ExecutorHibernationMode {
        self.executor_hibernation_mode
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn max_queued_requests(&self) -> usize {
        self.max_queued_requests
    }

    pub fn max_running_requests(&self) -> usize {
        self.scheduler_config.max_requests
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Qwen35ModelMode {
    Vanilla,
    MTP {
        model_dir: PathBuf,
        num_spec_tokens: NonZeroUsize,
    },
    DSpark {
        model_dir: PathBuf,
        num_spec_tokens: Option<NonZeroUsize>,
    },
}

impl Qwen35ModelMode {
    pub fn num_cache_lanes(&self) -> usize {
        match self {
            Self::MTP { num_spec_tokens, .. } => {
                num_spec_tokens
                    .get()
                    .checked_add(1)
                    .expect("validated Qwen3.5 MTP cache-lane count must fit usize")
            },
            Self::Vanilla | Self::DSpark { .. } => 1,
        }
    }
}

#[derive(Debug)]
pub struct Qwen35Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    model_mode: Qwen35ModelMode,
    telemetry: TelemetryConfig,
    executor_hibernation_timeout: Duration,
    executor_hibernation_mode: ExecutorHibernationMode,
    num_cache_pages: usize,
    max_queued_requests: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen35Config {
    pub fn from_args(args: Qwen35Args) -> Result<Self> {
        if args.hf_dspark_model_dir.is_some() && args.hf_mtp_model_dir.is_some() {
            return Err(log_info_invalid_argument!(
                "--hf-dspark-model-dir is mutually exclusive with Qwen3.5 MTP"
            ));
        }
        if args.num_spec_tokens.is_some() && args.hf_mtp_model_dir.is_none() && args.hf_dspark_model_dir.is_none() {
            return Err(log_info_invalid_argument!(
                "--hf-mtp-model-dir or --hf-dspark-model-dir is required when --num-spec-tokens is set"
            ));
        }
        let num_mtp_tokens = args
            .hf_mtp_model_dir
            .as_ref()
            .map(|_| args.num_spec_tokens.unwrap_or(NonZeroUsize::MIN));
        validate_scheduler_token_capacity(args.max_tokens, args.max_tokens_per_request)?;
        validate_mtp_scheduler_capacity(num_mtp_tokens, args.max_tokens_per_request)?;
        if u32::try_from(args.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(args.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        let model_mode = match (args.hf_mtp_model_dir, args.hf_dspark_model_dir, num_mtp_tokens) {
            (None, Some(model_dir), None) => {
                Qwen35ModelMode::DSpark {
                    model_dir,
                    num_spec_tokens: args.num_spec_tokens,
                }
            },
            (Some(model_dir), None, Some(num_spec_tokens)) => {
                Qwen35ModelMode::MTP {
                    model_dir,
                    num_spec_tokens,
                }
            },
            (None, None, None) => Qwen35ModelMode::Vanilla,
            _ => unreachable!("validated Qwen3.5 model mode must be complete and exclusive"),
        };
        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            model_mode,
            telemetry: TelemetryConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            executor_hibernation_timeout: Duration::from_secs(args.executor_hibernation_timeout_secs.get()),
            executor_hibernation_mode: args.executor_hibernation_mode,
            num_cache_pages: args.num_cache_pages.get(),
            max_queued_requests: MAX_QUEUED_REQUESTS,
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

    pub fn model_mode(&self) -> &Qwen35ModelMode {
        &self.model_mode
    }

    pub fn telemetry_config(&self) -> TelemetryConfig {
        self.telemetry
    }

    pub fn executor_hibernation_timeout(&self) -> Duration {
        self.executor_hibernation_timeout
    }

    pub fn executor_hibernation_mode(&self) -> ExecutorHibernationMode {
        self.executor_hibernation_mode
    }

    pub fn num_cache_lanes(&self) -> usize {
        self.model_mode.num_cache_lanes()
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn max_queued_requests(&self) -> usize {
        self.max_queued_requests
    }

    pub fn max_running_requests(&self) -> usize {
        self.scheduler_config.max_requests
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

fn validate_mtp_scheduler_capacity(
    num_spec_tokens: Option<NonZeroUsize>,
    max_tokens_per_request: NonZeroUsize,
) -> Result<()> {
    let Some(num_spec_tokens) = num_spec_tokens else {
        return Ok(());
    };
    if num_spec_tokens.get() > max_tokens_per_request.get() {
        return Err(log_info_invalid_argument!(
            "--max-tokens-per-request={max_tokens_per_request} must be at least --num-spec-tokens={num_spec_tokens} \
             for MTP cache-lane initialization"
        ));
    }
    Ok(())
}

fn validate_scheduler_token_capacity(max_tokens: NonZeroUsize, max_tokens_per_request: NonZeroUsize) -> Result<()> {
    if max_tokens_per_request.get() > max_tokens.get() {
        return Err(log_info_invalid_argument!(
            "--max-tokens-per-request={max_tokens_per_request} must not exceed --max-tokens={max_tokens}"
        ));
    }
    Ok(())
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
