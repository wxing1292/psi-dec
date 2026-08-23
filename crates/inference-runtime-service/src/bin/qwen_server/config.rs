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
use crate::qwen_server::args::QwenSpecType;
use crate::telemetry::ProfileMode;
use crate::telemetry::ProfilingConfig;
use crate::telemetry::TelemetryConfig;

const MAX_QUEUED_REQUESTS: usize = 32;

#[derive(Debug)]
struct SpecCheckpoint {
    model_dir: PathBuf,
    spec_type: QwenSpecType,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Qwen3ModelMode {
    Vanilla,
    DSpark { model_dir: PathBuf },
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
        let spec_checkpoint = normalize_spec_checkpoint(args.spec.hf_spec_model_dir, args.spec.spec_type)?;
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

        let model_mode = match spec_checkpoint {
            None => Qwen3ModelMode::Vanilla,
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::DSpark,
            }) => Qwen3ModelMode::DSpark { model_dir },
            Some(_) => {
                return Err(log_info_invalid_argument!("Qwen3 supports only --spec-type dspark"));
            },
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
    },
    DFlash2 {
        model_dir: PathBuf,
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
            Self::Vanilla | Self::DSpark { .. } | Self::DFlash2 { .. } => 1,
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
        let spec_checkpoint = normalize_spec_checkpoint(args.spec.hf_spec_model_dir, args.spec.spec_type)?;
        let is_mtp = matches!(
            spec_checkpoint.as_ref(),
            Some(SpecCheckpoint {
                spec_type: QwenSpecType::MTP,
                ..
            })
        );
        if args.num_spec_tokens.is_some() && !is_mtp {
            return Err(log_info_invalid_argument!(
                "--num-spec-tokens controls MTP proposal length and requires --spec-type mtp"
            ));
        }
        let num_mtp_tokens = is_mtp.then(|| args.num_spec_tokens.unwrap_or(NonZeroUsize::MIN));
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

        let model_mode = match spec_checkpoint {
            None => Qwen35ModelMode::Vanilla,
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::MTP,
            }) => {
                Qwen35ModelMode::MTP {
                    num_spec_tokens: num_mtp_tokens.expect("configured Qwen3.5 MTP must have a proposal length"),
                    model_dir,
                }
            },
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::DSpark,
            }) => Qwen35ModelMode::DSpark { model_dir },
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::DFlash2,
            }) => Qwen35ModelMode::DFlash2 { model_dir },
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

fn normalize_spec_checkpoint(
    hf_spec_model_dir: Option<PathBuf>,
    spec_type: Option<QwenSpecType>,
) -> Result<Option<SpecCheckpoint>> {
    match (hf_spec_model_dir, spec_type) {
        (None, None) => Ok(None),
        (Some(model_dir), Some(spec_type)) => Ok(Some(SpecCheckpoint { model_dir, spec_type })),
        _ => {
            Err(log_info_invalid_argument!(
                "--hf-spec-model-dir and --spec-type must be specified together"
            ))
        },
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
