use std::net::SocketAddr;
use std::path::Path;
use std::path::PathBuf;

use inference_runtime_core::Result;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_info_invalid_argument;
use inference_runtime_service::telemetry::ProfileMode;
use inference_runtime_service::telemetry::ProfilingConfig;
use inference_runtime_service::telemetry::TelemetryConfig;

use crate::qwen_server::args::Qwen3Args;
use crate::qwen_server::args::Qwen35Args;
use crate::qwen_server::args::QwenLogLevel;
use crate::qwen_server::args::QwenProfileMode;

const MAX_QUEUED_REQUESTS: usize = 32;

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
    num_cache_pages: usize,
    max_queued_requests: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen3Config {
    pub fn from_args(args: Qwen3Args) -> Result<Self> {
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
            Some(model_dir) => Qwen3ModelMode::DSpark { model_dir },
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
pub enum Qwen35ModelMode {
    Vanilla,
    Mtp { model_dir: PathBuf },
    DSpark { model_dir: PathBuf },
}

impl Qwen35ModelMode {
    pub fn num_mtp_modules(&self) -> usize {
        usize::from(matches!(self, Self::Mtp { .. }))
    }
}

#[derive(Debug)]
pub struct Qwen35Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    model_mode: Qwen35ModelMode,
    telemetry: TelemetryConfig,
    num_cache_pages: usize,
    max_queued_requests: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen35Config {
    pub fn from_args(args: Qwen35Args) -> Result<Self> {
        let num_mtp_modules = args
            .mtp_module
            .unwrap_or_else(|| usize::from(args.hf_mtp_model_dir.is_some()));
        if args.hf_dspark_model_dir.is_some() && (args.hf_mtp_model_dir.is_some() || num_mtp_modules != 0) {
            return Err(log_info_invalid_argument!(
                "--hf-dspark-model-dir is mutually exclusive with Qwen3.5 MTP"
            ));
        }
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
        if u32::try_from(args.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
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
        if u32::try_from(args.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        let model_mode = match (args.hf_mtp_model_dir, args.hf_dspark_model_dir, num_mtp_modules) {
            (_, Some(model_dir), 0) => Qwen35ModelMode::DSpark { model_dir },
            (Some(model_dir), None, 1) => Qwen35ModelMode::Mtp { model_dir },
            (_, None, 0) => Qwen35ModelMode::Vanilla,
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

    pub fn num_mtp_modules(&self) -> usize {
        self.model_mode.num_mtp_modules()
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
