use std::fmt::Display;
use std::fmt::Formatter;

use tracing_subscriber::EnvFilter;

use crate::perf_metrics::BatchRequestSummary;
use crate::perf_metrics::BatchResponseSummary;
use crate::perf_metrics::ExecutorBatchPerfMetrics;
use crate::perf_metrics::ms;
use crate::profiling;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileMode {
    Component,
    Operation,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProfilingConfig {
    pub mode: Option<ProfileMode>,
    pub summary_every: u64,
}

impl ProfilingConfig {
    pub fn is_requested(&self) -> bool {
        self.mode.is_some() || self.summary_every > 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TelemetryConfig {
    pub profiling: ProfilingConfig,
    pub debug_logging: bool,
}

impl TelemetryConfig {
    pub fn init(self) {
        let profiling_requested = self.profiling.is_requested();
        profiling::set_profiling_summary_every(self.profiling.summary_every);
        init_tracing(profiling_requested, self.debug_logging);

        tracing::info!(
            component = "telemetry",
            profile_mode = ?self.profiling.mode,
            profile_enabled = self.profiling.is_requested(),
            profile_summary_every_batches = self.profiling.summary_every,
            logging = if self.debug_logging { "debug" } else { "info" },
            "runtime service telemetry initialized"
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StartupLogger {
    component: &'static str,
}

impl StartupLogger {
    pub fn new(component: &'static str) -> Self {
        Self { component }
    }

    pub fn event(&self, message: &'static str) {
        tracing::info!(target: "inference-runtime-service::startup", component = self.component, message);
    }

    pub fn cache_lane_config(&self, summary: CacheLaneLogSummary) {
        tracing::info!(
            target: "inference-runtime-service::startup",
            component = self.component,
            cache_lane = summary.cache_lane,
            mtp = summary.mtp,
            num_kv_pages_per_block = summary.num_kv_pages_per_block,
            num_state_pages_per_block = summary.num_state_pages_per_block,
            block_cache_capacity = summary.block_cache_capacity,
            "runtime cache lane configured"
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CacheLaneLogSummary {
    pub cache_lane: usize,
    pub mtp: bool,
    pub num_kv_pages_per_block: usize,
    pub num_state_pages_per_block: usize,
    pub block_cache_capacity: usize,
}

struct FourDecimal(f64);

impl Display for FourDecimal {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:.4}", self.0)
    }
}

fn four_decimal(value: f64) -> impl Display {
    FourDecimal(value)
}

struct FourDecimalList<'a>(&'a [f64]);

impl Display for FourDecimalList<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "[")?;
        for (index, value) in self.0.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "{value:.4}")?;
        }
        write!(formatter, "]")
    }
}

pub fn emit_executor_batch_perf_metrics(
    model_name: &str,
    model_mode: &str,
    batch_seq: u64,
    batch_summary: BatchRequestSummary,
    response_summary: BatchResponseSummary,
    metrics: ExecutorBatchPerfMetrics,
) {
    let acceptance_rate = ratio(response_summary.num_verified_tokens, response_summary.num_spec_tokens);
    let acceptance_rate_by_index = response_summary
        .num_verified_token_by_index
        .iter()
        .zip(&response_summary.num_spec_token_by_index)
        .map(|(&verified, &spec)| ratio(verified, spec))
        .collect::<Vec<_>>();
    tracing::info!(
        target: "inference-runtime-service::perf",
        component = "executor",
        phase = "executor.batch.perf",
        model = model_name,
        model_mode,
        batch_seq,
        num_reqs = batch_summary.num_reqs,
        num_input_tokens = batch_summary.num_input_tokens,
        num_spec_tokens = response_summary.num_spec_tokens,
        num_verified_tokens = response_summary.num_verified_tokens,
        acceptance_rate = %four_decimal(acceptance_rate),
        num_spec_token_by_index = ?response_summary.num_spec_token_by_index,
        num_verified_token_by_index = ?response_summary.num_verified_token_by_index,
        acceptance_rate_by_index = %FourDecimalList(&acceptance_rate_by_index),
        main_ms = %four_decimal(ms(metrics.main_elapsed)),
        spec_ms = %four_decimal(ms(metrics.spec_elapsed)),
        spec_passes = metrics.spec_passes,
        "executor batch perf"
    );
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn init_tracing(profile: bool, debug_logging: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let service_level = if debug_logging { "debug" } else { "info" };
    let filter = filter
        .add_directive(
            format!("inference_runtime_service={service_level}")
                .parse()
                .expect("service log directive should be valid"),
        )
        .add_directive(
            format!("inference-runtime-service={service_level}")
                .parse()
                .expect("explicit service log target directive should be valid"),
        );
    let filter = if profile {
        filter.add_directive(
            "inference-runtime-service::profile=debug"
                .parse()
                .expect("profile log directive should be valid"),
        )
    } else {
        filter
    };
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_decimal_formats_scalars_and_lists() {
        assert_eq!(four_decimal(1.0).to_string(), "1.0000");
        assert_eq!(four_decimal(2.0 / 7.0).to_string(), "0.2857");
        assert_eq!(FourDecimalList(&[0.75, 2.0 / 3.0]).to_string(), "[0.7500, 0.6667]");
    }
}
