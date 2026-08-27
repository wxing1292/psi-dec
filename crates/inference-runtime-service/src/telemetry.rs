use std::fmt::Display;
use std::fmt::Formatter;
use std::time::Duration;

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

struct FourDecimal(f64);

impl Display for FourDecimal {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:.4}", self.0)
    }
}

fn four_decimal(value: f64) -> impl Display {
    FourDecimal(value)
}

struct OptionalMilliseconds(Option<Duration>);

impl Display for OptionalMilliseconds {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(elapsed) => write!(formatter, "{:.4}", ms(elapsed)),
            None => formatter.write_str("unavailable"),
        }
    }
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
    let executor_elapsed = metrics.main_elapsed + metrics.spec_elapsed;
    let acceptance_rate_by_index = response_summary
        .num_verified_token_by_index
        .iter()
        .zip(&response_summary.num_spec_token_by_index)
        .map(|(&verified, &spec)| ratio(verified, spec))
        .collect::<Vec<_>>();
    tracing::debug!(
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
        executor_cpu_ms = %four_decimal(ms(executor_elapsed)),
        main_cpu_ms = %four_decimal(ms(metrics.main_elapsed)),
        spec_cpu_ms = %four_decimal(ms(metrics.spec_elapsed)),
        main_gpu_ms = %OptionalMilliseconds(metrics.main_gpu_elapsed),
        rejection_gpu_ms = %OptionalMilliseconds(metrics.rejection_gpu_elapsed),
        spec_prepare_gpu_ms = %OptionalMilliseconds(metrics.spec_prepare_gpu_elapsed),
        spec_prefill_gpu_ms = %OptionalMilliseconds(metrics.spec_prefill_gpu_elapsed),
        spec_decode_gpu_ms = %OptionalMilliseconds(metrics.spec_decode_gpu_elapsed),
        spec_gpu_ms = %OptionalMilliseconds(metrics.spec_gpu_elapsed),
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
