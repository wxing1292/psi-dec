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

pub fn emit_executor_batch_perf_metrics(
    debug_logging: bool,
    model_name: &str,
    batch_seq: u64,
    batch_summary: BatchRequestSummary,
    response_summary: BatchResponseSummary,
    metrics: ExecutorBatchPerfMetrics,
) {
    if debug_logging {
        emit_executor_batch_perf_debug(model_name, batch_seq, &batch_summary, &response_summary, &metrics);
    } else {
        emit_executor_batch_perf_info(model_name, batch_seq, &batch_summary, &response_summary, &metrics);
    }
}

fn emit_executor_batch_perf_info(
    model_name: &str,
    batch_seq: u64,
    batch_summary: &BatchRequestSummary,
    response_summary: &BatchResponseSummary,
    metrics: &ExecutorBatchPerfMetrics,
) {
    let acceptance_rate = ratio(response_summary.accepted_spec_tokens, batch_summary.input_spec_tokens);
    tracing::info!(
        target: "inference-runtime-service::perf",
        phase = "executor.batch.perf",
        model = model_name,
        batch_seq,
        num_reqs = batch_summary.num_reqs,
        num_tokens = batch_summary.query_tokens,
        num_spec_tokens = batch_summary.input_spec_tokens,
        num_accepted_tokens = response_summary.accepted_spec_tokens,
        num_sampled_tokens = response_summary.sampled_tokens,
        acceptance_rate = %four_decimal(acceptance_rate),
        latency_ms = %four_decimal(ms(metrics.total_elapsed)),
        "executor batch perf"
    );
}

fn emit_executor_batch_perf_debug(
    model_name: &str,
    batch_seq: u64,
    batch_summary: &BatchRequestSummary,
    response_summary: &BatchResponseSummary,
    metrics: &ExecutorBatchPerfMetrics,
) {
    let rejected_spec_tokens = batch_summary
        .input_spec_tokens
        .saturating_sub(response_summary.accepted_spec_tokens);
    let acceptance_rate = ratio(response_summary.accepted_spec_tokens, batch_summary.input_spec_tokens);
    let rejection_rate = ratio(rejected_spec_tokens, batch_summary.input_spec_tokens);

    tracing::debug!(
        target: "inference-runtime-service::perf",
        phase = "executor.batch.perf",
        model = model_name,
        batch_seq,
        num_reqs = batch_summary.num_reqs,
        num_tokens = batch_summary.query_tokens,
        num_spec_tokens = batch_summary.input_spec_tokens,
        num_accepted_tokens = response_summary.accepted_spec_tokens,
        num_sampled_tokens = response_summary.sampled_tokens,
        acceptance_rate = %four_decimal(acceptance_rate),
        latency_ms = %four_decimal(ms(metrics.total_elapsed)),
        num_prefill_reqs = batch_summary.prefill_reqs,
        num_decode_reqs = batch_summary.decode_reqs,
        max_num_tokens = batch_summary.max_query_tokens,
        num_spec_reqs = batch_summary.spec_decode_reqs,
        num_rejected_tokens = rejected_spec_tokens,
        num_output_spec_tokens = response_summary.output_spec_tokens,
        rejection_rate = %four_decimal(rejection_rate),
        sampled_rows = metrics.sampled_rows,
        model_output_main_replay_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.main_replay_elapsed)))),
        model_output_main_sample_replay_ms =
            metrics.model_output_timing.map(|timing| {
                tracing::field::display(four_decimal(ms(timing.main_sample_replay_elapsed)))
            }),
        model_output_sample_read_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.sample_read_elapsed)))),
        model_output_rejection_build_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.rejection_build_elapsed)))),
        model_output_rejection_read_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.rejection_read_elapsed)))),
        model_output_spec_build_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.spec_build_elapsed)))),
        model_output_spec_replay_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.spec_replay_elapsed)))),
        model_output_spec_read_ms = metrics
            .model_output_timing
            .map(|timing| tracing::field::display(four_decimal(ms(timing.spec_read_elapsed)))),
        model_output_spec_passes = metrics.model_output_timing.map(|timing| timing.spec_passes),
        prepare_batch_ms = %four_decimal(ms(metrics.prepare_batch_elapsed)),
        input_ms = %four_decimal(ms(metrics.input_elapsed)),
        model_ms = %four_decimal(ms(metrics.model_elapsed)),
        output_ms = %four_decimal(ms(metrics.output_elapsed)),
        commit_batch_ms = %four_decimal(ms(metrics.commit_batch_elapsed)),
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
    fn four_decimal_pads_and_rounds() {
        assert_eq!(four_decimal(1.0).to_string(), "1.0000");
        assert_eq!(four_decimal(2.0 / 7.0).to_string(), "0.2857");
    }
}
