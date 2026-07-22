use tracing_subscriber::EnvFilter;

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
pub struct ServiceObservabilityConfig {
    pub profiling: ProfilingConfig,
    pub debug_logging: bool,
}

impl ServiceObservabilityConfig {
    pub fn init(self) -> ServiceObservability {
        let profiling_requested = self.profiling.is_requested();
        profiling::set_profiling_summary_every(self.profiling.summary_every);
        init_tracing(profiling_requested, self.debug_logging);

        tracing::info!(
            profile_mode = ?self.profiling.mode,
            profile_enabled = self.profiling.is_requested(),
            profile_summary_every_batches = self.profiling.summary_every,
            logging = if self.debug_logging { "debug" } else { "info" },
            "runtime service observability initialized"
        );

        ServiceObservability { profiling_requested }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ServiceObservability {
    profiling_requested: bool,
}

impl ServiceObservability {
    pub fn profiling_requested(&self) -> bool {
        self.profiling_requested
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
