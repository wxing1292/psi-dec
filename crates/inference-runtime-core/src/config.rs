use std::str::FromStr;
use std::time::Duration;

use crate::runtime::Token;

pub const DEFAULT_SAMPLING_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_SAMPLING_TOP_K: usize = 20;
pub const DEFAULT_SAMPLING_TOP_P: f32 = 0.8;
pub const DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT: Duration = Duration::from_secs(300);
pub const MAX_SAMPLING_TOP_K: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorHibernationMode {
    All,
    Selected,
}

impl FromStr for ExecutorHibernationMode {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "all" => Ok(Self::All),
            "selected" => Ok(Self::Selected),
            _ => Err("executor hibernation mode must be 'all' or 'selected'"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CacheLaneRuntimeConfig {
    pub num_pages_per_kv_block: usize,
    pub num_pages_per_state_block: usize,
    pub block_cache_capacity: usize,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub max_running_requests: usize,
    pub executor_hibernation_timeout: Duration,
    pub executor_hibernation_mode: ExecutorHibernationMode,
    pub context_window: usize,

    /// Logical token extent of one shared trie/GQA/GDN cache block.
    pub num_tokens_per_cache_block: usize,

    pub num_pages: usize,
    pub cache_lanes: Vec<CacheLaneRuntimeConfig>,
}

impl RuntimeConfig {
    pub fn num_tokens_per_cache_block(&self) -> usize {
        self.num_tokens_per_cache_block
    }

    pub fn cache_lane(&self, cache_lane: usize) -> &CacheLaneRuntimeConfig {
        self.cache_lanes
            .get(cache_lane)
            .unwrap_or_else(|| panic!("cache lane {cache_lane} is not configured"))
    }

    pub fn num_cache_lanes(&self) -> usize {
        self.cache_lanes.len()
    }

    pub fn num_pages_per_kv_block(&self, cache_lane: usize) -> usize {
        self.cache_lane(cache_lane).num_pages_per_kv_block
    }

    pub fn num_pages_per_state_block(&self, cache_lane: usize) -> usize {
        self.cache_lane(cache_lane).num_pages_per_state_block
    }

    pub fn block_cache_capacity(&self, cache_lane: usize) -> usize {
        self.cache_lane(cache_lane).block_cache_capacity
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub max_compute_slots: usize,
}

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub max_sampled_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u32>,
    pub stop_sequences: Vec<Vec<Token>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            max_sampled_tokens: 16,
            temperature: DEFAULT_SAMPLING_TEMPERATURE,
            top_k: DEFAULT_SAMPLING_TOP_K,
            top_p: DEFAULT_SAMPLING_TOP_P,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}
