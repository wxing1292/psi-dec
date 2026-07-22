use std::time::Duration;

use crate::runtime::Token;

pub const DEFAULT_SAMPLING_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_SAMPLING_TOP_K: usize = 20;
pub const DEFAULT_SAMPLING_TOP_P: f32 = 0.8;
pub const MAX_SAMPLING_TOP_K: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct CacheLaneRuntimeConfig {
    pub num_pages_per_kv_block: usize,
    pub num_pages_per_state_block: usize,
    pub block_cache_capacity: usize,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Logical token extent of one shared trie/GQA/GDN cache block.
    pub num_tokens_per_cache_block: usize,
    pub num_kv_heads: usize,
    pub kv_head_dim: usize,
    pub kv_dtype_bytes: usize,

    pub num_pages: usize,
    pub page_bytes: usize,
    pub cache_lanes: Vec<CacheLaneRuntimeConfig>,
}

impl RuntimeConfig {
    pub fn num_tokens_per_cache_block(&self) -> usize {
        self.num_tokens_per_cache_block
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.num_kv_heads * self.kv_head_dim * self.kv_dtype_bytes
    }

    pub fn num_tokens_per_page(&self) -> usize {
        let kv_bytes_per_token = self.kv_bytes_per_token();
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "page_bytes={} must be divisible by kv_bytes_per_token={}",
            self.page_bytes,
            kv_bytes_per_token
        );
        self.page_bytes / kv_bytes_per_token
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
    pub wait_duration: Duration,
    pub max_compute_slots: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ServiceConfig {
    pub user_req_queue_capacity: usize,
    pub batch_req_queue_capacity: usize,
    pub batch_resp_queue_capacity: usize,
    pub token_prob_channel_capacity: usize,
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
