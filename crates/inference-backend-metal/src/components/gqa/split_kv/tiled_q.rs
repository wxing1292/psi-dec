use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::components::gqa::kv_page_write::PageTableLayout;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../../metal/gqa_split_kv_tiled_q.metal");

/// SplitKV TiledQ SDPA (`T` = tokens, `H` = heads, `D` = head width):
///
/// ```text
/// Q: [Tq, Hq, D]       Q tile: [Tq_tile, Hq_tile, D]
/// K: [Tkv, Hkv, D]     K tile: [Tkv_tile, D]  (one fixed KV head)
/// V: [Tkv, Hkv, D]     V tile: [Tkv_tile, D]
/// O: [Tq, Hq, D]
/// Q tile [Tq_tile, Hq_tile, D] x K tile^T [D, Tkv_tile]
///   -> scores -> x V tile [Tkv_tile, D]
///   -> SDPAPartialOutput [Tq_tile, Hq_tile, D]
/// Map task template: { q_token_range_index, kv_token_begin, kv_token_end }
/// MapThreadBlockTask:
///   { q_token_range_index, kv_token_begin, kv_token_end } from the template
///   + { kv_head_index, q_head_range_index } from the grid
/// grid: (Hkv * Q-head ranges, Map task templates, 1)
/// threadblock: (Q-token fragments * Q-head range * 32, 1, 1)
/// parallel: Q-token ranges, KV heads, Q-head ranges, Q-token fragments
/// ordered/reduce: consecutive KV iterations merged with online softmax
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// Only the Map task template is materialized. It uses the shared three-`u32`
/// TaskTemplate ABI. The complete task is derived from the template, grid
/// coordinates, and specialization. A Q-token range never crosses a request
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_q_heads: u32,
    pub max_q_tokens: u32,
    pub kv_tokens_per_iteration: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub dtype: Dtype,
    pub page_table_layout: PageTableLayout,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.max_q_heads > 0);
        assert!(self.max_q_heads <= self.q_heads_per_kv_head());
        let tiled_q_profile = (self.head_dim, self.num_tokens_per_page());
        assert!(
            matches!(tiled_q_profile, (128, 8) | (256, 8 | 16)),
            "GQA SplitKV TiledQ supports only (head_dim, tokens_per_page) profiles (128, 8), (256, 8), and (256, 16), \
             got {tiled_q_profile:?}"
        );
        assert!(matches!(self.max_q_tokens, 8 | 16));
        assert!(matches!(self.kv_tokens_per_iteration, 8 | 16));
        assert!(self.required_threads() <= 256);
        assert!(self.scale > 0.0);
        assert_eq!(self.dtype, Dtype::Bfloat16, "GQA SplitKV TiledQ specializes bf16");
        self.page_table_layout.validate();
        assert_u32_count_domain(self.num_head_groups(), "GQA SplitKV TiledQ head groups");
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(self.dtype.item_size().try_into().expect("dtype size must fit u32")))
            .expect("GQA SplitKV TiledQ K/V bytes per token must fit u32");
        assert!(self.page_bytes.is_multiple_of(kv_bytes_per_token));
        self.page_bytes / kv_bytes_per_token
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }

    pub fn num_q_head_ranges_per_kv_head(self) -> u32 {
        self.q_heads_per_kv_head().div_ceil(self.max_q_heads)
    }

    fn num_head_groups(self) -> usize {
        checked_product(
            "GQA SplitKV TiledQ head-group count",
            &[
                self.num_kv_heads as usize,
                self.num_q_head_ranges_per_kv_head() as usize,
            ],
        )
    }

    pub fn required_threads(self) -> u32 {
        self.max_q_tokens
            .checked_div(8)
            .and_then(|threads| threads.checked_mul(self.max_q_heads))
            .and_then(|threads| threads.checked_mul(32))
            .expect("GQA SplitKV TiledQ threadblock width must fit u32")
    }

    pub fn q_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV TiledQ query byte length",
            &[
                shape.num_total_tokens as u64,
                self.num_q_heads as u64,
                self.head_dim as u64,
                self.dtype.item_size() as u64,
            ],
        )
    }

    pub fn partial_output_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV TiledQ partial output byte length",
            &[
                shape.num_total_sdpa_map_task_templates as u64,
                self.num_q_heads as u64,
                self.max_q_tokens as u64,
                self.head_dim as u64,
                self.dtype.item_size() as u64,
            ],
        )
    }

    pub fn partial_output_stats_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV TiledQ partial statistic byte length",
            &[
                shape.num_total_sdpa_map_task_templates as u64,
                self.num_q_heads as u64,
                self.max_q_tokens as u64,
                size_of::<f32>() as u64,
            ],
        )
    }

    pub fn map_threadblock_memory_bytes(self) -> usize {
        let padded_head_dim = self.head_dim as usize + 16 / self.dtype.item_size();
        checked_product(
            "GQA SplitKV TiledQ threadgroup memory byte length",
            &[
                2,
                self.kv_tokens_per_iteration as usize,
                padded_head_dim,
                self.dtype.item_size(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    pub num_total_q_token_tiles: u32,
    pub num_total_sdpa_map_task_templates: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_q_token_tiles > 0 && self.num_total_q_token_tiles <= self.num_total_tokens);
        assert!(self.num_total_sdpa_map_task_templates >= self.num_total_q_token_tiles);
        assert_u32_index_domain(
            self.num_q_token_range_values(),
            "GQA SplitKV TiledQ Q-token-range metadata",
        );
        assert_u32_index_domain(
            self.num_sdpa_map_task_template_values(),
            "GQA SplitKV TiledQ map TaskTemplate metadata",
        );
    }

    fn num_q_token_range_values(self) -> usize {
        checked_product(
            "GQA SplitKV TiledQ Q-token-range metadata element count",
            &[self.num_total_q_token_tiles as usize, 2],
        )
    }

    fn num_sdpa_map_task_template_values(self) -> usize {
        checked_product(
            "GQA SplitKV TiledQ map TaskTemplate metadata element count",
            &[self.num_total_sdpa_map_task_templates as usize, 3],
        )
    }

    fn num_cu_sdpa_partial_output_values(self) -> usize {
        self.num_total_q_token_tiles as usize + 1
    }
}

fn checked_product_u64(name: &str, factors: &[u64]) -> u64 {
    factors
        .iter()
        .try_fold(1u64, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit u64"))
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ReduceBuffers<'a> {
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    config: Config,
    shape: Shape,
    map: Kernel,
    reduce: Kernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config, shape: Shape) -> Self {
        shape.validate(config);
        assert!(
            config.map_threadblock_memory_bytes() <= device.max_threadblock_memory_length(),
            "GQA SplitKV TiledQ shape needs {} bytes of threadblock memory but device only supports {}",
            config.map_threadblock_memory_bytes(),
            device.max_threadblock_memory_length()
        );
        let source = source(config, shape);
        Self {
            config,
            shape,
            map: Kernel::new(device, &source, "gqa_split_kv_tiled_q_map"),
            reduce: Kernel::new(device, &source, "gqa_split_kv_tiled_q_reduce"),
        }
    }

    pub fn invoke_map<'a>(&self, buffers: MapBuffers<'a>, page_table_index: ReplayU32) -> MapInvocation<'a> {
        MapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            page_table_index,
            num_active_tokens: ReplayU32::Fixed(self.shape.num_total_tokens),
            num_active_q_token_tiles: ReplayU32::Fixed(self.shape.num_total_q_token_tiles),
            num_active_kv_splits: ReplayU32::Fixed(self.shape.num_total_sdpa_map_task_templates),
        }
    }

    pub fn invoke_map_bucketed<'a>(
        &self,
        buffers: MapBuffers<'a>,
        page_table_index: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_q_token_tiles: ReplayU32,
        num_active_kv_splits: ReplayU32,
    ) -> MapInvocation<'a> {
        MapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            page_table_index,
            num_active_tokens,
            num_active_q_token_tiles,
            num_active_kv_splits,
        }
    }

    pub fn invoke_reduce<'a>(&self, buffers: ReduceBuffers<'a>) -> ReduceInvocation<'a> {
        ReduceInvocation {
            pipeline: self.reduce.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            num_active_q_token_tiles: ReplayU32::Fixed(self.shape.num_total_q_token_tiles),
        }
    }

    pub fn invoke_reduce_bucketed<'a>(
        &self,
        buffers: ReduceBuffers<'a>,
        num_active_q_token_tiles: ReplayU32,
    ) -> ReduceInvocation<'a> {
        ReduceInvocation {
            pipeline: self.reduce.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            num_active_q_token_tiles,
        }
    }
}

pub struct MapInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    shape: Shape,
    buffers: MapBuffers<'a>,
    page_table_index: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_q_token_tiles: ReplayU32,
    num_active_kv_splits: ReplayU32,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        let config = self.config;
        shape.validate(config);
        assert!(self.buffers.q.len_bytes_u64() >= config.q_bytes(shape));
        assert!(self.buffers.kv_pages.len_bytes() >= config.page_bytes as usize);
        assert!(self.buffers.req_slots.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>());
        assert!(self.buffers.page_ids.len_bytes() >= config.page_table_layout.bytes());
        assert!(self.buffers.flat_token_indices.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>());
        assert!(self.buffers.q_token_ranges.len_bytes() >= shape.num_q_token_range_values() * size_of::<u32>());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes()
                >= shape.num_sdpa_map_task_template_values() * size_of::<u32>()
        );
        assert!(self.buffers.partial_output.len_bytes_u64() >= config.partial_output_bytes(shape));
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= config.partial_output_stats_bytes(shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= config.partial_output_stats_bytes(shape));
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.kv_pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.flat_token_indices, 0);
        recorder.set_buffer_read(5, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(6, self.buffers.sdpa_map_task_templates, 0);
        recorder.set_buffer_write(7, self.buffers.partial_output, 0);
        recorder.set_buffer_write(8, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(9, self.buffers.partial_max_logits, 0);
        let max_page_table_index = config.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => {
                assert!(
                    page_table_index <= max_page_table_index,
                    "GQA page-table index exceeds layer count"
                );
                recorder.set_u32(10, page_table_index);
            },
            ReplayU32::Parameter(key) => recorder.bind_u32(10, key, 0, max_page_table_index),
        }
        set_replay_u32(
            recorder,
            11,
            self.num_active_q_token_tiles,
            shape.num_total_q_token_tiles,
            "GQA SplitKV TiledQ active Q-token-tile count",
        );
        set_replay_u32(
            recorder,
            12,
            self.num_active_kv_splits,
            shape.num_total_sdpa_map_task_templates,
            "GQA SplitKV TiledQ active KV split count",
        );
        set_replay_u32(
            recorder,
            13,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA SplitKV TiledQ active token count",
        );
        recorder.set_threadblock_memory_length(0, config.map_threadblock_memory_bytes());
        recorder.dispatch_threadblocks(
            (
                config.num_head_groups(),
                shape.num_total_sdpa_map_task_templates as usize,
                1,
            ),
            (config.required_threads() as usize, 1, 1),
        );
    }
}

pub struct ReduceInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    shape: Shape,
    buffers: ReduceBuffers<'a>,
    num_active_q_token_tiles: ReplayU32,
}

impl Operator for ReduceInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        let config = self.config;
        shape.validate(config);
        assert!(self.buffers.partial_output.len_bytes_u64() >= config.partial_output_bytes(shape));
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= config.partial_output_stats_bytes(shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= config.partial_output_stats_bytes(shape));
        assert!(self.buffers.q_token_ranges.len_bytes() >= shape.num_q_token_range_values() * size_of::<u32>());
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes()
                >= shape.num_cu_sdpa_partial_output_values() * size_of::<u32>()
        );
        assert!(self.buffers.output.len_bytes_u64() >= config.q_bytes(shape));

        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.partial_output, 0);
        recorder.set_buffer_read(1, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_read(2, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_read(3, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(4, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(5, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            6,
            self.num_active_q_token_tiles,
            shape.num_total_q_token_tiles,
            "GQA SplitKV TiledQ active Q-token-tile count",
        );
        recorder.dispatch_threadblocks(
            (config.num_q_heads as usize, shape.num_total_q_token_tiles as usize, 1),
            (config.required_threads() as usize, 1, 1),
        );
    }
}

fn source(config: Config, shape: Shape) -> String {
    format!(
        r#"
#define NUM_TOKENS {num_tokens}
#define NUM_Q_HEADS {num_q_heads}
#define NUM_KV_HEADS {num_kv_heads}
#define MAX_Q_HEADS {max_q_heads}
#define NUM_Q_HEAD_RANGES_PER_KV_HEAD {num_q_head_ranges_per_kv_head}
#define HEAD_DIM {head_dim}
#define ATTENTION_SCALE {scale}
#define PAGE_BYTES {page_bytes}
#define NUM_TOKENS_PER_PAGE {num_tokens_per_page}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}
#define MAX_Q_TOKENS {max_q_tokens}
#define KV_TOKENS_PER_ITERATION {kv_tokens_per_iteration}
#define REQUIRED_THREADS {required_threads}
{body}
"#,
        num_tokens = shape.num_total_tokens,
        num_q_heads = config.num_q_heads,
        num_kv_heads = config.num_kv_heads,
        max_q_heads = config.max_q_heads,
        num_q_head_ranges_per_kv_head = config.num_q_head_ranges_per_kv_head(),
        head_dim = config.head_dim,
        scale = config.scale,
        page_bytes = config.page_bytes,
        num_tokens_per_page = config.num_tokens_per_page(),
        num_gqa_layers = config.page_table_layout.num_gqa_layers,
        num_blocks = config.page_table_layout.num_blocks,
        num_page_ids_per_block = config.page_table_layout.num_page_ids_per_block,
        max_q_tokens = config.max_q_tokens,
        kv_tokens_per_iteration = config.kv_tokens_per_iteration,
        required_threads = config.required_threads(),
        body = SOURCE,
    )
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

#[cfg(test)]
#[path = "tiled_q_test.rs"]
mod tests;
