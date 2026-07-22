use super::assert_u32_count_domain;
use super::assert_u32_index_domain;
use super::checked_product;
use crate::components::GQAPageTableLayout;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const SOURCE: &str = include_str!("metal/gqa_tiled_attention.metal");

/// Tiled SDPA (`T` = tokens, `H` = heads, `D` = head width):
///
/// ```text
/// Q: [Tq, Hq, D]       Q tile: [Tq_tile, Hq_tile, D]
/// K: [Tkv, Hkv, D]     K tile: [Tkv_tile, D]  (one fixed KV head)
/// V: [Tkv, Hkv, D]     V tile: [Tkv_tile, D]
/// O: [Tq, Hq, D]
/// Q tile [Tq_tile, Hq_tile, D] x K tile^T [D, Tkv_tile]
///   -> scores -> x V tile [Tkv_tile, D]
///   -> SDPAPartialOutput [Tq_tile, Hq_tile, D]
/// SDPAMapTile: (q_token_tile_index, kv_head_index,
///               q_head_tile_index, kv_token_tile_index)
/// SDPAMapTaskTemplate: { q_token_tile_index, kv_token_begin, kv_token_end }
/// SDPAMapTask / threadblock:
///   { q_token_tile_index, kv_token_begin, kv_token_end } from TaskTemplate
///   + { kv_head_index, q_head_tile_index } from grid
/// grid: (Hkv * Q-head tiles, TaskTemplates, 1)
/// threadblock: (Q-token fragments * Q-head tile * 32, 1, 1)
/// parallel: Q-token tiles, KV heads, Q-head tiles, Q-token fragments
/// ordered/reduce: consecutive KV tiles merged with online softmax
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// Only the TaskTemplate is materialized; the complete Task is comment-only
/// and does not change the three-`u32` ABI. A Q-token tile never crosses a
/// request boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQATiledSDPAShape {
    pub num_tokens: u32,
    pub num_q_token_tiles: u32,
    pub total_sdpa_map_task_templates: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub q_head_tile_size: u32,
    pub q_token_tile_size: u32,
    pub kv_token_tile_size: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub dtype: Dtype,
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
}

impl GQATiledSDPAShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_q_token_tiles > 0 && self.num_q_token_tiles <= self.num_tokens);
        assert!(self.total_sdpa_map_task_templates >= self.num_q_token_tiles);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.q_head_tile_size > 0);
        assert!(self.q_head_tile_size <= self.q_heads_per_kv_head());
        assert_eq!(self.head_dim, 256, "tiled GQA specializes head_dim=256");
        assert!(matches!(self.q_token_tile_size, 8 | 16));
        assert!(matches!(self.kv_token_tile_size, 8 | 16));
        assert!(self.num_threads_per_threadblock() <= 256);
        assert!(self.scale > 0.0);
        assert_eq!(self.dtype, Dtype::Bfloat16, "tiled GQA specializes bf16");
        assert_eq!(
            self.num_tokens_per_page(),
            16,
            "tiled GQA specializes 16 tokens per page"
        );
        self.page_table_layout.validate();
        assert!(self.gqa_layer_index < self.page_table_layout.num_gqa_layers);
        assert_u32_count_domain(self.num_head_groups(), "GQA tiled SDPA head groups");
        assert_u32_index_domain(self.num_q_token_tile_values(), "GQA tiled SDPA Q-token-tile metadata");
        assert_u32_index_domain(
            self.num_sdpa_map_task_template_values(),
            "GQA tiled SDPA map TaskTemplate metadata",
        );
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(self.dtype.item_size().try_into().expect("dtype size must fit u32")))
            .expect("GQA tiled K/V bytes per token must fit u32");
        assert!(self.page_bytes.is_multiple_of(kv_bytes_per_token));
        self.page_bytes / kv_bytes_per_token
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }

    pub fn num_q_head_tiles_per_kv_head(self) -> u32 {
        self.q_heads_per_kv_head().div_ceil(self.q_head_tile_size)
    }

    fn num_head_groups(self) -> usize {
        checked_product(
            "GQA tiled SDPA head-group count",
            &[self.num_kv_heads as usize, self.num_q_head_tiles_per_kv_head() as usize],
        )
    }

    fn num_q_token_tile_values(self) -> usize {
        checked_product(
            "GQA tiled SDPA Q-token-tile metadata element count",
            &[self.num_q_token_tiles as usize, 2],
        )
    }

    fn num_sdpa_map_task_template_values(self) -> usize {
        checked_product(
            "GQA tiled SDPA map TaskTemplate metadata element count",
            &[self.total_sdpa_map_task_templates as usize, 3],
        )
    }

    fn num_cu_sdpa_partial_output_values(self) -> usize {
        self.num_q_token_tiles as usize + 1
    }

    pub fn num_threads_per_threadblock(self) -> u32 {
        self.q_token_tile_size
            .checked_div(8)
            .and_then(|threads| threads.checked_mul(self.q_head_tile_size))
            .and_then(|threads| threads.checked_mul(32))
            .expect("GQA tiled threadblock width must fit u32")
    }

    pub fn q_bytes(self) -> u64 {
        checked_product_u64(
            "GQA tiled query byte length",
            &[
                u64::from(self.num_tokens),
                u64::from(self.num_q_heads),
                u64::from(self.head_dim),
                self.dtype.item_size().try_into().expect("dtype item size must fit u64"),
            ],
        )
    }

    pub fn partial_output_bytes(self) -> u64 {
        checked_product_u64(
            "GQA tiled partial output byte length",
            &[
                u64::from(self.total_sdpa_map_task_templates),
                u64::from(self.num_q_heads),
                u64::from(self.q_token_tile_size),
                u64::from(self.head_dim),
                self.dtype.item_size().try_into().expect("dtype item size must fit u64"),
            ],
        )
    }

    pub fn partial_output_stats_bytes(self) -> u64 {
        checked_product_u64(
            "GQA tiled partial statistic byte length",
            &[
                u64::from(self.total_sdpa_map_task_templates),
                u64::from(self.num_q_heads),
                u64::from(self.q_token_tile_size),
                size_of::<f32>().try_into().expect("f32 item size must fit u64"),
            ],
        )
    }

    pub fn map_threadblock_memory_bytes(self) -> usize {
        let padded_head_dim = self.head_dim as usize + 16 / self.dtype.item_size();
        checked_product(
            "GQA tiled threadgroup memory byte length",
            &[
                2,
                self.kv_token_tile_size as usize,
                padded_head_dim,
                self.dtype.item_size(),
            ],
        )
    }
}

fn checked_product_u64(name: &str, factors: &[u64]) -> u64 {
    factors
        .iter()
        .try_fold(1u64, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit u64"))
}

#[derive(Clone, Copy)]
pub struct GQATiledSDPAMapBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub q_token_tiles: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GQATiledSDPAReduceBuffers<'a> {
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub q_token_tiles: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct GQATiledSDPAKernels {
    device: Device,
}

impl GQATiledSDPAKernels {
    pub fn new(device: &Device) -> Self {
        Self { device: device.clone() }
    }

    pub fn invoke_map<'a>(
        &'a self,
        shape: GQATiledSDPAShape,
        buffers: GQATiledSDPAMapBuffers<'a>,
    ) -> GQATiledSDPAMapInvocation<'a> {
        GQATiledSDPAMapInvocation {
            device: &self.device,
            shape,
            buffers,
        }
    }

    pub fn invoke_reduce<'a>(
        &'a self,
        shape: GQATiledSDPAShape,
        buffers: GQATiledSDPAReduceBuffers<'a>,
    ) -> GQATiledSDPAReduceInvocation<'a> {
        GQATiledSDPAReduceInvocation {
            device: &self.device,
            shape,
            buffers,
        }
    }
}

pub struct GQATiledSDPAMapInvocation<'a> {
    device: &'a Device,
    shape: GQATiledSDPAShape,
    buffers: GQATiledSDPAMapBuffers<'a>,
}

impl Operator for GQATiledSDPAMapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        shape.validate();
        assert!(self.buffers.q.len_bytes_u64() >= shape.q_bytes());
        assert!(self.buffers.kv_pages.len_bytes() >= shape.page_bytes as usize);
        assert!(self.buffers.req_slots.len_bytes() >= shape.num_tokens as usize * size_of::<u32>());
        assert!(self.buffers.page_ids.len_bytes() >= shape.page_table_layout.bytes());
        assert!(self.buffers.flat_token_indices.len_bytes() >= shape.num_tokens as usize * size_of::<u32>());
        assert!(self.buffers.q_token_tiles.len_bytes() >= shape.num_q_token_tile_values() * size_of::<u32>());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes()
                >= shape.num_sdpa_map_task_template_values() * size_of::<u32>()
        );
        assert!(self.buffers.partial_output.len_bytes_u64() >= shape.partial_output_bytes());
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= shape.partial_output_stats_bytes());
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= shape.partial_output_stats_bytes());
        assert!(shape.map_threadblock_memory_bytes() <= self.device.max_threadblock_memory_length());

        let kernel = Kernel::new(self.device, &source(shape), "gqa_tiled_sdpa_map");
        recorder.set_kernel(&kernel);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.kv_pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.flat_token_indices, 0);
        recorder.set_buffer_read(5, self.buffers.q_token_tiles, 0);
        recorder.set_buffer_read(6, self.buffers.sdpa_map_task_templates, 0);
        recorder.set_buffer_write(7, self.buffers.partial_output, 0);
        recorder.set_buffer_write(8, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(9, self.buffers.partial_max_logits, 0);
        recorder.set_threadblock_memory_length(0, shape.map_threadblock_memory_bytes());
        recorder.dispatch_threadblocks(
            (shape.num_head_groups(), shape.total_sdpa_map_task_templates as usize, 1),
            (shape.num_threads_per_threadblock() as usize, 1, 1),
        );
    }
}

pub struct GQATiledSDPAReduceInvocation<'a> {
    device: &'a Device,
    shape: GQATiledSDPAShape,
    buffers: GQATiledSDPAReduceBuffers<'a>,
}

impl Operator for GQATiledSDPAReduceInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        shape.validate();
        assert!(self.buffers.partial_output.len_bytes_u64() >= shape.partial_output_bytes());
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= shape.partial_output_stats_bytes());
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= shape.partial_output_stats_bytes());
        assert!(self.buffers.q_token_tiles.len_bytes() >= shape.num_q_token_tile_values() * size_of::<u32>());
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes()
                >= shape.num_cu_sdpa_partial_output_values() * size_of::<u32>()
        );
        assert!(self.buffers.output.len_bytes_u64() >= shape.q_bytes());

        let kernel = Kernel::new(self.device, &source(shape), "gqa_tiled_sdpa_reduce");
        recorder.set_kernel(&kernel);
        recorder.set_buffer_read(0, self.buffers.partial_output, 0);
        recorder.set_buffer_read(1, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_read(2, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_read(3, self.buffers.q_token_tiles, 0);
        recorder.set_buffer_read(4, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(5, self.buffers.output, 0);
        recorder.dispatch_threadblocks(
            (shape.num_q_heads as usize, shape.num_q_token_tiles as usize, 1),
            (shape.num_threads_per_threadblock() as usize, 1, 1),
        );
    }
}

fn source(shape: GQATiledSDPAShape) -> String {
    format!(
        r#"
#define NUM_TOKENS {num_tokens}
#define NUM_Q_TOKEN_TILES {num_q_token_tiles}
#define NUM_Q_HEADS {num_q_heads}
#define NUM_KV_HEADS {num_kv_heads}
#define Q_HEAD_TILE_SIZE {q_head_tile_size}
#define NUM_Q_HEAD_TILES_PER_KV_HEAD {num_q_head_tiles_per_kv_head}
#define HEAD_DIM {head_dim}
#define ATTENTION_SCALE {scale}
#define PAGE_BYTES {page_bytes}
#define NUM_TOKENS_PER_PAGE {num_tokens_per_page}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}
#define GQA_LAYER_INDEX {gqa_layer_index}
#define Q_TOKEN_TILE_SIZE {q_token_tile_size}
#define KV_TOKEN_TILE_SIZE {kv_token_tile_size}
#define NUM_THREADS_PER_THREADBLOCK {num_threads_per_threadblock}
{body}
"#,
        num_tokens = shape.num_tokens,
        num_q_token_tiles = shape.num_q_token_tiles,
        num_q_heads = shape.num_q_heads,
        num_kv_heads = shape.num_kv_heads,
        q_head_tile_size = shape.q_head_tile_size,
        num_q_head_tiles_per_kv_head = shape.num_q_head_tiles_per_kv_head(),
        head_dim = shape.head_dim,
        scale = shape.scale,
        page_bytes = shape.page_bytes,
        num_tokens_per_page = shape.num_tokens_per_page(),
        num_gqa_layers = shape.page_table_layout.num_gqa_layers,
        num_blocks = shape.page_table_layout.num_blocks,
        num_page_ids_per_block = shape.page_table_layout.num_page_ids_per_block,
        gqa_layer_index = shape.gqa_layer_index,
        q_token_tile_size = shape.q_token_tile_size,
        kv_token_tile_size = shape.kv_token_tile_size,
        num_threads_per_threadblock = shape.num_threads_per_threadblock(),
        body = SOURCE,
    )
}

#[cfg(test)]
mod tests {
    use super::GQATiledSDPAShape;
    use crate::components::GQAPageTableLayout;
    use crate::metal::Dtype;

    #[test]
    #[should_panic(expected = "GQA tiled SDPA Q-token-tile metadata exceeds the shader u32 element-index domain")]
    fn test_shape_rejects_shader_index_overflow() {
        GQATiledSDPAShape {
            num_tokens: u32::MAX,
            num_q_token_tiles: u32::MAX,
            total_sdpa_map_task_templates: u32::MAX,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 256,
            q_head_tile_size: 1,
            q_token_tile_size: 8,
            kv_token_tile_size: 8,
            scale: 1.0,
            page_bytes: 16 * 2 * 256 * 2,
            dtype: Dtype::Bfloat16,
            page_table_layout: GQAPageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
            gqa_layer_index: 0,
        }
        .validate();
    }
}
