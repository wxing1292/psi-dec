use super::assert_u32_count_domain;
use super::assert_u32_index_domain;
use super::checked_product;
use crate::components::gqa_kv_pages::GQAPageTableLayout;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_PAGED_SDPA_MAP_BODY: &str = include_str!("metal/gqa_paged_sdpa_map.metal");
const GQA_PAGED_SDPA_REDUCE_SOURCE: &str = include_str!("metal/gqa_paged_sdpa_reduce.metal");
const GQA_ACTIVATION_GATE_SOURCE: &str = include_str!("metal/gqa_activation_gate.metal");

/// Paged context-parallel SDPA (`T` = tokens, `H` = heads, `D` = head width):
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
/// grid: (total TaskTemplates * Hkv * Q-head tiles, 1, 1), flattened
/// threadblock: (configured width, 1, 1)
/// parallel: TaskTemplates, KV heads, Q-head tiles
/// ordered/reduce: consecutive KV tiles merged with online softmax
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// This path uses `Tq_tile = 1`. Only the TaskTemplate is materialized; the
/// complete Task is comment-only and does not change the three-`u32` ABI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQAPagedSDPAConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub kv_token_tile_size: u32,
    pub num_threads_per_threadblock: u32,
    pub q_head_tile_size: u32,
    pub dtype: Dtype,
}

impl GQAPagedSDPAConfig {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.scale > 0.0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.num_tokens_per_page() > 0);
        self.page_table_layout.validate();
        assert!(self.gqa_layer_index < self.page_table_layout.num_gqa_layers);
        assert!(self.kv_token_tile_size > 0 && self.kv_token_tile_size <= 1024);
        assert!(self.num_threads_per_threadblock.is_power_of_two() && self.num_threads_per_threadblock <= 256);
        assert!(self.q_head_tile_size > 0 && self.q_head_tile_size <= 8);
        assert!(self.q_head_tile_size <= self.q_heads_per_kv_head());
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(self.dtype.item_size().try_into().expect("dtype size must fit u32")))
            .expect("GQA SDPA K/V bytes per token must fit u32");
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        self.page_bytes / kv_bytes_per_token
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }

    pub fn q_head_tile_size(self) -> u32 {
        self.q_head_tile_size
    }

    pub fn context_parallel_threadblock_memory_bytes(self) -> usize {
        (self.q_head_tile_size as usize * self.kv_token_tile_size as usize + self.num_threads_per_threadblock as usize)
            * size_of::<f32>()
    }

    pub fn num_q_head_tiles_per_kv_head(self) -> u32 {
        self.q_heads_per_kv_head().div_ceil(self.q_head_tile_size())
    }

    pub fn map_threads(self, shape: GQAPagedSDPAShape) -> usize {
        checked_product(
            "GQA SDPA map thread count",
            &[
                shape.total_sdpa_map_task_templates as usize,
                self.num_kv_heads as usize,
                self.num_q_head_tiles_per_kv_head() as usize,
                self.num_threads_per_threadblock as usize,
            ],
        )
    }

    pub fn num_output_values(self, shape: GQAPagedSDPAShape) -> usize {
        checked_product(
            "GQA SDPA output element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_sdpa_partial_output_stats(self, shape: GQAPagedSDPAShape) -> usize {
        checked_product(
            "GQA SDPA partial-output statistic count",
            &[shape.total_sdpa_map_task_templates as usize, self.num_q_heads as usize],
        )
    }

    pub fn num_partial_output_values(self, shape: GQAPagedSDPAShape) -> usize {
        self.num_sdpa_partial_output_stats(shape)
            .checked_mul(self.head_dim as usize)
            .expect("GQA SDPA partial output element count must fit usize")
    }

    pub fn q_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::try_from(self.num_output_values(shape))
            .expect("GQA SDPA query element count must fit u64")
            .checked_mul(self.dtype.item_size().try_into().expect("dtype item size must fit u64"))
            .expect("GQA SDPA query byte length must fit u64")
    }

    pub fn kv_pages_bytes(self) -> usize {
        self.page_bytes as usize
    }

    pub fn req_slots_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::from(shape.num_tokens)
            .checked_mul(size_of::<u32>().try_into().expect("u32 item size must fit u64"))
            .expect("GQA SDPA request-slot byte length must fit u64")
    }

    pub fn page_ids_bytes(self) -> u64 {
        self.page_table_layout
            .bytes()
            .try_into()
            .expect("GQA page-table byte length must fit u64")
    }

    pub fn sdpa_map_task_templates_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::from(shape.total_sdpa_map_task_templates)
            .checked_mul(3)
            .and_then(|count| count.checked_mul(size_of::<u32>().try_into().expect("u32 item size must fit u64")))
            .expect("GQA SDPA map TaskTemplate byte length must fit u64")
    }

    pub fn cu_sdpa_partial_outputs_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::from(shape.num_tokens)
            .checked_add(1)
            .and_then(|count| count.checked_mul(size_of::<u32>().try_into().expect("u32 item size must fit u64")))
            .expect("GQA SDPA partial-output cumulative-count byte length must fit u64")
    }

    pub fn partial_output_stats_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::try_from(self.num_sdpa_partial_output_stats(shape))
            .expect("GQA SDPA statistic element count must fit u64")
            .checked_mul(size_of::<f32>().try_into().expect("f32 item size must fit u64"))
            .expect("GQA SDPA statistic byte length must fit u64")
    }

    pub fn partial_output_bytes(self, shape: GQAPagedSDPAShape) -> u64 {
        u64::try_from(self.num_partial_output_values(shape))
            .expect("GQA SDPA partial output element count must fit u64")
            .checked_mul(self.dtype.item_size().try_into().expect("dtype item size must fit u64"))
            .expect("GQA SDPA partial output byte length must fit u64")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAPagedSDPAShape {
    pub num_tokens: u32,
    pub total_sdpa_map_task_templates: u32,
}

impl GQAPagedSDPAShape {
    pub fn validate(self, config: GQAPagedSDPAConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert!(self.total_sdpa_map_task_templates > 0);
        assert_u32_count_domain(config.num_output_values(self), "GQA SDPA query/output");
        assert_u32_index_domain(
            config.num_sdpa_partial_output_stats(self),
            "GQA SDPA partial-output stats",
        );
        assert_u32_index_domain(config.num_partial_output_values(self), "GQA SDPA partial output");
        assert_u32_count_domain(config.map_threads(self), "GQA SDPA map threads");
    }
}

#[derive(Clone, Copy)]
pub struct GQAPagedSDPAMapBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GQAPagedSDPAReduceBuffers<'a> {
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct GQAPagedSDPAScratch {
    pub partial_exp_sums: Buffer,
    pub partial_max_logits: Buffer,
    pub partial_output: Buffer,
}

impl GQAPagedSDPAScratch {
    pub fn new(device: &Device, config: GQAPagedSDPAConfig, shape: GQAPagedSDPAShape) -> Self {
        shape.validate(config);
        Self {
            partial_exp_sums: Buffer::new_zeroed(device, config.partial_output_stats_bytes(shape)),
            partial_max_logits: Buffer::new_zeroed(device, config.partial_output_stats_bytes(shape)),
            partial_output: Buffer::new_zeroed(device, config.partial_output_bytes(shape)),
        }
    }

    pub fn map_buffers<'a>(
        &'a self,
        q: &'a Buffer,
        kv_pages: &'a Buffer,
        req_slots: &'a Buffer,
        page_ids: &'a Buffer,
        sdpa_map_task_templates: &'a Buffer,
    ) -> GQAPagedSDPAMapBuffers<'a> {
        GQAPagedSDPAMapBuffers {
            q,
            kv_pages,
            req_slots,
            page_ids,
            sdpa_map_task_templates,
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
        }
    }

    pub fn reduce_buffers<'a>(
        &'a self,
        cu_sdpa_partial_outputs: &'a Buffer,
        output: &'a Buffer,
    ) -> GQAPagedSDPAReduceBuffers<'a> {
        GQAPagedSDPAReduceBuffers {
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
            cu_sdpa_partial_outputs,
            output,
        }
    }
}

pub struct GQAPagedSDPAKernels {
    device: Device,
}

impl GQAPagedSDPAKernels {
    pub fn new(device: &Device) -> Self {
        Self { device: device.clone() }
    }

    pub fn invoke_map<'a>(
        &'a self,
        config: GQAPagedSDPAConfig,
        shape: GQAPagedSDPAShape,
        buffers: GQAPagedSDPAMapBuffers<'a>,
    ) -> GQAPagedSDPAMapInvocation<'a> {
        GQAPagedSDPAMapInvocation {
            device: &self.device,
            config,
            shape,
            buffers,
        }
    }

    pub fn invoke_reduce<'a>(
        &'a self,
        config: GQAPagedSDPAConfig,
        shape: GQAPagedSDPAShape,
        buffers: GQAPagedSDPAReduceBuffers<'a>,
    ) -> GQAPagedSDPAReduceInvocation<'a> {
        GQAPagedSDPAReduceInvocation {
            device: &self.device,
            config,
            shape,
            buffers,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GQAActivationGateConfig {
    pub num_q_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl GQAActivationGateConfig {
    pub fn f32(num_q_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            head_dim,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_q_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            head_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.head_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_values(self, shape: GQAActivationGateShape) -> usize {
        checked_product(
            "GQA activation/gate element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn bytes(self, shape: GQAActivationGateShape) -> usize {
        self.num_values(shape)
            .checked_mul(self.dtype.item_size())
            .expect("GQA activation/gate byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GQAActivationGateShape {
    pub num_tokens: u32,
}

impl GQAActivationGateShape {
    pub fn validate(self, config: GQAActivationGateConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_u32_count_domain(config.num_values(self), "GQA activation/gate");
    }
}

#[derive(Clone, Copy)]
pub struct GQAActivationGateBuffers<'a> {
    pub attention_output: &'a Buffer,
    pub g: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct GQAActivationGateKernel {
    config: GQAActivationGateConfig,
    kernel: Kernel,
}

impl GQAActivationGateKernel {
    pub fn new(device: &Device, config: GQAActivationGateConfig) -> Self {
        config.validate();
        let source = activation_gate_source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_activation_gate_f32",
            Dtype::Bfloat16 => "gqa_activation_gate_bf16",
            dtype => panic!("unsupported GQA activation gate dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQAActivationGateShape,
        buffers: GQAActivationGateBuffers<'a>,
    ) -> GQAActivationGateInvocation<'a> {
        GQAActivationGateInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn activation_gate_source(config: GQAActivationGateConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    GQA_ACTIVATION_GATE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQAActivationGateInvocation<'a> {
    config: GQAActivationGateConfig,
    kernel: &'a Kernel,
    shape: GQAActivationGateShape,
    buffers: GQAActivationGateBuffers<'a>,
}

impl Operator for GQAActivationGateInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQAActivationGateInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.attention_output.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }

    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.attention_output, 0);
        builder.set_buffer_read(1, self.buffers.g, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, shape.num_tokens);
        builder.dispatch_1d(self.config.num_values(shape), 256);
    }
}

fn gqa_paged_sdpa_map_source(config: GQAPagedSDPAConfig, shape: GQAPagedSDPAShape) -> String {
    let dtype = metal_dtype_name(config.dtype);
    let body = GQA_PAGED_SDPA_MAP_BODY
        .replace("uint global_thread_index = thread_position_in_grid.x;\n", "")
        .replace(
            "int num_blocks = page_table_layout[2];",
            &format!("int num_blocks = {};", config.page_table_layout.num_blocks),
        );
    assert!(!body.contains("thread_position_in_grid"));
    assert!(!body.contains("q_shape"));
    assert!(!body.contains("page_table_layout"));
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
#define T {dtype}
#define KV_T {dtype}
#define NUM_Q_HEADS {num_q_heads}
#define NUM_KV_HEADS {num_kv_heads}
#define KV_HEAD_DIM {head_dim}
#define ATTENTION_SCALE {scale}
#define Q_HEADS_PER_KV_HEAD {q_heads_per_kv_head}
#define Q_HEAD_TILE_SIZE {q_head_tile_size}
#define NUM_Q_HEAD_TILES_PER_KV_HEAD {num_q_head_tiles_per_kv_head}
#define NUM_TOKENS {num_tokens}
#define NUM_ACTIVE_TOKENS {num_active_tokens}
#define PAGE_BYTES {page_bytes}
#define KV_TOKEN_TILE_SIZE {kv_token_tile_size}
#define TOTAL_SDPA_MAP_TASK_TEMPLATES {total_sdpa_map_task_templates}
#define NUM_THREADS_PER_THREADBLOCK {num_threads_per_threadblock}
#define GQA_LAYER_INDEX {gqa_layer_index}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}

kernel void gqa_paged_sdpa_map(
    device const T* q [[buffer(0)]],
    device const KV_T* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* sdpa_map_task_templates [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device T* partial_output [[buffer(7)]],
    uint global_thread_index [[thread_position_in_grid]]
) {{
{body}
}}
"#,
        dtype = dtype,
        head_dim = config.head_dim,
        scale = config.scale,
        q_heads_per_kv_head = config.q_heads_per_kv_head(),
        q_head_tile_size = config.q_head_tile_size(),
        num_q_head_tiles_per_kv_head = config.num_q_head_tiles_per_kv_head(),
        num_kv_heads = config.num_kv_heads,
        num_tokens = config.num_tokens_per_page(),
        num_active_tokens = shape.num_tokens,
        num_q_heads = config.num_q_heads,
        page_bytes = config.page_bytes,
        kv_token_tile_size = config.kv_token_tile_size,
        total_sdpa_map_task_templates = shape.total_sdpa_map_task_templates,
        num_threads_per_threadblock = config.num_threads_per_threadblock,
        gqa_layer_index = config.gqa_layer_index,
        num_gqa_layers = config.page_table_layout.num_gqa_layers,
        num_blocks = config.page_table_layout.num_blocks,
        num_page_ids_per_block = config.page_table_layout.num_page_ids_per_block,
        body = body,
    )
}

fn metal_dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Float32 => "float",
        Dtype::Bfloat16 => "bfloat16_t",
        unsupported_dtype => panic!("unsupported GQA paged SDPA map dtype {unsupported_dtype:?}"),
    }
}

pub struct GQAPagedSDPAMapInvocation<'a> {
    device: &'a Device,
    config: GQAPagedSDPAConfig,
    shape: GQAPagedSDPAShape,
    buffers: GQAPagedSDPAMapBuffers<'a>,
}

impl Operator for GQAPagedSDPAMapInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQAPagedSDPAMapInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.q.len_bytes_u64() >= self.config.q_bytes(self.shape));
        assert!(self.buffers.kv_pages.len_bytes_u64() >= u64::from(self.config.page_bytes));
        assert!(self.buffers.req_slots.len_bytes_u64() >= self.config.req_slots_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes_u64() >= self.config.page_ids_bytes());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes_u64()
                >= self.config.sdpa_map_task_templates_bytes(self.shape)
        );
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_output.len_bytes_u64() >= self.config.partial_output_bytes(self.shape));
        assert!(
            self.config.context_parallel_threadblock_memory_bytes() <= self.device.max_threadblock_memory_length(),
            "GQA context-parallel SDPA shape needs {} bytes of threadblock memory but device only supports {}",
            self.config.context_parallel_threadblock_memory_bytes(),
            self.device.max_threadblock_memory_length()
        );
    }

    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        let source = gqa_paged_sdpa_map_source(self.config, shape);
        let kernel = Kernel::new(self.device, &source, "gqa_paged_sdpa_map");
        builder.set_kernel(&kernel);
        builder.set_buffer_read(0, self.buffers.q, 0);
        builder.set_buffer_read(1, self.buffers.kv_pages, 0);
        builder.set_buffer_read(2, self.buffers.req_slots, 0);
        builder.set_buffer_read(3, self.buffers.page_ids, 0);
        builder.set_buffer_read(4, self.buffers.sdpa_map_task_templates, 0);
        builder.set_buffer_write(5, self.buffers.partial_exp_sums, 0);
        builder.set_buffer_write(6, self.buffers.partial_max_logits, 0);
        builder.set_buffer_write(7, self.buffers.partial_output, 0);
        builder.dispatch_1d(
            self.config.map_threads(shape),
            self.config.num_threads_per_threadblock as usize,
        );
    }
}

pub struct GQAPagedSDPAReduceInvocation<'a> {
    device: &'a Device,
    config: GQAPagedSDPAConfig,
    shape: GQAPagedSDPAShape,
    buffers: GQAPagedSDPAReduceBuffers<'a>,
}

impl Operator for GQAPagedSDPAReduceInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQAPagedSDPAReduceInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_output.len_bytes_u64() >= self.config.partial_output_bytes(self.shape));
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes_u64()
                >= self.config.cu_sdpa_partial_outputs_bytes(self.shape)
        );
        assert!(self.buffers.output.len_bytes_u64() >= self.config.q_bytes(self.shape));
    }

    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        let source = gqa_paged_sdpa_reduce_source(self.config);
        let function_name = match self.config.dtype {
            Dtype::Float32 => "gqa_paged_sdpa_reduce_f32",
            Dtype::Bfloat16 => "gqa_paged_sdpa_reduce_bf16",
            dtype => panic!("unsupported GQA paged SDPA reduce dtype {dtype:?}"),
        };
        let kernel = Kernel::new(self.device, &source, function_name);
        builder.set_kernel(&kernel);
        builder.set_buffer_read(0, self.buffers.partial_exp_sums, 0);
        builder.set_buffer_read(1, self.buffers.partial_max_logits, 0);
        builder.set_buffer_read(2, self.buffers.partial_output, 0);
        builder.set_buffer_read(3, self.buffers.cu_sdpa_partial_outputs, 0);
        builder.set_buffer_write(4, self.buffers.output, 0);
        builder.set_u32(5, shape.num_tokens);
        builder.dispatch_1d(self.config.num_output_values(shape), 256);
    }
}

fn gqa_paged_sdpa_reduce_source(config: GQAPagedSDPAConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    GQA_PAGED_SDPA_REDUCE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::gqa::GQACore;
    use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
    use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

    use super::*;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "GQA SDPA query/output exceeds the shader u32 count domain")]
    fn test_sdpa_shape_rejects_shader_count_overflow() {
        let config = GQAPagedSDPAConfig {
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            scale: 1.0,
            page_bytes: 8,
            page_table_layout: GQAPageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
            gqa_layer_index: 0,
            kv_token_tile_size: 1,
            num_threads_per_threadblock: 32,
            q_head_tile_size: 1,
            dtype: Dtype::Bfloat16,
        };
        GQAPagedSDPAShape {
            num_tokens: 1 << 30,
            total_sdpa_map_task_templates: 1,
        }
        .validate(config);
    }

    #[test]
    #[should_panic(expected = "GQA activation/gate exceeds the shader u32 count domain")]
    fn test_activation_gate_shape_rejects_shader_count_overflow() {
        GQAActivationGateShape { num_tokens: 1 << 30 }.validate(GQAActivationGateConfig::bf16(2, 2));
    }

    #[test]
    fn test_fixed() {
        let config = fixture_config();
        let shape = fixture_shape();
        let q = fixture_values(config.num_output_values(shape), 0.125, 3);
        let k = fixture_values(2 * config.num_kv_heads as usize * config.head_dim as usize, 0.0625, 5);
        let v = fixture_values(2 * config.num_kv_heads as usize * config.head_dim as usize, 0.25, 7);
        let kv_pages = kv_page_values(config, &[(&k, &v)]);
        let actual = run_gqa_paged_sdpa(
            config,
            shape,
            GQAPagedSDPATestInput {
                q: &q,
                kv_pages: &kv_pages,
                req_slots: &[0, 0],
                page_ids: &[0],
                flat_token_indices: &[0, 1],
            },
        );
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, 2],
                token_indices: &[0],
                q: &q,
                context_k_by_req: &[&k],
                context_v_by_req: &[&v],
            },
        );
        assert_close(&actual, &expected, 2.0e-5);
    }

    #[test]
    fn test_random() {
        let random_seed = 0x4C8F_17D2;
        let config = fixture_config();
        let shape = fixture_shape();
        let q = generated_values(config.num_output_values(shape), random_seed);
        let k = generated_values(
            2 * config.num_kv_heads as usize * config.head_dim as usize,
            random_seed.wrapping_add(1),
        );
        let v = generated_values(
            2 * config.num_kv_heads as usize * config.head_dim as usize,
            random_seed.wrapping_add(2),
        );
        let kv_pages = kv_page_values(config, &[(&k, &v)]);
        let actual = run_gqa_paged_sdpa(
            config,
            shape,
            GQAPagedSDPATestInput {
                q: &q,
                kv_pages: &kv_pages,
                req_slots: &[0, 0],
                page_ids: &[0],
                flat_token_indices: &[0, 1],
            },
        );
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, 2],
                token_indices: &[0],
                q: &q,
                context_k_by_req: &[&k],
                context_v_by_req: &[&v],
            },
        );
        assert_close(&actual, &expected, 2.0e-5);
    }

    #[test]
    fn test_ragged_random() {
        let random_seed = 0xD205_6AB9;
        let mut config = fixture_config();
        let mut shape = fixture_shape();
        shape.num_tokens = 3;
        shape.total_sdpa_map_task_templates = 4;
        config.page_table_layout.num_req_slots = 2;
        let q = generated_values(config.num_output_values(shape), random_seed);
        let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
        let req0_k = generated_values(2 * kv_stride, random_seed.wrapping_add(1));
        let req0_v = generated_values(2 * kv_stride, random_seed.wrapping_add(2));
        let req1_k = generated_values(2 * kv_stride, random_seed.wrapping_add(3));
        let req1_v = generated_values(2 * kv_stride, random_seed.wrapping_add(4));
        let kv_pages = kv_page_values(config, &[(&req0_k, &req0_v), (&req1_k, &req1_v)]);
        let actual = run_gqa_paged_sdpa(
            config,
            shape,
            GQAPagedSDPATestInput {
                q: &q,
                kv_pages: &kv_pages,
                req_slots: &[0, 1, 1],
                page_ids: &[0, 1],
                flat_token_indices: &[1, 0, 1],
            },
        );
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, 1, 3],
                token_indices: &[1, 0],
                q: &q,
                context_k_by_req: &[&req0_k, &req1_k],
                context_v_by_req: &[&req0_v, &req1_v],
            },
        );
        assert_close(&actual, &expected, 2.0e-5);
    }

    #[test]
    fn test_multiple_page_ids_per_block() {
        let mut config = fixture_config();
        let mut shape = fixture_shape();
        shape.num_tokens = 1;
        shape.total_sdpa_map_task_templates = 1;
        config.page_table_layout.num_page_ids_per_block = 2;
        let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
        let q = fixture_values(config.num_output_values(shape), 0.125, 3);
        let k = fixture_values(8 * kv_stride, 0.0625, 5);
        let v = fixture_values(8 * kv_stride, 0.25, 7);
        let kv_pages = kv_page_values(
            config,
            &[
                (&k[..4 * kv_stride], &v[..4 * kv_stride]),
                (&k[4 * kv_stride..], &v[4 * kv_stride..]),
            ],
        );
        let actual = run_gqa_paged_sdpa(
            config,
            shape,
            GQAPagedSDPATestInput {
                q: &q,
                kv_pages: &kv_pages,
                req_slots: &[0],
                page_ids: &[0, 1],
                flat_token_indices: &[7],
            },
        );
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, 1],
                token_indices: &[7],
                q: &q,
                context_k_by_req: &[&k],
                context_v_by_req: &[&v],
            },
        );
        assert_close(&actual, &expected, 2.0e-5);
    }

    fn fixture_config() -> GQAPagedSDPAConfig {
        let num_kv_heads = 2;
        let num_tokens_per_page = 4;
        let head_dim = 2;
        GQAPagedSDPAConfig {
            num_q_heads: 4,
            num_kv_heads,
            head_dim,
            scale: 0.5,
            page_bytes: 2 * num_kv_heads * num_tokens_per_page * head_dim * Dtype::Float32.item_size() as u32,
            page_table_layout: GQAPageTableLayout {
                num_req_slots: 1,
                num_blocks: 1,
                num_gqa_layers: 1,
                num_page_ids_per_block: 1,
            },
            gqa_layer_index: 0,
            kv_token_tile_size: 4,
            num_threads_per_threadblock: 64,
            q_head_tile_size: 2,
            dtype: Dtype::Float32,
        }
    }

    fn fixture_shape() -> GQAPagedSDPAShape {
        GQAPagedSDPAShape {
            num_tokens: 2,
            total_sdpa_map_task_templates: 2,
        }
    }

    fn fixture_core(config: GQAPagedSDPAConfig) -> GQACore {
        let q_dim = config.num_q_heads as usize * config.head_dim as usize;
        GQACore::new(
            0,
            q_dim,
            config.head_dim as usize,
            config.num_q_heads as usize,
            config.num_kv_heads as usize,
            config.scale,
        )
    }

    struct GQAPagedSDPATestInput<'a> {
        q: &'a [f32],
        kv_pages: &'a [f32],
        req_slots: &'a [u32],
        page_ids: &'a [u32],
        flat_token_indices: &'a [u32],
    }

    fn run_gqa_paged_sdpa(
        config: GQAPagedSDPAConfig,
        shape: GQAPagedSDPAShape,
        input: GQAPagedSDPATestInput<'_>,
    ) -> Vec<f32> {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let kernels = GQAPagedSDPAKernels::new(&device);
        let q = Buffer::from_slice(&device, input.q);
        let kv_pages = Buffer::from_slice(&device, input.kv_pages);
        let req_slots = Buffer::from_slice(&device, input.req_slots);
        let page_ids = Buffer::from_slice(&device, input.page_ids);
        let (sdpa_map_task_template_values, cu_sdpa_partial_output_values) =
            sdpa_map_task_template_buffers(config, shape, input.flat_token_indices);
        let sdpa_map_task_templates = Buffer::from_slice(&device, &sdpa_map_task_template_values);
        let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &cu_sdpa_partial_output_values);
        let output = Buffer::new_zeroed(&device, config.q_bytes(shape));
        let scratch = GQAPagedSDPAScratch::new(&device, config, shape);

        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_map(
            config,
            shape,
            scratch.map_buffers(&q, &kv_pages, &req_slots, &page_ids, &sdpa_map_task_templates),
        ));
        builder.record_with_barrier_before(kernels.invoke_reduce(
            config,
            shape,
            scratch.reduce_buffers(&cu_sdpa_partial_outputs, &output),
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();
        output.read_typed::<f32>(0, config.num_output_values(shape))
    }

    fn sdpa_map_task_template_buffers(
        config: GQAPagedSDPAConfig,
        shape: GQAPagedSDPAShape,
        flat_token_indices: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        let num_kv_token_tiles = flat_token_indices
            .iter()
            .map(|&token_index| (token_index + 1).div_ceil(config.kv_token_tile_size) as usize)
            .collect::<Vec<_>>();
        let mut num_sdpa_map_task_templates_by_q_token_tile = vec![1_usize; flat_token_indices.len()];
        let mut num_sdpa_map_task_templates = num_sdpa_map_task_templates_by_q_token_tile.len();
        while num_sdpa_map_task_templates < shape.total_sdpa_map_task_templates as usize {
            let Some(q_token_tile_index) = (0..num_kv_token_tiles.len())
                .filter(|&q_token_tile_index| {
                    num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index]
                        < num_kv_token_tiles[q_token_tile_index]
                })
                .max_by_key(|&q_token_tile_index| {
                    num_kv_token_tiles[q_token_tile_index]
                        .div_ceil(num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index])
                })
            else {
                break;
            };
            num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] += 1;
            num_sdpa_map_task_templates += 1;
        }
        let mut sdpa_map_task_templates = Vec::new();
        let mut cu_sdpa_partial_outputs = vec![0];
        for (q_token_tile_index, &token_index) in flat_token_indices.iter().enumerate() {
            let context_len = token_index + 1;
            for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] {
                let kv_token_tile_begin = num_kv_token_tiles[q_token_tile_index] * sdpa_map_task_template_index
                    / num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index];
                let kv_token_tile_end = num_kv_token_tiles[q_token_tile_index] * (sdpa_map_task_template_index + 1)
                    / num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index];
                let kv_token_begin = kv_token_tile_begin as u32 * config.kv_token_tile_size;
                sdpa_map_task_templates.extend_from_slice(&[
                    q_token_tile_index as u32,
                    kv_token_begin,
                    context_len.min(kv_token_tile_end as u32 * config.kv_token_tile_size),
                ]);
            }
            cu_sdpa_partial_outputs.push((sdpa_map_task_templates.len() / 3) as u32);
        }
        assert!(sdpa_map_task_templates.len() / 3 <= shape.total_sdpa_map_task_templates as usize);
        sdpa_map_task_templates.resize(shape.total_sdpa_map_task_templates as usize * 3, u32::MAX);
        (sdpa_map_task_templates, cu_sdpa_partial_outputs)
    }

    fn kv_page_values(config: GQAPagedSDPAConfig, pages: &[(&[f32], &[f32])]) -> Vec<f32> {
        let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
        let page_f32_values = config.page_bytes as usize / size_of::<f32>();
        let mut v = vec![0.0_f32; pages.len() * page_f32_values];
        for (page_index, (k, page_v)) in pages.iter().enumerate() {
            assert_eq!(k.len(), page_v.len());
            assert_eq!(k.len() % kv_stride, 0);
            let num_tokens = k.len() / kv_stride;
            let num_tokens_per_page = config.num_tokens_per_page() as usize;
            assert!(num_tokens <= num_tokens_per_page);
            let page_base = page_index * page_f32_values;
            for token in 0..num_tokens {
                for kv_head in 0..config.num_kv_heads as usize {
                    for dim in 0..config.head_dim as usize {
                        let source = (token * config.num_kv_heads as usize + kv_head) * config.head_dim as usize + dim;
                        let k_target =
                            page_base + (kv_head * num_tokens_per_page + token) * config.head_dim as usize + dim;
                        let v_target = page_base
                            + ((config.num_kv_heads as usize + kv_head) * num_tokens_per_page + token)
                                * config.head_dim as usize
                            + dim;
                        v[k_target] = k[source];
                        v[v_target] = page_v[source];
                    }
                }
            }
        }
        v
    }

    fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 11 + pattern_offset) % 23) as f32 * scale - 11.0 * scale)
            .collect()
    }

    fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual_value - expected_value).abs();
            assert!(
                diff <= tolerance,
                "GQA reference mismatch at {index}: expected={expected_value} actual={actual_value} diff={diff} \
                 tolerance={tolerance}"
            );
        }
    }
}
