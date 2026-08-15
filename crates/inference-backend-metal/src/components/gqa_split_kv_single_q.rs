use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::components::gqa_kv_page_write::GQAPageTableLayout;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const GQA_SPLIT_KV_SINGLE_Q_MAP_BODY: &str = include_str!("metal/gqa_split_kv_single_q_map.metal");
const GQA_SPLIT_KV_SINGLE_Q_REDUCE_SOURCE: &str = include_str!("metal/gqa_split_kv_single_q_reduce.metal");
const GQA_ACTIVATION_GATE_SOURCE: &str = include_str!("metal/gqa_activation_gate.metal");

/// SplitKV SingleQ SDPA (`T` = tokens, `H` = heads, `D` = head width):
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
/// KVSplit: { q_token_tile_index, kv_token_begin, kv_token_end }
/// SDPAMapTask / threadblock:
///   { q_token_tile_index, kv_token_begin, kv_token_end } from KVSplit
///   + { kv_head_index, q_head_tile_index } from grid
/// grid: (total KV splits * Hkv * Q-head tiles, 1, 1), flattened
/// threadblock: (configured width, 1, 1)
/// parallel: KV splits, KV heads, Q-head tiles
/// ordered/reduce: consecutive KV tiles merged with online softmax
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// This variant uses `Tq_tile = 1`. Only the KV split is materialized. It uses
/// the shared three-`u32` TaskTemplate ABI. The complete Task is comment-only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQASplitKVSingleQConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub kv_token_tile_size: u32,
    pub num_threads_per_threadblock: u32,
    pub q_head_tile_size: u32,
    pub dtype: Dtype,
}

impl GQASplitKVSingleQConfig {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.scale > 0.0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.num_tokens_per_page() > 0);
        self.page_table_layout.validate();
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

    pub fn threadblock_memory_bytes(self) -> usize {
        (self.q_head_tile_size as usize * self.kv_token_tile_size as usize + self.num_threads_per_threadblock as usize)
            * size_of::<f32>()
    }

    pub fn num_q_head_tiles_per_kv_head(self) -> u32 {
        self.q_heads_per_kv_head().div_ceil(self.q_head_tile_size())
    }

    pub fn map_threads(self, shape: GQASplitKVSingleQShape) -> usize {
        checked_product(
            "GQA SDPA map thread count",
            &[
                shape.num_total_sdpa_map_task_templates as usize,
                self.num_kv_heads as usize,
                self.num_q_head_tiles_per_kv_head() as usize,
                self.num_threads_per_threadblock as usize,
            ],
        )
    }

    pub fn num_output_values(self, shape: GQASplitKVSingleQShape) -> usize {
        checked_product(
            "GQA SDPA output element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_sdpa_partial_output_stats(self, shape: GQASplitKVSingleQShape) -> usize {
        checked_product(
            "GQA SDPA partial-output statistic count",
            &[
                shape.num_total_sdpa_map_task_templates as usize,
                self.num_q_heads as usize,
            ],
        )
    }

    pub fn num_partial_output_values(self, shape: GQASplitKVSingleQShape) -> usize {
        self.num_sdpa_partial_output_stats(shape)
            .checked_mul(self.head_dim as usize)
            .expect("GQA SDPA partial output element count must fit usize")
    }

    pub fn q_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (self.num_output_values(shape) as u64)
            .checked_mul(self.dtype.item_size() as u64)
            .expect("GQA SDPA query byte length must fit u64")
    }

    pub fn kv_pages_bytes(self) -> usize {
        self.page_bytes as usize
    }

    pub fn req_slots_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (shape.num_total_tokens as u64)
            .checked_mul(size_of::<u32>() as u64)
            .expect("GQA SDPA request-slot byte length must fit u64")
    }

    pub fn page_ids_bytes(self) -> u64 {
        self.page_table_layout.bytes() as u64
    }

    pub fn sdpa_map_task_templates_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (shape.num_total_sdpa_map_task_templates as u64)
            .checked_mul(3)
            .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
            .expect("GQA SDPA map TaskTemplate byte length must fit u64")
    }

    pub fn cu_sdpa_partial_outputs_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (shape.num_total_tokens as u64)
            .checked_add(1)
            .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
            .expect("GQA SDPA partial-output cumulative-count byte length must fit u64")
    }

    pub fn partial_output_stats_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (self.num_sdpa_partial_output_stats(shape) as u64)
            .checked_mul(size_of::<f32>() as u64)
            .expect("GQA SDPA statistic byte length must fit u64")
    }

    pub fn partial_output_bytes(self, shape: GQASplitKVSingleQShape) -> u64 {
        (self.num_partial_output_values(shape) as u64)
            .checked_mul(self.dtype.item_size() as u64)
            .expect("GQA SDPA partial output byte length must fit u64")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQASplitKVSingleQShape {
    pub num_total_tokens: u32,
    pub num_total_sdpa_map_task_templates: u32,
}

impl GQASplitKVSingleQShape {
    pub fn validate(self, config: GQASplitKVSingleQConfig) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_sdpa_map_task_templates > 0);
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
pub struct GQASplitKVSingleQMapBuffers<'a> {
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
pub struct GQASplitKVSingleQReduceBuffers<'a> {
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct GQASplitKVSingleQScratch {
    pub partial_exp_sums: Buffer,
    pub partial_max_logits: Buffer,
    pub partial_output: Buffer,
}

impl GQASplitKVSingleQScratch {
    pub fn new(device: &Device, config: GQASplitKVSingleQConfig, shape: GQASplitKVSingleQShape) -> Self {
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
    ) -> GQASplitKVSingleQMapBuffers<'a> {
        GQASplitKVSingleQMapBuffers {
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
    ) -> GQASplitKVSingleQReduceBuffers<'a> {
        GQASplitKVSingleQReduceBuffers {
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
            cu_sdpa_partial_outputs,
            output,
        }
    }
}

pub struct GQASplitKVSingleQKernels {
    config: GQASplitKVSingleQConfig,
    shape: GQASplitKVSingleQShape,
    map: Kernel,
    reduce: Kernel,
}

impl GQASplitKVSingleQKernels {
    pub fn new(device: &Device, config: GQASplitKVSingleQConfig, shape: GQASplitKVSingleQShape) -> Self {
        shape.validate(config);
        assert!(
            config.threadblock_memory_bytes() <= device.max_threadblock_memory_length(),
            "GQA SplitKV SingleQ shape needs {} bytes of threadblock memory but device only supports {}",
            config.threadblock_memory_bytes(),
            device.max_threadblock_memory_length()
        );
        let reduce_function_name = match config.dtype {
            Dtype::Float32 => "gqa_split_kv_single_q_reduce_f32",
            Dtype::Bfloat16 => "gqa_split_kv_single_q_reduce_bf16",
            dtype => panic!("unsupported GQA SplitKV SingleQ reduce dtype {dtype:?}"),
        };
        Self {
            config,
            shape,
            map: Kernel::new(
                device,
                &gqa_split_kv_single_q_map_source(config, shape),
                "gqa_split_kv_single_q_map",
            ),
            reduce: Kernel::new(
                device,
                &gqa_split_kv_single_q_reduce_source(config),
                reduce_function_name,
            ),
        }
    }

    pub fn invoke_map<'a>(
        &self,
        buffers: GQASplitKVSingleQMapBuffers<'a>,
        page_table_index: ReplayU32,
    ) -> GQASplitKVSingleQMapInvocation<'a> {
        GQASplitKVSingleQMapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            page_table_index,
            num_active_tokens: ReplayU32::Fixed(self.shape.num_total_tokens),
            num_active_kv_splits: ReplayU32::Fixed(self.shape.num_total_sdpa_map_task_templates),
        }
    }

    pub fn invoke_map_bucketed<'a>(
        &self,
        buffers: GQASplitKVSingleQMapBuffers<'a>,
        page_table_index: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_kv_splits: ReplayU32,
    ) -> GQASplitKVSingleQMapInvocation<'a> {
        GQASplitKVSingleQMapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            page_table_index,
            num_active_tokens,
            num_active_kv_splits,
        }
    }

    pub fn invoke_reduce<'a>(
        &self,
        buffers: GQASplitKVSingleQReduceBuffers<'a>,
    ) -> GQASplitKVSingleQReduceInvocation<'a> {
        GQASplitKVSingleQReduceInvocation {
            pipeline: self.reduce.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            num_active_tokens: ReplayU32::Fixed(self.shape.num_total_tokens),
        }
    }

    pub fn invoke_reduce_bucketed<'a>(
        &self,
        buffers: GQASplitKVSingleQReduceBuffers<'a>,
        num_active_tokens: ReplayU32,
    ) -> GQASplitKVSingleQReduceInvocation<'a> {
        GQASplitKVSingleQReduceInvocation {
            pipeline: self.reduce.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers,
            num_active_tokens,
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
                shape.num_total_tokens as usize,
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
    pub num_total_tokens: u32,
}

impl GQAActivationGateShape {
    pub fn validate(self, config: GQAActivationGateConfig) {
        config.validate();
        assert!(self.num_total_tokens > 0);
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
            num_active_tokens: ReplayU32::Fixed(shape.num_total_tokens),
        }
    }

    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: GQAActivationGateShape,
        buffers: GQAActivationGateBuffers<'a>,
        num_active_tokens: ReplayU32,
    ) -> GQAActivationGateInvocation<'a> {
        GQAActivationGateInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
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
    num_active_tokens: ReplayU32,
}

impl Operator for GQAActivationGateInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.attention_output, 0);
        recorder.set_buffer_read(1, self.buffers.g, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            3,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA activation-gate active token count",
        );
        recorder.dispatch_1d(self.config.num_values(shape), 256);
    }
}

impl GQAActivationGateInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.attention_output.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }
}

fn gqa_split_kv_single_q_map_source(config: GQASplitKVSingleQConfig, shape: GQASplitKVSingleQShape) -> String {
    let dtype = metal_dtype_name(config.dtype);
    let body = GQA_SPLIT_KV_SINGLE_Q_MAP_BODY
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
#define PAGE_BYTES {page_bytes}
#define KV_TOKEN_TILE_SIZE {kv_token_tile_size}
#define TOTAL_KV_SPLITS {num_total_kv_splits}
#define NUM_THREADS_PER_THREADBLOCK {num_threads_per_threadblock}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}

kernel void gqa_split_kv_single_q_map(
    device const T* q [[buffer(0)]],
    device const KV_T* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* kv_splits [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device T* partial_output [[buffer(7)]],
    constant uint& gqa_layer_index [[buffer(8)]],
    constant uint& num_active_tokens [[buffer(9)]],
    constant uint& num_active_kv_splits [[buffer(10)]],
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
        num_q_heads = config.num_q_heads,
        page_bytes = config.page_bytes,
        kv_token_tile_size = config.kv_token_tile_size,
        num_total_kv_splits = shape.num_total_sdpa_map_task_templates,
        num_threads_per_threadblock = config.num_threads_per_threadblock,
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
        unsupported_dtype => panic!("unsupported GQA SplitKV SingleQ map dtype {unsupported_dtype:?}"),
    }
}

pub struct GQASplitKVSingleQMapInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: GQASplitKVSingleQConfig,
    shape: GQASplitKVSingleQShape,
    buffers: GQASplitKVSingleQMapBuffers<'a>,
    page_table_index: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_kv_splits: ReplayU32,
}

impl Operator for GQASplitKVSingleQMapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.kv_pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.sdpa_map_task_templates, 0);
        recorder.set_buffer_write(5, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(6, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_write(7, self.buffers.partial_output, 0);
        let max_page_table_index = self.config.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => {
                assert!(
                    page_table_index <= max_page_table_index,
                    "GQA page-table index exceeds layer count"
                );
                recorder.set_u32(8, page_table_index);
            },
            ReplayU32::Parameter(key) => recorder.bind_u32(8, key, 0, max_page_table_index),
        }
        set_replay_u32(
            recorder,
            9,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA SplitKV SingleQ active token count",
        );
        set_replay_u32(
            recorder,
            10,
            self.num_active_kv_splits,
            shape.num_total_sdpa_map_task_templates,
            "GQA SplitKV SingleQ active KV split count",
        );
        recorder.dispatch_1d(
            self.config.map_threads(shape),
            self.config.num_threads_per_threadblock as usize,
        );
    }
}

impl GQASplitKVSingleQMapInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.q.len_bytes_u64() >= self.config.q_bytes(self.shape));
        assert!(self.buffers.kv_pages.len_bytes_u64() >= self.config.page_bytes as u64);
        assert!(self.buffers.req_slots.len_bytes_u64() >= self.config.req_slots_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes_u64() >= self.config.page_ids_bytes());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes_u64()
                >= self.config.sdpa_map_task_templates_bytes(self.shape)
        );
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= self.config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_output.len_bytes_u64() >= self.config.partial_output_bytes(self.shape));
    }
}

pub struct GQASplitKVSingleQReduceInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: GQASplitKVSingleQConfig,
    shape: GQASplitKVSingleQShape,
    buffers: GQASplitKVSingleQReduceBuffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for GQASplitKVSingleQReduceInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_read(1, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_read(2, self.buffers.partial_output, 0);
        recorder.set_buffer_read(3, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(4, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            5,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA SplitKV SingleQ active token count",
        );
        recorder.dispatch_1d(self.config.num_output_values(shape), 256);
    }
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

impl GQASplitKVSingleQReduceInvocation<'_> {
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
}

fn gqa_split_kv_single_q_reduce_source(config: GQASplitKVSingleQConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    GQA_SPLIT_KV_SINGLE_Q_REDUCE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

#[cfg(test)]
#[path = "gqa_split_kv_single_q_test.rs"]
mod tests;
