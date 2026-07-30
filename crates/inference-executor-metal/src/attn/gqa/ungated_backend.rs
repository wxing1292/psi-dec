use inference_backend_metal::components::GQACompute;
use inference_backend_metal::components::GQAComputePath;
use inference_backend_metal::components::GQAKVPageWrite;
use inference_backend_metal::components::GQAKVPageWriteBuffers;
use inference_backend_metal::components::GQAKVPageWriteConfig;
use inference_backend_metal::components::GQAKVPageWriteShape;
use inference_backend_metal::components::GQANormRopeBuffers;
use inference_backend_metal::components::GQANormRopeConfig;
use inference_backend_metal::components::GQANormRopeKernel;
use inference_backend_metal::components::GQANormRopeShape;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::components::GQAPagedSDPAConfig;
use inference_backend_metal::components::GQAPagedSDPAKernels;
use inference_backend_metal::components::GQAPagedSDPAMapBuffers;
use inference_backend_metal::components::GQAPagedSDPAReduceBuffers;
use inference_backend_metal::components::GQAPagedSDPAShape;
use inference_backend_metal::components::GQAQKVSplitBuffers;
use inference_backend_metal::components::GQAQKVSplitConfig;
use inference_backend_metal::components::GQAQKVSplitKernel;
use inference_backend_metal::components::GQAQKVSplitShape;
use inference_backend_metal::components::GQATiledSDPAConfig;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::backend::gqa_compute_config;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratchBindings;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy)]
pub struct UngatedGQAWeights<'a> {
    pub qkv_weight: &'a Buffer,
    pub qkv_scales: &'a Buffer,
    pub qkv_biases: &'a Buffer,
    pub q_norm_weight: &'a Buffer,
    pub k_norm_weight: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one ungated GQA replay recording. The replay shape
/// belongs to `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct UngatedGQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub batch_metadata: &'a GQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: UngatedGQAWeights<'a>,
    pub scratch: UngatedGQAScratchBindings<'a>,
}

/// The caller-owned next-hidden-state buffer returned by one ungated GQA
/// recording.
pub type UngatedGQAOutput<'a> = &'a Buffer;

/// Ungated GQA data flow:
///
/// ```text
/// hidden_state
///   -> qkv
///   -> scratch.qkv
///   -> qkv_to_q_k_v
///      |- scratch.q -> q_norm_rope -> scratch.q_norm_rope -----------+
///      |- scratch.k -> k_norm_rope -> scratch.k_norm_rope --+        |
///      `- scratch.v -----------------------------------------+        |
///                                                           |        |
///                                                           v        |
///                                                    kv_page_write    |
///                                                           |        |
///                                                           v        |
///                                                   kv_cache.kv_pages |
///                                                           |        |
///                                                           +--------+
///                                                                    |
///                                                                    v
///                              GQAPagedSDPAKernels or GQATiledSDPAKernels
///                                -> scratch.sdpa_partial_exp_sums
///                                -> scratch.sdpa_partial_max_logits
///                                -> scratch.sdpa_partial_output
///                                -> scratch.attention_output
///                                -> output
///                                -> next_hidden_state
/// ```
pub struct UngatedGQA {
    device: Device,
    core: UngatedGQACore,
    config: GQAMetalConfig,
    compute: GQACompute,
    qkv: AffineQuantizedMatmul,
    qkv_to_q_k_v: GQAQKVSplitKernel,
    q_norm_rope: GQANormRopeKernel,
    k_norm_rope: GQANormRopeKernel,
    kv_page_write: GQAKVPageWrite,
    output: AffineQuantizedMatmul,
}

impl GQAMetalConfig {
    pub fn num_ungated_tokens_per_page(self, core: &UngatedGQACore) -> u32 {
        gqa_compute_config(self, core.num_q_heads, core.num_kv_heads, core.head_dim).num_tokens_per_page()
    }
}

impl UngatedGQA {
    fn validate_input(&self, input: &UngatedGQAInput<'_>) {
        input.batch_metadata.replay_shape().validate();
        input.page_table_layout.validate();
        assert!(input.gqa_layer_index < input.page_table_layout.num_gqa_layers);
    }

    pub fn new(device: &Device, core: UngatedGQACore, config: GQAMetalConfig) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qkv = core.qkv_shape();
        let output = core.output_shape();
        Self {
            device: device.clone(),
            core: core.clone(),
            config,
            compute: GQACompute::new(gqa_compute_config(
                config,
                core.num_q_heads,
                core.num_kv_heads,
                core.head_dim,
            )),
            qkv: AffineQuantizedMatmul::new(device, affine_config(qkv.out_dim, qkv.in_dim, config)),
            qkv_to_q_k_v: GQAQKVSplitKernel::new(device, qkv_to_q_k_v_config(&core, config)),
            q_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: GQAKVPageWrite::new(device, kv_page_write_config(&core, config)),
            output: AffineQuantizedMatmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_ungated_tokens_per_page(&self.core)
    }

    pub fn new_scratch(&self, max_tokens: usize) -> UngatedGQAScratch {
        UngatedGQAScratch::new(&self.device, &self.core, self.config, self.compute, max_tokens)
    }

    pub fn prepare(
        &self,
        batch_metadata: &GQAMetadataBuffers,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
    ) -> GQAReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        let num_q_token_tiles = cu_tokens
            .windows(2)
            .map(|cu| {
                assert!(cu[0] <= cu[1], "ungated GQA batch cu_tokens must be nondecreasing");
                (cu[1] - cu[0]).div_ceil(self.compute.tiled_query_token_tile_size())
            })
            .sum();
        let compute_path = self.compute.select(num_tokens, num_q_token_tiles);
        batch_metadata.update(req_slots, token_indices, cu_tokens, compute_path)
    }
}

impl ReplayLayer for UngatedGQA {
    type Input<'a> = UngatedGQAInput<'a>;
    type Output<'a> = UngatedGQAOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let gqa_layer_index = input.gqa_layer_index;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let kv_cache = input.kv_cache;
        let weights = input.weights;
        let batch_metadata = input.batch_metadata;
        let scratch = input.scratch;
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.qkv.invoke(
                shape
                    .num_tokens
                    .try_into()
                    .expect("ungated GQA token count must fit i32"),
                scratch.qkv,
                0,
                hidden_state,
                0,
                weights.qkv_weight,
                0,
                weights.qkv_scales,
                0,
                weights.qkv_biases,
                0,
            ),
        ));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv_to_q_k_v.invoke(
            self.qkv_to_q_k_v_shape(shape),
            GQAQKVSplitBuffers {
                qkv: scratch.qkv,
                q: scratch.q,
                k: scratch.k,
                v: scratch.v,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            self.norm_rope_shape(shape),
            GQANormRopeBuffers {
                input: scratch.q,
                norm_weight: weights.q_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            self.norm_rope_shape(shape),
            GQANormRopeBuffers {
                input: scratch.k,
                norm_weight: weights.k_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_page_write.invoke(
            self.kv_page_write_shape(shape, page_table_layout, gqa_layer_index),
            GQAKVPageWriteBuffers {
                pages: kv_cache.kv_pages,
                flat_k: scratch.k_norm_rope,
                flat_v: scratch.v,
                req_slots: batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: kv_cache.page_ids,
            },
        )));
        let attention_output = match batch_metadata.compute_path() {
            GQAComputePath::SingleQueryToken {
                kv_token_tile_size,
                num_threads_per_threadblock,
                q_head_tile_size,
            } => {
                let sdpa_config = self.paged_sdpa_config(
                    page_table_layout,
                    gqa_layer_index,
                    kv_token_tile_size,
                    num_threads_per_threadblock,
                    q_head_tile_size,
                );
                let sdpa_shape = self.paged_sdpa_shape(shape);
                let sdpa = GQAPagedSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
                recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(GQAPagedSDPAMapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                })));
                recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(GQAPagedSDPAReduceBuffers {
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                })));
                scratch.attention_output
            },
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size,
                kv_token_tile_size,
                q_head_tile_size,
            } => {
                let sdpa_config = self.tiled_sdpa_config(
                    page_table_layout,
                    gqa_layer_index,
                    q_token_tile_size,
                    kv_token_tile_size,
                    q_head_tile_size,
                );
                let sdpa_shape = self.tiled_sdpa_shape(shape);
                let sdpa = GQATiledSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
                recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(GQATiledSDPAMapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    flat_token_indices: batch_metadata.flat_token_indices(),
                    q_token_tiles: batch_metadata.q_token_tiles(),
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_output: scratch.sdpa_partial_output,
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                })));
                recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(GQATiledSDPAReduceBuffers {
                    partial_output: scratch.sdpa_partial_output,
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    q_token_tiles: batch_metadata.q_token_tiles(),
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                })));
                scratch.attention_output
            },
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.output.invoke(
                shape
                    .num_tokens
                    .try_into()
                    .expect("ungated GQA token count must fit i32"),
                next_hidden_state,
                0,
                attention_output,
                0,
                weights.output_weight,
                0,
                weights.output_scales,
                0,
                weights.output_biases,
                0,
            ),
        ));
        next_hidden_state
    }
}

impl UngatedGQA {
    fn qkv_to_q_k_v_shape(&self, shape: GQAReplayShape) -> GQAQKVSplitShape {
        GQAQKVSplitShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn norm_rope_shape(&self, shape: GQAReplayShape) -> GQANormRopeShape {
        GQANormRopeShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn kv_page_write_shape(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
    ) -> GQAKVPageWriteShape {
        GQAKVPageWriteShape {
            num_token_writes: shape.num_tokens,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
        }
    }

    fn paged_sdpa_config(
        &self,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
        kv_token_tile_size: u32,
        num_threads_per_threadblock: u32,
        q_head_tile_size: u32,
    ) -> GQAPagedSDPAConfig {
        GQAPagedSDPAConfig {
            num_q_heads: self
                .core
                .num_q_heads
                .try_into()
                .expect("ungated GQA q heads must fit u32"),
            num_kv_heads: self
                .core
                .num_kv_heads
                .try_into()
                .expect("ungated GQA KV heads must fit u32"),
            head_dim: self
                .core
                .head_dim
                .try_into()
                .expect("ungated GQA head_dim must fit u32"),
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
            kv_token_tile_size,
            num_threads_per_threadblock,
            q_head_tile_size,
            dtype: self.config.io_dtype,
        }
    }

    fn paged_sdpa_shape(&self, shape: GQAReplayShape) -> GQAPagedSDPAShape {
        GQAPagedSDPAShape {
            num_tokens: shape.num_tokens,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
        }
    }

    fn tiled_sdpa_config(
        &self,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
        q_token_tile_size: u32,
        kv_token_tile_size: u32,
        q_head_tile_size: u32,
    ) -> GQATiledSDPAConfig {
        GQATiledSDPAConfig {
            num_q_heads: self
                .core
                .num_q_heads
                .try_into()
                .expect("ungated GQA q heads must fit u32"),
            num_kv_heads: self
                .core
                .num_kv_heads
                .try_into()
                .expect("ungated GQA KV heads must fit u32"),
            head_dim: self
                .core
                .head_dim
                .try_into()
                .expect("ungated GQA head_dim must fit u32"),
            q_head_tile_size,
            q_token_tile_size,
            kv_token_tile_size,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            dtype: self.config.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
        }
    }

    fn tiled_sdpa_shape(&self, shape: GQAReplayShape) -> GQATiledSDPAShape {
        GQATiledSDPAShape {
            num_tokens: shape.num_tokens,
            num_q_token_tiles: shape.num_q_token_tiles,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
        }
    }
}

fn backend_page_table_layout(shape: GQAPageTableLayout) -> MetalGQAPageTableLayout {
    MetalGQAPageTableLayout {
        num_req_slots: shape.num_req_slots,
        num_blocks: shape.num_blocks,
        num_gqa_layers: shape.num_gqa_layers,
        num_page_ids_per_block: shape.num_page_ids_per_block,
    }
}

fn qkv_to_q_k_v_config(core: &UngatedGQACore, config: GQAMetalConfig) -> GQAQKVSplitConfig {
    let num_q_heads = core.num_q_heads.try_into().expect("ungated GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("ungated GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => GQAQKVSplitConfig::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => GQAQKVSplitConfig::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &UngatedGQACore, config: GQAMetalConfig, num_heads: usize) -> GQANormRopeConfig {
    let num_heads_u32 = num_heads.try_into().expect("ungated GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => {
            GQANormRopeConfig::f32(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
                config.rope_scale,
            )
        },
        Dtype::Bfloat16 => {
            GQANormRopeConfig::bf16(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
                config.rope_scale,
            )
        },
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    }
}

fn kv_page_write_config(core: &UngatedGQACore, config: GQAMetalConfig) -> GQAKVPageWriteConfig {
    GQAKVPageWriteConfig {
        num_kv_heads: core.num_kv_heads.try_into().expect("ungated GQA KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("ungated GQA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
    }
}

fn validate_config_for_core(core: &UngatedGQACore, config: GQAMetalConfig) {
    config.validate();
    assert!(config.rope_dim as usize <= core.head_dim);
    assert!(config.num_ungated_tokens_per_page(core) > 0);
}

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("ungated GQA affine n must fit i32"),
        k: k.try_into().expect("ungated GQA affine k must fit i32"),
        group_size: config
            .group_size
            .try_into()
            .expect("ungated GQA group_size must fit i32"),
        bits: config.bits.try_into().expect("ungated GQA bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}
