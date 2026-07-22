use inference_backend_metal::components::GQAActivationGateBuffers;
use inference_backend_metal::components::GQAActivationGateConfig;
use inference_backend_metal::components::GQAActivationGateKernel;
use inference_backend_metal::components::GQAActivationGateShape;
use inference_backend_metal::components::GQAKVPageUpdate;
use inference_backend_metal::components::GQAKVPageUpdateBuffers;
use inference_backend_metal::components::GQAKVPageUpdateConfig;
use inference_backend_metal::components::GQAKVPageUpdateShape;
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
use inference_backend_metal::components::GQAProjectionSplitBuffers;
use inference_backend_metal::components::GQAProjectionSplitConfig;
use inference_backend_metal::components::GQAProjectionSplitKernel;
use inference_backend_metal::components::GQAProjectionSplitShape;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::scratch::GQAScratchBindings;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQAMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub page_bytes: u32,
    pub context_parallel_kv_token_tile_size: u32,
    pub context_parallel_num_threads_per_threadblock: u32,
    pub context_parallel_max_q_head_tile_size: u32,
    pub q_token_tile_size: u32,
    pub tiled_kv_token_tile_size: u32,
    pub rope_dim: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub dtype: Dtype,
}

impl GQAMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.page_bytes > 0);
        assert!(self.context_parallel_kv_token_tile_size > 0 && self.context_parallel_kv_token_tile_size <= 1024);
        assert!(
            self.context_parallel_num_threads_per_threadblock.is_power_of_two()
                && self.context_parallel_num_threads_per_threadblock <= 256
        );
        assert!(self.context_parallel_max_q_head_tile_size > 0 && self.context_parallel_max_q_head_tile_size <= 8);
        assert!(matches!(self.q_token_tile_size, 8 | 16));
        assert!(matches!(self.tiled_kv_token_tile_size, 8 | 16));
        assert!(self.rope_dim > 0);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.norm_eps > 0.0);
        assert!(self.rope_theta > 0.0);
        assert!(self.rope_scale > 0.0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_tokens_per_page(self, core: &GQACore) -> u32 {
        let kv_bytes_per_token = core
            .num_kv_heads
            .checked_mul(core.head_dim)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(self.dtype.item_size()))
            .expect("GQA K/V bytes per token must fit usize");
        assert!(
            (self.page_bytes as usize).is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        (self.page_bytes as usize / kv_bytes_per_token)
            .try_into()
            .expect("GQA tokens per page must fit u32")
    }

    pub fn supports_tiled(self, core: &GQACore) -> bool {
        self.dtype == Dtype::Bfloat16
            && core.head_dim == 256
            && self.num_tokens_per_page(core) == 16
            && core.num_q_heads / core.num_kv_heads <= 8
    }

    fn tiled_max_q_head_tile_size(self) -> usize {
        (256 / (self.q_token_tile_size / 8 * 32)) as usize
    }
}

#[derive(Clone, Copy)]
pub struct GQAKVCacheBindings<'a> {
    pub kv_pages: &'a Buffer,
    pub page_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GQAWeights<'a> {
    pub qgkv_weight: &'a Buffer,
    pub qgkv_scales: &'a Buffer,
    pub qgkv_biases: &'a Buffer,
    pub q_norm_weight: &'a Buffer,
    pub k_norm_weight: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one GQA replay recording. The replay shape belongs to
/// `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct GQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub batch_metadata: &'a GQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: GQAWeights<'a>,
    pub scratch: GQAScratchBindings<'a>,
}

/// The caller-owned next-hidden-state buffer returned by one GQA recording.
pub type GQAOutput<'a> = &'a Buffer;

pub struct GQA {
    core: GQACore,
    config: GQAMetalConfig,
    qgkv_projection_qmv: AffineQuantizedMatmulKernel,
    qgkv_projection_qmm: AffineQuantizedMatmulKernel,
    projection_split: GQAProjectionSplitKernel,
    q_norm_rope: GQANormRopeKernel,
    k_norm_rope: GQANormRopeKernel,
    kv_update: GQAKVPageUpdate,
    paged_sdpa: GQAPagedSDPAKernels,
    tiled_sdpa: GQATiledSDPAKernels,
    activation_gate: GQAActivationGateKernel,
    output_projection_qmv: AffineQuantizedMatmulKernel,
    output_projection_qmm: AffineQuantizedMatmulKernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GQASDPAPath {
    ContextParallel,
    Tiled { q_head_tile_size: u32 },
}

impl GQA {
    fn validate_input(&self, input: &GQAInput<'_>) {
        input.batch_metadata.replay_shape().validate();
        input.page_table_layout.validate();
        assert!(input.gqa_layer_index < input.page_table_layout.num_gqa_layers);
    }

    fn sdpa_path(&self, num_tokens: u32, num_q_token_tiles: u32) -> GQASDPAPath {
        // Tiled SDPA uses Tq_tile=`q_token_tile_size`,
        // Tkv_tile=`tiled_kv_token_tile_size`, and Hq_tile=`q_head_tile_size`.
        // By average useful Q tokens per Tq tile, the selector uses
        // context-parallel below 2, half-group Hq below 4, and full-group Hq
        // otherwise, subject to the 256-threadblock cap.
        assert!(num_tokens > 0);
        assert!(num_q_token_tiles > 0 && num_q_token_tiles <= num_tokens);
        let q_heads_per_kv_head = self.core.num_q_heads / self.core.num_kv_heads;
        if !self.config.supports_tiled(&self.core) || (num_tokens as u64) < 2 * num_q_token_tiles as u64 {
            return GQASDPAPath::ContextParallel;
        }
        let desired_q_head_tile_size = if (num_tokens as u64) < 4 * num_q_token_tiles as u64 {
            q_heads_per_kv_head.div_ceil(2)
        } else {
            q_heads_per_kv_head
        };
        GQASDPAPath::Tiled {
            q_head_tile_size: desired_q_head_tile_size
                .min(self.config.tiled_max_q_head_tile_size())
                .try_into()
                .expect("GQA Q-head tile size must fit u32"),
        }
    }

    pub fn new(device: &Device, core: GQACore, config: GQAMetalConfig) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qgkv = core.qgkv_shape();
        let output = core.output_shape();
        let qgkv_qmm_m = qmv_batch_limit(qgkv.in_dim, qgkv.out_dim);
        let output_qmm_m = qmv_batch_limit(output.in_dim, output.out_dim);
        Self {
            core: core.clone(),
            config,
            qgkv_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(1, qgkv.out_dim, qgkv.in_dim, config),
            ),
            qgkv_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(qgkv_qmm_m, qgkv.out_dim, qgkv.in_dim, config),
            ),
            projection_split: GQAProjectionSplitKernel::new(device, projection_split_config(&core, config)),
            q_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_update: GQAKVPageUpdate::new(device, kv_update_config(&core, config)),
            paged_sdpa: GQAPagedSDPAKernels::new(device),
            tiled_sdpa: GQATiledSDPAKernels::new(device),
            activation_gate: GQAActivationGateKernel::new(device, activation_gate_config(&core, config)),
            output_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(1, output.out_dim, output.in_dim, config),
            ),
            output_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape(output_qmm_m, output.out_dim, output.in_dim, config),
            ),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_tokens_per_page(&self.core)
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
                assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
                (cu[1] - cu[0]).div_ceil(self.config.q_token_tile_size)
            })
            .sum();
        match self.sdpa_path(num_tokens, num_q_token_tiles) {
            GQASDPAPath::ContextParallel => {
                batch_metadata.update_context_parallel(
                    req_slots,
                    token_indices,
                    cu_tokens,
                    self.config.context_parallel_kv_token_tile_size,
                )
            },
            GQASDPAPath::Tiled { .. } => {
                batch_metadata.update_tiled(
                    req_slots,
                    token_indices,
                    cu_tokens,
                    self.config.q_token_tile_size,
                    self.config.tiled_kv_token_tile_size,
                )
            },
        }
    }
}

impl Layer for GQA {
    type Input<'a> = GQAInput<'a>;
    type Output<'a> = GQAOutput<'a>;

    type InputShape = GQACore;
    type OutputShape = GQACore;

    fn input_shape(&self) -> Self::InputShape {
        self.core.clone()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.core.clone()
    }
}

impl ReplayLayer for GQA {
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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qgkv_projection(shape).invoke_with_shape(
            self.qgkv_affine_shape(shape),
            scratch.qgkv_proj,
            0,
            hidden_state,
            0,
            weights.qgkv_weight,
            0,
            weights.qgkv_scales,
            0,
            weights.qgkv_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.projection_split.invoke(
            self.projection_split_shape(shape),
            GQAProjectionSplitBuffers {
                qgkv: scratch.qgkv_proj,
                q: scratch.q,
                g: scratch.g,
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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_update.invoke(
            self.kv_update_shape(shape, page_table_layout, gqa_layer_index),
            GQAKVPageUpdateBuffers {
                pages: kv_cache.kv_pages,
                flat_k: scratch.k_norm_rope,
                flat_v: scratch.v,
                req_slots: batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: kv_cache.page_ids,
            },
        )));
        let attention_output = match self.sdpa_path(shape.num_tokens, shape.num_q_token_tiles) {
            GQASDPAPath::ContextParallel => {
                let sdpa_config = self.paged_sdpa_config(page_table_layout, gqa_layer_index);
                let sdpa_shape = self.paged_sdpa_shape(shape);
                recorder.record_with_barrier_before(ReplayOp::opaque(self.paged_sdpa.invoke_map(
                    sdpa_config,
                    sdpa_shape,
                    GQAPagedSDPAMapBuffers {
                        q: scratch.q_norm_rope,
                        kv_pages: kv_cache.kv_pages,
                        req_slots: batch_metadata.req_slots(),
                        page_ids: kv_cache.page_ids,
                        sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                        partial_exp_sums: scratch.sdpa_partial_exp_sums,
                        partial_max_logits: scratch.sdpa_partial_max_logits,
                        partial_output: scratch.sdpa_partial_output,
                    },
                )));
                recorder.record_with_barrier_before(ReplayOp::opaque(self.paged_sdpa.invoke_reduce(
                    sdpa_config,
                    sdpa_shape,
                    GQAPagedSDPAReduceBuffers {
                        partial_exp_sums: scratch.sdpa_partial_exp_sums,
                        partial_max_logits: scratch.sdpa_partial_max_logits,
                        partial_output: scratch.sdpa_partial_output,
                        cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                        output: scratch.attention_output,
                    },
                )));
                scratch.attention_output
            },
            GQASDPAPath::Tiled { q_head_tile_size } => {
                let sdpa_shape = self.tiled_sdpa_shape(shape, page_table_layout, gqa_layer_index, q_head_tile_size);
                recorder.record_with_barrier_before(ReplayOp::opaque(self.tiled_sdpa.invoke_map(
                    sdpa_shape,
                    GQATiledSDPAMapBuffers {
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
                    },
                )));
                recorder.record_with_barrier_before(ReplayOp::opaque(self.tiled_sdpa.invoke_reduce(
                    sdpa_shape,
                    GQATiledSDPAReduceBuffers {
                        partial_output: scratch.sdpa_partial_output,
                        partial_exp_sums: scratch.sdpa_partial_exp_sums,
                        partial_max_logits: scratch.sdpa_partial_max_logits,
                        q_token_tiles: batch_metadata.q_token_tiles(),
                        cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                        output: scratch.attention_output,
                    },
                )));
                scratch.attention_output
            },
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(self.activation_gate.invoke(
            self.activation_gate_shape(shape),
            GQAActivationGateBuffers {
                attention_output,
                g: scratch.g,
                output: scratch.gated_attention_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output_projection(shape).invoke_with_shape(
            self.output_affine_shape(shape),
            next_hidden_state,
            0,
            scratch.gated_attention_output,
            0,
            weights.output_weight,
            0,
            weights.output_scales,
            0,
            weights.output_biases,
            0,
        )));
        next_hidden_state
    }
}

impl GQA {
    fn qgkv_projection(&self, shape: GQAReplayShape) -> &AffineQuantizedMatmulKernel {
        let qgkv = self.core.qgkv_shape();
        if shape.num_tokens >= qmv_batch_limit(qgkv.in_dim, qgkv.out_dim) {
            &self.qgkv_projection_qmm
        } else {
            &self.qgkv_projection_qmv
        }
    }

    fn output_projection(&self, shape: GQAReplayShape) -> &AffineQuantizedMatmulKernel {
        let output = self.core.output_shape();
        if shape.num_tokens >= qmv_batch_limit(output.in_dim, output.out_dim) {
            &self.output_projection_qmm
        } else {
            &self.output_projection_qmv
        }
    }

    fn qgkv_affine_shape(&self, shape: GQAReplayShape) -> AffineQuantizedMatmulShape {
        let qgkv = self.core.qgkv_shape();
        affine_shape(shape.num_tokens, qgkv.out_dim, qgkv.in_dim, self.config)
    }

    fn output_affine_shape(&self, shape: GQAReplayShape) -> AffineQuantizedMatmulShape {
        let output = self.core.output_shape();
        affine_shape(shape.num_tokens, output.out_dim, output.in_dim, self.config)
    }

    fn projection_split_shape(&self, shape: GQAReplayShape) -> GQAProjectionSplitShape {
        GQAProjectionSplitShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn norm_rope_shape(&self, shape: GQAReplayShape) -> GQANormRopeShape {
        GQANormRopeShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn kv_update_shape(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
    ) -> GQAKVPageUpdateShape {
        GQAKVPageUpdateShape {
            num_token_writes: shape.num_tokens,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
        }
    }

    fn paged_sdpa_config(&self, page_table_layout: GQAPageTableLayout, gqa_layer_index: u32) -> GQAPagedSDPAConfig {
        let q_heads_per_kv_head = self.core.num_q_heads / self.core.num_kv_heads;
        GQAPagedSDPAConfig {
            num_q_heads: self.core.num_q_heads.try_into().expect("GQA q heads must fit u32"),
            num_kv_heads: self.core.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
            head_dim: self.core.head_dim.try_into().expect("GQA head_dim must fit u32"),
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
            kv_token_tile_size: self.config.context_parallel_kv_token_tile_size,
            num_threads_per_threadblock: self.config.context_parallel_num_threads_per_threadblock,
            q_head_tile_size: q_heads_per_kv_head
                .min(self.config.context_parallel_max_q_head_tile_size as usize)
                .try_into()
                .expect("GQA Q-head tile size must fit u32"),
            dtype: self.config.dtype,
        }
    }

    fn paged_sdpa_shape(&self, shape: GQAReplayShape) -> GQAPagedSDPAShape {
        GQAPagedSDPAShape {
            num_tokens: shape.num_tokens,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
        }
    }

    fn tiled_sdpa_shape(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
        q_head_tile_size: u32,
    ) -> GQATiledSDPAShape {
        GQATiledSDPAShape {
            num_tokens: shape.num_tokens,
            num_q_token_tiles: shape.num_q_token_tiles,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
            num_q_heads: self.core.num_q_heads.try_into().expect("GQA q heads must fit u32"),
            num_kv_heads: self.core.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
            head_dim: self.core.head_dim.try_into().expect("GQA head_dim must fit u32"),
            q_head_tile_size,
            q_token_tile_size: self.config.q_token_tile_size,
            kv_token_tile_size: self.config.tiled_kv_token_tile_size,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            dtype: self.config.dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
            gqa_layer_index,
        }
    }

    fn activation_gate_shape(&self, shape: GQAReplayShape) -> GQAActivationGateShape {
        GQAActivationGateShape {
            num_tokens: shape.num_tokens,
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

fn projection_split_config(core: &GQACore, config: GQAMetalConfig) -> GQAProjectionSplitConfig {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.dtype {
        Dtype::Float32 => GQAProjectionSplitConfig::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => GQAProjectionSplitConfig::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &GQACore, config: GQAMetalConfig, num_heads: usize) -> GQANormRopeConfig {
    let num_heads_u32 = num_heads.try_into().expect("GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.dtype {
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
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn kv_update_config(core: &GQACore, config: GQAMetalConfig) -> GQAKVPageUpdateConfig {
    GQAKVPageUpdateConfig {
        num_kv_heads: core.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("GQA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.dtype,
    }
}

fn activation_gate_config(core: &GQACore, config: GQAMetalConfig) -> GQAActivationGateConfig {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.dtype {
        Dtype::Float32 => GQAActivationGateConfig::f32(num_q_heads, head_dim),
        Dtype::Bfloat16 => GQAActivationGateConfig::bf16(num_q_heads, head_dim),
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn validate_config_for_core(core: &GQACore, config: GQAMetalConfig) {
    config.validate();
    assert!(config.rope_dim as usize <= core.head_dim);
    assert!(config.num_tokens_per_page(core) > 0);
}

fn affine_shape(m: u32, n: usize, k: usize, config: GQAMetalConfig) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: m.try_into().expect("GQA affine m must fit i32"),
        n: n.try_into().expect("GQA affine n must fit i32"),
        k: k.try_into().expect("GQA affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GQA group_size must fit i32"),
        bits: config.bits.try_into().expect("GQA bits must fit i32"),
        input_dtype: config.dtype,
        output_dtype: config.dtype,
        affine_dtype: config.dtype,
    }
}

fn qmv_batch_limit(input_dim: usize, output_dim: usize) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::GQACore;

    use super::GQA;
    use super::GQAMetalConfig;
    use super::GQASDPAPath;

    #[test]
    #[should_panic(expected = "GQA K/V bytes per token must fit usize")]
    fn test_tokens_per_page_rejects_kv_byte_overflow() {
        let num_heads = usize::MAX / 6;
        let core = GQACore::new(0, 1, 1, num_heads, num_heads, 1.0);
        GQAMetalConfig {
            group_size: 64,
            bits: 4,
            page_bytes: 65536,
            context_parallel_kv_token_tile_size: 256,
            context_parallel_num_threads_per_threadblock: 256,
            context_parallel_max_q_head_tile_size: 8,
            q_token_tile_size: 8,
            tiled_kv_token_tile_size: 16,
            rope_dim: 64,
            norm_eps: 1.0e-6,
            rope_theta: 10_000_000.0,
            rope_scale: 1.0,
            dtype: Dtype::Float32,
        }
        .num_tokens_per_page(&core);
    }

    #[test]
    fn test_path() {
        let core = GQACore::new(3, 5120, 256, 24, 4, 0.0625);
        let config = GQAMetalConfig {
            group_size: 64,
            bits: 4,
            page_bytes: 65536,
            context_parallel_kv_token_tile_size: 256,
            context_parallel_num_threads_per_threadblock: 256,
            context_parallel_max_q_head_tile_size: 8,
            q_token_tile_size: 8,
            tiled_kv_token_tile_size: 16,
            rope_dim: 64,
            norm_eps: 1.0e-6,
            rope_theta: 10_000_000.0,
            rope_scale: 1.0,
            dtype: Dtype::Bfloat16,
        };
        let gqa_q_token_tile_8 = GQA::new(&Device::system_default(), core.clone(), config);

        assert_eq!(gqa_q_token_tile_8.sdpa_path(4, 4), GQASDPAPath::ContextParallel);
        assert_eq!(
            gqa_q_token_tile_8.sdpa_path(8, 4),
            GQASDPAPath::Tiled { q_head_tile_size: 3 }
        );
        assert_eq!(
            gqa_q_token_tile_8.sdpa_path(16, 4),
            GQASDPAPath::Tiled { q_head_tile_size: 6 }
        );

        let gqa_q_token_tile_16 = GQA::new(
            &Device::system_default(),
            core,
            GQAMetalConfig {
                q_token_tile_size: 16,
                ..config
            },
        );
        assert_eq!(
            gqa_q_token_tile_16.sdpa_path(32, 2),
            GQASDPAPath::Tiled { q_head_tile_size: 4 }
        );
    }
}
