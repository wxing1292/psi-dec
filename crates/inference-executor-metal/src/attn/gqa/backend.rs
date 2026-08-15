use inference_backend_metal::components::GQAActivationGateBuffers;
use inference_backend_metal::components::GQAActivationGateConfig;
use inference_backend_metal::components::GQAActivationGateKernel;
use inference_backend_metal::components::GQAActivationGateShape;
use inference_backend_metal::components::GQACompute;
use inference_backend_metal::components::GQAComputePath;
use inference_backend_metal::components::GQAKVPageWrite;
use inference_backend_metal::components::GQAKVPageWriteBuffers;
use inference_backend_metal::components::GQAKVPageWriteConfig;
use inference_backend_metal::components::GQAKVPageWriteShape;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::components::GQAPagedSDPAConfig;
use inference_backend_metal::components::GQAPagedSDPAKernels;
use inference_backend_metal::components::GQAPagedSDPAMapBuffers;
use inference_backend_metal::components::GQAPagedSDPAReduceBuffers;
use inference_backend_metal::components::GQAPagedSDPAShape;
use inference_backend_metal::components::GQAQGKVSplitBuffers;
use inference_backend_metal::components::GQAQGKVSplitConfig;
use inference_backend_metal::components::GQAQGKVSplitKernel;
use inference_backend_metal::components::GQAQGKVSplitShape;
use inference_backend_metal::components::GQATiledSDPAConfig;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::components::RMSNormRopeBuffers;
use inference_backend_metal::components::RMSNormRopeConfig;
use inference_backend_metal::components::RMSNormRopeKernel;
use inference_backend_metal::components::RMSNormRopeShape;
use inference_backend_metal::components::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;

use super::gqa_compute_config;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::scratch::GQAScratch;
use crate::attn::gqa::scratch::GQAScratchBindings;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

pub const GQA_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("gqa.num_active_tokens");
pub const GQA_NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey = ReplayParameterKey::new("gqa.num_active_q_token_tiles");
pub const GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES: ReplayParameterKey =
    ReplayParameterKey::new("gqa.num_active_sdpa_map_task_templates");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GQAReplayMode {
    Exact,
    Bucketed,
    BucketedWithTokenKey(ReplayParameterKey),
}

impl GQAReplayMode {
    fn active_tokens_key(self) -> Option<ReplayParameterKey> {
        match self {
            Self::Exact => None,
            Self::Bucketed => Some(GQA_NUM_ACTIVE_TOKENS),
            Self::BucketedWithTokenKey(key) => {
                assert_ne!(
                    key, GQA_NUM_ACTIVE_Q_TOKEN_TILES,
                    "GQA active-token key must differ from the private Q-token-tile key"
                );
                assert_ne!(
                    key, GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES,
                    "GQA active-token key must differ from the private SDPA map TaskTemplate key"
                );
                Some(key)
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQAReplayTopology {
    pub compute_path: GQAComputePath,
    pub qgkv_affine: AffineQuantizedMatmulKernelKind,
    pub output_affine: AffineQuantizedMatmulKernelKind,
}

pub fn add_gqa_replay_arguments(shape: GQAReplayShape, topology: GQAReplayTopology, arguments: &mut ReplayArguments) {
    add_gqa_private_replay_arguments(shape, topology, arguments);
    arguments.set_u32(GQA_NUM_ACTIVE_TOKENS, shape.num_tokens);
}

pub fn add_gqa_private_replay_arguments(
    shape: GQAReplayShape,
    topology: GQAReplayTopology,
    arguments: &mut ReplayArguments,
) {
    shape.validate();
    if matches!(topology.compute_path, GQAComputePath::TiledQueryTokens { .. }) {
        arguments.set_u32(GQA_NUM_ACTIVE_Q_TOKEN_TILES, shape.num_q_token_tiles);
    }
    arguments.set_u32(
        GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES,
        shape.num_sdpa_map_task_templates,
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQAMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub page_bytes: u32,
    pub rope_dim: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub io_dtype: Dtype,
}

impl GQAMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.page_bytes > 0);
        assert!(self.rope_dim > 0);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        self.rope_scaling.validate();
        assert!(
            !matches!(self.rope_scaling, RopeScaling::Yarn { .. }) || self.rope_theta > 1.0,
            "Yarn rope_theta must be greater than 1"
        );
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 GQA model boundary is not supported"),
            dtype => panic!("unsupported GQA model boundary dtype {dtype:?}"),
        }
    }

    pub fn num_tokens_per_page(self, core: &GQACore) -> u32 {
        gqa_compute_config(self, core.num_q_heads, core.num_kv_heads, core.head_dim).num_tokens_per_page()
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
    pub gqa_layer_index: ReplayU32,
    pub batch_metadata: &'a GQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: GQAWeights<'a>,
    pub scratch: GQAScratchBindings<'a>,
    pub replay_mode: GQAReplayMode,
}

/// The caller-owned next-hidden-state buffer returned by one GQA recording.
pub type GQAOutput<'a> = &'a Buffer;

/// Gated GQA data flow:
///
/// ```text
/// hidden_state
///   -> qgkv
///   -> scratch.qgkv
///   -> qgkv_to_q_g_k_v
///      |- scratch.q -> q_norm_rope -> scratch.q_norm_rope -----------+
///      |- scratch.g -------------------------------------------------|
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
///                                -> gate with scratch.g
///                                -> scratch.gated_attention_output
///                                -> output
///                                -> next_hidden_state
/// ```
pub struct GQA {
    device: Device,
    core: GQACore,
    config: GQAMetalConfig,
    compute: GQACompute,
    qgkv: AffineQuantizedMatmul,
    qgkv_to_q_g_k_v: GQAQGKVSplitKernel,
    q_norm_rope: RMSNormRopeKernel,
    k_norm_rope: RMSNormRopeKernel,
    kv_page_write: GQAKVPageWrite,
    gate: GQAActivationGateKernel,
    output: AffineQuantizedMatmul,
}

impl GQA {
    fn validate_input(&self, input: &GQAInput<'_>) {
        let shape = input.batch_metadata.replay_shape();
        shape.validate();
        match input.replay_mode {
            GQAReplayMode::Exact => {
                assert_eq!(shape.num_tokens, shape.num_total_tokens);
                assert_eq!(shape.num_q_token_tiles, shape.num_total_q_token_tiles);
            },
            GQAReplayMode::Bucketed => {
                self.validate_token_capacity_topology(shape.num_tokens, shape.num_total_tokens);
            },
            GQAReplayMode::BucketedWithTokenKey(_) => {
                let _ = input.replay_mode.active_tokens_key();
                self.validate_token_capacity_topology(shape.num_tokens, shape.num_total_tokens);
            },
        }
        input.page_table_layout.validate();
        if let ReplayU32::Fixed(index) = input.gqa_layer_index {
            assert!(index < input.page_table_layout.num_gqa_layers);
        }
    }

    pub fn new(device: &Device, core: GQACore, config: GQAMetalConfig) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qgkv = core.qgkv_shape();
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
            qgkv: AffineQuantizedMatmul::new(device, affine_config(qgkv.out_dim, qgkv.in_dim, config)),
            qgkv_to_q_g_k_v: GQAQGKVSplitKernel::new(device, qgkv_to_q_g_k_v_config(&core, config)),
            q_norm_rope: RMSNormRopeKernel::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: RMSNormRopeKernel::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: GQAKVPageWrite::new(device, kv_page_write_config(&core, config)),
            gate: GQAActivationGateKernel::new(device, gate_config(&core, config)),
            output: AffineQuantizedMatmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_tokens_per_page(&self.core)
    }

    pub fn new_scratch(&self, max_tokens: usize) -> GQAScratch {
        GQAScratch::new(&self.device, &self.core, self.config, self.compute, max_tokens)
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
                (cu[1] - cu[0]).div_ceil(self.compute.tiled_query_token_tile_size())
            })
            .sum();
        let compute_path = self.compute.select(num_tokens, num_q_token_tiles);
        batch_metadata.update(req_slots, token_indices, cu_tokens, compute_path)
    }

    pub fn prepare_bucketed(
        &self,
        batch_metadata: &GQAMetadataBuffers,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        policy: &GQAReplayBucketPolicy,
    ) -> GQAReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        let num_q_token_tiles = cu_tokens
            .windows(2)
            .map(|cu| {
                assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
                (cu[1] - cu[0]).div_ceil(self.compute.tiled_query_token_tile_size())
            })
            .sum();
        let compute_path = self.compute.select(num_tokens, num_q_token_tiles);
        batch_metadata.update_bucketed(req_slots, token_indices, cu_tokens, compute_path, policy)
    }

    pub fn prepare_bucketed_with_token_capacity(
        &self,
        batch_metadata: &GQAMetadataBuffers,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> GQAReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        assert!(num_tokens > 0, "GQA replay requires active tokens");
        assert!(
            num_tokens <= num_total_tokens,
            "GQA active token count must not exceed the caller-owned token capacity"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GQA caller-owned token capacity must not exceed the metadata capacity"
        );
        self.validate_token_capacity_topology(num_tokens, num_total_tokens);
        let num_q_token_tiles = cu_tokens
            .windows(2)
            .map(|cu| {
                assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
                (cu[1] - cu[0]).div_ceil(self.compute.tiled_query_token_tile_size())
            })
            .sum();
        let compute_path = self.compute.select(num_tokens, num_q_token_tiles);
        batch_metadata.update_bucketed_with_token_capacity(
            req_slots,
            token_indices,
            cu_tokens,
            compute_path,
            policy,
            num_total_tokens,
        )
    }

    pub fn replay_bucket_policy(&self, max_tokens: u32) -> GQAReplayBucketPolicy {
        GQAReplayBucketPolicy::new(max_tokens, &self.replay_token_topology_boundaries())
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qgkv.topology_boundaries().into_vec();
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    fn validate_token_capacity_topology(&self, num_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.qgkv.topology(num_tokens),
            self.qgkv.topology(num_total_tokens),
            "GQA caller-owned token capacity must preserve the active QGKV affine topology"
        );
        assert_eq!(
            self.output.topology(num_tokens),
            self.output.topology(num_total_tokens),
            "GQA caller-owned token capacity must preserve the active output affine topology"
        );
    }

    pub fn replay_topology(&self, batch_metadata: &GQAMetadataBuffers) -> GQAReplayTopology {
        let shape = batch_metadata.replay_shape();
        shape.validate();
        GQAReplayTopology {
            compute_path: batch_metadata.compute_path(),
            qgkv_affine: self.qgkv.topology(shape.num_total_tokens),
            output_affine: self.output.topology(shape.num_total_tokens),
        }
    }
}

impl ReplayLayer for GQA {
    type Input<'a> = GQAInput<'a>;
    type Output<'a> = GQAOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let page_table_index = input.gqa_layer_index;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let kv_cache = input.kv_cache;
        let weights = input.weights;
        let batch_metadata = input.batch_metadata;
        let scratch = input.scratch;
        let active_tokens_key = input.replay_mode.active_tokens_key();
        let bucketed = active_tokens_key.is_some();
        let active_tokens_key = active_tokens_key.unwrap_or(GQA_NUM_ACTIVE_TOKENS);
        let active_tokens = ReplayU32::Parameter(active_tokens_key);
        let active_q_token_tiles = ReplayU32::Parameter(GQA_NUM_ACTIVE_Q_TOKEN_TILES);
        let active_sdpa_map_task_templates = ReplayU32::Parameter(GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES);
        let qgkv = if bucketed {
            self.qgkv.invoke_bucketed(
                shape.num_total_tokens,
                active_tokens_key,
                scratch.qgkv,
                0,
                hidden_state,
                0,
                weights.qgkv_weight,
                0,
                weights.qgkv_scales,
                0,
                weights.qgkv_biases,
                0,
            )
        } else {
            self.qgkv.invoke(
                shape.num_total_tokens.try_into().expect("GQA token count must fit i32"),
                scratch.qgkv,
                0,
                hidden_state,
                0,
                weights.qgkv_weight,
                0,
                weights.qgkv_scales,
                0,
                weights.qgkv_biases,
                0,
            )
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(qgkv));
        let qgkv_to_q_g_k_v_shape = self.qgkv_to_q_g_k_v_shape(shape);
        let qgkv_to_q_g_k_v_buffers = GQAQGKVSplitBuffers {
            qgkv: scratch.qgkv,
            q: scratch.q,
            g: scratch.g,
            k: scratch.k,
            v: scratch.v,
        };
        let qgkv_to_q_g_k_v = if bucketed {
            self.qgkv_to_q_g_k_v
                .invoke_bucketed(qgkv_to_q_g_k_v_shape, qgkv_to_q_g_k_v_buffers, active_tokens)
        } else {
            self.qgkv_to_q_g_k_v
                .invoke(qgkv_to_q_g_k_v_shape, qgkv_to_q_g_k_v_buffers)
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(qgkv_to_q_g_k_v));
        let q_norm_rope_shape = self.norm_rope_shape(shape);
        let q_norm_rope_buffers = RMSNormRopeBuffers {
            input: scratch.q,
            norm_weight: weights.q_norm_weight,
            flat_token_indices: batch_metadata.flat_token_indices(),
            output: scratch.q_norm_rope,
        };
        let q_norm_rope = if bucketed {
            self.q_norm_rope
                .invoke_bucketed(q_norm_rope_shape, q_norm_rope_buffers, active_tokens)
        } else {
            self.q_norm_rope.invoke(q_norm_rope_shape, q_norm_rope_buffers)
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(q_norm_rope));
        let k_norm_rope_shape = self.norm_rope_shape(shape);
        let k_norm_rope_buffers = RMSNormRopeBuffers {
            input: scratch.k,
            norm_weight: weights.k_norm_weight,
            flat_token_indices: batch_metadata.flat_token_indices(),
            output: scratch.k_norm_rope,
        };
        let k_norm_rope = if bucketed {
            self.k_norm_rope
                .invoke_bucketed(k_norm_rope_shape, k_norm_rope_buffers, active_tokens)
        } else {
            self.k_norm_rope.invoke(k_norm_rope_shape, k_norm_rope_buffers)
        };
        recorder.record(ReplayOp::opaque(k_norm_rope));
        let kv_page_write_shape = self.kv_page_write_shape(shape, page_table_layout);
        let kv_page_write_buffers = GQAKVPageWriteBuffers {
            pages: kv_cache.kv_pages,
            flat_k: scratch.k_norm_rope,
            flat_v: scratch.v,
            req_slots: batch_metadata.req_slots(),
            flat_token_indices: batch_metadata.flat_token_indices(),
            page_ids: kv_cache.page_ids,
        };
        let kv_page_write = if bucketed {
            self.kv_page_write.invoke_bucketed(
                kv_page_write_shape,
                kv_page_write_buffers,
                active_tokens,
                page_table_index,
            )
        } else {
            self.kv_page_write
                .invoke(kv_page_write_shape, kv_page_write_buffers, page_table_index)
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(kv_page_write));
        let attention_output = match batch_metadata.compute_path() {
            GQAComputePath::SingleQueryToken {
                kv_token_tile_size,
                num_threads_per_threadblock,
                q_head_tile_size,
            } => {
                let sdpa_config = self.paged_sdpa_config(
                    page_table_layout,
                    kv_token_tile_size,
                    num_threads_per_threadblock,
                    q_head_tile_size,
                );
                let sdpa_shape = self.paged_sdpa_shape(shape);
                let sdpa = GQAPagedSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
                let buffers = GQAPagedSDPAMapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                };
                let map = if bucketed {
                    sdpa.invoke_map_bucketed(buffers, page_table_index, active_tokens, active_sdpa_map_task_templates)
                } else {
                    sdpa.invoke_map(buffers, page_table_index)
                };
                recorder.record_with_barrier_before(ReplayOp::opaque(map));
                let reduce_buffers = GQAPagedSDPAReduceBuffers {
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                };
                let reduce = if bucketed {
                    sdpa.invoke_reduce_bucketed(reduce_buffers, active_tokens)
                } else {
                    sdpa.invoke_reduce(reduce_buffers)
                };
                recorder.record_with_barrier_before(ReplayOp::opaque(reduce));
                scratch.attention_output
            },
            GQAComputePath::TiledQueryTokens {
                q_token_tile_size,
                kv_token_tile_size,
                q_head_tile_size,
            } => {
                let sdpa_config = self.tiled_sdpa_config(
                    page_table_layout,
                    q_token_tile_size,
                    kv_token_tile_size,
                    q_head_tile_size,
                );
                let sdpa_shape = self.tiled_sdpa_shape(shape);
                let sdpa = GQATiledSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
                let buffers = GQATiledSDPAMapBuffers {
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
                };
                let map = if bucketed {
                    sdpa.invoke_map_bucketed(
                        buffers,
                        page_table_index,
                        active_tokens,
                        active_q_token_tiles,
                        active_sdpa_map_task_templates,
                    )
                } else {
                    sdpa.invoke_map(buffers, page_table_index)
                };
                recorder.record_with_barrier_before(ReplayOp::opaque(map));
                let reduce_buffers = GQATiledSDPAReduceBuffers {
                    partial_output: scratch.sdpa_partial_output,
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    q_token_tiles: batch_metadata.q_token_tiles(),
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                };
                let reduce = if bucketed {
                    sdpa.invoke_reduce_bucketed(reduce_buffers, active_q_token_tiles)
                } else {
                    sdpa.invoke_reduce(reduce_buffers)
                };
                recorder.record_with_barrier_before(ReplayOp::opaque(reduce));
                scratch.attention_output
            },
        };
        let gate_shape = self.gate_shape(shape);
        let gate_buffers = GQAActivationGateBuffers {
            attention_output,
            g: scratch.g,
            output: scratch.gated_attention_output,
        };
        let gate = if bucketed {
            self.gate.invoke_bucketed(gate_shape, gate_buffers, active_tokens)
        } else {
            self.gate.invoke(gate_shape, gate_buffers)
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(gate));
        let output = if bucketed {
            self.output.invoke_bucketed(
                shape.num_total_tokens,
                active_tokens_key,
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
            )
        } else {
            self.output.invoke(
                shape.num_total_tokens.try_into().expect("GQA token count must fit i32"),
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
            )
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(output));
        next_hidden_state
    }
}

impl GQA {
    fn qgkv_to_q_g_k_v_shape(&self, shape: GQAReplayShape) -> GQAQGKVSplitShape {
        GQAQGKVSplitShape {
            num_total_tokens: shape.num_total_tokens,
        }
    }

    fn norm_rope_shape(&self, shape: GQAReplayShape) -> RMSNormRopeShape {
        RMSNormRopeShape {
            num_total_tokens: shape.num_total_tokens,
        }
    }

    fn kv_page_write_shape(&self, shape: GQAReplayShape, page_table_layout: GQAPageTableLayout) -> GQAKVPageWriteShape {
        GQAKVPageWriteShape {
            num_total_token_writes: shape.num_total_tokens,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn paged_sdpa_config(
        &self,
        page_table_layout: GQAPageTableLayout,
        kv_token_tile_size: u32,
        num_threads_per_threadblock: u32,
        q_head_tile_size: u32,
    ) -> GQAPagedSDPAConfig {
        debug_assert!(u32::try_from(self.core.num_q_heads).is_ok());
        debug_assert!(u32::try_from(self.core.num_kv_heads).is_ok());
        debug_assert!(u32::try_from(self.core.head_dim).is_ok());
        GQAPagedSDPAConfig {
            num_q_heads: self.core.num_q_heads as u32,
            num_kv_heads: self.core.num_kv_heads as u32,
            head_dim: self.core.head_dim as u32,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            kv_token_tile_size,
            num_threads_per_threadblock,
            q_head_tile_size,
            dtype: self.config.io_dtype,
        }
    }

    fn paged_sdpa_shape(&self, shape: GQAReplayShape) -> GQAPagedSDPAShape {
        GQAPagedSDPAShape {
            num_total_tokens: shape.num_total_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn tiled_sdpa_config(
        &self,
        page_table_layout: GQAPageTableLayout,
        q_token_tile_size: u32,
        kv_token_tile_size: u32,
        q_head_tile_size: u32,
    ) -> GQATiledSDPAConfig {
        debug_assert!(u32::try_from(self.core.num_q_heads).is_ok());
        debug_assert!(u32::try_from(self.core.num_kv_heads).is_ok());
        debug_assert!(u32::try_from(self.core.head_dim).is_ok());
        GQATiledSDPAConfig {
            num_q_heads: self.core.num_q_heads as u32,
            num_kv_heads: self.core.num_kv_heads as u32,
            head_dim: self.core.head_dim as u32,
            q_head_tile_size,
            q_token_tile_size,
            kv_token_tile_size,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            dtype: self.config.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn tiled_sdpa_shape(&self, shape: GQAReplayShape) -> GQATiledSDPAShape {
        GQATiledSDPAShape {
            num_total_tokens: shape.num_total_tokens,
            num_total_q_token_tiles: shape.num_total_q_token_tiles,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn gate_shape(&self, shape: GQAReplayShape) -> GQAActivationGateShape {
        GQAActivationGateShape {
            num_total_tokens: shape.num_total_tokens,
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

fn qgkv_to_q_g_k_v_config(core: &GQACore, config: GQAMetalConfig) -> GQAQGKVSplitConfig {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => GQAQGKVSplitConfig::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => GQAQGKVSplitConfig::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &GQACore, config: GQAMetalConfig, num_heads: usize) -> RMSNormRopeConfig {
    let num_heads_u32 = num_heads.try_into().expect("GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    let norm_rope = match config.io_dtype {
        Dtype::Float32 => {
            RMSNormRopeConfig::f32(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        Dtype::Bfloat16 => {
            RMSNormRopeConfig::bf16(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    };
    norm_rope.with_rope_scaling(config.rope_scaling)
}

fn kv_page_write_config(core: &GQACore, config: GQAMetalConfig) -> GQAKVPageWriteConfig {
    GQAKVPageWriteConfig {
        num_kv_heads: core.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("GQA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
    }
}

fn gate_config(core: &GQACore, config: GQAMetalConfig) -> GQAActivationGateConfig {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.io_dtype {
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

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("GQA affine n must fit i32"),
        k: k.try_into().expect("GQA affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GQA group_size must fit i32"),
        bits: config.bits.try_into().expect("GQA bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_backend_metal::components::RopeScaling;
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayU32;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::attn::GQACore;

    use super::GQA_NUM_ACTIVE_TOKENS;
    use super::GQAActivationGateBuffers;
    use super::GQAActivationGateConfig;
    use super::GQAActivationGateKernel;
    use super::GQAActivationGateShape;
    use super::GQAMetalConfig;
    use super::GQAQGKVSplitBuffers;
    use super::GQAQGKVSplitConfig;
    use super::GQAQGKVSplitKernel;
    use super::GQAQGKVSplitShape;
    use super::RMSNormRopeBuffers;
    use super::RMSNormRopeConfig;
    use super::RMSNormRopeKernel;
    use super::RMSNormRopeShape;

    #[test]
    #[should_panic(expected = "GQA Q-head count must fit u32")]
    fn test_tokens_per_page_rejects_head_count_outside_shader_domain() {
        let num_heads = usize::MAX / 6;
        let core = GQACore::new(0, 1, 1, num_heads, num_heads, 1.0);
        GQAMetalConfig {
            group_size: 64,
            bits: 4,
            page_bytes: 65536,
            rope_dim: 64,
            norm_eps: 1.0e-6,
            rope_theta: 10_000_000.0,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        }
        .num_tokens_per_page(&core);
    }

    #[test]
    fn test_bucketed_scratch_leaves_shrink_expand_and_guard_poisoned_tail() {
        let device = inference_backend_metal::metal::Device::system_default();
        let stream = Stream::new(&device);
        let split_config = GQAQGKVSplitConfig::f32(1, 1, 2);
        let split_shape = GQAQGKVSplitShape { num_total_tokens: 2 };
        let norm_config = RMSNormRopeConfig::f32(1, 2, 2, 1.0e-6, 1_000_000.0);
        let norm_shape = RMSNormRopeShape { num_total_tokens: 2 };
        let gate_config = GQAActivationGateConfig::f32(1, 2);
        let gate_shape = GQAActivationGateShape { num_total_tokens: 2 };
        let split = GQAQGKVSplitKernel::new(&device, split_config);
        let q_norm = RMSNormRopeKernel::new(&device, norm_config);
        let k_norm = RMSNormRopeKernel::new(&device, norm_config);
        let gate = GQAActivationGateKernel::new(&device, gate_config);
        let input = Buffer::new_zeroed_elements(&device, split_config.num_qgkv_slots(split_shape), Dtype::Float32);
        let q = Buffer::new_zeroed_elements(&device, split_config.num_q_slots(split_shape), Dtype::Float32);
        let g = Buffer::new_zeroed_elements(&device, split_config.num_q_slots(split_shape), Dtype::Float32);
        let k = Buffer::new_zeroed_elements(&device, split_config.num_kv_slots(split_shape), Dtype::Float32);
        let v = Buffer::new_zeroed_elements(&device, split_config.num_kv_slots(split_shape), Dtype::Float32);
        let norm_weight = Buffer::from_slice(&device, &[bf16::from_f32(1.0).to_bits(), bf16::from_f32(1.0).to_bits()]);
        let flat_token_indices = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
        let q_norm_output = Buffer::new_zeroed_elements(&device, norm_config.num_slots(norm_shape), Dtype::Float32);
        let k_norm_output = Buffer::new_zeroed_elements(&device, norm_config.num_slots(norm_shape), Dtype::Float32);
        let gated = Buffer::new_zeroed_elements(&device, gate_config.num_values(gate_shape), Dtype::Float32);
        let mut builder = stream.create_replay_program();
        builder.record(split.invoke_bucketed(
            split_shape,
            GQAQGKVSplitBuffers {
                qgkv: &input,
                q: &q,
                g: &g,
                k: &k,
                v: &v,
            },
            ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
        ));
        builder.record_with_barrier_before(q_norm.invoke_bucketed(
            norm_shape,
            RMSNormRopeBuffers {
                input: &q,
                norm_weight: &norm_weight,
                flat_token_indices: &flat_token_indices,
                output: &q_norm_output,
            },
            ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
        ));
        builder.record_with_barrier_before(k_norm.invoke_bucketed(
            norm_shape,
            RMSNormRopeBuffers {
                input: &k,
                norm_weight: &norm_weight,
                flat_token_indices: &flat_token_indices,
                output: &k_norm_output,
            },
            ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
        ));
        builder.record_with_barrier_before(gate.invoke_bucketed(
            gate_shape,
            GQAActivationGateBuffers {
                attention_output: &q_norm_output,
                g: &g,
                output: &gated,
            },
            ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
        ));
        let replay = builder.build();
        let valid_rows = [
            [1.0_f32, 2.0, 0.5, -0.5, 3.0, 4.0, 5.0, 6.0],
            [2.0_f32, 1.0, -1.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        ];
        for num_active_tokens in [1_usize, 2, 1] {
            let mut input_values = valid_rows.into_iter().flatten().collect::<Vec<_>>();
            let mut token_indices = vec![0_u32, 1];
            if num_active_tokens == 1 {
                input_values[8..].fill(f32::NAN);
                token_indices[1] = u32::MAX;
            }
            input.write_typed(0, &input_values);
            flat_token_indices.write_typed(0, &token_indices);
            for buffer in [&q, &g, &k, &v, &q_norm_output, &k_norm_output, &gated] {
                buffer.write_typed(0, &[-777.0_f32; 4]);
            }
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(GQA_NUM_ACTIVE_TOKENS, num_active_tokens as u32),
                )
                .wait();

            let expected_q = valid_rows[..num_active_tokens]
                .iter()
                .flat_map(|row| [row[0], row[1]])
                .collect::<Vec<_>>();
            let expected_g = valid_rows[..num_active_tokens]
                .iter()
                .flat_map(|row| [row[2], row[3]])
                .collect::<Vec<_>>();
            let expected_k = valid_rows[..num_active_tokens]
                .iter()
                .flat_map(|row| [row[4], row[5]])
                .collect::<Vec<_>>();
            let expected_v = valid_rows[..num_active_tokens]
                .iter()
                .flat_map(|row| [row[6], row[7]])
                .collect::<Vec<_>>();
            let expected_q_norm = expected_q
                .as_chunks::<2>()
                .0
                .iter()
                .enumerate()
                .flat_map(|(token_index, row)| norm_rope_2(row, token_index as f32))
                .collect::<Vec<_>>();
            let expected_k_norm = expected_k
                .as_chunks::<2>()
                .0
                .iter()
                .enumerate()
                .flat_map(|(token_index, row)| norm_rope_2(row, token_index as f32))
                .collect::<Vec<_>>();
            let expected_gated = expected_q_norm
                .iter()
                .zip(&expected_g)
                .map(|(&value, &gate)| value / (1.0 + (-gate).exp()))
                .collect::<Vec<_>>();
            let num_active_values = num_active_tokens * 2;
            for (buffer, expected) in [
                (&q, &expected_q),
                (&g, &expected_g),
                (&k, &expected_k),
                (&v, &expected_v),
                (&q_norm_output, &expected_q_norm),
                (&k_norm_output, &expected_k_norm),
                (&gated, &expected_gated),
            ] {
                assert_close(&buffer.read_typed::<f32>(0, num_active_values), expected, 2.0e-5);
                assert_eq!(
                    buffer.read_typed::<f32>(num_active_values, 4 - num_active_values),
                    vec![-777.0; 4 - num_active_values]
                );
            }
        }
    }

    fn norm_rope_2(row: &[f32], position: f32) -> [f32; 2] {
        let inv_rms = ((row[0] * row[0] + row[1] * row[1]) / 2.0 + 1.0e-6).sqrt().recip();
        let x0 = row[0] * inv_rms;
        let x1 = row[1] * inv_rms;
        let (sin, cos) = position.sin_cos();
        [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
