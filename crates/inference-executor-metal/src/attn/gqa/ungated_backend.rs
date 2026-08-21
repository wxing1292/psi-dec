use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qkv_split as backend_qkv_split;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::split_kv::single_q as backend_single_q;
use inference_backend_metal::components::gqa::split_kv::tiled_q as backend_tiled_q;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::backend::recorder::Recorder;

use super::gqa_sdpa_config;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::sdpa::RequestShape;
use crate::attn::gqa::sdpa::Selector;
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
///                              backend_single_q::Compute or backend_tiled_q::Compute
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
    sdpa_selector: Selector,
    qkv: affine_quantized::Matmul,
    qkv_to_q_k_v: backend_qkv_split::Compute,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    kv_page_write: backend_kv_page_write::Compute,
    output: affine_quantized::Matmul,
}

impl GQAMetalConfig {
    pub fn num_ungated_tokens_per_page(self, core: &UngatedGQACore) -> u32 {
        gqa_sdpa_config(self, core.num_q_heads, core.num_kv_heads, core.head_dim).tokens_per_page
    }
}

impl UngatedGQA {
    fn validate_input(&self, input: &UngatedGQAInput<'_>) {
        input.batch_metadata.replay_shape().validate();
        input.page_table_layout.validate();
        assert!(input.gqa_layer_index < input.page_table_layout.num_gqa_layers);
    }

    pub fn new(device: &Device, core: UngatedGQACore, config: GQAMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qkv = core.qkv_shape();
        let output = core.output_shape();
        Self {
            device: device.clone(),
            core: core.clone(),
            config,
            sdpa_selector: Selector::new(
                backend_sdpa::Registry::new(gqa_sdpa_config(
                    config,
                    core.num_q_heads,
                    core.num_kv_heads,
                    core.head_dim,
                )),
                max_tokens,
            ),
            qkv: affine_quantized::Matmul::new(device, affine_config(qkv.out_dim, qkv.in_dim, config)),
            qkv_to_q_k_v: backend_qkv_split::Compute::new(device, qkv_to_q_k_v_config(&core, config)),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: backend_kv_page_write::Compute::new(device, kv_page_write_config(&core, config)),
            output: affine_quantized::Matmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_ungated_tokens_per_page(&self.core)
    }

    pub fn new_scratch(&self) -> UngatedGQAScratch {
        UngatedGQAScratch::new(&self.device, &self.core, self.config, &self.sdpa_selector)
    }

    pub fn prepare(
        &self,
        batch_metadata: &GQAMetadataBuffers,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> GQAReplayShape {
        assert_eq!(
            batch_metadata.max_tokens(),
            self.sdpa_selector.limits().max_map_task_templates as usize
        );
        let request_shapes = RequestShape::from_batch(token_indices, cu_tokens);
        let selection = self.sdpa_selector.select(&request_shapes, policy, num_total_tokens);
        batch_metadata.update(req_slots, token_indices, cu_tokens, &selection)
    }

    pub fn replay_bucket_policy(&self, max_tokens: u32) -> GQAReplayBucketPolicy {
        GQAReplayBucketPolicy::new(max_tokens, &self.replay_token_topology_boundaries())
    }

    fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qkv.topology_boundaries().into_vec();
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
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
            backend_qkv_split::Buffers {
                qkv: scratch.qkv,
                q: scratch.q,
                k: scratch.k,
                v: scratch.v,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            self.norm_rope_shape(shape),
            rms_norm_rope::Buffers {
                input: scratch.q,
                norm_weight: weights.q_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            self.norm_rope_shape(shape),
            rms_norm_rope::Buffers {
                input: scratch.k,
                norm_weight: weights.k_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_page_write.invoke(
            self.kv_page_write_shape(shape, page_table_layout),
            backend_kv_page_write::Buffers {
                pages: kv_cache.kv_pages,
                flat_k: scratch.k_norm_rope,
                flat_v: scratch.v,
                req_slots: batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: kv_cache.page_ids,
            },
            ReplayU32::Fixed(gqa_layer_index),
        )));
        let attention_output = self.record_sdpa(recorder, input);
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
    fn record_sdpa<'a, R>(&'a self, recorder: &mut R, input: UngatedGQAInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let gqa_layer_index = input.gqa_layer_index;
        let batch_metadata = input.batch_metadata;
        let kv_cache = input.kv_cache;
        let scratch = input.scratch;
        let execution = batch_metadata.variant();
        let map_constants = execution.map.thread_block;
        if map_constants.max_q_tokens == 1 {
            let sdpa_config = self.split_kv_single_q_config(page_table_layout);
            let sdpa = backend_single_q::Compute::new(
                &self.device,
                sdpa_config,
                execution,
                self.split_kv_single_q_shape(shape),
            );
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_single_q::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                },
                ReplayU32::Fixed(gqa_layer_index),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_single_q::ReduceBuffers {
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
            )));
        } else {
            let sdpa_config = self.split_kv_tiled_q_config(page_table_layout);
            let sdpa =
                backend_tiled_q::Compute::new(&self.device, sdpa_config, execution, self.split_kv_tiled_q_shape(shape));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_tiled_q::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    flat_token_indices: batch_metadata.flat_token_indices(),
                    q_token_ranges: batch_metadata.q_token_ranges(),
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_output: scratch.sdpa_partial_output,
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                },
                ReplayU32::Fixed(gqa_layer_index),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(backend_tiled_q::ReduceBuffers {
                partial_output: scratch.sdpa_partial_output,
                partial_exp_sums: scratch.sdpa_partial_exp_sums,
                partial_max_logits: scratch.sdpa_partial_max_logits,
                q_token_ranges: batch_metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                output: scratch.attention_output,
            })));
        }
        scratch.attention_output
    }

    fn qkv_to_q_k_v_shape(&self, shape: GQAReplayShape) -> backend_qkv_split::Shape {
        backend_qkv_split::Shape {
            num_tokens: shape.num_tokens,
        }
    }

    fn norm_rope_shape(&self, shape: GQAReplayShape) -> rms_norm_rope::Shape {
        rms_norm_rope::Shape {
            num_total_tokens: shape.num_tokens,
        }
    }

    fn kv_page_write_shape(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
    ) -> backend_kv_page_write::Shape {
        backend_kv_page_write::Shape {
            num_total_token_writes: shape.num_tokens,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn split_kv_single_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_single_q::Config {
        backend_single_q::Config {
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
            dtype: self.config.io_dtype,
        }
    }

    fn split_kv_single_q_shape(&self, shape: GQAReplayShape) -> backend_single_q::Shape {
        backend_single_q::Shape {
            num_total_tokens: shape.num_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn split_kv_tiled_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_tiled_q::Config {
        backend_tiled_q::Config {
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
            dtype: self.config.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn split_kv_tiled_q_shape(&self, shape: GQAReplayShape) -> backend_tiled_q::Shape {
        backend_tiled_q::Shape {
            num_total_tokens: shape.num_tokens,
            num_total_q_token_tiles: shape.num_q_token_tiles,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }
}

fn backend_page_table_layout(shape: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: shape.num_req_slots,
        num_blocks: shape.num_blocks,
        num_gqa_layers: shape.num_gqa_layers,
        num_page_ids_per_block: shape.num_page_ids_per_block,
    }
}

fn qkv_to_q_k_v_config(core: &UngatedGQACore, config: GQAMetalConfig) -> backend_qkv_split::Config {
    let num_q_heads = core.num_q_heads.try_into().expect("ungated GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("ungated GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => backend_qkv_split::Config::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => backend_qkv_split::Config::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &UngatedGQACore, config: GQAMetalConfig, num_heads: usize) -> rms_norm_rope::Config {
    let num_heads_u32 = num_heads.try_into().expect("ungated GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
    let norm_rope = match config.io_dtype {
        Dtype::Float32 => {
            rms_norm_rope::Config::f32(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        Dtype::Bfloat16 => {
            rms_norm_rope::Config::bf16(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    };
    norm_rope.with_rope_scaling(config.rope_scaling)
}

fn kv_page_write_config(core: &UngatedGQACore, config: GQAMetalConfig) -> backend_kv_page_write::Config {
    backend_kv_page_write::Config {
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

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
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
