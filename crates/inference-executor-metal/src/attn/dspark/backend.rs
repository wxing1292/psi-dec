use inference_backend_metal::components::gqa::block_sdpa as backend_block_sdpa;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qkv_split as backend_qkv_split;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_backend_metal::components::gqa::split_kv::single_q as backend_single_q;
use inference_backend_metal::components::gqa::split_kv::tiled_q as backend_tiled_q;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::DSparkGQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratchBindings;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy)]
pub struct DSparkGQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub metadata: &'a DSparkGQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: UngatedGQAWeights<'a>,
    pub scratch: DSparkBlockScratchBindings<'a>,
}

pub struct DSparkGQA {
    device: Device,
    core: DSparkGQACore,
    metal: GQAMetalConfig,
    qkv: affine_quantized::Matmul,
    qkv_to_q_k_v: backend_qkv_split::Compute,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    sdpa_execution: ExecutionVariant,
    block_sdpa: backend_block_sdpa::Compute,
    output: affine_quantized::Matmul,
}

impl DSparkGQA {
    pub fn new(device: &Device, core: DSparkGQACore, metal: GQAMetalConfig, sdpa_execution: ExecutionVariant) -> Self {
        core.validate();
        metal.validate();
        let attention = &core.attention;
        assert!(metal.rope_dim as usize <= attention.head_dim);
        assert!(metal.num_ungated_tokens_per_page(attention) > 0);
        let sdpa_config = backend_sdpa::Config {
            io_dtype: metal.io_dtype,
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("DSpark GQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("DSpark GQA KV-head count must fit u32"),
            head_dim: attention.head_dim.try_into().expect("DSpark GQA head_dim must fit u32"),
            tokens_per_page: metal.num_ungated_tokens_per_page(attention),
        };
        assert!(
            sdpa_execution.supports(sdpa_config),
            "DSpark history attention execution must support the layer geometry"
        );
        let qkv = attention.qkv_shape();
        let output = attention.output_shape();
        Self {
            device: device.clone(),
            qkv: affine_quantized::Matmul::new(device, affine_config(qkv.out_dim, qkv.in_dim, metal)),
            qkv_to_q_k_v: backend_qkv_split::Compute::new(device, qkv_to_q_k_v_config(attention, metal)),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(attention, metal, attention.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(
                device,
                norm_rope_config(attention, metal, attention.num_kv_heads),
            ),
            sdpa_execution,
            block_sdpa: backend_block_sdpa::Compute::new(
                device,
                backend_block_sdpa::Config {
                    block_size: core.block_size.try_into().expect("DSpark GQA block size must fit u32"),
                    max_q_tokens: sdpa_execution.map.thread_block.max_q_tokens,
                    num_q_heads: attention
                        .num_q_heads
                        .try_into()
                        .expect("DSpark GQA Q-head count must fit u32"),
                    num_kv_heads: attention
                        .num_kv_heads
                        .try_into()
                        .expect("DSpark GQA KV-head count must fit u32"),
                    head_dim: attention.head_dim.try_into().expect("DSpark GQA head_dim must fit u32"),
                    scale: attention.scale,
                    dtype: metal.io_dtype,
                },
            ),
            output: affine_quantized::Matmul::new(device, affine_config(output.out_dim, output.in_dim, metal)),
            core,
            metal,
        }
    }

    fn validate_input(&self, input: &DSparkGQAInput<'_>) -> GQAReplayShape {
        input.page_table_layout.validate();
        assert!(
            input.gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "DSpark GQA layer index exceeds the page table"
        );
        let shape = input.metadata.replay_shape();
        let sdpa_execution = input.metadata.sdpa_execution();
        shape.validate();
        assert!(shape.reduce_sdpa_partial_outputs);
        assert!(
            shape.num_tokens as usize <= input.scratch.capacity.block.max_tokens,
            "DSpark GQA replay token count exceeds scratch"
        );
        assert!(
            shape.num_total_sdpa_map_task_templates as usize <= input.scratch.capacity.max_sdpa_map_task_templates,
            "DSpark GQA replay partial count exceeds scratch"
        );
        assert_eq!(
            input.scratch.capacity.block.block_size, self.core.block_size,
            "DSpark GQA scratch block size must match the backend"
        );
        assert_eq!(
            sdpa_execution, self.sdpa_execution,
            "DSpark history attention metadata must match the frozen backend execution"
        );
        assert_eq!(
            self.sdpa_execution.map.thread_block.max_q_tokens as usize, input.scratch.capacity.max_q_tokens,
            "DSpark history attention execution must match scratch capacity"
        );
        shape
    }

    fn split_kv_single_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_single_q::Config {
        let attention = &self.core.attention;
        backend_single_q::Config {
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("DSpark GQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("DSpark GQA KV-head count must fit u32"),
            head_dim: attention.head_dim.try_into().expect("DSpark GQA head_dim must fit u32"),
            scale: attention.scale,
            page_bytes: self.metal.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            dtype: self.metal.io_dtype,
        }
    }

    fn split_kv_tiled_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_tiled_q::Config {
        let attention = &self.core.attention;
        backend_tiled_q::Config {
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("DSpark GQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("DSpark GQA KV-head count must fit u32"),
            head_dim: attention.head_dim.try_into().expect("DSpark GQA head_dim must fit u32"),
            scale: attention.scale,
            page_bytes: self.metal.page_bytes,
            dtype: self.metal.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn record_block_sdpa<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: GQAReplayShape,
        metadata: &'a DSparkGQAMetadataBuffers,
        scratch: DSparkBlockScratchBindings<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::opaque(self.block_sdpa.invoke(
            backend_block_sdpa::Shape {
                num_tokens: shape.num_tokens,
                num_q_token_ranges: shape.num_q_token_tiles,
                num_total_partial_output_slots: shape.num_total_sdpa_map_task_templates,
            },
            backend_block_sdpa::Buffers {
                q: scratch.q_norm_rope,
                local_k: scratch.k_norm_rope,
                local_v: scratch.v,
                q_token_ranges: metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: metadata.cu_sdpa_partial_outputs(),
                partial_exp_sums: scratch.partial_exp_sums,
                partial_max_logits: scratch.partial_max_logits,
                partial_output: scratch.partial_output,
            },
        )));
    }
}

impl ReplayLayer for DSparkGQA {
    type Input<'a> = DSparkGQAInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = self.validate_input(&input);
        let sdpa_execution = self.sdpa_execution;
        let attention = &self.core.attention;
        let scratch = input.scratch;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv.invoke(
            shape.num_tokens,
            ReplayU32::Fixed(shape.num_tokens),
            scratch.qkv,
            0,
            input.hidden_state,
            0,
            input.weights.qkv_weight,
            0,
            input.weights.qkv_scales,
            0,
            input.weights.qkv_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv_to_q_k_v.invoke(
            backend_qkv_split::Shape {
                num_tokens: shape.num_tokens,
            },
            backend_qkv_split::Buffers {
                qkv: scratch.qkv,
                q: scratch.q,
                k: scratch.k,
                v: scratch.v,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.q,
                norm_weight: input.weights.q_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
            ReplayU32::Fixed(shape.num_tokens),
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.k,
                norm_weight: input.weights.k_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
            ReplayU32::Fixed(shape.num_tokens),
        )));

        if sdpa_execution.map.thread_block.max_q_tokens == 1 {
            let sdpa_config = self.split_kv_single_q_config(input.page_table_layout);
            let sdpa_shape = backend_single_q::Shape {
                num_total_tokens: shape.num_tokens,
                num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
            };
            let sdpa = backend_single_q::Compute::new(&self.device, sdpa_config, sdpa_execution, sdpa_shape);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_single_q::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: input.kv_cache.kv_pages,
                    req_slots: input.metadata.req_slots(),
                    page_ids: input.kv_cache.page_ids,
                    sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    partial_output: scratch.partial_output,
                },
                ReplayU32::Fixed(input.gqa_layer_index),
                ReplayU32::Fixed(shape.num_tokens),
                ReplayU32::Fixed(shape.num_sdpa_map_task_templates),
            )));
            self.record_block_sdpa(recorder, shape, input.metadata, scratch);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_single_q::ReduceBuffers {
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    partial_output: scratch.partial_output,
                    cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                ReplayU32::Fixed(shape.num_tokens),
            )));
        } else {
            let sdpa_config = self.split_kv_tiled_q_config(input.page_table_layout);
            let sdpa_shape = backend_tiled_q::Shape {
                num_total_tokens: shape.num_tokens,
                num_total_q_token_tiles: shape.num_total_q_token_tiles,
                num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
            };
            let sdpa = backend_tiled_q::Compute::new(&self.device, sdpa_config, sdpa_execution, sdpa_shape);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_tiled_q::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: input.kv_cache.kv_pages,
                    req_slots: input.metadata.req_slots(),
                    page_ids: input.kv_cache.page_ids,
                    visible_kv_token_ranges: input.metadata.visible_kv_token_ranges(),
                    q_token_ranges: input.metadata.q_token_ranges(),
                    sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
                    partial_output: scratch.partial_output,
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                },
                ReplayU32::Fixed(input.gqa_layer_index),
                ReplayU32::Fixed(shape.num_tokens),
                ReplayU32::Fixed(shape.num_q_token_tiles),
                ReplayU32::Fixed(shape.num_sdpa_map_task_templates),
            )));
            self.record_block_sdpa(recorder, shape, input.metadata, scratch);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_tiled_q::ReduceBuffers {
                    partial_output: scratch.partial_output,
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    q_token_ranges: input.metadata.q_token_ranges(),
                    cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                ReplayU32::Fixed(shape.num_q_token_tiles),
            )));
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output.invoke(
            shape.num_tokens,
            ReplayU32::Fixed(shape.num_tokens),
            input.next_hidden_state,
            0,
            scratch.attention_output,
            0,
            input.weights.output_weight,
            0,
            input.weights.output_scales,
            0,
            input.weights.output_biases,
            0,
        )));
        input.next_hidden_state
    }
}

fn backend_page_table_layout(layout: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: layout.num_req_slots,
        num_blocks: layout.num_blocks,
        num_gqa_layers: layout.num_gqa_layers,
        num_page_ids_per_block: layout.num_page_ids_per_block,
    }
}

fn qkv_to_q_k_v_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
) -> backend_qkv_split::Config {
    let num_q_heads = core
        .num_q_heads
        .try_into()
        .expect("DSpark GQA Q-head count must fit u32");
    let num_kv_heads = core
        .num_kv_heads
        .try_into()
        .expect("DSpark GQA KV-head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("DSpark GQA head_dim must fit u32");
    match metal.io_dtype {
        Dtype::Float32 => backend_qkv_split::Config::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => backend_qkv_split::Config::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported DSpark GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
    num_heads: usize,
) -> rms_norm_rope::Config {
    let num_heads = num_heads.try_into().expect("DSpark GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("DSpark GQA head_dim must fit u32");
    let norm_rope = match metal.io_dtype {
        Dtype::Float32 => {
            rms_norm_rope::Config::f32(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        Dtype::Bfloat16 => {
            rms_norm_rope::Config::bf16(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        dtype => panic!("unsupported DSpark GQA dtype {dtype:?}"),
    };
    norm_rope.with_rope_scaling(metal.rope_scaling)
}

fn affine_config(n: usize, k: usize, metal: GQAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("DSpark GQA affine n must fit i32"),
        k: k.try_into().expect("DSpark GQA affine k must fit i32"),
        group_size: metal.group_size.try_into().expect("DSpark GQA group_size must fit i32"),
        bits: metal.bits.try_into().expect("DSpark GQA bits must fit i32"),
        input_dtype: metal.io_dtype,
        output_dtype: metal.io_dtype,
        scale_bias_dtype: metal.io_dtype,
    }
}
