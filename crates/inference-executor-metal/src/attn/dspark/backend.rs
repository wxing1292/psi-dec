use inference_backend_metal::components::GQABlockSDPABuffers;
use inference_backend_metal::components::GQABlockSDPAConfig;
use inference_backend_metal::components::GQABlockSDPAKernel;
use inference_backend_metal::components::GQABlockSDPAShape;
use inference_backend_metal::components::GQACompute;
use inference_backend_metal::components::GQAComputePath;
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
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::attn::DSparkBlockMetadata;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::dspark::metadata::DSparkGQAMetadataBuffers;
use crate::attn::dspark::scratch::DSparkBlockScratchBindings;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::backend::gqa_compute_config;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy)]
pub struct UngatedDSparkGQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub metadata: &'a DSparkGQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: UngatedGQAWeights<'a>,
    pub scratch: DSparkBlockScratchBindings<'a>,
}

pub struct UngatedDSparkGQA {
    device: Device,
    core: UngatedDSparkGQACore,
    metal: GQAMetalConfig,
    compute: GQACompute,
    qkv: AffineQuantizedMatmul,
    qkv_to_q_k_v: GQAQKVSplitKernel,
    q_norm_rope: GQANormRopeKernel,
    k_norm_rope: GQANormRopeKernel,
    block_sdpa: GQABlockSDPAKernel,
    output: AffineQuantizedMatmul,
}

impl UngatedDSparkGQA {
    pub fn new(device: &Device, core: UngatedDSparkGQACore, metal: GQAMetalConfig) -> Self {
        core.validate();
        metal.validate();
        let attention = &core.attention;
        assert!(metal.rope_dim as usize <= attention.head_dim);
        assert!(metal.num_ungated_tokens_per_page(attention) > 0);
        let qkv = attention.qkv_shape();
        let output = attention.output_shape();
        Self {
            device: device.clone(),
            qkv: AffineQuantizedMatmul::new(device, affine_config(qkv.out_dim, qkv.in_dim, metal)),
            qkv_to_q_k_v: GQAQKVSplitKernel::new(device, qkv_to_q_k_v_config(attention, metal)),
            q_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(attention, metal, attention.num_q_heads)),
            k_norm_rope: GQANormRopeKernel::new(device, norm_rope_config(attention, metal, attention.num_kv_heads)),
            block_sdpa: GQABlockSDPAKernel::new(
                device,
                GQABlockSDPAConfig {
                    block_size: core.block_size.try_into().expect("DSpark GQA block size must fit u32"),
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
            output: AffineQuantizedMatmul::new(device, affine_config(output.out_dim, output.in_dim, metal)),
            compute: GQACompute::new_dspark_history(gqa_compute_config(
                metal,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
            )),
            core,
            metal,
        }
    }

    pub fn prepare(&self, metadata: &DSparkGQAMetadataBuffers, block: &DSparkBlockMetadata) -> GQAReplayShape {
        let num_tokens = block
            .num_tokens()
            .try_into()
            .expect("DSpark GQA token count must fit u32");
        let GQAComputePath::SingleQueryToken { kv_token_tile_size, .. } = self.compute.select(num_tokens, num_tokens)
        else {
            unreachable!("DSpark history attention requires the single-query-token compute path")
        };
        metadata.update(block, kv_token_tile_size)
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.metal.num_ungated_tokens_per_page(&self.core.attention)
    }

    fn validate_input(&self, input: &UngatedDSparkGQAInput<'_>) -> GQAReplayShape {
        input.page_table_layout.validate();
        assert!(
            input.gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "DSpark GQA layer index exceeds the page table"
        );
        let shape = input.metadata.replay_shape();
        shape.validate();
        assert_eq!(
            shape.num_q_token_tiles, shape.num_tokens,
            "DSpark first milestone requires single-Q history attention"
        );
        assert!(shape.reduce_sdpa_partial_outputs);
        assert!(
            shape.num_tokens as usize <= input.scratch.capacity.max_tokens,
            "DSpark GQA replay token count exceeds scratch"
        );
        assert!(
            shape.total_sdpa_map_task_templates as usize <= input.scratch.capacity.max_sdpa_map_task_templates,
            "DSpark GQA replay partial count exceeds scratch"
        );
        assert_eq!(
            input.scratch.capacity.block_size, self.core.block_size,
            "DSpark GQA scratch block size must match the backend"
        );
        shape
    }

    fn paged_sdpa_config(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
        gqa_layer_index: u32,
    ) -> GQAPagedSDPAConfig {
        let attention = &self.core.attention;
        let GQAComputePath::SingleQueryToken {
            kv_token_tile_size,
            num_threads_per_threadblock,
            q_head_tile_size,
        } = self.compute.select(shape.num_tokens, shape.num_q_token_tiles)
        else {
            unreachable!("DSpark history attention requires the single-query-token compute path")
        };
        GQAPagedSDPAConfig {
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
            gqa_layer_index,
            kv_token_tile_size,
            num_threads_per_threadblock,
            q_head_tile_size,
            dtype: self.metal.io_dtype,
        }
    }
}

impl ReplayLayer for UngatedDSparkGQA {
    type Input<'a> = UngatedDSparkGQAInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = self.validate_input(&input);
        let attention = &self.core.attention;
        let scratch = input.scratch;
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.qkv.invoke(
                shape
                    .num_tokens
                    .try_into()
                    .expect("DSpark GQA token count must fit i32"),
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
            ),
        ));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv_to_q_k_v.invoke(
            GQAQKVSplitShape {
                num_tokens: shape.num_tokens,
            },
            GQAQKVSplitBuffers {
                qkv: scratch.qkv,
                q: scratch.q,
                k: scratch.k,
                v: scratch.v,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            GQANormRopeShape {
                num_tokens: shape.num_tokens,
            },
            GQANormRopeBuffers {
                input: scratch.q,
                norm_weight: input.weights.q_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            GQANormRopeShape {
                num_tokens: shape.num_tokens,
            },
            GQANormRopeBuffers {
                input: scratch.k,
                norm_weight: input.weights.k_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
        )));

        let sdpa_config = self.paged_sdpa_config(shape, input.page_table_layout, input.gqa_layer_index);
        let sdpa_shape = GQAPagedSDPAShape {
            num_tokens: shape.num_tokens,
            total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
        };
        let sdpa = GQAPagedSDPAKernels::new(&self.device, sdpa_config, sdpa_shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(GQAPagedSDPAMapBuffers {
            q: scratch.q_norm_rope,
            kv_pages: input.kv_cache.kv_pages,
            req_slots: input.metadata.req_slots(),
            page_ids: input.kv_cache.page_ids,
            sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
            partial_exp_sums: scratch.partial_exp_sums,
            partial_max_logits: scratch.partial_max_logits,
            partial_output: scratch.partial_output,
        })));
        recorder.record(ReplayOp::opaque(self.block_sdpa.invoke(
            GQABlockSDPAShape {
                num_tokens: shape.num_tokens,
                total_sdpa_map_task_templates: shape.total_sdpa_map_task_templates,
            },
            GQABlockSDPABuffers {
                q: scratch.q_norm_rope,
                local_k: scratch.k_norm_rope,
                local_v: scratch.v,
                block_sdpa_map_task_template_indices: input.metadata.block_sdpa_map_task_template_indices(),
                partial_exp_sums: scratch.partial_exp_sums,
                partial_max_logits: scratch.partial_max_logits,
                partial_output: scratch.partial_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(GQAPagedSDPAReduceBuffers {
            partial_exp_sums: scratch.partial_exp_sums,
            partial_max_logits: scratch.partial_max_logits,
            partial_output: scratch.partial_output,
            cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
            output: scratch.attention_output,
        })));
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.output.invoke(
                shape
                    .num_tokens
                    .try_into()
                    .expect("DSpark GQA token count must fit i32"),
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
            ),
        ));
        input.next_hidden_state
    }
}

fn backend_page_table_layout(layout: GQAPageTableLayout) -> MetalGQAPageTableLayout {
    MetalGQAPageTableLayout {
        num_req_slots: layout.num_req_slots,
        num_blocks: layout.num_blocks,
        num_gqa_layers: layout.num_gqa_layers,
        num_page_ids_per_block: layout.num_page_ids_per_block,
    }
}

fn qkv_to_q_k_v_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
) -> GQAQKVSplitConfig {
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
        Dtype::Float32 => GQAQKVSplitConfig::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => GQAQKVSplitConfig::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported DSpark GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
    num_heads: usize,
) -> GQANormRopeConfig {
    let num_heads = num_heads.try_into().expect("DSpark GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("DSpark GQA head_dim must fit u32");
    match metal.io_dtype {
        Dtype::Float32 => {
            GQANormRopeConfig::f32(
                num_heads,
                head_dim,
                metal.rope_dim,
                metal.norm_eps,
                metal.rope_theta,
                metal.rope_scale,
            )
        },
        Dtype::Bfloat16 => {
            GQANormRopeConfig::bf16(
                num_heads,
                head_dim,
                metal.rope_dim,
                metal.norm_eps,
                metal.rope_theta,
                metal.rope_scale,
            )
        },
        dtype => panic!("unsupported DSpark GQA dtype {dtype:?}"),
    }
}

fn affine_config(n: usize, k: usize, metal: GQAMetalConfig) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("DSpark GQA affine n must fit i32"),
        k: k.try_into().expect("DSpark GQA affine k must fit i32"),
        group_size: metal.group_size.try_into().expect("DSpark GQA group_size must fit i32"),
        bits: metal.bits.try_into().expect("DSpark GQA bits must fit i32"),
        input_dtype: metal.io_dtype,
        output_dtype: metal.io_dtype,
        scale_bias_dtype: metal.io_dtype,
    }
}
