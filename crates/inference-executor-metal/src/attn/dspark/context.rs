use inference_backend_metal::components::GQAKVPageWrite;
use inference_backend_metal::components::GQAKVPageWriteBuffers;
use inference_backend_metal::components::GQAKVPageWriteConfig;
use inference_backend_metal::components::GQAKVPageWriteShape;
use inference_backend_metal::components::GQANormRopeBuffers;
use inference_backend_metal::components::GQANormRopeConfig;
use inference_backend_metal::components::GQANormRopeKernel;
use inference_backend_metal::components::GQANormRopeShape;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::UngatedDSparkGQACore;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::def::replay_op::ReplayOp;

pub struct DSparkGQAContextScratch {
    max_tokens: usize,
    k: Buffer,
    v: Buffer,
    k_norm_rope: Buffer,
}

#[derive(Clone, Copy)]
pub struct DSparkGQAContextScratchBindings<'a> {
    pub max_tokens: usize,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct UngatedDSparkGQAContextInput<'a> {
    pub num_tokens: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub main_feature: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: UngatedGQAWeights<'a>,
    pub scratch: DSparkGQAContextScratchBindings<'a>,
}

pub struct UngatedDSparkGQAContextAppender {
    core: UngatedDSparkGQACore,
    metal: GQAMetalConfig,
    k: AffineQuantizedMatmul,
    v: AffineQuantizedMatmul,
    k_norm_rope: GQANormRopeKernel,
    kv_page_write: GQAKVPageWrite,
}

impl DSparkGQAContextScratch {
    pub fn new(device: &Device, core: &UngatedDSparkGQACore, metal: GQAMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        metal.validate();
        assert!(max_tokens > 0, "DSpark context scratch requires tokens");
        let kv_elements = max_tokens
            .checked_mul(core.attention.k_dim())
            .expect("DSpark context scratch K/V element count must fit usize");
        Self {
            max_tokens,
            k: Buffer::new_zeroed_elements(device, kv_elements, metal.io_dtype),
            v: Buffer::new_zeroed_elements(device, kv_elements, metal.io_dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, kv_elements, metal.io_dtype),
        }
    }

    pub fn bindings(&self) -> DSparkGQAContextScratchBindings<'_> {
        DSparkGQAContextScratchBindings {
            max_tokens: self.max_tokens,
            k: &self.k,
            v: &self.v,
            k_norm_rope: &self.k_norm_rope,
        }
    }
}

impl UngatedDSparkGQAContextAppender {
    pub fn new(device: &Device, core: UngatedDSparkGQACore, metal: GQAMetalConfig) -> Self {
        core.validate();
        metal.validate();
        let attention = &core.attention;
        assert!(metal.rope_dim as usize <= attention.head_dim);
        let kv_config = attention_kv_config(attention, metal);
        Self {
            k: AffineQuantizedMatmul::new(device, kv_config),
            v: AffineQuantizedMatmul::new(device, kv_config),
            k_norm_rope: GQANormRopeKernel::new(device, k_norm_rope_config(attention, metal)),
            kv_page_write: GQAKVPageWrite::new(
                device,
                GQAKVPageWriteConfig {
                    num_kv_heads: attention
                        .num_kv_heads
                        .try_into()
                        .expect("DSpark context KV-head count must fit u32"),
                    head_dim: attention
                        .head_dim
                        .try_into()
                        .expect("DSpark context head_dim must fit u32"),
                    page_bytes: metal.page_bytes,
                    dtype: metal.io_dtype,
                },
            ),
            core,
            metal,
        }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: UngatedDSparkGQAContextInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(input.num_tokens > 0, "DSpark context append requires tokens");
        assert!(
            input.num_tokens as usize <= input.scratch.max_tokens,
            "DSpark context append exceeds scratch"
        );
        input.page_table_layout.validate();
        assert!(
            input.gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "DSpark context layer index exceeds the page table"
        );
        let attention = &self.core.attention;
        let offsets = QKVOffsets::new(attention, self.metal);
        let num_tokens = input
            .num_tokens
            .try_into()
            .expect("DSpark context token count must fit i32");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.k.invoke(
            num_tokens,
            input.scratch.k,
            0,
            input.main_feature,
            0,
            input.weights.qkv_weight,
            offsets.k_weight,
            input.weights.qkv_scales,
            offsets.k_affine,
            input.weights.qkv_biases,
            offsets.k_affine,
        )));
        recorder.record(ReplayOp::opaque(self.v.invoke(
            num_tokens,
            input.scratch.v,
            0,
            input.main_feature,
            0,
            input.weights.qkv_weight,
            offsets.v_weight,
            input.weights.qkv_scales,
            offsets.v_affine,
            input.weights.qkv_biases,
            offsets.v_affine,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.k_norm_rope.invoke(
            GQANormRopeShape {
                num_tokens: input.num_tokens,
            },
            GQANormRopeBuffers {
                input: input.scratch.k,
                norm_weight: input.weights.k_norm_weight,
                flat_token_indices: input.flat_token_indices,
                output: input.scratch.k_norm_rope,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_page_write.invoke(
            GQAKVPageWriteShape {
                num_token_writes: input.num_tokens,
                page_table_layout: MetalGQAPageTableLayout {
                    num_req_slots: input.page_table_layout.num_req_slots,
                    num_gqa_layers: input.page_table_layout.num_gqa_layers,
                    num_blocks: input.page_table_layout.num_blocks,
                    num_page_ids_per_block: input.page_table_layout.num_page_ids_per_block,
                },
                gqa_layer_index: input.gqa_layer_index,
            },
            GQAKVPageWriteBuffers {
                pages: input.kv_cache.kv_pages,
                flat_k: input.scratch.k_norm_rope,
                flat_v: input.scratch.v,
                req_slots: input.req_slots,
                flat_token_indices: input.flat_token_indices,
                page_ids: input.kv_cache.page_ids,
            },
        )));
    }
}

struct QKVOffsets {
    k_weight: usize,
    k_affine: usize,
    v_weight: usize,
    v_affine: usize,
}

impl QKVOffsets {
    fn new(core: &inference_executor_core::attn::UngatedGQACore, metal: GQAMetalConfig) -> Self {
        let q_config = AffineQuantizedMatmulConfig {
            n: core.q_dim().try_into().expect("DSpark Q dimension must fit i32"),
            k: core
                .hidden_dim
                .try_into()
                .expect("DSpark hidden dimension must fit i32"),
            group_size: metal
                .group_size
                .try_into()
                .expect("DSpark quantization group_size must fit i32"),
            bits: metal.bits.try_into().expect("DSpark quantization bits must fit i32"),
            input_dtype: metal.io_dtype,
            output_dtype: metal.io_dtype,
            scale_bias_dtype: metal.io_dtype,
        };
        let k_config = AffineQuantizedMatmulConfig {
            n: core.k_dim().try_into().expect("DSpark K dimension must fit i32"),
            ..q_config
        };
        Self {
            k_weight: q_config.weight_bytes(),
            k_affine: q_config.scale_or_bias_bytes(),
            v_weight: q_config
                .weight_bytes()
                .checked_add(k_config.weight_bytes())
                .expect("DSpark V weight offset must fit usize"),
            v_affine: q_config
                .scale_or_bias_bytes()
                .checked_add(k_config.scale_or_bias_bytes())
                .expect("DSpark V affine offset must fit usize"),
        }
    }
}

fn attention_kv_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: core
            .k_dim()
            .try_into()
            .expect("DSpark context K dimension must fit i32"),
        k: core
            .hidden_dim
            .try_into()
            .expect("DSpark context hidden dimension must fit i32"),
        group_size: metal
            .group_size
            .try_into()
            .expect("DSpark context group_size must fit i32"),
        bits: metal.bits.try_into().expect("DSpark context bits must fit i32"),
        input_dtype: metal.io_dtype,
        output_dtype: metal.io_dtype,
        scale_bias_dtype: metal.io_dtype,
    }
}

fn k_norm_rope_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: GQAMetalConfig,
) -> GQANormRopeConfig {
    let num_heads = core
        .num_kv_heads
        .try_into()
        .expect("DSpark context KV-head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("DSpark context head_dim must fit u32");
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
        dtype => panic!("unsupported DSpark context dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::UngatedGQACore;

    use super::*;

    #[test]
    fn test_qkv_offsets_follow_fused_q_k_v_row_order() {
        let core = UngatedGQACore::new(0, 5120, 128, 40, 8, 1.0);
        let metal = GQAMetalConfig {
            group_size: 64,
            bits: 4,
            page_bytes: 32 * 1024,
            rope_dim: 128,
            norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scale: 1.0,
            io_dtype: Dtype::Bfloat16,
        };

        let offsets = QKVOffsets::new(&core, metal);

        assert_eq!(offsets.k_weight, 13_107_200);
        assert_eq!(offsets.v_weight, 15_728_640);
        assert_eq!(offsets.k_affine, 819_200);
        assert_eq!(offsets.v_affine, 983_040);
    }
}
