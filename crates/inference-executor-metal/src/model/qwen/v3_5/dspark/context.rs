use inference_backend_metal::components::GQAKVPageUpdate;
use inference_backend_metal::components::GQAKVPageUpdateBuffers;
use inference_backend_metal::components::GQAKVPageUpdateConfig;
use inference_backend_metal::components::GQAKVPageUpdateShape;
use inference_backend_metal::components::GQANormRopeBuffers;
use inference_backend_metal::components::GQANormRopeConfig;
use inference_backend_metal::components::GQANormRopeKernel;
use inference_backend_metal::components::GQANormRopeShape;
use inference_backend_metal::components::GQAPageTableLayout as MetalGQAPageTableLayout;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;

use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_5::dspark::attention::Qwen35DSparkQKVLayout;
use crate::model::qwen::v3_5::dspark::weights::Qwen35DSparkAttentionWeights;
use crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35DSparkContextGeometry {
    dspark_layer_index: u32,
    max_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    k_dim: u32,
    v_dim: u32,
    page_bytes: u32,
}

impl Qwen35DSparkContextGeometry {
    fn new(plan: &Qwen35DSparkLayerPlan, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "DSpark context append requires token capacity");
        plan.attention_core.validate();
        plan.attention_metal.validate();
        assert_eq!(
            plan.attention_metal.dtype,
            Dtype::Bfloat16,
            "DSpark context append requires BF16 storage"
        );
        let qkv = Qwen35DSparkQKVLayout::from_plan(plan);
        let geometry = Self {
            dspark_layer_index: plan
                .dspark_layer_index
                .try_into()
                .expect("DSpark layer index must fit u32"),
            max_tokens: max_tokens.try_into().expect("DSpark context max_tokens must fit u32"),
            num_kv_heads: plan
                .attention_core
                .num_kv_heads
                .try_into()
                .expect("DSpark KV head count must fit u32"),
            head_dim: plan
                .attention_core
                .head_dim
                .try_into()
                .expect("DSpark head dimension must fit u32"),
            k_dim: qkv.k_dim(),
            v_dim: qkv.v_dim(),
            page_bytes: plan.attention_metal.page_bytes,
        };
        assert_eq!(geometry.k_dim, geometry.v_dim, "DSpark K/V dimensions must match");
        geometry
    }

    fn context_gqa_layer_index(self, num_target_gqa_layers: u32) -> u32 {
        num_target_gqa_layers
            .checked_add(self.dspark_layer_index)
            .expect("DSpark context GQA layer index must fit u32")
    }

    fn scratch_compatible(self, other: Self) -> bool {
        self.max_tokens == other.max_tokens
            && self.num_kv_heads == other.num_kv_heads
            && self.head_dim == other.head_dim
            && self.k_dim == other.k_dim
            && self.v_dim == other.v_dim
    }

    fn tensor_elements(self, dimension: u32) -> usize {
        (self.max_tokens as usize)
            .checked_mul(dimension as usize)
            .expect("DSpark context scratch element count must fit usize")
    }
}

pub struct Qwen35DSparkContextScratch {
    geometry: Qwen35DSparkContextGeometry,
    k: Buffer,
    v: Buffer,
    k_norm_rope: Buffer,
}

impl Qwen35DSparkContextScratch {
    pub fn new(device: &Device, plan: &Qwen35DSparkLayerPlan, max_tokens: usize) -> Self {
        let geometry = Qwen35DSparkContextGeometry::new(plan, max_tokens);
        Self {
            geometry,
            k: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.k_dim), Dtype::Bfloat16),
            v: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.v_dim), Dtype::Bfloat16),
            k_norm_rope: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.k_dim), Dtype::Bfloat16),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Qwen35DSparkContextAppendInput<'a> {
    pub num_tokens: u32,
    pub num_target_gqa_layers: u32,
    pub projected_target: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub page_table_layout: GQAPageTableLayout,
    pub weights: &'a Qwen35DSparkAttentionWeights,
    pub scratch: &'a Qwen35DSparkContextScratch,
}

pub struct Qwen35DSparkContextAppender {
    geometry: Qwen35DSparkContextGeometry,
    qkv_layout: Qwen35DSparkQKVLayout,
    single_token_projection: AffineQuantizedMatmulKernel,
    multi_token_projection: AffineQuantizedMatmulKernel,
    k_norm_rope: GQANormRopeKernel,
    kv_update: GQAKVPageUpdate,
}

impl Qwen35DSparkContextAppender {
    pub fn new(device: &Device, plan: &Qwen35DSparkLayerPlan, max_tokens: usize) -> Self {
        let geometry = Qwen35DSparkContextGeometry::new(plan, max_tokens);
        let qkv_layout = Qwen35DSparkQKVLayout::from_plan(plan);
        let single_token_projection = AffineQuantizedMatmulKernel::new(device, qkv_layout.k_shape(1));
        let multi_token_projection = AffineQuantizedMatmulKernel::new(device, qkv_layout.k_shape(geometry.max_tokens));
        let metal = plan.attention_metal;
        let k_norm_rope = GQANormRopeKernel::new(
            device,
            GQANormRopeConfig::bf16(
                geometry.num_kv_heads,
                geometry.head_dim,
                metal.rope_dim,
                metal.norm_eps,
                metal.rope_theta,
                metal.rope_scale,
            ),
        );
        let kv_update = GQAKVPageUpdate::new(
            device,
            GQAKVPageUpdateConfig {
                num_kv_heads: geometry.num_kv_heads,
                page_bytes: geometry.page_bytes,
                head_dim: geometry.head_dim,
                dtype: Dtype::Bfloat16,
            },
        );
        Self {
            geometry,
            qkv_layout,
            single_token_projection,
            multi_token_projection,
            k_norm_rope,
            kv_update,
        }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: Qwen35DSparkContextAppendInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(input.num_tokens > 0, "DSpark context append requires tokens");
        assert!(
            input.num_tokens <= self.geometry.max_tokens,
            "DSpark context append num_tokens={} exceed max_tokens={}",
            input.num_tokens,
            self.geometry.max_tokens
        );
        assert!(
            input.scratch.geometry.scratch_compatible(self.geometry),
            "DSpark context scratch geometry must match the layer"
        );
        input.page_table_layout.validate();
        let context_gqa_layer_index = self.geometry.context_gqa_layer_index(input.num_target_gqa_layers);
        assert!(
            context_gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "DSpark context layer index must fit the lane-0 page table"
        );
        let projection = if input.num_tokens == 1 {
            &self.single_token_projection
        } else {
            &self.multi_token_projection
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(projection.invoke_with_shape(
            self.qkv_layout.k_shape(input.num_tokens),
            &input.scratch.k,
            0,
            input.projected_target,
            0,
            &input.weights.qkv_weight,
            self.qkv_layout.k_weight_offset_bytes(),
            &input.weights.qkv_scales,
            self.qkv_layout.k_affine_offset_bytes(),
            &input.weights.qkv_biases,
            self.qkv_layout.k_affine_offset_bytes(),
        )));
        recorder.record(ReplayOp::opaque(projection.invoke_with_shape(
            self.qkv_layout.v_shape(input.num_tokens),
            &input.scratch.v,
            0,
            input.projected_target,
            0,
            &input.weights.qkv_weight,
            self.qkv_layout.v_weight_offset_bytes(),
            &input.weights.qkv_scales,
            self.qkv_layout.v_affine_offset_bytes(),
            &input.weights.qkv_biases,
            self.qkv_layout.v_affine_offset_bytes(),
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.k_norm_rope.invoke(
            GQANormRopeShape {
                num_tokens: input.num_tokens,
            },
            GQANormRopeBuffers {
                input: &input.scratch.k,
                norm_weight: &input.weights.k_norm_weight,
                flat_token_indices: input.flat_token_indices,
                output: &input.scratch.k_norm_rope,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_update.invoke(
            GQAKVPageUpdateShape {
                num_token_writes: input.num_tokens,
                page_table_layout: MetalGQAPageTableLayout {
                    num_req_slots: input.page_table_layout.num_req_slots,
                    num_gqa_layers: input.page_table_layout.num_gqa_layers,
                    num_blocks: input.page_table_layout.num_blocks,
                    num_page_ids_per_block: input.page_table_layout.num_page_ids_per_block,
                },
                gqa_layer_index: context_gqa_layer_index,
            },
            GQAKVPageUpdateBuffers {
                pages: input.pages,
                flat_k: &input.scratch.k_norm_rope,
                flat_v: &input.scratch.v,
                req_slots: input.req_slots,
                flat_token_indices: input.flat_token_indices,
                page_ids: input.page_ids,
            },
        )));
    }
}

#[cfg(test)]
mod tests {
    use inference_executor_core::attn::GQACore;
    use inference_executor_core::mlp::dense::DenseMLPCore;

    use super::*;
    use crate::attn::gqa::backend::GQAMetalConfig;
    use crate::mlp::dense::backend::DenseMLPMetalConfig;

    #[test]
    fn context_geometry_appends_dspark_layers_after_target_gqa_layers() {
        let plan = test_plan(4);
        let geometry = Qwen35DSparkContextGeometry::new(&plan, 128);
        assert_eq!(geometry.max_tokens, 128);
        assert_eq!(geometry.num_kv_heads, 8);
        assert_eq!(geometry.head_dim, 128);
        assert_eq!(geometry.k_dim, 1024);
        assert_eq!(geometry.v_dim, 1024);
        assert_eq!(geometry.context_gqa_layer_index(16), 20);
        assert_eq!(geometry.tensor_elements(geometry.k_dim), 131_072);
        assert!(geometry.scratch_compatible(Qwen35DSparkContextGeometry::new(&test_plan(0), 128)));
        assert!(!geometry.scratch_compatible(Qwen35DSparkContextGeometry::new(&test_plan(0), 64)));
    }

    fn test_plan(dspark_layer_index: usize) -> Qwen35DSparkLayerPlan {
        Qwen35DSparkLayerPlan {
            dspark_layer_index,
            input_layernorm_eps: 1e-6,
            post_attention_layernorm_eps: 1e-6,
            attention_core: GQACore::new(dspark_layer_index, 5120, 128, 40, 8, 128.0_f32.sqrt().recip()),
            attention_metal: GQAMetalConfig {
                group_size: 64,
                bits: 4,
                page_bytes: 65_536,
                context_parallel_kv_token_tile_size: 256,
                context_parallel_num_threads_per_threadblock: 256,
                context_parallel_max_q_head_tile_size: 8,
                q_token_tile_size: 8,
                tiled_kv_token_tile_size: 16,
                rope_dim: 128,
                norm_eps: 1e-6,
                rope_theta: 10_000_000.0,
                rope_scale: 1.0,
                dtype: Dtype::Bfloat16,
            },
            mlp_core: DenseMLPCore {
                model_layer_index: dspark_layer_index,
                hidden_dim: 5120,
                intermediate_dim: 17_408,
            },
            mlp_metal: DenseMLPMetalConfig {
                group_size: 64,
                bits: 4,
                dtype: Dtype::Bfloat16,
            },
        }
    }
}
