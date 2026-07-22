use inference_backend_metal::components::GQALocalSDPABuffers;
use inference_backend_metal::components::GQALocalSDPAConfig;
use inference_backend_metal::components::GQALocalSDPAKernel;
use inference_backend_metal::components::GQALocalSDPAShape;
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
use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::components::ResidualBuffers;
use inference_backend_metal::components::ResidualKernel;
use inference_backend_metal::components::ResidualShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPReplayShape;
use inference_executor_core::model::qwen::v3_5::dspark_weight_layout::Qwen35DSparkLayerWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::backend::DenseMLP;
use crate::mlp::dense::backend::DenseMLPReplayInput;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_5::dspark::attention::Qwen35DSparkQKVLayout;
use crate::model::qwen::v3_5::dspark::block_request::Qwen35DSparkBlockRequest;
use crate::model::qwen::v3_5::dspark::context::Qwen35DSparkContextAppendInput;
use crate::model::qwen::v3_5::dspark::context::Qwen35DSparkContextAppender;
use crate::model::qwen::v3_5::dspark::context::Qwen35DSparkContextScratch;
use crate::model::qwen::v3_5::dspark::weights::Qwen35DSparkLayerWeights;
use crate::model::qwen::v3_5::plan::Qwen35DSparkLayerPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35DSparkLayerGeometry {
    dspark_layer_index: u32,
    hidden_dim: u32,
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_block_tokens: u32,
    max_sdpa_map_task_templates: u32,
}

impl Qwen35DSparkLayerGeometry {
    fn new(plan: &Qwen35DSparkLayerPlan, max_block_tokens: usize, max_sdpa_map_task_templates: usize) -> Self {
        plan.attention_core.validate();
        plan.attention_metal.validate();
        plan.mlp_core.validate();
        plan.mlp_metal.validate();
        assert_eq!(plan.attention_metal.dtype, Dtype::Bfloat16);
        assert_eq!(plan.mlp_metal.dtype, Dtype::Bfloat16);
        assert_eq!(plan.attention_core.hidden_dim, plan.mlp_core.hidden_dim);
        let qkv = Qwen35DSparkQKVLayout::from_plan(plan);
        let geometry = Self {
            dspark_layer_index: to_u32("DSpark layer index", plan.dspark_layer_index),
            hidden_dim: to_u32("DSpark hidden dimension", plan.attention_core.hidden_dim),
            q_dim: qkv.q_dim(),
            k_dim: qkv.k_dim(),
            v_dim: qkv.v_dim(),
            num_q_heads: to_u32("DSpark Q head count", plan.attention_core.num_q_heads),
            num_kv_heads: to_u32("DSpark KV head count", plan.attention_core.num_kv_heads),
            head_dim: to_u32("DSpark head dimension", plan.attention_core.head_dim),
            max_block_tokens: to_u32("DSpark maximum block tokens", max_block_tokens),
            max_sdpa_map_task_templates: to_u32("DSpark maximum SDPA map TaskTemplates", max_sdpa_map_task_templates),
        };
        assert!(geometry.max_block_tokens > 0);
        assert!(geometry.max_sdpa_map_task_templates >= geometry.max_block_tokens);
        assert_eq!(geometry.q_dim, geometry.num_q_heads * geometry.head_dim);
        assert_eq!(geometry.k_dim, geometry.num_kv_heads * geometry.head_dim);
        assert_eq!(geometry.v_dim, geometry.k_dim);
        geometry
    }

    fn tensor_elements(self, dimension: u32) -> usize {
        (self.max_block_tokens as usize)
            .checked_mul(dimension as usize)
            .expect("DSpark layer scratch tensor size must fit usize")
    }

    fn partials(self) -> usize {
        (self.max_sdpa_map_task_templates as usize)
            .checked_mul(self.num_q_heads as usize)
            .expect("DSpark attention partial count must fit usize")
    }

    fn compatible(self, other: Self) -> bool {
        self.hidden_dim == other.hidden_dim
            && self.q_dim == other.q_dim
            && self.k_dim == other.k_dim
            && self.v_dim == other.v_dim
            && self.num_q_heads == other.num_q_heads
            && self.num_kv_heads == other.num_kv_heads
            && self.head_dim == other.head_dim
            && self.max_block_tokens == other.max_block_tokens
            && self.max_sdpa_map_task_templates == other.max_sdpa_map_task_templates
    }
}

pub struct Qwen35DSparkLayerScratch {
    geometry: Qwen35DSparkLayerGeometry,
    normalized_hidden: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    sdpa_partial_exp_sums: Buffer,
    sdpa_partial_max_logits: Buffer,
    sdpa_partial_output: Buffer,
    attention_output: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
    residual_stream: [Buffer; 2],
    mlp: DenseMLPScratch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35DSparkLayerCapacities {
    pub max_target_tokens: usize,
    pub max_block_tokens: usize,
    pub max_sdpa_map_task_templates: usize,
    pub local_block_size: u32,
}

impl Qwen35DSparkLayerScratch {
    pub fn new(
        device: &Device,
        plan: &Qwen35DSparkLayerPlan,
        max_block_tokens: usize,
        max_sdpa_map_task_templates: usize,
    ) -> Self {
        let geometry = Qwen35DSparkLayerGeometry::new(plan, max_block_tokens, max_sdpa_map_task_templates);
        let hidden_elements = geometry.tensor_elements(geometry.hidden_dim);
        let partials = geometry.partials();
        let partial_output_elements = partials
            .checked_mul(geometry.head_dim as usize)
            .expect("DSpark attention partial output size must fit usize");
        Self {
            geometry,
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            q: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.q_dim), Dtype::Bfloat16),
            k: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.k_dim), Dtype::Bfloat16),
            v: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.v_dim), Dtype::Bfloat16),
            q_norm_rope: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.q_dim), Dtype::Bfloat16),
            k_norm_rope: Buffer::new_zeroed_elements(device, geometry.tensor_elements(geometry.k_dim), Dtype::Bfloat16),
            sdpa_partial_exp_sums: Buffer::new_zeroed_elements(device, partials, Dtype::Float32),
            sdpa_partial_max_logits: Buffer::new_zeroed_elements(device, partials, Dtype::Float32),
            sdpa_partial_output: Buffer::new_zeroed_elements(device, partial_output_elements, Dtype::Bfloat16),
            attention_output: Buffer::new_zeroed_elements(
                device,
                geometry.tensor_elements(geometry.q_dim),
                Dtype::Bfloat16,
            ),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            residual_stream: [
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            ],
            mlp: DenseMLPScratch::new(device, &plan.mlp_core, plan.mlp_metal, max_block_tokens),
        }
    }

    pub fn residual_stream(&self, layer_index: usize) -> &Buffer {
        &self.residual_stream[layer_index % self.residual_stream.len()]
    }
}

#[derive(Clone, Copy)]
pub struct Qwen35DSparkLayerInput<'a> {
    pub block_request: &'a Qwen35DSparkBlockRequest,
    pub local_block_size: u32,
    pub num_target_gqa_layers: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: &'a Qwen35DSparkLayerScratch,
}

#[derive(Clone, Copy)]
pub struct Qwen35DSparkLayerContextInput<'a> {
    pub num_tokens: u32,
    pub num_target_gqa_layers: u32,
    pub projected_target: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub page_table_layout: GQAPageTableLayout,
    pub scratch: &'a Qwen35DSparkContextScratch,
}

pub struct Qwen35DSparkLayer {
    geometry: Qwen35DSparkLayerGeometry,
    qkv_layout: Qwen35DSparkQKVLayout,
    input_layernorm_eps: f32,
    post_attention_layernorm_eps: f32,
    attention_scale: f32,
    page_bytes: u32,
    kv_token_tile_size: u32,
    sdpa_map_threads: u32,
    max_q_head_tile_size: u32,
    group_size: u32,
    bits: u32,
    context_appender: Qwen35DSparkContextAppender,
    weights: Qwen35DSparkLayerWeights,
    rms_norm: RMSNormKernel,
    q_projection_qmv: AffineQuantizedMatmulKernel,
    q_projection_qmm: AffineQuantizedMatmulKernel,
    kv_projection_qmv: AffineQuantizedMatmulKernel,
    kv_projection_qmm: AffineQuantizedMatmulKernel,
    q_norm_rope: GQANormRopeKernel,
    k_norm_rope: GQANormRopeKernel,
    paged_sdpa: GQAPagedSDPAKernels,
    local_sdpa: GQALocalSDPAKernel,
    output_projection_qmv: AffineQuantizedMatmulKernel,
    output_projection_qmm: AffineQuantizedMatmulKernel,
    residual: ResidualKernel,
    mlp: DenseMLP,
}

impl Qwen35DSparkLayer {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        plan: &Qwen35DSparkLayerPlan,
        weight_bindings: &Qwen35DSparkLayerWeightBindings,
        capacities: Qwen35DSparkLayerCapacities,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35DSparkLayerCapacities {
            max_target_tokens,
            max_block_tokens,
            max_sdpa_map_task_templates,
            local_block_size,
        } = capacities;
        let geometry = Qwen35DSparkLayerGeometry::new(plan, max_block_tokens, max_sdpa_map_task_templates);
        assert!(local_block_size > 0);
        assert_eq!(geometry.max_block_tokens % local_block_size, 0);
        let qkv_layout = Qwen35DSparkQKVLayout::from_plan(plan);
        let attention = plan.attention_metal;
        let q_qmm_rows = qmv_batch_limit(geometry.hidden_dim, geometry.q_dim);
        let kv_qmm_rows = qmv_batch_limit(geometry.hidden_dim, geometry.k_dim);
        let output_qmm_rows = qmv_batch_limit(geometry.q_dim, geometry.hidden_dim);
        Ok(Self {
            geometry,
            qkv_layout,
            input_layernorm_eps: plan.input_layernorm_eps,
            post_attention_layernorm_eps: plan.post_attention_layernorm_eps,
            attention_scale: plan.attention_core.scale,
            page_bytes: attention.page_bytes,
            kv_token_tile_size: attention.context_parallel_kv_token_tile_size,
            sdpa_map_threads: attention.context_parallel_num_threads_per_threadblock,
            max_q_head_tile_size: attention.context_parallel_max_q_head_tile_size,
            group_size: attention.group_size,
            bits: attention.bits,
            context_appender: Qwen35DSparkContextAppender::new(device, plan, max_target_tokens),
            weights: Qwen35DSparkLayerWeights::load(device, store, plan, weight_bindings)?,
            rms_norm: RMSNormKernel::new(device),
            q_projection_qmv: AffineQuantizedMatmulKernel::new(device, qkv_layout.q_shape(1)),
            q_projection_qmm: AffineQuantizedMatmulKernel::new(device, qkv_layout.q_shape(q_qmm_rows)),
            kv_projection_qmv: AffineQuantizedMatmulKernel::new(device, qkv_layout.k_shape(1)),
            kv_projection_qmm: AffineQuantizedMatmulKernel::new(device, qkv_layout.k_shape(kv_qmm_rows)),
            q_norm_rope: GQANormRopeKernel::new(
                device,
                GQANormRopeConfig::bf16(
                    geometry.num_q_heads,
                    geometry.head_dim,
                    attention.rope_dim,
                    attention.norm_eps,
                    attention.rope_theta,
                    attention.rope_scale,
                ),
            ),
            k_norm_rope: GQANormRopeKernel::new(
                device,
                GQANormRopeConfig::bf16(
                    geometry.num_kv_heads,
                    geometry.head_dim,
                    attention.rope_dim,
                    attention.norm_eps,
                    attention.rope_theta,
                    attention.rope_scale,
                ),
            ),
            paged_sdpa: GQAPagedSDPAKernels::new(device),
            local_sdpa: GQALocalSDPAKernel::new(
                device,
                GQALocalSDPAConfig {
                    local_block_size,
                    num_q_heads: geometry.num_q_heads,
                    num_kv_heads: geometry.num_kv_heads,
                    head_dim: geometry.head_dim,
                    scale: plan.attention_core.scale,
                    num_threads_per_threadblock: attention.context_parallel_num_threads_per_threadblock,
                    dtype: Dtype::Bfloat16,
                },
            ),
            output_projection_qmv: AffineQuantizedMatmulKernel::new(device, output_shape(plan, 1)),
            output_projection_qmm: AffineQuantizedMatmulKernel::new(device, output_shape(plan, output_qmm_rows)),
            residual: ResidualKernel::new(device),
            mlp: DenseMLP::new(device, plan.mlp_core.clone(), plan.mlp_metal),
        })
    }

    pub fn record_target_context<'a, R>(&'a self, recorder: &mut R, input: Qwen35DSparkLayerContextInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.context_appender.record(
            recorder,
            Qwen35DSparkContextAppendInput {
                num_tokens: input.num_tokens,
                num_target_gqa_layers: input.num_target_gqa_layers,
                projected_target: input.projected_target,
                req_slots: input.req_slots,
                flat_token_indices: input.flat_token_indices,
                pages: input.pages,
                page_ids: input.page_ids,
                page_table_layout: input.page_table_layout,
                weights: &self.weights.attention,
                scratch: input.scratch,
            },
        );
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, input: Qwen35DSparkLayerInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let replay_shape = input.block_request.replay_shape();
        assert!(replay_shape.reduce_sdpa_partial_outputs);
        assert_eq!(replay_shape.num_q_token_tiles, replay_shape.num_tokens);
        assert!(replay_shape.num_tokens <= self.geometry.max_block_tokens);
        assert!(replay_shape.total_sdpa_map_task_templates <= self.geometry.max_sdpa_map_task_templates);
        assert_eq!(input.local_block_size, input.block_request.local_block_size());
        assert_eq!(replay_shape.num_tokens % input.local_block_size, 0);
        assert!(input.scratch.geometry.compatible(self.geometry));
        input.page_table_layout.validate();
        let gqa_layer_index = input
            .num_target_gqa_layers
            .checked_add(self.geometry.dspark_layer_index)
            .expect("DSpark context layer index must fit u32");
        assert!(gqa_layer_index < input.page_table_layout.num_gqa_layers);

        let num_tokens = replay_shape.num_tokens;
        let hidden_shape = RMSNormShape::bf16(num_tokens, self.geometry.hidden_dim);
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.rms_norm.invoke(
            hidden_shape,
            RMSNormBuffers {
                input: input.hidden_state,
                weight: &self.weights.input_norm_weight,
                output: &input.scratch.normalized_hidden,
            },
            self.input_layernorm_eps,
        )));

        let q_projection = select_projection(
            num_tokens,
            self.geometry.hidden_dim,
            self.geometry.q_dim,
            &self.q_projection_qmv,
            &self.q_projection_qmm,
        );
        let kv_projection = select_projection(
            num_tokens,
            self.geometry.hidden_dim,
            self.geometry.k_dim,
            &self.kv_projection_qmv,
            &self.kv_projection_qmm,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(q_projection.invoke_with_shape(
            self.qkv_layout.q_shape(num_tokens),
            &input.scratch.q,
            0,
            &input.scratch.normalized_hidden,
            0,
            &self.weights.attention.qkv_weight,
            0,
            &self.weights.attention.qkv_scales,
            0,
            &self.weights.attention.qkv_biases,
            0,
        )));
        recorder.record(ReplayOp::opaque(kv_projection.invoke_with_shape(
            self.qkv_layout.k_shape(num_tokens),
            &input.scratch.k,
            0,
            &input.scratch.normalized_hidden,
            0,
            &self.weights.attention.qkv_weight,
            self.qkv_layout.k_weight_offset_bytes(),
            &self.weights.attention.qkv_scales,
            self.qkv_layout.k_affine_offset_bytes(),
            &self.weights.attention.qkv_biases,
            self.qkv_layout.k_affine_offset_bytes(),
        )));
        recorder.record(ReplayOp::opaque(kv_projection.invoke_with_shape(
            self.qkv_layout.v_shape(num_tokens),
            &input.scratch.v,
            0,
            &input.scratch.normalized_hidden,
            0,
            &self.weights.attention.qkv_weight,
            self.qkv_layout.v_weight_offset_bytes(),
            &self.weights.attention.qkv_scales,
            self.qkv_layout.v_affine_offset_bytes(),
            &self.weights.attention.qkv_biases,
            self.qkv_layout.v_affine_offset_bytes(),
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            GQANormRopeShape { num_tokens },
            GQANormRopeBuffers {
                input: &input.scratch.q,
                norm_weight: &self.weights.attention.q_norm_weight,
                flat_token_indices: input.block_request.flat_token_indices(),
                output: &input.scratch.q_norm_rope,
            },
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            GQANormRopeShape { num_tokens },
            GQANormRopeBuffers {
                input: &input.scratch.k,
                norm_weight: &self.weights.attention.k_norm_weight,
                flat_token_indices: input.block_request.flat_token_indices(),
                output: &input.scratch.k_norm_rope,
            },
        )));

        let sdpa_config = self.paged_sdpa_config(input.page_table_layout, gqa_layer_index);
        let sdpa_shape = self.paged_sdpa_shape(replay_shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.paged_sdpa.invoke_map(
            sdpa_config,
            sdpa_shape,
            GQAPagedSDPAMapBuffers {
                q: &input.scratch.q_norm_rope,
                kv_pages: input.pages,
                req_slots: input.block_request.req_slots(),
                page_ids: input.page_ids,
                sdpa_map_task_templates: input.block_request.sdpa_map_task_templates(),
                partial_exp_sums: &input.scratch.sdpa_partial_exp_sums,
                partial_max_logits: &input.scratch.sdpa_partial_max_logits,
                partial_output: &input.scratch.sdpa_partial_output,
            },
        )));
        recorder.record(ReplayOp::opaque(self.local_sdpa.invoke(
            GQALocalSDPAShape {
                num_tokens,
                total_sdpa_map_task_templates: replay_shape.total_sdpa_map_task_templates,
            },
            GQALocalSDPABuffers {
                q: &input.scratch.q_norm_rope,
                local_k: &input.scratch.k_norm_rope,
                local_v: &input.scratch.v,
                local_sdpa_map_task_template_indices: input.block_request.local_sdpa_map_task_template_indices(),
                partial_exp_sums: &input.scratch.sdpa_partial_exp_sums,
                partial_max_logits: &input.scratch.sdpa_partial_max_logits,
                partial_output: &input.scratch.sdpa_partial_output,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.paged_sdpa.invoke_reduce(
            sdpa_config,
            sdpa_shape,
            GQAPagedSDPAReduceBuffers {
                partial_exp_sums: &input.scratch.sdpa_partial_exp_sums,
                partial_max_logits: &input.scratch.sdpa_partial_max_logits,
                partial_output: &input.scratch.sdpa_partial_output,
                cu_sdpa_partial_outputs: input.block_request.cu_sdpa_partial_outputs(),
                output: &input.scratch.attention_output,
            },
        )));

        let output_projection = select_projection(
            num_tokens,
            self.geometry.q_dim,
            self.geometry.hidden_dim,
            &self.output_projection_qmv,
            &self.output_projection_qmm,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(output_projection.invoke_with_shape(
            self.output_shape(num_tokens),
            &input.scratch.branch_output,
            0,
            &input.scratch.attention_output,
            0,
            &self.weights.attention.output_weight,
            0,
            &self.weights.attention.output_scales,
            0,
            &self.weights.attention.output_biases,
            0,
        )));

        let residual_shape = ResidualShape::bf16(
            num_tokens
                .checked_mul(self.geometry.hidden_dim)
                .expect("DSpark residual value count must fit u32"),
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add(self.residual.invoke(
            residual_shape,
            ResidualBuffers {
                lhs: input.hidden_state,
                rhs: &input.scratch.branch_output,
                output: &input.scratch.post_attention_hidden,
            },
        )));
        recorder.record(ReplayOp::rms_norm(self.rms_norm.invoke(
            hidden_shape,
            RMSNormBuffers {
                input: &input.scratch.post_attention_hidden,
                weight: &self.weights.post_attention_norm_weight,
                output: &input.scratch.normalized_hidden,
            },
            self.post_attention_layernorm_eps,
        )));
        let _ = <DenseMLP as ReplayLayer>::record(
            &self.mlp,
            recorder,
            DenseMLPReplayInput {
                shape: DenseMLPReplayShape { num_tokens },
                hidden_state: &input.scratch.normalized_hidden,
                next_hidden_state: &input.scratch.branch_output,
                scratch: input.scratch.mlp.bindings(),
                weights: self.weights.mlp.as_borrowed(),
            },
        );
        recorder.record_with_barrier_before(ReplayOp::residual_add(self.residual.invoke(
            residual_shape,
            ResidualBuffers {
                lhs: &input.scratch.post_attention_hidden,
                rhs: &input.scratch.branch_output,
                output: input.next_hidden_state,
            },
        )));
        input.next_hidden_state
    }

    fn paged_sdpa_config(&self, page_table_layout: GQAPageTableLayout, gqa_layer_index: u32) -> GQAPagedSDPAConfig {
        let q_heads_per_kv_head = self.geometry.num_q_heads / self.geometry.num_kv_heads;
        GQAPagedSDPAConfig {
            num_q_heads: self.geometry.num_q_heads,
            head_dim: self.geometry.head_dim,
            scale: self.attention_scale,
            num_kv_heads: self.geometry.num_kv_heads,
            page_bytes: self.page_bytes,
            kv_token_tile_size: self.kv_token_tile_size,
            num_threads_per_threadblock: self.sdpa_map_threads,
            q_head_tile_size: q_heads_per_kv_head.min(self.max_q_head_tile_size),
            dtype: Dtype::Bfloat16,
            page_table_layout: MetalGQAPageTableLayout {
                num_req_slots: page_table_layout.num_req_slots,
                num_gqa_layers: page_table_layout.num_gqa_layers,
                num_blocks: page_table_layout.num_blocks,
                num_page_ids_per_block: page_table_layout.num_page_ids_per_block,
            },
            gqa_layer_index,
        }
    }

    fn paged_sdpa_shape(&self, replay_shape: inference_executor_core::attn::GQAReplayShape) -> GQAPagedSDPAShape {
        GQAPagedSDPAShape {
            num_tokens: replay_shape.num_tokens,
            total_sdpa_map_task_templates: replay_shape.total_sdpa_map_task_templates,
        }
    }

    fn output_shape(&self, num_tokens: u32) -> AffineQuantizedMatmulShape {
        AffineQuantizedMatmulShape::same_dtype(
            num_tokens.try_into().expect("DSpark output rows must fit i32"),
            self.geometry
                .hidden_dim
                .try_into()
                .expect("DSpark output dimension must fit i32"),
            self.geometry.q_dim.try_into().expect("DSpark Q dimension must fit i32"),
            self.group_size.try_into().expect("DSpark group size must fit i32"),
            self.bits.try_into().expect("DSpark bits must fit i32"),
            Dtype::Bfloat16,
        )
    }
}

fn output_shape(plan: &Qwen35DSparkLayerPlan, num_tokens: u32) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape::same_dtype(
        num_tokens.try_into().expect("DSpark output rows must fit i32"),
        plan.attention_core
            .hidden_dim
            .try_into()
            .expect("DSpark output dimension must fit i32"),
        plan.attention_core
            .q_dim()
            .try_into()
            .expect("DSpark Q dimension must fit i32"),
        plan.attention_metal
            .group_size
            .try_into()
            .expect("DSpark group size must fit i32"),
        plan.attention_metal.bits.try_into().expect("DSpark bits must fit i32"),
        Dtype::Bfloat16,
    )
}

fn select_projection<'a>(
    num_tokens: u32,
    input_dim: u32,
    output_dim: u32,
    qmv: &'a AffineQuantizedMatmulKernel,
    qmm: &'a AffineQuantizedMatmulKernel,
) -> &'a AffineQuantizedMatmulKernel {
    if num_tokens >= qmv_batch_limit(input_dim, output_dim) {
        qmm
    } else {
        qmv
    }
}

fn qmv_batch_limit(input_dim: u32, output_dim: u32) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

fn to_u32(name: &str, value: usize) -> u32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit u32"))
}
