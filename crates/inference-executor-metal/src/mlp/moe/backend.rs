use inference_backend_metal::components::MoECombineKernel;
use inference_backend_metal::components::MoECombineWithCommonBuffers;
use inference_backend_metal::components::MoECombineWithCommonShape;
use inference_backend_metal::components::MoECombineWithoutCommonBuffers;
use inference_backend_metal::components::MoECombineWithoutCommonShape;
use inference_backend_metal::components::MoEExpertMajorKernels;
use inference_backend_metal::components::MoEExpertMajorLayoutBuffers;
use inference_backend_metal::components::MoEExpertMajorPackInputBuffers;
use inference_backend_metal::components::MoEExpertMajorScatterWithCommonBuffers;
use inference_backend_metal::components::MoEExpertMajorScatterWithoutCommonBuffers;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingBuffers;
use inference_backend_metal::components::MoERoutingKernel;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPKernels;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::components::QuantizedSparseMLP;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorScratch;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorScratch;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_backend_metal::operators::SoftmaxKernel;
use inference_backend_metal::operators::SoftmaxShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::Layer;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::mlp::moe::GatedMoEReplayShape;
use inference_executor_core::mlp::moe::MoEExecutionPolicy;
use inference_executor_core::mlp::moe::MoEExecutionPolicyConfig;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::moe::scratch::CommonExpertScratchBindings;
use crate::mlp::moe::scratch::MoEScratchBindings;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedMoEMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub router_bits: u32,
    pub common_gate_bits: u32,
    pub dtype: Dtype,
    pub execution_policy: MoEExecutionPolicyConfig,
}

impl GatedMoEMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.router_bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.common_gate_bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(
            self.dtype,
            Dtype::Bfloat16,
            "gated MoE Metal path currently supports bf16 only"
        );
    }
}

#[derive(Clone, Copy)]
pub struct GatedMoEWeights<'a> {
    pub router_weight: &'a Buffer,
    pub router_scales: &'a Buffer,
    pub router_biases: &'a Buffer,
    pub topk_experts: QuantizedSparseMLPWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoECommonExpertWeights<'a> {
    pub common_gate_weight: &'a Buffer,
    pub common_gate_scales: &'a Buffer,
    pub common_gate_biases: &'a Buffer,
    pub common_expert: QuantizedDenseMLPWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoECommonExpertReplayInput<'a> {
    pub scratch: CommonExpertScratchBindings<'a>,
    pub weights: GatedMoECommonExpertWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoEReplayInput<'a> {
    pub shape: GatedMoEReplayShape,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: MoEScratchBindings<'a>,
    pub weights: GatedMoEWeights<'a>,
    pub common_expert: Option<GatedMoECommonExpertReplayInput<'a>>,
}

pub struct GatedMoE {
    core: GatedMoECore,
    config: GatedMoEMetalConfig,
    router_projection_qmv: AffineQuantizedMatmulKernel,
    router_projection_qmm: AffineQuantizedMatmulKernel,
    router_softmax: SoftmaxKernel,
    common_gate_projection_qmv: AffineQuantizedMatmulKernel,
    common_gate_projection_qmm: AffineQuantizedMatmulKernel,
    routing: MoERoutingKernel,
    expert_major: MoEExpertMajorKernels,
    topk_experts_mlp: QuantizedSparseMLP,
    common_expert_mlp: Option<QuantizedDenseMLPKernels>,
    combine: MoECombineKernel,
}

impl GatedMoE {
    fn validate_input(&self, input: &GatedMoEReplayInput<'_>) {
        input.shape.validate();
        assert!(
            common_expert_input_matches_core(&self.core, input.common_expert.is_some()),
            "gated MoE replay common expert must match core configuration"
        );
    }

    pub fn new(device: &Device, core: GatedMoECore, config: GatedMoEMetalConfig) -> Self {
        core.validate();
        config.validate();
        let router_shape = core.router_shape();
        let router_qmm_m = qmv_batch_limit(router_shape.in_dim, router_shape.out_dim);
        let common_gate_qmm_m = qmv_batch_limit(core.hidden_dim, 1);
        Self {
            router_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape_with_bits(1, router_shape.out_dim, router_shape.in_dim, config.router_bits, config),
            ),
            router_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape_with_bits(
                    router_qmm_m,
                    router_shape.out_dim,
                    router_shape.in_dim,
                    config.router_bits,
                    config,
                ),
            ),
            router_softmax: SoftmaxKernel::new(
                device,
                SoftmaxShape {
                    num_rows: 1,
                    num_values_per_row: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    dtype: config.dtype,
                },
            ),
            common_gate_projection_qmv: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape_with_bits(1, 1, core.hidden_dim, config.common_gate_bits, config),
            ),
            common_gate_projection_qmm: AffineQuantizedMatmulKernel::new(
                device,
                affine_shape_with_bits(common_gate_qmm_m, 1, core.hidden_dim, config.common_gate_bits, config),
            ),
            routing: MoERoutingKernel::new(device),
            expert_major: MoEExpertMajorKernels::new(device),
            topk_experts_mlp: QuantizedSparseMLP::new(device, topk_experts_config(&core, config)),
            common_expert_mlp: core.common_expert_intermediate_dim.map(|intermediate_dim| {
                QuantizedDenseMLPKernels::new(device, common_expert_config(&core, intermediate_dim, config))
            }),
            combine: MoECombineKernel::new(device),
            core,
            config,
        }
    }

    fn record_token_major_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: GatedMoEReplayShape,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
    ) {
        self.record_router(builder, shape, hidden_state, scratch, weights);
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts_mlp.invoke_token_major(
            self.token_major_shape(shape),
            QuantizedSparseMLPTokenMajorBuffers {
                input: hidden_state,
                token_indices: scratch.topk_experts.token_indices,
                expert_indices: scratch.routing.expert_indices,
                route_indices: scratch.topk_experts.route_indices,
                output: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPTokenMajorScratch {
                activation: scratch.topk_experts.sparse_activation,
            },
            weights.topk_experts,
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.combine.invoke_without_common(
            self.combine_shape(shape),
            MoECombineWithoutCommonBuffers {
                routed_hidden: scratch.topk_experts.routed_hidden,
                routed_probs: scratch.routing.expert_probs,
                output: next_hidden_state,
            },
        )));
    }

    fn record_expert_major_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: GatedMoEReplayShape,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
    ) {
        let expert_major_shape = self.expert_major_shape(shape);
        self.record_router(builder, shape, hidden_state, scratch, weights);
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_layout(
            expert_major_shape,
            MoEExpertMajorLayoutBuffers {
                expert_indices: scratch.routing.expert_indices,
                expert_counts: scratch.topk_experts.expert_counts,
                expert_offsets: scratch.topk_experts.expert_offsets,
                expert_cursors: scratch.topk_experts.expert_cursors,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                routes_by_token: scratch.topk_experts.routes_by_token,
                experts_by_route: scratch.topk_experts.experts_by_route,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_pack_input(
            expert_major_shape,
            MoEExpertMajorPackInputBuffers {
                input: hidden_state,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                packed_input: scratch.topk_experts.packed_input,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts_mlp.invoke_expert_major(
            QuantizedSparseMLPExpertMajorShape {
                num_experts: self.core.num_experts.try_into().expect("MoE num_experts must fit u32"),
                num_routes: self.num_routes(shape),
            },
            QuantizedSparseMLPExpertMajorBuffers {
                packed_input: scratch.topk_experts.packed_input,
                experts_by_route: scratch.topk_experts.experts_by_route,
                route_output: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPExpertMajorScratch {
                activation: scratch.topk_experts.sparse_activation,
            },
            weights.topk_experts,
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_scatter_without_common(
            expert_major_shape,
            MoEExpertMajorScatterWithoutCommonBuffers {
                route_output: scratch.topk_experts.routed_hidden,
                routes_by_token: scratch.topk_experts.routes_by_token,
                routed_probs: scratch.routing.expert_probs,
                output: next_hidden_state,
            },
        )));
    }

    #[allow(clippy::too_many_arguments)]
    fn record_token_major_with_common_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: GatedMoEReplayShape,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        common_scratch: CommonExpertScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
        common_weights: GatedMoECommonExpertWeights<'a>,
    ) {
        self.record_router(builder, shape, hidden_state, scratch, weights);
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts_mlp.invoke_token_major(
            self.token_major_shape(shape),
            QuantizedSparseMLPTokenMajorBuffers {
                input: hidden_state,
                token_indices: scratch.topk_experts.token_indices,
                expert_indices: scratch.routing.expert_indices,
                route_indices: scratch.topk_experts.route_indices,
                output: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPTokenMajorScratch {
                activation: scratch.topk_experts.sparse_activation,
            },
            weights.topk_experts,
        )));
        self.record_common_expert(builder, shape, hidden_state, common_scratch, common_weights);
        builder.record_with_barrier_before(ReplayOp::opaque(self.combine.invoke_with_common(
            self.combine_with_common_shape(shape),
            MoECombineWithCommonBuffers {
                routed_hidden: scratch.topk_experts.routed_hidden,
                routed_probs: scratch.routing.expert_probs,
                common_hidden: common_scratch.hidden,
                common_gate_logits: common_scratch.gate_logits,
                output: next_hidden_state,
            },
        )));
    }

    #[allow(clippy::too_many_arguments)]
    fn record_expert_major_with_common_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: GatedMoEReplayShape,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        common_scratch: CommonExpertScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
        common_weights: GatedMoECommonExpertWeights<'a>,
    ) {
        let expert_major_shape = self.expert_major_shape(shape);
        self.record_router(builder, shape, hidden_state, scratch, weights);
        self.record_common_expert(builder, shape, hidden_state, common_scratch, common_weights);
        builder.record(ReplayOp::opaque(self.expert_major.invoke_layout(
            expert_major_shape,
            MoEExpertMajorLayoutBuffers {
                expert_indices: scratch.routing.expert_indices,
                expert_counts: scratch.topk_experts.expert_counts,
                expert_offsets: scratch.topk_experts.expert_offsets,
                expert_cursors: scratch.topk_experts.expert_cursors,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                routes_by_token: scratch.topk_experts.routes_by_token,
                experts_by_route: scratch.topk_experts.experts_by_route,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_pack_input(
            expert_major_shape,
            MoEExpertMajorPackInputBuffers {
                input: hidden_state,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                packed_input: scratch.topk_experts.packed_input,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts_mlp.invoke_expert_major(
            QuantizedSparseMLPExpertMajorShape {
                num_experts: self.core.num_experts.try_into().expect("MoE num_experts must fit u32"),
                num_routes: self.num_routes(shape),
            },
            QuantizedSparseMLPExpertMajorBuffers {
                packed_input: scratch.topk_experts.packed_input,
                experts_by_route: scratch.topk_experts.experts_by_route,
                route_output: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPExpertMajorScratch {
                activation: scratch.topk_experts.sparse_activation,
            },
            weights.topk_experts,
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_scatter_with_common(
            expert_major_shape,
            MoEExpertMajorScatterWithCommonBuffers {
                route_output: scratch.topk_experts.routed_hidden,
                routes_by_token: scratch.topk_experts.routes_by_token,
                routed_probs: scratch.routing.expert_probs,
                common_hidden: common_scratch.hidden,
                common_gate_logits: common_scratch.gate_logits,
                output: next_hidden_state,
            },
        )));
    }

    fn record_router<'a, I>(
        &'a self,
        builder: &mut I,
        shape: GatedMoEReplayShape,
        input: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        builder.record_with_barrier_before(ReplayOp::opaque(self.router_projection(shape).invoke_with_shape(
            self.router_projection_shape(shape),
            scratch.routing.router_logits,
            0,
            input,
            0,
            weights.router_weight,
            0,
            weights.router_scales,
            0,
            weights.router_biases,
            0,
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.router_softmax.invoke_with_shape(
            self.router_softmax_shape(shape),
            scratch.routing.router_probs,
            scratch.routing.router_logits,
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.routing.invoke(
            self.routing_shape(shape),
            MoERoutingBuffers {
                router_probs: scratch.routing.router_probs,
                expert_indices: scratch.routing.expert_indices,
                expert_probs: scratch.routing.expert_probs,
            },
        )));
    }

    fn record_common_expert<'a, I>(
        &'a self,
        builder: &mut I,
        shape: GatedMoEReplayShape,
        input: &'a Buffer,
        scratch: CommonExpertScratchBindings<'a>,
        weights: GatedMoECommonExpertWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let common_expert_mlp = self
            .common_expert_mlp
            .as_ref()
            .expect("gated MoE common replay requires a configured common expert");
        builder.record(ReplayOp::opaque(common_expert_mlp.invoke(
            self.common_expert_dense_shape(shape),
            QuantizedDenseMLPBuffers {
                hidden_state: input,
                next_hidden_state: scratch.hidden,
            },
            QuantizedDenseMLPScratch {
                gate_up_proj: scratch.dense_mlp.gate_up_proj,
                activation: scratch.dense_mlp.activation,
            },
            weights.common_expert,
        )));
        builder.record(ReplayOp::opaque(self.common_gate_projection(shape).invoke_with_shape(
            self.common_gate_shape(shape),
            scratch.gate_logits,
            0,
            input,
            0,
            weights.common_gate_weight,
            0,
            weights.common_gate_scales,
            0,
            weights.common_gate_biases,
            0,
        )));
    }

    fn router_projection(&self, shape: GatedMoEReplayShape) -> &AffineQuantizedMatmulKernel {
        let router = self.core.router_shape();
        if shape.num_tokens >= qmv_batch_limit(router.in_dim, router.out_dim) {
            &self.router_projection_qmm
        } else {
            &self.router_projection_qmv
        }
    }

    fn router_projection_shape(&self, shape: GatedMoEReplayShape) -> AffineQuantizedMatmulShape {
        let router = self.core.router_shape();
        affine_shape_with_bits(
            shape.num_tokens,
            router.out_dim,
            router.in_dim,
            self.config.router_bits,
            self.config,
        )
    }

    fn common_gate_projection(&self, shape: GatedMoEReplayShape) -> &AffineQuantizedMatmulKernel {
        if shape.num_tokens >= qmv_batch_limit(self.core.hidden_dim, 1) {
            &self.common_gate_projection_qmm
        } else {
            &self.common_gate_projection_qmv
        }
    }

    fn common_gate_shape(&self, shape: GatedMoEReplayShape) -> AffineQuantizedMatmulShape {
        affine_shape_with_bits(
            shape.num_tokens,
            1,
            self.core.hidden_dim,
            self.config.common_gate_bits,
            self.config,
        )
    }

    fn common_expert_dense_shape(&self, shape: GatedMoEReplayShape) -> QuantizedDenseMLPShape {
        assert!(
            self.core.has_common_expert(),
            "gated MoE common shape requires a configured common expert"
        );
        QuantizedDenseMLPShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn routing_shape(&self, shape: GatedMoEReplayShape) -> MoERoutingShape {
        MoERoutingShape {
            num_tokens: shape.num_tokens,
            num_experts: self.core.num_experts.try_into().expect("MoE expert count must fit u32"),
            num_experts_per_token: self
                .core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k must fit u32"),
            norm_topk_prob: self.core.norm_topk_prob,
        }
    }

    fn router_softmax_shape(&self, shape: GatedMoEReplayShape) -> SoftmaxShape {
        SoftmaxShape {
            num_rows: shape.num_tokens,
            num_values_per_row: self.core.num_experts.try_into().expect("MoE expert count must fit u32"),
            dtype: self.config.dtype,
        }
    }

    fn token_major_shape(&self, shape: GatedMoEReplayShape) -> QuantizedSparseMLPTokenMajorShape {
        QuantizedSparseMLPTokenMajorShape {
            num_routes: self.num_routes(shape),
            num_tokens: shape.num_tokens,
        }
    }

    fn expert_major_shape(&self, shape: GatedMoEReplayShape) -> MoEExpertMajorShape {
        MoEExpertMajorShape::bf16(
            shape.num_tokens,
            self.core.num_experts.try_into().expect("MoE expert count must fit u32"),
            self.core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k must fit u32"),
            self.core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
        )
    }

    fn combine_shape(&self, shape: GatedMoEReplayShape) -> MoECombineWithoutCommonShape {
        MoECombineWithoutCommonShape::bf16(
            shape.num_tokens,
            self.core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k must fit u32"),
            self.core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
        )
    }

    fn combine_with_common_shape(&self, shape: GatedMoEReplayShape) -> MoECombineWithCommonShape {
        MoECombineWithCommonShape::bf16(
            shape.num_tokens,
            self.core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k must fit u32"),
            self.core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
        )
    }

    fn num_routes(&self, shape: GatedMoEReplayShape) -> u32 {
        shape
            .num_tokens
            .checked_mul(
                self.core
                    .num_experts_per_token
                    .try_into()
                    .expect("MoE top-k must fit u32"),
            )
            .expect("MoE route count must fit u32")
    }
}

impl Layer for GatedMoE {
    type Input<'a> = GatedMoEReplayInput<'a>;
    type Output<'a> = &'a Buffer;

    type InputShape = GatedMoECore;
    type OutputShape = GatedMoECore;

    fn input_shape(&self) -> Self::InputShape {
        self.core.clone()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.core.clone()
    }
}

impl ReplayLayer for GatedMoE {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let shape = input.shape;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let weights = input.weights;
        match (
            self.config.execution_policy.resolve(shape.num_tokens),
            input.common_expert,
        ) {
            (MoEExecutionPolicy::Auto, _) => {
                unreachable!("auto policy must resolve to a concrete MoE policy")
            },
            (MoEExecutionPolicy::TokenMajor, None) => {
                self.record_token_major_replay(recorder, shape, hidden_state, next_hidden_state, scratch, weights);
            },
            (MoEExecutionPolicy::ExpertMajor, None) => {
                self.record_expert_major_replay(recorder, shape, hidden_state, next_hidden_state, scratch, weights);
            },
            (MoEExecutionPolicy::TokenMajor, Some(common_expert)) => {
                self.record_token_major_with_common_replay(
                    recorder,
                    shape,
                    hidden_state,
                    next_hidden_state,
                    scratch,
                    common_expert.scratch,
                    weights,
                    common_expert.weights,
                );
            },
            (MoEExecutionPolicy::ExpertMajor, Some(common_expert)) => {
                self.record_expert_major_with_common_replay(
                    recorder,
                    shape,
                    hidden_state,
                    next_hidden_state,
                    scratch,
                    common_expert.scratch,
                    weights,
                    common_expert.weights,
                );
            },
        }
        next_hidden_state
    }
}

fn topk_experts_config(core: &GatedMoECore, config: GatedMoEMetalConfig) -> QuantizedSparseMLPConfig {
    QuantizedSparseMLPConfig {
        hidden_dim: core
            .hidden_dim
            .try_into()
            .expect("MoE sparse expert hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("MoE sparse expert intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.dtype,
    }
}

fn common_expert_config(
    core: &GatedMoECore,
    intermediate_dim: usize,
    config: GatedMoEMetalConfig,
) -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: core
            .hidden_dim
            .try_into()
            .expect("common expert hidden_dim must fit u32"),
        intermediate_dim: intermediate_dim
            .try_into()
            .expect("common expert intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.dtype,
    }
}

fn affine_shape(m: u32, n: usize, k: usize, config: GatedMoEMetalConfig) -> AffineQuantizedMatmulShape {
    affine_shape_with_bits(m, n, k, config.bits, config)
}

fn affine_shape_with_bits(
    m: u32,
    n: usize,
    k: usize,
    bits: u32,
    config: GatedMoEMetalConfig,
) -> AffineQuantizedMatmulShape {
    AffineQuantizedMatmulShape {
        m: m.try_into().expect("MoE affine m must fit i32"),
        n: n.try_into().expect("MoE affine n must fit i32"),
        k: k.try_into().expect("MoE affine k must fit i32"),
        group_size: config.group_size.try_into().expect("MoE group size must fit i32"),
        bits: bits.try_into().expect("MoE bits must fit i32"),
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

fn common_expert_input_matches_core(core: &GatedMoECore, input_has_common_expert: bool) -> bool {
    core.has_common_expert() == input_has_common_expert
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_expert_dim() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            common_expert_intermediate_dim: Some(1024),
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };
        let metal = GatedMoEMetalConfig {
            group_size: 64,
            bits: 4,
            router_bits: 8,
            common_gate_bits: 8,
            dtype: Dtype::Bfloat16,
            execution_policy: MoEExecutionPolicyConfig::default(),
        };

        let common = common_expert_config(&core, core.common_expert_intermediate_dim.unwrap(), metal);

        assert_eq!(common.hidden_dim, 2048);
        assert_eq!(common.intermediate_dim, 1024);
    }

    #[test]
    fn test_common_expert_input_rejects_missing_resource() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            common_expert_intermediate_dim: Some(1024),
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };

        assert!(!common_expert_input_matches_core(&core, false));
    }

    #[test]
    fn test_common_expert_input_rejects_unconfigured_resource() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            common_expert_intermediate_dim: None,
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };

        assert!(!common_expert_input_matches_core(&core, true));
    }
}
