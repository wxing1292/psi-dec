use inference_backend_metal::components::MoECombineConfig;
use inference_backend_metal::components::MoECombineKernels;
use inference_backend_metal::components::MoECombineShape;
use inference_backend_metal::components::MoECombineWithSharedExpertsBuffers;
use inference_backend_metal::components::MoECombineWithoutSharedExpertsBuffers;
use inference_backend_metal::components::MoEExpertMajorConfig;
use inference_backend_metal::components::MoEExpertMajorKernels;
use inference_backend_metal::components::MoEExpertMajorLayoutBuffers;
use inference_backend_metal::components::MoEExpertMajorPackInputBuffers;
use inference_backend_metal::components::MoEExpertMajorScatterWithSharedExpertsBuffers;
use inference_backend_metal::components::MoEExpertMajorScatterWithoutSharedExpertsBuffers;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingBuffers;
use inference_backend_metal::components::MoERoutingConfig;
use inference_backend_metal::components::MoERoutingKernel;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::QuantizedDenseMLP;
use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::components::QuantizedSparseMLP;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPScratch;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
use inference_backend_metal::operators::SoftmaxBuffers;
use inference_backend_metal::operators::SoftmaxConfig;
use inference_backend_metal::operators::SoftmaxKernel;
use inference_backend_metal::operators::SoftmaxShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::mlp::moe::GatedMoEReplayShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::moe::scratch::MoERoutingScratchBindings;
use crate::mlp::moe::scratch::MoEScratchBindings;
use crate::mlp::moe::scratch::SharedExpertsScratchBindings;

const TOKEN_MAJOR_MAX_TOKENS: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedMoEMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub router_bits: u32,
    pub shared_expert_gate_bits: u32,
    pub io_dtype: Dtype,
}

impl GatedMoEMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.router_bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.shared_expert_gate_bits, 2 | 3 | 4 | 6 | 8));
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 gated MoE model boundary is not supported"),
            dtype => panic!("unsupported gated MoE model boundary dtype {dtype:?}"),
        }
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
pub struct GatedMoERoutingWeights<'a> {
    pub router_weight: &'a Buffer,
    pub router_scales: &'a Buffer,
    pub router_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GatedMoERoutingBucketedReplayInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens_key: ReplayParameterKey,
    pub hidden_state: &'a Buffer,
    pub scratch: MoERoutingScratchBindings<'a>,
    pub weights: GatedMoERoutingWeights<'a>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GatedMoERoutingReplayTopology {
    pub router_affine: AffineQuantizedMatmulKernelKind,
}

impl<'a> GatedMoEWeights<'a> {
    fn routing(self) -> GatedMoERoutingWeights<'a> {
        GatedMoERoutingWeights {
            router_weight: self.router_weight,
            router_scales: self.router_scales,
            router_biases: self.router_biases,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GatedMoESharedExpertsWeights<'a> {
    pub shared_expert_gate_weight: &'a Buffer,
    pub shared_expert_gate_scales: &'a Buffer,
    pub shared_expert_gate_biases: &'a Buffer,
    pub shared_experts: QuantizedDenseMLPWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoESharedExpertsReplayInput<'a> {
    pub scratch: SharedExpertsScratchBindings<'a>,
    pub weights: GatedMoESharedExpertsWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoEReplayInput<'a> {
    pub shape: GatedMoEReplayShape,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: MoEScratchBindings<'a>,
    pub weights: GatedMoEWeights<'a>,
    pub shared_experts: Option<GatedMoESharedExpertsReplayInput<'a>>,
}

/// Records one gated MoE layer:
///
/// ```text
/// hidden_state
///      |
///      v
/// router -> router_logits -> router_softmax -> router_probs
///                                               |
///                                               v
///                                            routing
///                                               |
///                                expert_indices + expert_probs
///
/// TokenMajor
///
/// hidden_state + routing
///      -> topk_experts.invoke_token_major
///      -> routed_hidden
///      -> combine
///      -> next_hidden_state
///
/// ExpertMajor
///
/// routing
///      -> expert_major.layout
/// hidden_state + layout
///      -> expert_major.pack_input
///      -> topk_experts.invoke_expert_major
///      -> packed_output
///      -> expert_major.scatter
///      -> next_hidden_state
///
/// Optional shared branch
///
/// hidden_state -> shared_experts.mlp --------------> shared_hidden
/// hidden_state -> shared_experts.shared_expert_gate -> gate_logits
///
/// combine and expert_major.scatter consume both shared outputs.
/// ```
pub struct GatedMoE {
    core: GatedMoECore,
    router: AffineQuantizedMatmul,
    router_softmax: SoftmaxKernel,
    routing: MoERoutingKernel,
    expert_major: MoEExpertMajorKernels,
    topk_experts: QuantizedSparseMLP,
    shared_experts: Option<GatedMoESharedExperts>,
    combine: MoECombineKernels,
}

struct GatedMoESharedExperts {
    shared_expert_gate: AffineQuantizedMatmul,
    mlp: QuantizedDenseMLP,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GatedMoEComputePath {
    TokenMajor,
    ExpertMajor,
}

impl GatedMoEComputePath {
    fn select(shape: GatedMoEReplayShape) -> Self {
        shape.validate();
        if shape.num_tokens <= TOKEN_MAJOR_MAX_TOKENS {
            Self::TokenMajor
        } else {
            Self::ExpertMajor
        }
    }
}

impl GatedMoE {
    fn validate_input(&self, input: &GatedMoEReplayInput<'_>) {
        input.shape.validate();
        assert!(
            shared_experts_input_matches_core(&self.core, input.shared_experts.is_some()),
            "gated MoE replay shared expert must match core configuration"
        );
    }

    pub fn new(device: &Device, core: GatedMoECore, config: GatedMoEMetalConfig) -> Self {
        core.validate();
        config.validate();
        let router_shape = core.router_shape();
        Self {
            router: AffineQuantizedMatmul::new(
                device,
                affine_config_with_bits(router_shape.out_dim, router_shape.in_dim, config.router_bits, config),
            ),
            router_softmax: SoftmaxKernel::new(
                device,
                SoftmaxConfig {
                    num_values_per_row: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    dtype: config.io_dtype,
                },
            ),
            routing: MoERoutingKernel::new(
                device,
                MoERoutingConfig {
                    num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    num_experts_per_token: core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                    norm_topk_prob: core.norm_topk_prob,
                },
            ),
            expert_major: MoEExpertMajorKernels::new(
                device,
                MoEExpertMajorConfig::bf16(
                    core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                    core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
                ),
            ),
            topk_experts: QuantizedSparseMLP::new(device, topk_experts_config(&core, config)),
            shared_experts: core.shared_experts_core().map(|shared_core| {
                let gate_shape = core
                    .shared_expert_gate_shape()
                    .expect("shared-expert MLP requires a shared-expert gate shape");
                GatedMoESharedExperts {
                    shared_expert_gate: AffineQuantizedMatmul::new(
                        device,
                        affine_config_with_bits(
                            gate_shape.out_dim,
                            gate_shape.in_dim,
                            config.shared_expert_gate_bits,
                            config,
                        ),
                    ),
                    mlp: QuantizedDenseMLP::new(device, shared_experts_config(&shared_core, config)),
                }
            }),
            combine: MoECombineKernels::new(
                device,
                MoECombineConfig::bf16(
                    core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                    core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
                ),
            ),
            core,
        }
    }

    /// Returns the recorded topology of the routing-only replay chain.
    pub fn routing_replay_topology(&self, num_total_tokens: u32) -> GatedMoERoutingReplayTopology {
        GatedMoERoutingReplayTopology {
            router_affine: self.router.topology(num_total_tokens),
        }
    }

    /// Returns the first token count for each routing-only topology change.
    pub fn routing_replay_topology_boundaries(&self) -> Box<[u32]> {
        self.router.topology_boundaries()
    }

    /// Records only `router affine -> softmax -> top-k routing` for bucket-readiness tests.
    ///
    /// This method does not enable bucketed replay for the full gated MoE layer.
    pub fn record_routing_bucketed<'a, R>(&'a self, recorder: &mut R, input: GatedMoERoutingBucketedReplayInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens_key = input.num_active_tokens_key;
        let scratch = input.scratch;
        let weights = input.weights;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.router.invoke_bucketed(
            num_total_tokens,
            num_active_tokens_key,
            scratch.router_logits,
            0,
            input.hidden_state,
            0,
            weights.router_weight,
            0,
            weights.router_scales,
            0,
            weights.router_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.router_softmax.invoke_bucketed(
            self.router_softmax_shape(num_total_tokens),
            num_active_tokens_key,
            SoftmaxBuffers {
                input: scratch.router_logits,
                output: scratch.router_probs,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.routing.invoke_bucketed(
            self.routing_shape(num_total_tokens),
            num_active_tokens_key,
            MoERoutingBuffers {
                router_probs: scratch.router_probs,
                expert_indices: scratch.expert_indices,
                expert_probs: scratch.expert_probs,
            },
        )));
    }

    fn record_token_major_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        input: GatedMoEReplayInput<'a>,
    ) {
        let shape = input.shape;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let weights = input.weights;
        let shared_experts = input.shared_experts;
        self.record_router(builder, shape, hidden_state, scratch.routing, weights.routing());
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts.invoke_token_major(
            self.token_major_shape(shape),
            QuantizedSparseMLPTokenMajorBuffers {
                input: hidden_state,
                token_indices: scratch.topk_experts.token_indices,
                expert_indices: scratch.routing.expert_indices,
                route_indices: scratch.topk_experts.route_indices,
                routed_hidden: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPScratch {
                swiglu: scratch.topk_experts.sparse_swiglu,
            },
            weights.topk_experts,
        )));
        match shared_experts {
            None => {
                builder.record_with_barrier_before(ReplayOp::opaque(self.combine.invoke_without_shared_experts(
                    self.combine_shape(shape),
                    MoECombineWithoutSharedExpertsBuffers {
                        routed_hidden: scratch.topk_experts.routed_hidden,
                        routed_probs: scratch.routing.expert_probs,
                        output: next_hidden_state,
                    },
                )));
            },
            Some(shared_experts) => {
                self.record_shared_experts(
                    builder,
                    shape,
                    hidden_state,
                    shared_experts.scratch,
                    shared_experts.weights,
                );
                builder.record_with_barrier_before(ReplayOp::opaque(self.combine.invoke_with_shared_experts(
                    self.combine_shape(shape),
                    MoECombineWithSharedExpertsBuffers {
                        routed_hidden: scratch.topk_experts.routed_hidden,
                        routed_probs: scratch.routing.expert_probs,
                        shared_hidden: shared_experts.scratch.hidden,
                        shared_expert_gate_logits: shared_experts.scratch.gate_logits,
                        output: next_hidden_state,
                    },
                )));
            },
        }
    }

    fn record_expert_major_replay<'a>(
        &'a self,
        builder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        input: GatedMoEReplayInput<'a>,
    ) {
        let shape = input.shape;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let weights = input.weights;
        let shared_experts = input.shared_experts;
        let expert_major_shape = self.expert_major_shape(shape);
        self.record_router(builder, shape, hidden_state, scratch.routing, weights.routing());
        if let Some(shared_experts) = shared_experts {
            self.record_shared_experts(
                builder,
                shape,
                hidden_state,
                shared_experts.scratch,
                shared_experts.weights,
            );
        }
        let layout = ReplayOp::opaque(self.expert_major.invoke_layout(
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
        ));
        if shared_experts.is_some() {
            builder.record(layout);
        } else {
            builder.record_with_barrier_before(layout);
        }
        builder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_pack_input(
            expert_major_shape,
            MoEExpertMajorPackInputBuffers {
                input: hidden_state,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                packed_input: scratch.topk_experts.packed_input,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts.invoke_expert_major(
            QuantizedSparseMLPExpertMajorShape {
                num_routes: self.num_routes(shape),
            },
            QuantizedSparseMLPExpertMajorBuffers {
                packed_input: scratch.topk_experts.packed_input,
                experts_by_route: scratch.topk_experts.experts_by_route,
                packed_output: scratch.topk_experts.routed_hidden,
            },
            QuantizedSparseMLPScratch {
                swiglu: scratch.topk_experts.sparse_swiglu,
            },
            weights.topk_experts,
        )));
        match shared_experts {
            None => {
                builder.record_with_barrier_before(ReplayOp::opaque(
                    self.expert_major.invoke_scatter_without_shared_experts(
                        expert_major_shape,
                        MoEExpertMajorScatterWithoutSharedExpertsBuffers {
                            packed_output: scratch.topk_experts.routed_hidden,
                            routes_by_token: scratch.topk_experts.routes_by_token,
                            routed_probs: scratch.routing.expert_probs,
                            output: next_hidden_state,
                        },
                    ),
                ));
            },
            Some(shared_experts) => {
                builder.record_with_barrier_before(ReplayOp::opaque(
                    self.expert_major.invoke_scatter_with_shared_experts(
                        expert_major_shape,
                        MoEExpertMajorScatterWithSharedExpertsBuffers {
                            packed_output: scratch.topk_experts.routed_hidden,
                            routes_by_token: scratch.topk_experts.routes_by_token,
                            routed_probs: scratch.routing.expert_probs,
                            shared_hidden: shared_experts.scratch.hidden,
                            shared_expert_gate_logits: shared_experts.scratch.gate_logits,
                            output: next_hidden_state,
                        },
                    ),
                ));
            },
        }
    }

    fn record_router<'a, I>(
        &'a self,
        builder: &mut I,
        shape: GatedMoEReplayShape,
        input: &'a Buffer,
        scratch: MoERoutingScratchBindings<'a>,
        weights: GatedMoERoutingWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        builder.record_with_barrier_before(ReplayOp::opaque(self.router.invoke(
            shape.num_tokens.try_into().expect("MoE token count must fit i32"),
            scratch.router_logits,
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
        builder.record_with_barrier_before(ReplayOp::opaque(self.router_softmax.invoke(
            self.router_softmax_shape(shape.num_tokens),
            SoftmaxBuffers {
                input: scratch.router_logits,
                output: scratch.router_probs,
            },
        )));
        builder.record_with_barrier_before(ReplayOp::opaque(self.routing.invoke(
            self.routing_shape(shape.num_tokens),
            MoERoutingBuffers {
                router_probs: scratch.router_probs,
                expert_indices: scratch.expert_indices,
                expert_probs: scratch.expert_probs,
            },
        )));
    }

    fn record_shared_experts<'a, I>(
        &'a self,
        builder: &mut I,
        shape: GatedMoEReplayShape,
        input: &'a Buffer,
        scratch: SharedExpertsScratchBindings<'a>,
        weights: GatedMoESharedExpertsWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shared_experts = self
            .shared_experts
            .as_ref()
            .expect("gated MoE shared replay requires a configured shared expert");
        builder.record(ReplayOp::opaque(shared_experts.mlp.invoke(
            self.shared_experts_dense_shape(shape),
            QuantizedDenseMLPBuffers {
                hidden_state: input,
                next_hidden_state: scratch.hidden,
            },
            QuantizedDenseMLPScratch {
                gate_up: scratch.dense_mlp.gate_up,
                swiglu: scratch.dense_mlp.swiglu,
            },
            weights.shared_experts,
        )));
        builder.record(ReplayOp::opaque(shared_experts.shared_expert_gate.invoke(
            shape.num_tokens.try_into().expect("MoE token count must fit i32"),
            scratch.gate_logits,
            0,
            input,
            0,
            weights.shared_expert_gate_weight,
            0,
            weights.shared_expert_gate_scales,
            0,
            weights.shared_expert_gate_biases,
            0,
        )));
    }

    fn shared_experts_dense_shape(&self, shape: GatedMoEReplayShape) -> QuantizedDenseMLPShape {
        assert!(
            self.core.has_shared_experts(),
            "gated MoE shared shape requires a configured shared expert"
        );
        QuantizedDenseMLPShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn routing_shape(&self, num_tokens: u32) -> MoERoutingShape {
        MoERoutingShape { num_tokens }
    }

    fn router_softmax_shape(&self, num_tokens: u32) -> SoftmaxShape {
        SoftmaxShape { num_rows: num_tokens }
    }

    fn token_major_shape(&self, shape: GatedMoEReplayShape) -> QuantizedSparseMLPTokenMajorShape {
        QuantizedSparseMLPTokenMajorShape {
            num_routes: self.num_routes(shape),
            num_tokens: shape.num_tokens,
        }
    }

    fn expert_major_shape(&self, shape: GatedMoEReplayShape) -> MoEExpertMajorShape {
        MoEExpertMajorShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn combine_shape(&self, shape: GatedMoEReplayShape) -> MoECombineShape {
        MoECombineShape {
            num_tokens: shape.num_tokens,
        }
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

impl ReplayLayer for GatedMoE {
    type Input<'a> = GatedMoEReplayInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let shape = input.shape;
        let next_hidden_state = input.next_hidden_state;
        match GatedMoEComputePath::select(shape) {
            GatedMoEComputePath::TokenMajor => {
                self.record_token_major_replay(recorder, input);
            },
            GatedMoEComputePath::ExpertMajor => {
                self.record_expert_major_replay(recorder, input);
            },
        }
        next_hidden_state
    }
}

fn topk_experts_config(core: &GatedMoECore, config: GatedMoEMetalConfig) -> QuantizedSparseMLPConfig {
    QuantizedSparseMLPConfig {
        num_experts: core
            .num_experts
            .try_into()
            .expect("MoE sparse expert count must fit u32"),
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
        dtype: config.io_dtype,
    }
}

fn shared_experts_config(core: &DenseMLPCore, config: GatedMoEMetalConfig) -> QuantizedDenseMLPConfig {
    QuantizedDenseMLPConfig {
        hidden_dim: core
            .hidden_dim
            .try_into()
            .expect("shared expert hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("shared expert intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.io_dtype,
    }
}

fn affine_config_with_bits(n: usize, k: usize, bits: u32, config: GatedMoEMetalConfig) -> AffineQuantizedMatmulConfig {
    AffineQuantizedMatmulConfig {
        n: n.try_into().expect("MoE affine n must fit i32"),
        k: k.try_into().expect("MoE affine k must fit i32"),
        group_size: config.group_size.try_into().expect("MoE group size must fit i32"),
        bits: bits.try_into().expect("MoE bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

fn shared_experts_input_matches_core(core: &GatedMoECore, input_has_shared_experts: bool) -> bool {
    core.has_shared_experts() == input_has_shared_experts
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayParameterKey;
    use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;
    use inference_executor_core::replay::ReplayBucketPolicy;

    use super::*;
    use crate::def::replay_op::MetalReplayRuntime;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gated_moe.routing.num_active_tokens");

    #[test]
    fn test_shared_experts_dim() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            shared_experts_intermediate_dim: Some(1024),
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };
        let metal = GatedMoEMetalConfig {
            group_size: 64,
            bits: 4,
            router_bits: 8,
            shared_expert_gate_bits: 8,
            io_dtype: Dtype::Bfloat16,
        };

        let shared_core = core.shared_experts_core().unwrap();
        let shared = shared_experts_config(&shared_core, metal);

        assert_eq!(shared.hidden_dim, 2048);
        assert_eq!(shared.intermediate_dim, 1024);
    }

    #[test]
    fn test_compute_path_selection_boundary() {
        assert_eq!(
            GatedMoEComputePath::select(GatedMoEReplayShape { num_tokens: 1 }),
            GatedMoEComputePath::TokenMajor
        );
        assert_eq!(
            GatedMoEComputePath::select(GatedMoEReplayShape { num_tokens: 4 }),
            GatedMoEComputePath::TokenMajor
        );
        assert_eq!(
            GatedMoEComputePath::select(GatedMoEReplayShape { num_tokens: 5 }),
            GatedMoEComputePath::ExpertMajor
        );
    }

    #[test]
    fn test_routing_bucketed_chain_reuses_one_parameter_and_preserves_inactive_rows() {
        let device = Device::system_default();
        let stream = inference_backend_metal::metal::Stream::new(&device);
        let (core, metal) = routing_test_config(true);
        let moe = GatedMoE::new(&device, core.clone(), metal);
        let router_config = affine_config_with_bits(core.num_experts, core.hidden_dim, metal.router_bits, metal);
        let num_total_tokens = 4_u32;
        let num_active_tokens = 3_u32;
        let hidden_dim = core.hidden_dim;
        let num_experts = core.num_experts;
        let topk = core.num_experts_per_token;
        let all_hidden = bf16_values(
            &(0..num_total_tokens as usize * hidden_dim)
                .map(|index| ((index * 17 + 5) % 41) as f32 * 0.03125 - 0.625)
                .collect::<Vec<_>>(),
        );
        let mut active_hidden = all_hidden.clone();
        active_hidden[num_active_tokens as usize * hidden_dim..].fill(f32::NAN);
        let router_weight_values = (0..num_experts * hidden_dim)
            .map(|index| ((index * 29 + 11) % 251) as u8)
            .collect::<Vec<_>>();
        let router_scale_values = bf16_values(
            &(0..num_experts)
                .map(|expert| 0.000_976_562_5 + expert as f32 * 0.000_061_035_156)
                .collect::<Vec<_>>(),
        );
        let router_bias_values = bf16_values(
            &(0..num_experts)
                .map(|expert| -0.011_718_75 + expert as f32 * 0.003_906_25)
                .collect::<Vec<_>>(),
        );
        assert_eq!(router_weight_values.len(), router_config.weight_bytes());
        let hidden_state = bf16_buffer(&device, &active_hidden);
        let router_weight = Buffer::from_slice(&device, &router_weight_values);
        let router_scales = bf16_buffer(&device, &router_scale_values);
        let router_biases = bf16_buffer(&device, &router_bias_values);
        let bf16_sentinel = bf16::from_f32(-777.0).to_f32();
        let index_sentinel = 0xDEAD_BEEF_u32;
        let num_router_values = num_total_tokens as usize * num_experts;
        let num_routes = num_total_tokens as usize * topk;
        let router_logits = bf16_buffer(&device, &vec![bf16_sentinel; num_router_values]);
        let router_probs = bf16_buffer(&device, &vec![bf16_sentinel; num_router_values]);
        let expert_indices = Buffer::from_slice(&device, &vec![index_sentinel; num_routes]);
        let expert_probs = Buffer::from_slice(&device, &vec![-777.0_f32; num_routes]);
        let scratch = MoERoutingScratchBindings {
            router_logits: &router_logits,
            router_probs: &router_probs,
            expert_indices: &expert_indices,
            expert_probs: &expert_probs,
        };
        let weights = GatedMoERoutingWeights {
            router_weight: &router_weight,
            router_scales: &router_scales,
            router_biases: &router_biases,
        };

        let expected_logits = cpu_router_logits(
            router_config,
            num_total_tokens as usize,
            &all_hidden,
            &router_weight_values,
            &router_scale_values,
            &router_bias_values,
        );
        let expected_probs = cpu_softmax_bf16_rows(&expected_logits, num_total_tokens as usize, num_experts);
        let expected_routes = moe_routing_from_bf16_probs_reference(
            &expected_probs,
            num_total_tokens as usize,
            num_experts,
            topk,
            core.norm_topk_prob,
        );

        let runtime = MetalReplayRuntime::new(&stream);
        let mut exact_recorder = runtime.create_recorder();
        moe.record_router(
            &mut exact_recorder,
            GatedMoEReplayShape {
                num_tokens: num_active_tokens,
            },
            &hidden_state,
            scratch,
            weights,
        );
        let exact_replay = exact_recorder.build();
        assert_eq!(exact_replay.stats().parameter_count, 0);
        runtime.submit_replay(&exact_replay).wait();
        let active_router_values = num_active_tokens as usize * num_experts;
        let active_routes = num_active_tokens as usize * topk;
        let exact_logits = read_bf16_values(&router_logits, num_router_values);
        let exact_probs = read_bf16_values(&router_probs, num_router_values);
        let exact_indices = expert_indices.read_typed::<u32>(0, num_routes);
        let exact_expert_probs = expert_probs.read_typed::<f32>(0, num_routes);
        assert_close(
            &exact_logits[..active_router_values],
            &expected_logits[..active_router_values],
            0.25,
        );
        assert_close(
            &exact_probs[..active_router_values],
            &expected_probs[..active_router_values],
            0.02,
        );
        assert_eq!(
            &exact_indices[..active_routes],
            &expected_routes.expert_indices[..active_routes]
        );
        assert_close(
            &exact_expert_probs[..active_routes],
            &expected_routes.expert_probs[..active_routes],
            0.02,
        );
        assert_eq!(&exact_logits[active_router_values..], &vec![bf16_sentinel; num_experts]);
        assert_eq!(&exact_probs[active_router_values..], &vec![bf16_sentinel; num_experts]);
        assert_eq!(&exact_indices[active_routes..], &vec![index_sentinel; topk]);
        assert_eq!(&exact_expert_probs[active_routes..], &vec![-777.0_f32; topk]);

        reset_routing_outputs(
            &router_logits,
            &router_probs,
            &expert_indices,
            &expert_probs,
            num_router_values,
            num_routes,
            bf16_sentinel,
            index_sentinel,
        );
        let mut bucketed_recorder = runtime.create_recorder();
        moe.record_routing_bucketed(
            &mut bucketed_recorder,
            GatedMoERoutingBucketedReplayInput {
                num_total_tokens,
                num_active_tokens_key: NUM_ACTIVE_TOKENS,
                hidden_state: &hidden_state,
                scratch,
                weights,
            },
        );
        let bucketed_replay = bucketed_recorder.build();
        assert_eq!(bucketed_replay.stats().parameter_count, 1);

        assert_invalid_routing_arguments(&stream, &bucketed_replay, num_total_tokens);
        runtime
            .submit_replay_with_arguments(
                &bucketed_replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();
        let first_logits = read_bf16_values(&router_logits, num_router_values);
        let first_probs = read_bf16_values(&router_probs, num_router_values);
        let first_indices = expert_indices.read_typed::<u32>(0, num_routes);
        let first_expert_probs = expert_probs.read_typed::<f32>(0, num_routes);
        assert_close(
            &first_logits[..active_router_values],
            &exact_logits[..active_router_values],
            0.0,
        );
        assert_close(
            &first_probs[..active_router_values],
            &exact_probs[..active_router_values],
            0.0,
        );
        assert_eq!(&first_indices[..active_routes], &exact_indices[..active_routes]);
        assert_close(
            &first_expert_probs[..active_routes],
            &exact_expert_probs[..active_routes],
            0.0,
        );
        assert_eq!(&first_logits[active_router_values..], &vec![bf16_sentinel; num_experts]);
        assert_eq!(&first_probs[active_router_values..], &vec![bf16_sentinel; num_experts]);
        assert_eq!(&first_indices[active_routes..], &vec![index_sentinel; topk]);
        assert_eq!(&first_expert_probs[active_routes..], &vec![-777.0_f32; topk]);

        write_bf16_values(&hidden_state, &all_hidden);
        runtime
            .submit_replay_with_arguments(
                &bucketed_replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_total_tokens),
            )
            .wait();
        let full_logits = read_bf16_values(&router_logits, num_router_values);
        let full_probs = read_bf16_values(&router_probs, num_router_values);
        let full_indices = expert_indices.read_typed::<u32>(0, num_routes);
        let full_expert_probs = expert_probs.read_typed::<f32>(0, num_routes);
        assert_close(&full_logits, &expected_logits, 0.25);
        assert_close(&full_probs, &expected_probs, 0.02);
        assert_eq!(full_indices, expected_routes.expert_indices);
        assert_close(&full_expert_probs, &expected_routes.expert_probs, 0.02);

        write_bf16_values(&hidden_state, &active_hidden);
        runtime
            .submit_replay_with_arguments(
                &bucketed_replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();
        let shrunk_logits = read_bf16_values(&router_logits, num_router_values);
        let shrunk_probs = read_bf16_values(&router_probs, num_router_values);
        let shrunk_indices = expert_indices.read_typed::<u32>(0, num_routes);
        let shrunk_expert_probs = expert_probs.read_typed::<f32>(0, num_routes);
        assert_close(
            &shrunk_logits[..active_router_values],
            &exact_logits[..active_router_values],
            0.0,
        );
        assert_close(
            &shrunk_probs[..active_router_values],
            &exact_probs[..active_router_values],
            0.0,
        );
        assert_eq!(&shrunk_indices[..active_routes], &exact_indices[..active_routes]);
        assert_close(
            &shrunk_expert_probs[..active_routes],
            &exact_expert_probs[..active_routes],
            0.0,
        );
        assert_eq!(
            &shrunk_logits[active_router_values..],
            &full_logits[active_router_values..]
        );
        assert_eq!(
            &shrunk_probs[active_router_values..],
            &full_probs[active_router_values..]
        );
        assert_eq!(&shrunk_indices[active_routes..], &full_indices[active_routes..]);
        assert_eq!(
            &shrunk_expert_probs[active_routes..],
            &full_expert_probs[active_routes..]
        );
    }

    #[test]
    fn test_routing_bucket_policy_preserves_router_affine_topology() {
        let device = Device::system_default();
        let (core, metal) = routing_test_config(false);
        let moe = GatedMoE::new(&device, core, metal);
        let boundaries = moe.routing_replay_topology_boundaries();
        let policy = ReplayBucketPolicy::with_topology_boundaries(64, &boundaries);

        for num_active_tokens in 1..=64 {
            let num_total_tokens = policy.capacity(num_active_tokens);
            assert_eq!(
                moe.routing_replay_topology(num_active_tokens),
                moe.routing_replay_topology(num_total_tokens),
                "num_active_tokens={num_active_tokens} num_total_tokens={num_total_tokens}"
            );
        }
    }

    #[test]
    fn test_routing_bucketed_buffer_validation_uses_total_tokens() {
        let device = Device::system_default();
        let stream = inference_backend_metal::metal::Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let (core, metal) = routing_test_config(true);
        let moe = GatedMoE::new(&device, core.clone(), metal);
        let router_config = affine_config_with_bits(core.num_experts, core.hidden_dim, metal.router_bits, metal);
        let num_total_tokens = 4_u32;
        let num_active_tokens = 3_u32;
        let num_total_router_values = num_total_tokens as usize * core.num_experts;
        let num_active_router_values = num_active_tokens as usize * core.num_experts;
        let num_total_routes = num_total_tokens as usize * core.num_experts_per_token;
        let num_active_routes = num_active_tokens as usize * core.num_experts_per_token;
        let full_hidden = Buffer::new_zeroed(
            &device,
            num_total_tokens as usize * core.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let short_hidden = Buffer::new_zeroed(
            &device,
            num_active_tokens as usize * core.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let full_router_logits = Buffer::new_zeroed_elements(&device, num_total_router_values, Dtype::Bfloat16);
        let short_router_logits = Buffer::new_zeroed_elements(&device, num_active_router_values, Dtype::Bfloat16);
        let full_router_probs = Buffer::new_zeroed_elements(&device, num_total_router_values, Dtype::Bfloat16);
        let short_router_probs = Buffer::new_zeroed_elements(&device, num_active_router_values, Dtype::Bfloat16);
        let full_expert_indices = Buffer::new_zeroed_elements(&device, num_total_routes, Dtype::Uint32);
        let short_expert_indices = Buffer::new_zeroed_elements(&device, num_active_routes, Dtype::Uint32);
        let full_expert_probs = Buffer::new_zeroed_elements(&device, num_total_routes, Dtype::Float32);
        let short_expert_probs = Buffer::new_zeroed_elements(&device, num_active_routes, Dtype::Float32);
        let router_weight = Buffer::new_zeroed(&device, router_config.weight_bytes());
        let router_scales = Buffer::new_zeroed(&device, router_config.scale_or_bias_bytes());
        let router_biases = Buffer::new_zeroed(&device, router_config.scale_or_bias_bytes());
        let weights = GatedMoERoutingWeights {
            router_weight: &router_weight,
            router_scales: &router_scales,
            router_biases: &router_biases,
        };
        let valid_scratch = MoERoutingScratchBindings {
            router_logits: &full_router_logits,
            router_probs: &full_router_probs,
            expert_indices: &full_expert_indices,
            expert_probs: &full_expert_probs,
        };
        let cases = [
            (&short_hidden, valid_scratch),
            (
                &full_hidden,
                MoERoutingScratchBindings {
                    router_logits: &short_router_logits,
                    ..valid_scratch
                },
            ),
            (
                &full_hidden,
                MoERoutingScratchBindings {
                    router_probs: &short_router_probs,
                    ..valid_scratch
                },
            ),
            (
                &full_hidden,
                MoERoutingScratchBindings {
                    expert_indices: &short_expert_indices,
                    ..valid_scratch
                },
            ),
            (
                &full_hidden,
                MoERoutingScratchBindings {
                    expert_probs: &short_expert_probs,
                    ..valid_scratch
                },
            ),
        ];

        for (hidden_state, scratch) in cases {
            assert_panics(|| {
                let mut recorder = runtime.create_recorder();
                moe.record_routing_bucketed(
                    &mut recorder,
                    GatedMoERoutingBucketedReplayInput {
                        num_total_tokens,
                        num_active_tokens_key: NUM_ACTIVE_TOKENS,
                        hidden_state,
                        scratch,
                        weights,
                    },
                );
            });
        }
    }

    #[test]
    fn test_shared_experts_input_rejects_missing_resource() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            shared_experts_intermediate_dim: Some(1024),
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };

        assert!(!shared_experts_input_matches_core(&core, false));
    }

    #[test]
    fn test_shared_experts_input_rejects_unconfigured_resource() {
        let core = GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 2048,
            intermediate_dim: 512,
            shared_experts_intermediate_dim: None,
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
        };

        assert!(!shared_experts_input_matches_core(&core, true));
    }

    fn routing_test_config(norm_topk_prob: bool) -> (GatedMoECore, GatedMoEMetalConfig) {
        (
            GatedMoECore {
                model_layer_index: 0,
                hidden_dim: 32,
                intermediate_dim: 32,
                shared_experts_intermediate_dim: None,
                num_experts: 8,
                num_experts_per_token: 3,
                norm_topk_prob,
            },
            GatedMoEMetalConfig {
                group_size: 32,
                bits: 8,
                router_bits: 8,
                shared_expert_gate_bits: 8,
                io_dtype: Dtype::Bfloat16,
            },
        )
    }

    fn cpu_router_logits(
        config: AffineQuantizedMatmulConfig,
        num_tokens: usize,
        hidden: &[f32],
        weights: &[u8],
        scales: &[f32],
        biases: &[f32],
    ) -> Vec<f32> {
        assert_eq!(config.bits, 8);
        let num_experts = config.n as usize;
        let hidden_dim = config.k as usize;
        let group_size = config.group_size as usize;
        let num_groups = hidden_dim / group_size;
        assert_eq!(hidden.len(), num_tokens * hidden_dim);
        assert_eq!(weights.len(), num_experts * hidden_dim);
        assert_eq!(scales.len(), num_experts * num_groups);
        assert_eq!(biases.len(), num_experts * num_groups);
        let mut logits = Vec::with_capacity(num_tokens * num_experts);
        for token in 0..num_tokens {
            let hidden_row = &hidden[token * hidden_dim..(token + 1) * hidden_dim];
            for expert in 0..num_experts {
                let weight_row = &weights[expert * hidden_dim..(expert + 1) * hidden_dim];
                let mut value = 0.0_f32;
                for group in 0..num_groups {
                    let start = group * group_size;
                    let end = start + group_size;
                    let input_sum = hidden_row[start..end].iter().sum::<f32>();
                    let dot = hidden_row[start..end]
                        .iter()
                        .zip(&weight_row[start..end])
                        .map(|(input, weight)| *input * f32::from(*weight))
                        .sum::<f32>();
                    let affine_index = expert * num_groups + group;
                    value += scales[affine_index] * dot + input_sum * biases[affine_index];
                }
                logits.push(bf16::from_f32(value).to_f32());
            }
        }
        logits
    }

    fn cpu_softmax_bf16_rows(values: &[f32], num_rows: usize, num_values_per_row: usize) -> Vec<f32> {
        assert_eq!(values.len(), num_rows * num_values_per_row);
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks_exact(num_values_per_row) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps = row.iter().map(|value| (*value - max).exp()).collect::<Vec<_>>();
            let sum = exps.iter().sum::<f32>();
            output.extend(exps.into_iter().map(|value| bf16::from_f32(value / sum).to_f32()));
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn reset_routing_outputs(
        router_logits: &Buffer,
        router_probs: &Buffer,
        expert_indices: &Buffer,
        expert_probs: &Buffer,
        num_router_values: usize,
        num_routes: usize,
        bf16_sentinel: f32,
        index_sentinel: u32,
    ) {
        write_bf16_values(router_logits, &vec![bf16_sentinel; num_router_values]);
        write_bf16_values(router_probs, &vec![bf16_sentinel; num_router_values]);
        expert_indices.write_typed(0, &vec![index_sentinel; num_routes]);
        expert_probs.write_typed(0, &vec![-777.0_f32; num_routes]);
    }

    fn assert_invalid_routing_arguments(
        stream: &inference_backend_metal::metal::Stream,
        replay: &inference_backend_metal::metal::ReplayProgram,
        num_total_tokens: u32,
    ) {
        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(replay, &ReplayArguments::new());
        });
        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(replay, &ReplayArguments::new().with_i32(NUM_ACTIVE_TOKENS, 3));
        });
        for value in [0, num_total_tokens + 1] {
            assert_panics(|| {
                let _ = stream
                    .submit_replay_with_arguments(replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, value));
            });
        }
    }

    fn bf16_values(values: &[f32]) -> Vec<f32> {
        values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }

    fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        buffer.write_typed(0, &bits);
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
