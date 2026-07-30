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
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
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
        self.record_router(builder, shape, hidden_state, scratch, weights);
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
        self.record_router(builder, shape, hidden_state, scratch, weights);
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
        scratch: MoEScratchBindings<'a>,
        weights: GatedMoEWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        builder.record_with_barrier_before(ReplayOp::opaque(self.router.invoke(
            shape.num_tokens.try_into().expect("MoE token count must fit i32"),
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
        builder.record_with_barrier_before(ReplayOp::opaque(self.router_softmax.invoke(
            self.router_softmax_shape(shape),
            SoftmaxBuffers {
                input: scratch.routing.router_logits,
                output: scratch.routing.router_probs,
            },
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

    fn routing_shape(&self, shape: GatedMoEReplayShape) -> MoERoutingShape {
        MoERoutingShape {
            num_tokens: shape.num_tokens,
        }
    }

    fn router_softmax_shape(&self, shape: GatedMoEReplayShape) -> SoftmaxShape {
        SoftmaxShape {
            num_rows: shape.num_tokens,
        }
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
    use super::*;

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
}
