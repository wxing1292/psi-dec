use inference_backend_metal::components::MoECombineWithCommonShape;
use inference_backend_metal::components::MoECombineWithoutCommonShape;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;

use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::dense::scratch::DenseMLPScratchBindings;
use crate::mlp::moe::backend::GatedMoEMetalConfig;

pub struct MoEScratch {
    routing: MoERoutingScratch,
    topk_experts: TopKExpertsScratch,
    common_expert: Option<CommonExpertScratch>,
}

struct MoERoutingScratch {
    router_logits: Buffer,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
}

struct TopKExpertsScratch {
    token_indices: Buffer,
    route_indices: Buffer,
    routed_hidden: Buffer,
    sparse_activation: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
}

struct CommonExpertScratch {
    hidden: Buffer,
    gate_logits: Buffer,
    dense_mlp: DenseMLPScratch,
}

#[derive(Clone, Copy)]
pub struct MoEScratchBindings<'a> {
    pub routing: MoERoutingScratchBindings<'a>,
    pub topk_experts: TopKExpertsScratchBindings<'a>,
}

#[derive(Clone, Copy)]
pub struct MoERoutingScratchBindings<'a> {
    pub router_logits: &'a Buffer,
    pub router_probs: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub expert_probs: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct TopKExpertsScratchBindings<'a> {
    pub token_indices: &'a Buffer,
    pub route_indices: &'a Buffer,
    pub routed_hidden: &'a Buffer,
    pub sparse_activation: &'a Buffer,
    pub expert_counts: &'a Buffer,
    pub expert_offsets: &'a Buffer,
    pub expert_cursors: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub experts_by_route: &'a Buffer,
    pub packed_input: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct CommonExpertScratchBindings<'a> {
    pub hidden: &'a Buffer,
    pub gate_logits: &'a Buffer,
    pub dense_mlp: DenseMLPScratchBindings<'a>,
}

impl MoEScratch {
    pub fn new(device: &Device, core: &GatedMoECore, config: GatedMoEMetalConfig, max_tokens: usize) -> Self {
        core.validate();
        config.validate();
        assert!(max_tokens > 0);

        let max_tokens_u32: u32 = max_tokens.try_into().expect("MoE scratch token capacity must fit u32");
        let num_routes = max_tokens_u32
            .checked_mul(
                core.num_experts_per_token
                    .try_into()
                    .expect("MoE top-k expert count must fit u32"),
            )
            .expect("MoE scratch route capacity must fit u32");
        let routing_shape = MoERoutingShape {
            num_tokens: max_tokens_u32,
            num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
            num_experts_per_token: core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k expert count must fit u32"),
            norm_topk_prob: core.norm_topk_prob,
        };
        routing_shape.validate();
        let router_shape = affine_shape(
            max_tokens_u32,
            core.num_experts,
            core.hidden_dim,
            config.router_bits,
            config,
        );
        let expert_major_shape = MoEExpertMajorShape::bf16(
            max_tokens_u32,
            core.num_experts.try_into().expect("MoE expert count must fit u32"),
            core.num_experts_per_token
                .try_into()
                .expect("MoE top-k expert count must fit u32"),
            core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
        );
        expert_major_shape.validate();
        if core.has_common_expert() {
            MoECombineWithCommonShape::bf16(
                expert_major_shape.num_tokens,
                expert_major_shape.num_experts_per_token,
                expert_major_shape.hidden_dim,
            )
            .validate();
        } else {
            MoECombineWithoutCommonShape::bf16(
                expert_major_shape.num_tokens,
                expert_major_shape.num_experts_per_token,
                expert_major_shape.hidden_dim,
            )
            .validate();
        }
        let sparse_config = QuantizedSparseMLPConfig {
            hidden_dim: core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
            intermediate_dim: core
                .intermediate_dim
                .try_into()
                .expect("MoE intermediate_dim must fit u32"),
            group_size: config.group_size,
            bits: config.bits,
            dtype: config.dtype,
        };
        let token_major_shape = QuantizedSparseMLPTokenMajorShape {
            num_routes,
            num_tokens: max_tokens_u32,
        };
        let routed_hidden_bytes =
            sparse_config
                .token_major_output_bytes(token_major_shape)
                .max(
                    sparse_config.expert_major_output_bytes(QuantizedSparseMLPExpertMajorShape {
                        num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                        num_routes: expert_major_shape.num_routes(),
                    }),
                );
        let topk: u32 = core
            .num_experts_per_token
            .try_into()
            .expect("MoE top-k expert count must fit u32");
        let token_route_indices = (0..num_routes).map(|route| route / topk).collect::<Vec<_>>();
        let identity_indices = (0..num_routes).collect::<Vec<_>>();

        let routing = MoERoutingScratch {
            router_logits: Buffer::new_zeroed(device, router_shape.output_bytes()),
            router_probs: Buffer::new_zeroed(device, router_shape.output_bytes()),
            expert_indices: Buffer::new_zeroed(device, routing_shape.expert_indices_bytes()),
            expert_probs: Buffer::new_zeroed(device, routing_shape.expert_probs_bytes()),
        };
        let topk_experts = TopKExpertsScratch {
            token_indices: Buffer::from_slice(device, &token_route_indices),
            route_indices: Buffer::from_slice(device, &identity_indices),
            routed_hidden: Buffer::new_zeroed(device, routed_hidden_bytes),
            sparse_activation: Buffer::new_zeroed(device, sparse_config.activation_bytes(num_routes)),
            expert_counts: Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes()),
            expert_offsets: Buffer::new_zeroed(device, expert_major_shape.expert_offsets_bytes()),
            expert_cursors: Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes()),
            routes_by_expert: Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes()),
            routes_by_token: Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes()),
            experts_by_route: Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes()),
            packed_input: Buffer::new_zeroed(device, expert_major_shape.route_hidden_bytes()),
        };
        let common_expert = core.common_expert_intermediate_dim.map(|intermediate_dim| {
            let dense_core = DenseMLPCore {
                model_layer_index: core.model_layer_index,
                hidden_dim: core.hidden_dim,
                intermediate_dim,
            };
            let dense_config = DenseMLPMetalConfig {
                group_size: config.group_size,
                bits: config.bits,
                dtype: config.dtype,
            };
            let gate_shape = affine_shape(max_tokens_u32, 1, core.hidden_dim, config.common_gate_bits, config);
            CommonExpertScratch {
                hidden: Buffer::new_zeroed_elements(
                    device,
                    max_tokens
                        .checked_mul(core.hidden_dim)
                        .expect("MoE common-expert hidden element capacity must fit usize"),
                    config.dtype,
                ),
                gate_logits: Buffer::new_zeroed(device, gate_shape.output_bytes()),
                dense_mlp: DenseMLPScratch::new(device, &dense_core, dense_config, max_tokens),
            }
        });

        Self {
            routing,
            topk_experts,
            common_expert,
        }
    }

    pub fn bindings(&self) -> MoEScratchBindings<'_> {
        MoEScratchBindings {
            routing: MoERoutingScratchBindings {
                router_logits: &self.routing.router_logits,
                router_probs: &self.routing.router_probs,
                expert_indices: &self.routing.expert_indices,
                expert_probs: &self.routing.expert_probs,
            },
            topk_experts: TopKExpertsScratchBindings {
                token_indices: &self.topk_experts.token_indices,
                route_indices: &self.topk_experts.route_indices,
                routed_hidden: &self.topk_experts.routed_hidden,
                sparse_activation: &self.topk_experts.sparse_activation,
                expert_counts: &self.topk_experts.expert_counts,
                expert_offsets: &self.topk_experts.expert_offsets,
                expert_cursors: &self.topk_experts.expert_cursors,
                routes_by_expert: &self.topk_experts.routes_by_expert,
                routes_by_token: &self.topk_experts.routes_by_token,
                experts_by_route: &self.topk_experts.experts_by_route,
                packed_input: &self.topk_experts.packed_input,
            },
        }
    }

    pub fn common_expert_bindings(&self) -> Option<CommonExpertScratchBindings<'_>> {
        self.common_expert.as_ref().map(|scratch| {
            CommonExpertScratchBindings {
                hidden: &scratch.hidden,
                gate_logits: &scratch.gate_logits,
                dense_mlp: scratch.dense_mlp.bindings(),
            }
        })
    }
}

fn affine_shape(m: u32, n: usize, k: usize, bits: u32, config: GatedMoEMetalConfig) -> AffineQuantizedMatmulShape {
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
