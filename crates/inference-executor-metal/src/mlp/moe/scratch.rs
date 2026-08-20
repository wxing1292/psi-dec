use inference_backend_metal::components::MoECombineConfig;
use inference_backend_metal::components::MoECombineShape;
use inference_backend_metal::components::MoEExpertMajorConfig;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingConfig;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::mlp::moe::GatedMoECore;

use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::dense::scratch::DenseMLPScratchBindings;
use crate::mlp::moe::backend::GatedMoEMetalConfig;

pub struct MoEScratch {
    routing: MoERoutingScratch,
    topk_experts: TopKExpertsScratch,
    shared_experts: Option<SharedExpertsScratch>,
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
    sparse_swiglu: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
}

struct SharedExpertsScratch {
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
    pub sparse_swiglu: &'a Buffer,
    pub expert_counts: &'a Buffer,
    pub expert_offsets: &'a Buffer,
    pub expert_cursors: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub experts_by_route: &'a Buffer,
    pub packed_input: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct SharedExpertsScratchBindings<'a> {
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
        let max_tokens_i32: i32 = max_tokens.try_into().expect("MoE scratch token capacity must fit i32");
        let num_routes = max_tokens_u32
            .checked_mul(
                core.num_experts_per_token
                    .try_into()
                    .expect("MoE top-k expert count must fit u32"),
            )
            .expect("MoE scratch route capacity must fit u32");
        let routing_config = MoERoutingConfig {
            num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
            num_experts_per_token: core
                .num_experts_per_token
                .try_into()
                .expect("MoE top-k expert count must fit u32"),
            norm_topk_prob: core.norm_topk_prob,
        };
        let routing_shape = MoERoutingShape {
            num_total_tokens: max_tokens_u32,
        };
        routing_config.validate_shape(routing_shape);
        let router_config = affine_config(core.num_experts, core.hidden_dim, config.router_bits, config);
        let expert_major_config = MoEExpertMajorConfig::bf16(
            core.num_experts.try_into().expect("MoE expert count must fit u32"),
            core.num_experts_per_token
                .try_into()
                .expect("MoE top-k expert count must fit u32"),
            core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
        );
        let expert_major_shape = MoEExpertMajorShape {
            num_total_tokens: max_tokens_u32,
        };
        expert_major_config.validate_shape(expert_major_shape);
        MoECombineConfig::bf16(
            expert_major_config.num_experts_per_token,
            expert_major_config.hidden_dim,
        )
        .validate_shape(MoECombineShape {
            num_total_tokens: expert_major_shape.num_total_tokens,
        });
        let sparse_config = sparse_mlp::Config {
            num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
            hidden_dim: core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
            intermediate_dim: core
                .intermediate_dim
                .try_into()
                .expect("MoE intermediate_dim must fit u32"),
            group_size: config.group_size,
            bits: config.bits,
            dtype: config.io_dtype,
        };
        let token_major_shape = sparse_mlp::TokenMajorShape {
            num_total_routes: num_routes,
            num_total_tokens: max_tokens_u32,
        };
        let routed_hidden_bytes =
            sparse_config
                .token_major_output_bytes(token_major_shape)
                .max(sparse_config.expert_major_output_bytes(sparse_mlp::ExpertMajorShape {
                    num_total_routes: expert_major_config.num_routes(expert_major_shape),
                }));
        let topk: u32 = core
            .num_experts_per_token
            .try_into()
            .expect("MoE top-k expert count must fit u32");
        let token_route_indices = (0..num_routes).map(|route| route / topk).collect::<Vec<_>>();
        let identity_indices = (0..num_routes).collect::<Vec<_>>();

        let routing = MoERoutingScratch {
            router_logits: Buffer::new_zeroed(device, router_config.output_bytes(max_tokens_i32)),
            router_probs: Buffer::new_zeroed(device, router_config.output_bytes(max_tokens_i32)),
            expert_indices: Buffer::new_zeroed(device, routing_config.expert_indices_bytes(routing_shape)),
            expert_probs: Buffer::new_zeroed(device, routing_config.expert_probs_bytes(routing_shape)),
        };
        let topk_experts = TopKExpertsScratch {
            token_indices: Buffer::from_slice(device, &token_route_indices),
            route_indices: Buffer::from_slice(device, &identity_indices),
            routed_hidden: Buffer::new_zeroed(device, routed_hidden_bytes),
            sparse_swiglu: Buffer::new_zeroed(device, sparse_config.swiglu_bytes(num_routes)),
            expert_counts: Buffer::new_zeroed(device, expert_major_config.expert_counts_bytes()),
            expert_offsets: Buffer::new_zeroed(device, expert_major_config.expert_offsets_bytes()),
            expert_cursors: Buffer::new_zeroed(device, expert_major_config.expert_counts_bytes()),
            routes_by_expert: Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape)),
            routes_by_token: Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape)),
            experts_by_route: Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape)),
            packed_input: Buffer::new_zeroed(device, expert_major_config.route_hidden_bytes(expert_major_shape)),
        };
        let shared_experts = core.shared_experts_core().map(|dense_core| {
            let gate_config = affine_config(1, core.hidden_dim, config.shared_expert_gate_bits, config);
            SharedExpertsScratch {
                hidden: Buffer::new_zeroed_elements(
                    device,
                    max_tokens
                        .checked_mul(core.hidden_dim)
                        .expect("MoE shared-expert hidden element capacity must fit usize"),
                    config.io_dtype,
                ),
                gate_logits: Buffer::new_zeroed(device, gate_config.output_bytes(max_tokens_i32)),
                dense_mlp: DenseMLPScratch::new(device, &dense_core, config.io_dtype, max_tokens),
            }
        });

        Self {
            routing,
            topk_experts,
            shared_experts,
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
                sparse_swiglu: &self.topk_experts.sparse_swiglu,
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

    pub fn shared_experts_bindings(&self) -> Option<SharedExpertsScratchBindings<'_>> {
        self.shared_experts.as_ref().map(|scratch| {
            SharedExpertsScratchBindings {
                hidden: &scratch.hidden,
                gate_logits: &scratch.gate_logits,
                dense_mlp: scratch.dense_mlp.bindings(),
            }
        })
    }
}

fn affine_config(n: usize, k: usize, bits: u32, config: GatedMoEMetalConfig) -> AffineQuantizedMatmulConfig {
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
