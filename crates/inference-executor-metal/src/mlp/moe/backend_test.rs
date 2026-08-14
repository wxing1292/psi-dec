use std::mem::size_of;

use half::bf16;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;

use super::*;
use crate::def::replay_op::MetalReplayRuntime;
use crate::mlp::moe::scratch::MoEScratch;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gated_moe.num_active_tokens");

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
fn test_routing_bucketed_chain_preserves_inactive_rows_across_reuse() {
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
        GatedMoERoutingBucketedInput {
            num_total_tokens,
            num_active_tokens_key: NUM_ACTIVE_TOKENS,
            hidden_state: &hidden_state,
            scratch,
            weights,
        },
    );
    let bucketed_replay = bucketed_recorder.build();
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
fn test_full_bucketed_replay_matches_exact_and_preserves_inactive_rows() {
    let device = Device::system_default();
    let stream = inference_backend_metal::metal::Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);

    for with_shared_experts in [false, true] {
        let (core, metal) = full_test_config(with_shared_experts);
        let moe = GatedMoE::new(&device, core.clone(), metal);
        let weights = FullMoETestWeights::new(&device, &core, metal);
        let max_tokens = 6_u32;
        let scratch = MoEScratch::new(&device, &core, metal, max_tokens as usize);
        let all_hidden = bf16_values(
            &(0..max_tokens as usize * core.hidden_dim)
                .map(|index| ((index * 23 + 7) % 67) as f32 * 0.015_625 - 0.5)
                .collect::<Vec<_>>(),
        );
        let hidden_state = bf16_buffer(&device, &all_hidden);
        let sentinel = bf16::from_f32(91.0).to_bits();
        let output_elements = max_tokens as usize * core.hidden_dim;
        let exact_output = Buffer::from_slice(&device, &vec![sentinel; output_elements]);
        let bucketed_output = Buffer::from_slice(&device, &vec![sentinel; output_elements]);

        for (num_active_tokens, num_total_tokens) in [(3_u32, 4_u32), (5_u32, 6_u32)] {
            write_bf16_bits(&exact_output, &vec![sentinel; output_elements]);
            write_bf16_bits(&bucketed_output, &vec![sentinel; output_elements]);
            let mut active_hidden = all_hidden.clone();
            active_hidden[num_active_tokens as usize * core.hidden_dim..].fill(f32::NAN);
            write_bf16_values(&hidden_state, &active_hidden);

            let mut exact_active_recorder = runtime.create_recorder();
            let _ = <GatedMoE as ReplayLayer>::record(
                &moe,
                &mut exact_active_recorder,
                weights.exact_input(&scratch, &hidden_state, &exact_output, num_active_tokens),
            );
            let exact_active_replay = exact_active_recorder.build();

            let mut exact_total_recorder = runtime.create_recorder();
            let _ = <GatedMoE as ReplayLayer>::record(
                &moe,
                &mut exact_total_recorder,
                weights.exact_input(&scratch, &hidden_state, &exact_output, num_total_tokens),
            );
            let exact_total_replay = exact_total_recorder.build();

            let mut bucketed_recorder = runtime.create_recorder();
            let _ = moe.record_bucketed(
                &mut bucketed_recorder,
                weights.bucketed_input(&scratch, &hidden_state, &bucketed_output, num_total_tokens),
            );
            let bucketed_replay = bucketed_recorder.build();

            runtime.submit_replay(&exact_active_replay).wait();
            let num_active_values = num_active_tokens as usize * core.hidden_dim;
            let num_total_values = num_total_tokens as usize * core.hidden_dim;
            let exact_active = exact_output.read_typed::<u16>(0, num_total_values);
            assert_finite_nonzero_bf16_bits(&exact_active[..num_active_values]);
            runtime
                .submit_replay_with_arguments(
                    &bucketed_replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                )
                .wait();
            let first_bucketed = bucketed_output.read_typed::<u16>(0, num_total_values);
            assert_eq!(&first_bucketed[..num_active_values], &exact_active[..num_active_values]);
            assert_eq!(
                &first_bucketed[num_active_values..],
                &vec![sentinel; num_total_values - num_active_values]
            );

            write_bf16_values(&hidden_state, &all_hidden);
            runtime.submit_replay(&exact_total_replay).wait();
            let exact_total = exact_output.read_typed::<u16>(0, num_total_values);
            runtime
                .submit_replay_with_arguments(
                    &bucketed_replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_total_tokens),
                )
                .wait();
            let full_bucketed = bucketed_output.read_typed::<u16>(0, num_total_values);
            assert_eq!(full_bucketed, exact_total);

            write_bf16_values(&hidden_state, &active_hidden);
            runtime.submit_replay(&exact_active_replay).wait();
            let exact_shrunk = exact_output.read_typed::<u16>(0, num_total_values);
            runtime
                .submit_replay_with_arguments(
                    &bucketed_replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                )
                .wait();
            let shrunk_bucketed = bucketed_output.read_typed::<u16>(0, num_total_values);
            assert_eq!(
                &shrunk_bucketed[..num_active_values],
                &exact_shrunk[..num_active_values]
            );
            assert_eq!(
                &shrunk_bucketed[num_active_values..],
                &full_bucketed[num_active_values..]
            );
        }
    }
}

struct FullMoETestWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    topk_experts: FullMoESparseTestWeights,
    shared_experts: Option<FullMoESharedTestWeights>,
}

impl FullMoETestWeights {
    fn new(device: &Device, core: &GatedMoECore, metal: GatedMoEMetalConfig) -> Self {
        let router_shape = core.router_shape();
        let router_config =
            affine_config_with_bits(router_shape.out_dim, router_shape.in_dim, metal.router_bits, metal);
        Self {
            router_weight: filled_u8_buffer(device, router_config.weight_bytes(), 9),
            router_scales: filled_bf16_buffer(device, router_config.scale_or_bias_bytes(), 0.000_976_562_5),
            router_biases: Buffer::new_zeroed(device, router_config.scale_or_bias_bytes()),
            topk_experts: FullMoESparseTestWeights::new(device, topk_experts_config(core, metal)),
            shared_experts: core
                .shared_experts_core()
                .map(|shared_core| FullMoESharedTestWeights::new(device, core, &shared_core, metal)),
        }
    }

    fn exact_input<'a>(
        &'a self,
        scratch: &'a MoEScratch,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        num_tokens: u32,
    ) -> GatedMoEInput<'a> {
        GatedMoEInput {
            shape: GatedMoEReplayShape { num_tokens },
            hidden_state,
            next_hidden_state,
            scratch: scratch.bindings(),
            weights: self.as_borrowed(),
            shared_experts: self.shared_experts_input(scratch),
        }
    }

    fn bucketed_input<'a>(
        &'a self,
        scratch: &'a MoEScratch,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        num_total_tokens: u32,
    ) -> GatedMoEBucketedInput<'a> {
        GatedMoEBucketedInput {
            num_total_tokens,
            num_active_tokens_key: NUM_ACTIVE_TOKENS,
            hidden_state,
            next_hidden_state,
            scratch: scratch.bindings(),
            weights: self.as_borrowed(),
            shared_experts: self.shared_experts_input(scratch),
        }
    }

    fn as_borrowed(&self) -> GatedMoEWeights<'_> {
        GatedMoEWeights {
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            topk_experts: self.topk_experts.as_borrowed(),
        }
    }

    fn shared_experts_input<'a>(&'a self, scratch: &'a MoEScratch) -> Option<GatedMoESharedExpertsInput<'a>> {
        self.shared_experts.as_ref().map(|weights| {
            GatedMoESharedExpertsInput {
                scratch: scratch
                    .shared_experts_bindings()
                    .expect("full MoE shared test weights require shared scratch"),
                weights: weights.as_borrowed(),
            }
        })
    }
}

struct FullMoESparseTestWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl FullMoESparseTestWeights {
    fn new(device: &Device, config: QuantizedSparseMLPConfig) -> Self {
        let num_experts = config.num_experts as usize;
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        let gate_up_weight_bytes = num_experts
            .checked_mul(gate_up.weight_bytes_per_expert())
            .expect("full MoE test sparse gate/up weight bytes must fit usize");
        let gate_up_param_bytes = num_experts
            .checked_mul(gate_up.affine_param_bytes_per_expert())
            .expect("full MoE test sparse gate/up parameter bytes must fit usize");
        let down_weight_bytes = num_experts
            .checked_mul(down.weight_bytes_per_expert())
            .expect("full MoE test sparse down weight bytes must fit usize");
        let down_param_bytes = num_experts
            .checked_mul(down.affine_param_bytes_per_expert())
            .expect("full MoE test sparse down parameter bytes must fit usize");
        Self {
            gate_weight: filled_u8_buffer(device, gate_up_weight_bytes, 7),
            gate_scales: filled_bf16_buffer(device, gate_up_param_bytes, 0.000_976_562_5),
            gate_biases: Buffer::new_zeroed(device, gate_up_param_bytes),
            up_weight: filled_u8_buffer(device, gate_up_weight_bytes, 11),
            up_scales: filled_bf16_buffer(device, gate_up_param_bytes, 0.000_976_562_5),
            up_biases: Buffer::new_zeroed(device, gate_up_param_bytes),
            down_weight: filled_u8_buffer(device, down_weight_bytes, 13),
            down_scales: filled_bf16_buffer(device, down_param_bytes, 0.000_976_562_5),
            down_biases: Buffer::new_zeroed(device, down_param_bytes),
        }
    }

    fn as_borrowed(&self) -> QuantizedSparseMLPWeights<'_> {
        QuantizedSparseMLPWeights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct FullMoESharedTestWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    dense: FullMoEDenseTestWeights,
}

impl FullMoESharedTestWeights {
    fn new(device: &Device, core: &GatedMoECore, shared_core: &DenseMLPCore, metal: GatedMoEMetalConfig) -> Self {
        let gate_shape = core
            .shared_expert_gate_shape()
            .expect("full MoE shared test requires a shared gate shape");
        let gate_config = affine_config_with_bits(
            gate_shape.out_dim,
            gate_shape.in_dim,
            metal.shared_expert_gate_bits,
            metal,
        );
        Self {
            gate_weight: filled_u8_buffer(device, gate_config.weight_bytes(), 17),
            gate_scales: filled_bf16_buffer(device, gate_config.scale_or_bias_bytes(), 0.000_976_562_5),
            gate_biases: Buffer::new_zeroed(device, gate_config.scale_or_bias_bytes()),
            dense: FullMoEDenseTestWeights::new(device, shared_experts_config(shared_core, metal)),
        }
    }

    fn as_borrowed(&self) -> GatedMoESharedExpertsWeights<'_> {
        GatedMoESharedExpertsWeights {
            shared_expert_gate_weight: &self.gate_weight,
            shared_expert_gate_scales: &self.gate_scales,
            shared_expert_gate_biases: &self.gate_biases,
            shared_experts: self.dense.as_borrowed(),
        }
    }
}

struct FullMoEDenseTestWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl FullMoEDenseTestWeights {
    fn new(device: &Device, config: QuantizedDenseMLPConfig) -> Self {
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        Self {
            gate_up_weight: filled_u8_buffer(device, gate_up.weight_bytes(), 19),
            gate_up_scales: filled_bf16_buffer(device, gate_up.scale_or_bias_bytes(), 0.000_976_562_5),
            gate_up_biases: Buffer::new_zeroed(device, gate_up.scale_or_bias_bytes()),
            down_weight: filled_u8_buffer(device, down.weight_bytes(), 23),
            down_scales: filled_bf16_buffer(device, down.scale_or_bias_bytes(), 0.000_976_562_5),
            down_biases: Buffer::new_zeroed(device, down.scale_or_bias_bytes()),
        }
    }

    fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

fn full_test_config(with_shared_experts: bool) -> (GatedMoECore, GatedMoEMetalConfig) {
    (
        GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 32,
            intermediate_dim: 32,
            shared_experts_intermediate_dim: with_shared_experts.then_some(32),
            num_experts: 8,
            num_experts_per_token: 2,
            norm_topk_prob: true,
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

fn write_bf16_bits(buffer: &Buffer, bits: &[u16]) {
    buffer.write_typed(0, bits);
}

fn filled_u8_buffer(device: &Device, len: usize, value: u8) -> Buffer {
    Buffer::from_slice(device, &vec![value; len])
}

fn filled_bf16_buffer(device: &Device, bytes: usize, value: f32) -> Buffer {
    assert_eq!(bytes % size_of::<u16>(), 0);
    Buffer::from_slice(device, &vec![bf16::from_f32(value).to_bits(); bytes / size_of::<u16>()])
}

fn assert_finite_nonzero_bf16_bits(bits: &[u16]) {
    let values = bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| *value != 0.0));
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
