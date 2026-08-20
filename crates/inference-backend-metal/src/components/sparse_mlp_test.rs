use half::bf16;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceInput;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceWeights;
use inference_executor_core::mlp::moe::reference::moe_combine_without_shared_experts_bf16_reference;
use inference_executor_core::mlp::moe::reference::quantized_sparse_mlp_reference;

use super::*;
use crate::components::MoEExpertMajorConfig;
use crate::components::MoEExpertMajorKernels;
use crate::components::MoEExpertMajorLayoutBuffers;
use crate::components::MoEExpertMajorPackInputBuffers;
use crate::components::MoEExpertMajorScatterWithoutSharedExpertsBuffers;
use crate::components::MoEExpertMajorShape;
use crate::metal::Buffer;
use crate::metal::ReplayArguments;
use crate::metal::ReplayProgram;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.sparse_mlp.num_active_tokens");
const INPUT_POISON: u16 = 0x7fc1;
const SWIGLU_CANARY: u16 = 0x3555;
const OUTPUT_CANARY: u16 = 0x3aaa;

#[test]
fn test_token_major_fixed() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 4,
        hidden_dim: 64,
        intermediate_dim: 64,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    };
    let shape = TokenMajorShape {
        num_total_routes: 4,
        num_total_tokens: 2,
    };
    let gate_up_config = config.gate_up_config();
    let down_config = config.down_config();
    let num_experts = config.num_experts as usize;
    let input_values = hidden_fixture(shape.num_total_tokens as usize, config.hidden_dim as usize);
    let input = bf16_buffer(&device, &input_values);
    let token_index_values = vec![0_u32, 0, 1, 1];
    let expert_index_values = vec![0_u32, 2, 1, 3];
    let route_index_values = vec![0_u32, 1, 2, 3];
    let token_indices = Buffer::from_slice(&device, &token_index_values);
    let expert_indices = Buffer::from_slice(&device, &expert_index_values);
    let route_indices = Buffer::from_slice(&device, &route_index_values);
    let gate_weight_values = quantized_weight_stack_values(num_experts, gate_up_config.weight_bytes_per_expert());
    let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
    let gate_scale_values =
        affine_param_fixture(num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let gate_scales = bf16_buffer(&device, &gate_scale_values);
    let gate_bias_values =
        zero_fixture(num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let gate_biases = bf16_buffer(&device, &gate_bias_values);
    let up_weight_values = quantized_weight_stack_values(num_experts, gate_up_config.weight_bytes_per_expert());
    let up_weight = Buffer::from_slice(&device, &up_weight_values);
    let up_scale_values =
        affine_param_fixture(num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let up_scales = bf16_buffer(&device, &up_scale_values);
    let up_bias_values = zero_fixture(num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let up_biases = bf16_buffer(&device, &up_bias_values);
    let down_weight_values = quantized_weight_stack_values(num_experts, down_config.weight_bytes_per_expert());
    let down_weight = Buffer::from_slice(&device, &down_weight_values);
    let down_scale_values =
        affine_param_fixture(num_experts * down_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let down_scales = bf16_buffer(&device, &down_scale_values);
    let down_bias_values = zero_fixture(num_experts * down_config.affine_param_bytes_per_expert() / size_of::<u16>());
    let down_biases = bf16_buffer(&device, &down_bias_values);

    let actual_output = Buffer::new_zeroed(&device, config.token_major_output_bytes(shape));
    let actual_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape.num_total_routes));
    let sparse_mlp = TokenMajorKernels::new(&device, config);
    let mut builder = stream.create_replay_program();
    builder.record(sparse_mlp.invoke(
        shape,
        TokenMajorBuffers {
            input: &input,
            token_indices: &token_indices,
            expert_indices: &expert_indices,
            route_indices: &route_indices,
            routed_hidden: &actual_output,
        },
        Scratch { swiglu: &actual_swiglu },
        Weights {
            gate_weight: &gate_weight,
            gate_scales: &gate_scales,
            gate_biases: &gate_biases,
            up_weight: &up_weight,
            up_scales: &up_scales,
            up_biases: &up_biases,
            down_weight: &down_weight,
            down_scales: &down_scales,
            down_biases: &down_biases,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    let expected = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
        hidden: &bf16_values(&input_values),
        token_indices: &token_index_values,
        expert_indices: &expert_index_values,
        swiglu_indices: &route_index_values,
        hidden_dim: config.hidden_dim as usize,
        intermediate_dim: config.intermediate_dim as usize,
        group_size: config.group_size as usize,
        bits: config.bits as usize,
        num_experts,
        weights: QuantizedSparseMLPReferenceWeights {
            gate_weight: &gate_weight_values,
            gate_scales: &bf16_values(&gate_scale_values),
            gate_biases: &bf16_values(&gate_bias_values),
            up_weight: &up_weight_values,
            up_scales: &bf16_values(&up_scale_values),
            up_biases: &bf16_values(&up_bias_values),
            down_weight: &down_weight_values,
            down_scales: &bf16_values(&down_scale_values),
            down_biases: &bf16_values(&down_bias_values),
        },
    })
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_bf16_close_rel_values(
        &expected,
        &actual_output,
        config.token_major_output_bytes(shape),
        2.0e-5,
        8.0e-3,
    );
}

#[test]
fn test_token_major_random() {
    let random_seed = 0x3A7D_C921;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 5,
        hidden_dim: 64,
        intermediate_dim: 64,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    };
    let shape = TokenMajorShape {
        num_total_routes: 6,
        num_total_tokens: 3,
    };
    let num_experts = config.num_experts as usize;
    let gate_up_config = config.gate_up_config();
    let down_config = config.down_config();
    let input_values = generated_values(
        shape.num_total_tokens as usize * config.hidden_dim as usize,
        random_seed,
    );
    let input = bf16_buffer(&device, &input_values);
    let token_index_values = generated_indices(
        shape.num_total_routes as usize,
        shape.num_total_tokens as usize,
        random_seed.wrapping_add(1),
    );
    let expert_index_values = generated_indices(
        shape.num_total_routes as usize,
        num_experts,
        random_seed.wrapping_add(2),
    );
    let route_index_values = identity_indices(shape.num_total_routes as usize);
    let token_indices = Buffer::from_slice(&device, &token_index_values);
    let expert_indices = Buffer::from_slice(&device, &expert_index_values);
    let route_indices = Buffer::from_slice(&device, &route_index_values);
    let gate_weight_values = generated_bytes(
        num_experts * gate_up_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(3),
    );
    let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
    let gate_scale_values = generated_scales(
        num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(4),
    );
    let gate_scales = bf16_buffer(&device, &gate_scale_values);
    let gate_bias_values = generated_biases(
        num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(5),
    );
    let gate_biases = bf16_buffer(&device, &gate_bias_values);
    let up_weight_values = generated_bytes(
        num_experts * gate_up_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(6),
    );
    let up_weight = Buffer::from_slice(&device, &up_weight_values);
    let up_scale_values = generated_scales(
        num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(7),
    );
    let up_scales = bf16_buffer(&device, &up_scale_values);
    let up_bias_values = generated_biases(
        num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(8),
    );
    let up_biases = bf16_buffer(&device, &up_bias_values);
    let down_weight_values = generated_bytes(
        num_experts * down_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(9),
    );
    let down_weight = Buffer::from_slice(&device, &down_weight_values);
    let down_scale_values = generated_scales(
        num_experts * down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(10),
    );
    let down_scales = bf16_buffer(&device, &down_scale_values);
    let down_bias_values = generated_biases(
        num_experts * down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(11),
    );
    let down_biases = bf16_buffer(&device, &down_bias_values);

    let actual_output = Buffer::new_zeroed(&device, config.token_major_output_bytes(shape));
    let actual_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape.num_total_routes));
    let sparse_mlp = TokenMajorKernels::new(&device, config);
    let mut builder = stream.create_replay_program();
    builder.record(sparse_mlp.invoke(
        shape,
        TokenMajorBuffers {
            input: &input,
            token_indices: &token_indices,
            expert_indices: &expert_indices,
            route_indices: &route_indices,
            routed_hidden: &actual_output,
        },
        Scratch { swiglu: &actual_swiglu },
        Weights {
            gate_weight: &gate_weight,
            gate_scales: &gate_scales,
            gate_biases: &gate_biases,
            up_weight: &up_weight,
            up_scales: &up_scales,
            up_biases: &up_biases,
            down_weight: &down_weight,
            down_scales: &down_scales,
            down_biases: &down_biases,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    let expected = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
        hidden: &bf16_values(&input_values),
        token_indices: &token_index_values,
        expert_indices: &expert_index_values,
        swiglu_indices: &route_index_values,
        hidden_dim: config.hidden_dim as usize,
        intermediate_dim: config.intermediate_dim as usize,
        group_size: config.group_size as usize,
        bits: config.bits as usize,
        num_experts,
        weights: QuantizedSparseMLPReferenceWeights {
            gate_weight: &gate_weight_values,
            gate_scales: &bf16_values(&gate_scale_values),
            gate_biases: &bf16_values(&gate_bias_values),
            up_weight: &up_weight_values,
            up_scales: &bf16_values(&up_scale_values),
            up_biases: &bf16_values(&up_bias_values),
            down_weight: &down_weight_values,
            down_scales: &bf16_values(&down_scale_values),
            down_biases: &bf16_values(&down_bias_values),
        },
    })
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_bf16_close_rel_values(
        &expected,
        &actual_output,
        config.token_major_output_bytes(shape),
        2.0e-5,
        8.0e-3,
    );
}

#[test]
fn test_expert_major_fixed() {
    assert_expert_major_pipeline_matches_reference(0x51a7_2026);
}

#[test]
fn test_expert_major_random() {
    let random_seed = 0xB650_2FE8;
    assert_expert_major_pipeline_matches_reference(random_seed);
}

#[test]
fn test_bucketed_shape_rejects_invalid_route_domain() {
    let token_major = BucketedSparseMLPFixture::new(4, 2);
    assert_panics(|| {
        let _ = token_major.compute.invoke_token_major_bucketed(
            token_major.num_total_tokens,
            0,
            NUM_ACTIVE_TOKENS,
            token_major.token_major_buffers(),
            Scratch {
                swiglu: &token_major.token_major_swiglu,
            },
            token_major.weights.as_borrowed(),
        );
    });
    assert_panics(|| {
        let _ = token_major.compute.invoke_token_major_bucketed(
            token_major.num_total_tokens,
            token_major.config.num_experts + 1,
            NUM_ACTIVE_TOKENS,
            token_major.token_major_buffers(),
            Scratch {
                swiglu: &token_major.token_major_swiglu,
            },
            token_major.weights.as_borrowed(),
        );
    });
    assert_panics(|| {
        let _ = token_major.compute.invoke_token_major_bucketed(
            u32::MAX,
            2,
            NUM_ACTIVE_TOKENS,
            token_major.token_major_buffers(),
            Scratch {
                swiglu: &token_major.token_major_swiglu,
            },
            token_major.weights.as_borrowed(),
        );
    });
}

#[test]
fn test_bucketed_token_major_preserves_inactive_routes_and_shrink() {
    let fixture = BucketedSparseMLPFixture::new(4, 2);
    let replay = fixture.bucketed_token_major_replay();
    fixture.reset_token_major_canaries();
    let first = fixture.write_work(3, 0x8100_0001);
    fixture.submit(&replay, 3);
    fixture.assert_active_output(&fixture.token_major_output, &first);
    fixture.assert_token_major_canary_tails(3);

    let full = fixture.write_work(4, 0x8100_0002);
    fixture.submit(&replay, 4);
    fixture.assert_active_output(&fixture.token_major_output, &full);
    let full_swiglu = fixture.read_swiglu(&fixture.token_major_swiglu);
    let full_output = fixture.read_output(&fixture.token_major_output);

    let smaller = fixture.write_work(3, 0x8100_0003);
    fixture.submit(&replay, 3);
    fixture.assert_active_output(&fixture.token_major_output, &smaller);
    fixture.assert_preserved_tails(
        3,
        &fixture.token_major_swiglu,
        &fixture.token_major_output,
        &full_swiglu,
        &full_output,
    );
}

#[test]
fn test_bucketed_token_major_fast_down_preserves_inactive_routes() {
    let fixture = BucketedSparseMLPFixture::new_with_intermediate_dim(2, 2, 512);
    let replay = fixture.bucketed_token_major_replay();
    fixture.reset_token_major_canaries();
    let input = fixture.write_work(1, 0x8150_0001);
    fixture.submit(&replay, 1);
    fixture.assert_active_output(&fixture.token_major_output, &input);
    fixture.assert_token_major_canary_tails(1);
}

#[test]
fn test_bucketed_expert_major_preserves_inactive_routes_and_shrink() {
    let fixture = BucketedSparseMLPFixture::new(6, 2);
    let replay = fixture.bucketed_expert_major_replay();
    fixture.reset_expert_major_canaries();
    let first = fixture.write_work(5, 0x8200_0001);
    fixture.submit(&replay, 5);
    fixture.assert_active_output(&fixture.expert_major_output, &first);
    fixture.assert_expert_major_canary_tails(5);

    let full = fixture.write_work(6, 0x8200_0002);
    fixture.submit(&replay, 6);
    fixture.assert_active_output(&fixture.expert_major_output, &full);
    let full_swiglu = fixture.read_swiglu(&fixture.expert_major_swiglu);
    let full_output = fixture.read_output(&fixture.expert_major_output);

    let smaller = fixture.write_work(5, 0x8200_0003);
    fixture.submit(&replay, 5);
    fixture.assert_active_output(&fixture.expert_major_output, &smaller);
    fixture.assert_preserved_tails(
        5,
        &fixture.expert_major_swiglu,
        &fixture.expert_major_output,
        &full_swiglu,
        &full_output,
    );
}

fn assert_expert_major_pipeline_matches_reference(random_seed: u32) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let num_experts = 5_u32;
    let config = Config {
        num_experts,
        hidden_dim: 64,
        intermediate_dim: 64,
        group_size: 32,
        bits: 4,
        dtype: Dtype::Bfloat16,
    };
    let num_tokens = 3_u32;
    let num_experts_per_token = 2_u32;
    let num_routes = num_tokens * num_experts_per_token;
    let layout_config = MoEExpertMajorConfig::bf16(num_experts, num_experts_per_token, config.hidden_dim);
    let layout_shape = MoEExpertMajorShape {
        num_total_tokens: num_tokens,
    };
    let sparse_shape = ExpertMajorShape {
        num_total_routes: num_routes,
    };
    let gate_up_config = config.gate_up_config();
    let down_config = config.down_config();

    let input_values = generated_values(num_tokens as usize * config.hidden_dim as usize, random_seed);
    let expert_index_values = generated_indices(num_routes as usize, num_experts as usize, random_seed.wrapping_add(1));
    let routed_prob_values = generated_probs(
        num_tokens as usize,
        num_experts_per_token as usize,
        random_seed.wrapping_add(2),
    );
    let token_index_values = (0..num_routes)
        .map(|route_index| route_index / num_experts_per_token)
        .collect::<Vec<_>>();
    let route_index_values = identity_indices(num_routes as usize);
    let gate_weight_values = generated_bytes(
        num_experts as usize * gate_up_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(3),
    );
    let gate_scale_values = generated_scales(
        num_experts as usize * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(4),
    );
    let gate_bias_values = generated_biases(
        num_experts as usize * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(5),
    );
    let up_weight_values = generated_bytes(
        num_experts as usize * gate_up_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(6),
    );
    let up_scale_values = generated_scales(
        num_experts as usize * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(7),
    );
    let up_bias_values = generated_biases(
        num_experts as usize * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(8),
    );
    let down_weight_values = generated_bytes(
        num_experts as usize * down_config.weight_bytes_per_expert(),
        random_seed.wrapping_add(9),
    );
    let down_scale_values = generated_scales(
        num_experts as usize * down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(10),
    );
    let down_bias_values = generated_biases(
        num_experts as usize * down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
        random_seed.wrapping_add(11),
    );

    let input = bf16_buffer(&device, &input_values);
    let expert_indices = Buffer::from_slice(&device, &expert_index_values);
    let routed_probs = Buffer::from_slice(&device, &routed_prob_values);
    let expert_counts = Buffer::new_zeroed(&device, layout_config.expert_counts_bytes());
    let expert_offsets = Buffer::new_zeroed(&device, layout_config.expert_offsets_bytes());
    let expert_cursors = Buffer::new_zeroed(&device, layout_config.expert_counts_bytes());
    let routes_by_expert = Buffer::new_zeroed(&device, layout_config.route_indices_bytes(layout_shape));
    let routes_by_token = Buffer::new_zeroed(&device, layout_config.route_indices_bytes(layout_shape));
    let experts_by_route = Buffer::new_zeroed(&device, layout_config.route_indices_bytes(layout_shape));
    let packed_input = Buffer::new_zeroed(&device, layout_config.route_hidden_bytes(layout_shape));
    let packed_output = Buffer::new_zeroed(&device, config.expert_major_output_bytes(sparse_shape));
    let swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(num_routes));
    let output = Buffer::new_zeroed(&device, layout_config.token_hidden_bytes(layout_shape));
    let gate_weight = Buffer::from_slice(&device, &gate_weight_values);
    let gate_scales = bf16_buffer(&device, &gate_scale_values);
    let gate_biases = bf16_buffer(&device, &gate_bias_values);
    let up_weight = Buffer::from_slice(&device, &up_weight_values);
    let up_scales = bf16_buffer(&device, &up_scale_values);
    let up_biases = bf16_buffer(&device, &up_bias_values);
    let down_weight = Buffer::from_slice(&device, &down_weight_values);
    let down_scales = bf16_buffer(&device, &down_scale_values);
    let down_biases = bf16_buffer(&device, &down_bias_values);

    let layout = MoEExpertMajorKernels::new(&device, layout_config);
    let sparse_mlp = Compute::new(&device, config);
    let weights = Weights {
        gate_weight: &gate_weight,
        gate_scales: &gate_scales,
        gate_biases: &gate_biases,
        up_weight: &up_weight,
        up_scales: &up_scales,
        up_biases: &up_biases,
        down_weight: &down_weight,
        down_scales: &down_scales,
        down_biases: &down_biases,
    };
    let mut builder = stream.create_replay_program();
    builder.record(layout.invoke_layout(
        layout_shape,
        MoEExpertMajorLayoutBuffers {
            expert_indices: &expert_indices,
            expert_counts: &expert_counts,
            expert_offsets: &expert_offsets,
            expert_cursors: &expert_cursors,
            routes_by_expert: &routes_by_expert,
            routes_by_token: &routes_by_token,
            experts_by_route: &experts_by_route,
        },
    ));
    builder.record_with_barrier_before(layout.invoke_pack_input(
        layout_shape,
        MoEExpertMajorPackInputBuffers {
            input: &input,
            routes_by_expert: &routes_by_expert,
            packed_input: &packed_input,
        },
    ));
    builder.record_with_barrier_before(sparse_mlp.invoke_expert_major(
        sparse_shape,
        ExpertMajorBuffers {
            packed_input: &packed_input,
            experts_by_route: &experts_by_route,
            packed_output: &packed_output,
        },
        Scratch { swiglu: &swiglu },
        weights,
    ));
    builder.record_with_barrier_before(layout.invoke_scatter_without_shared_experts(
        layout_shape,
        MoEExpertMajorScatterWithoutSharedExpertsBuffers {
            packed_output: &packed_output,
            routes_by_token: &routes_by_token,
            routed_probs: &routed_probs,
            output: &output,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    let routed_hidden = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
        hidden: &bf16_values(&input_values),
        token_indices: &token_index_values,
        expert_indices: &expert_index_values,
        swiglu_indices: &route_index_values,
        hidden_dim: config.hidden_dim as usize,
        intermediate_dim: config.intermediate_dim as usize,
        group_size: config.group_size as usize,
        bits: config.bits as usize,
        num_experts: num_experts as usize,
        weights: QuantizedSparseMLPReferenceWeights {
            gate_weight: &gate_weight_values,
            gate_scales: &bf16_values(&gate_scale_values),
            gate_biases: &bf16_values(&gate_bias_values),
            up_weight: &up_weight_values,
            up_scales: &bf16_values(&up_scale_values),
            up_biases: &bf16_values(&up_bias_values),
            down_weight: &down_weight_values,
            down_scales: &bf16_values(&down_scale_values),
            down_biases: &bf16_values(&down_bias_values),
        },
    })
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    let expected_bits = moe_combine_without_shared_experts_bf16_reference(
        &routed_hidden,
        &routed_prob_values,
        num_tokens as usize,
        num_experts_per_token as usize,
        config.hidden_dim as usize,
    );
    let expected = expected_bits
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    assert_bf16_close_rel_values(
        &expected,
        &output,
        layout_config.token_hidden_bytes(layout_shape),
        2.0e-5,
        8.0e-3,
    );
}

struct BucketedSparseMLPFixture {
    stream: Stream,
    config: Config,
    compute: Compute,
    num_total_tokens: u32,
    num_experts_per_token: u32,
    token_major_input: Buffer,
    token_indices: Buffer,
    expert_indices: Buffer,
    route_indices: Buffer,
    token_major_output: Buffer,
    token_major_swiglu: Buffer,
    expert_major_input: Buffer,
    experts_by_route: Buffer,
    expert_major_output: Buffer,
    expert_major_swiglu: Buffer,
    weights: BucketedSparseMLPWeights,
}

impl BucketedSparseMLPFixture {
    fn new(num_total_tokens: u32, num_experts_per_token: u32) -> Self {
        Self::new_with_intermediate_dim(num_total_tokens, num_experts_per_token, 64)
    }

    fn new_with_intermediate_dim(num_total_tokens: u32, num_experts_per_token: u32, intermediate_dim: u32) -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_experts: 5,
            hidden_dim: 64,
            intermediate_dim,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let num_total_routes = num_total_tokens.checked_mul(num_experts_per_token).unwrap();
        let token_shape = TokenMajorShape {
            num_total_routes,
            num_total_tokens,
        };
        let expert_shape = ExpertMajorShape { num_total_routes };
        let route_index_bytes = num_total_routes as usize * size_of::<u32>();
        let compute = Compute::new(&device, config);
        let weights = BucketedSparseMLPWeights::new(&device, config);
        Self {
            token_major_input: Buffer::new_zeroed(&device, config.token_major_input_bytes(token_shape)),
            token_indices: Buffer::new_zeroed(&device, route_index_bytes),
            expert_indices: Buffer::new_zeroed(&device, route_index_bytes),
            route_indices: Buffer::new_zeroed(&device, route_index_bytes),
            token_major_output: Buffer::new_zeroed(&device, config.token_major_output_bytes(token_shape)),
            token_major_swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(num_total_routes)),
            expert_major_input: Buffer::new_zeroed(&device, config.expert_major_input_bytes(expert_shape)),
            experts_by_route: Buffer::new_zeroed(&device, route_index_bytes),
            expert_major_output: Buffer::new_zeroed(&device, config.expert_major_output_bytes(expert_shape)),
            expert_major_swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(num_total_routes)),
            stream,
            config,
            compute,
            num_total_tokens,
            num_experts_per_token,
            weights,
        }
    }

    fn num_total_routes(&self) -> u32 {
        self.num_total_tokens.checked_mul(self.num_experts_per_token).unwrap()
    }

    fn num_active_routes(&self, num_active_tokens: u32) -> u32 {
        assert!(num_active_tokens <= self.num_total_tokens);
        num_active_tokens.checked_mul(self.num_experts_per_token).unwrap()
    }

    fn token_major_shape(&self) -> TokenMajorShape {
        TokenMajorShape {
            num_total_routes: self.num_total_routes(),
            num_total_tokens: self.num_total_tokens,
        }
    }

    fn expert_major_shape(&self) -> ExpertMajorShape {
        ExpertMajorShape {
            num_total_routes: self.num_total_routes(),
        }
    }

    fn token_major_buffers(&self) -> TokenMajorBuffers<'_> {
        TokenMajorBuffers {
            input: &self.token_major_input,
            token_indices: &self.token_indices,
            expert_indices: &self.expert_indices,
            route_indices: &self.route_indices,
            routed_hidden: &self.token_major_output,
        }
    }

    fn expert_major_buffers(&self) -> ExpertMajorBuffers<'_> {
        ExpertMajorBuffers {
            packed_input: &self.expert_major_input,
            experts_by_route: &self.experts_by_route,
            packed_output: &self.expert_major_output,
        }
    }

    fn bucketed_token_major_replay(&self) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke_token_major_bucketed(
            self.num_total_tokens,
            self.num_experts_per_token,
            NUM_ACTIVE_TOKENS,
            self.token_major_buffers(),
            Scratch {
                swiglu: &self.token_major_swiglu,
            },
            self.weights.as_borrowed(),
        ));
        builder.build()
    }

    fn bucketed_expert_major_replay(&self) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke_expert_major_bucketed(
            self.num_total_tokens,
            self.num_experts_per_token,
            NUM_ACTIVE_TOKENS,
            self.expert_major_buffers(),
            Scratch {
                swiglu: &self.expert_major_swiglu,
            },
            self.weights.as_borrowed(),
        ));
        builder.build()
    }

    fn write_work(&self, num_active_tokens: u32, seed: u32) -> ActiveSparseMLPInput {
        let num_active_routes = self.num_active_routes(num_active_tokens);
        let hidden_dim = self.config.hidden_dim as usize;
        let active_input = bf16_values(&generated_values(num_active_tokens as usize * hidden_dim, seed));
        let mut input_bits = vec![INPUT_POISON; self.num_total_tokens as usize * hidden_dim];
        for (bits, value) in input_bits.iter_mut().zip(&active_input) {
            *bits = bf16::from_f32(*value).to_bits();
        }
        self.token_major_input.write_typed(0, &input_bits);

        let mut token_indices = vec![u32::MAX; self.num_total_routes() as usize];
        let mut expert_indices = vec![u32::MAX; self.num_total_routes() as usize];
        let mut route_indices = vec![u32::MAX; self.num_total_routes() as usize];
        let mut packed_input_bits = vec![INPUT_POISON; self.num_total_routes() as usize * hidden_dim];
        for route in 0..num_active_routes {
            let token = route / self.num_experts_per_token;
            let expert = (route.wrapping_mul(3).wrapping_add(token).wrapping_add(1)) % self.config.num_experts;
            token_indices[route as usize] = token;
            expert_indices[route as usize] = expert;
            route_indices[route as usize] = route;
            let source_start = token as usize * hidden_dim;
            let target_start = route as usize * hidden_dim;
            for (target, value) in packed_input_bits[target_start..target_start + hidden_dim]
                .iter_mut()
                .zip(&active_input[source_start..source_start + hidden_dim])
            {
                *target = bf16::from_f32(*value).to_bits();
            }
        }
        self.token_indices.write_typed(0, &token_indices);
        self.expert_indices.write_typed(0, &expert_indices);
        self.route_indices.write_typed(0, &route_indices);
        self.expert_major_input.write_typed(0, &packed_input_bits);
        self.experts_by_route.write_typed(0, &expert_indices);

        ActiveSparseMLPInput {
            input: active_input,
            token_indices: token_indices[..num_active_routes as usize].to_vec(),
            expert_indices: expert_indices[..num_active_routes as usize].to_vec(),
            route_indices: route_indices[..num_active_routes as usize].to_vec(),
        }
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, output: &Buffer, input: &ActiveSparseMLPInput) {
        let expected = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
            hidden: &input.input,
            token_indices: &input.token_indices,
            expert_indices: &input.expert_indices,
            swiglu_indices: &input.route_indices,
            hidden_dim: self.config.hidden_dim as usize,
            intermediate_dim: self.config.intermediate_dim as usize,
            group_size: self.config.group_size as usize,
            bits: self.config.bits as usize,
            num_experts: self.config.num_experts as usize,
            weights: self.weights.as_reference(),
        })
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        assert_bf16_close_rel_values(&expected, output, expected.len() * size_of::<u16>(), 2.0e-5, 8.0e-3);
    }

    fn reset_token_major_canaries(&self) {
        self.reset_canaries(&self.token_major_swiglu, &self.token_major_output);
    }

    fn reset_expert_major_canaries(&self) {
        self.reset_canaries(&self.expert_major_swiglu, &self.expert_major_output);
    }

    fn reset_canaries(&self, swiglu: &Buffer, output: &Buffer) {
        swiglu.write_typed(
            0,
            &vec![SWIGLU_CANARY; self.num_total_routes() as usize * self.config.intermediate_dim as usize],
        );
        output.write_typed(
            0,
            &vec![OUTPUT_CANARY; self.num_total_routes() as usize * self.config.hidden_dim as usize],
        );
    }

    fn assert_token_major_canary_tails(&self, num_active_tokens: u32) {
        self.assert_canary_tails(num_active_tokens, &self.token_major_swiglu, &self.token_major_output);
    }

    fn assert_expert_major_canary_tails(&self, num_active_tokens: u32) {
        self.assert_canary_tails(num_active_tokens, &self.expert_major_swiglu, &self.expert_major_output);
    }

    fn assert_canary_tails(&self, num_active_tokens: u32, swiglu: &Buffer, output: &Buffer) {
        let num_active_routes = self.num_active_routes(num_active_tokens) as usize;
        let swiglu_tail = num_active_routes * self.config.intermediate_dim as usize;
        let output_tail = num_active_routes * self.config.hidden_dim as usize;
        assert!(
            self.read_swiglu(swiglu)[swiglu_tail..]
                .iter()
                .all(|&bits| bits == SWIGLU_CANARY)
        );
        assert!(
            self.read_output(output)[output_tail..]
                .iter()
                .all(|&bits| bits == OUTPUT_CANARY)
        );
    }

    fn assert_preserved_tails(
        &self,
        num_active_tokens: u32,
        swiglu: &Buffer,
        output: &Buffer,
        expected_swiglu: &[u16],
        expected_output: &[u16],
    ) {
        let num_active_routes = self.num_active_routes(num_active_tokens) as usize;
        let swiglu_tail = num_active_routes * self.config.intermediate_dim as usize;
        let output_tail = num_active_routes * self.config.hidden_dim as usize;
        assert_eq!(
            &self.read_swiglu(swiglu)[swiglu_tail..],
            &expected_swiglu[swiglu_tail..]
        );
        assert_eq!(
            &self.read_output(output)[output_tail..],
            &expected_output[output_tail..]
        );
    }

    fn read_swiglu(&self, buffer: &Buffer) -> Vec<u16> {
        buffer.read_typed(
            0,
            self.num_total_routes() as usize * self.config.intermediate_dim as usize,
        )
    }

    fn read_output(&self, buffer: &Buffer) -> Vec<u16> {
        buffer.read_typed(0, self.num_total_routes() as usize * self.config.hidden_dim as usize)
    }
}

struct ActiveSparseMLPInput {
    input: Vec<f32>,
    token_indices: Vec<u32>,
    expert_indices: Vec<u32>,
    route_indices: Vec<u32>,
}

struct BucketedSparseMLPWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
    gate_weight_values: Vec<u8>,
    gate_scale_values: Vec<f32>,
    gate_bias_values: Vec<f32>,
    up_weight_values: Vec<u8>,
    up_scale_values: Vec<f32>,
    up_bias_values: Vec<f32>,
    down_weight_values: Vec<u8>,
    down_scale_values: Vec<f32>,
    down_bias_values: Vec<f32>,
}

impl BucketedSparseMLPWeights {
    fn new(device: &Device, config: Config) -> Self {
        let num_experts = config.num_experts as usize;
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        let gate_weight_values = generated_bytes(num_experts * gate_up.weight_bytes_per_expert(), 0x8300_0001);
        let gate_scale_values = bf16_values(&generated_scales(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0002,
        ));
        let gate_bias_values = bf16_values(&generated_biases(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0003,
        ));
        let up_weight_values = generated_bytes(num_experts * gate_up.weight_bytes_per_expert(), 0x8300_0004);
        let up_scale_values = bf16_values(&generated_scales(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0005,
        ));
        let up_bias_values = bf16_values(&generated_biases(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0006,
        ));
        let down_weight_values = generated_bytes(num_experts * down.weight_bytes_per_expert(), 0x8300_0007);
        let down_scale_values = bf16_values(&generated_scales(
            num_experts * down.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0008,
        ));
        let down_bias_values = bf16_values(&generated_biases(
            num_experts * down.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0009,
        ));
        Self {
            gate_weight: Buffer::from_slice(device, &gate_weight_values),
            gate_scales: bf16_buffer(device, &gate_scale_values),
            gate_biases: bf16_buffer(device, &gate_bias_values),
            up_weight: Buffer::from_slice(device, &up_weight_values),
            up_scales: bf16_buffer(device, &up_scale_values),
            up_biases: bf16_buffer(device, &up_bias_values),
            down_weight: Buffer::from_slice(device, &down_weight_values),
            down_scales: bf16_buffer(device, &down_scale_values),
            down_biases: bf16_buffer(device, &down_bias_values),
            gate_weight_values,
            gate_scale_values,
            gate_bias_values,
            up_weight_values,
            up_scale_values,
            up_bias_values,
            down_weight_values,
            down_scale_values,
            down_bias_values,
        }
    }

    fn as_borrowed(&self) -> Weights<'_> {
        Weights {
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

    fn as_reference(&self) -> QuantizedSparseMLPReferenceWeights<'_> {
        QuantizedSparseMLPReferenceWeights {
            gate_weight: &self.gate_weight_values,
            gate_scales: &self.gate_scale_values,
            gate_biases: &self.gate_bias_values,
            up_weight: &self.up_weight_values,
            up_scales: &self.up_scale_values,
            up_biases: &self.up_bias_values,
            down_weight: &self.down_weight_values,
            down_scales: &self.down_scale_values,
            down_biases: &self.down_bias_values,
        }
    }
}

fn assert_panics(f: impl FnOnce()) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits = values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &bits)
}

fn assert_bf16_close_rel_values(
    expected: &[f32],
    actual: &Buffer,
    len_bytes: usize,
    abs_tolerance: f32,
    rel_tolerance: f32,
) {
    let actual = actual.read_typed::<u16>(0, len_bytes / size_of::<u16>());
    assert_eq!(expected.len(), actual.len());
    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let mut max_index = 0;
    let mut max_expected = 0.0_f32;
    let mut max_actual = 0.0_f32;
    for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let expected = *expected;
        let actual = bf16::from_bits(*actual).to_f32();
        let diff = (actual - expected).abs();
        let rel_diff = if expected == 0.0 { diff } else { diff / expected.abs() };
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_rel_diff = rel_diff;
            max_index = index;
            max_expected = expected;
            max_actual = actual;
        }
        let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
        assert!(
            diff <= tolerance,
            "fused sparse MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
             tolerance={tolerance} abs_tolerance={abs_tolerance} rel_tolerance={rel_tolerance}"
        );
    }
    eprintln!(
        "fused sparse MLP max_abs_diff={max_abs_diff} max_rel_diff={max_rel_diff} index={max_index} \
         expected={max_expected} actual={max_actual}"
    );
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<f32> {
    (0..num_tokens * hidden_dim)
        .map(|index| ((index * 13 + 5) % 31) as f32 * 0.0625 - 1.0)
        .collect()
}

fn quantized_weight_stack_values(num_experts: usize, bytes_per_expert: usize) -> Vec<u8> {
    let total_bytes = num_experts * bytes_per_expert;
    (0..total_bytes).map(|index| ((index * 13 + 17) & 0xff) as u8).collect()
}

fn identity_indices(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| u32::try_from(index).expect("identity index must fit u32"))
        .collect()
}

fn affine_param_fixture(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.001 + ((index * 3) % 7) as f32 * 0.0001)
        .collect()
}

fn zero_fixture(len: usize) -> Vec<f32> {
    vec![0.0; len]
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn generated_scales(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| 0.0005 + value.abs() * 0.001)
        .collect()
}

fn generated_biases(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| value * 0.0002)
        .collect()
}

fn generated_bytes(count: usize, random_seed: u32) -> Vec<u8> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn generated_indices(count: usize, upper_bound: usize, random_seed: u32) -> Vec<u32> {
    assert!(upper_bound > 0);
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(22_695_477).wrapping_add(1);
            (state as usize % upper_bound) as u32
        })
        .collect()
}

fn generated_probs(num_tokens: usize, num_experts_per_token: usize, random_seed: u32) -> Vec<f32> {
    let mut values = generated_values(num_tokens * num_experts_per_token, random_seed)
        .into_iter()
        .map(|value| value.abs() + 0.05)
        .collect::<Vec<_>>();
    for row in values.chunks_mut(num_experts_per_token) {
        let sum = row.iter().sum::<f32>();
        for value in row {
            *value /= sum;
        }
    }
    values
}
