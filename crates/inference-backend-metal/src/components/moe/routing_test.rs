use half::bf16;
use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.routing.num_active_tokens");

#[test]
fn test_constants_have_explicit_thread_block_scope() {
    let constants = KernelConstants::current();
    assert_eq!(constants.thread_block.required_threads, 256);
}

#[test]
#[should_panic(expected = "MoE routing supports at most 256 experts")]
fn test_config_rejects_more_than_256_experts() {
    Config {
        num_experts: 257,
        num_experts_per_token: 1,
        norm_topk_prob: false,
    }
    .validate();
}

#[test]
#[should_panic(expected = "MoE routing supports at most 16 experts per token")]
fn test_config_rejects_more_than_16_experts_per_token() {
    Config {
        num_experts: 256,
        num_experts_per_token: 17,
        norm_topk_prob: false,
    }
    .validate();
}

#[test]
#[should_panic(expected = "MoE routing probability elements exceeds the shader u32 element-index domain")]
fn test_shape_rejects_shader_index_overflow() {
    let config = Config {
        num_experts: 256,
        num_experts_per_token: 1,
        norm_topk_prob: false,
    };
    config.validate_shape(Shape {
        num_total_tokens: (u32::MAX / 256) + 2,
    });
}

#[test]
fn test_topk_renorm() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 4,
        num_experts_per_token: 2,
        norm_topk_prob: true,
    };
    let shape = Shape { num_total_tokens: 2 };
    let router_probs_values = [
        softmax_prob(0.25, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(2.0, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(-1.0, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(1.0, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(3.0, &[3.0, 3.0, 0.5, -2.0]),
        softmax_prob(3.0, &[3.0, 3.0, 0.5, -2.0]),
        softmax_prob(0.5, &[3.0, 3.0, 0.5, -2.0]),
        softmax_prob(-2.0, &[3.0, 3.0, 0.5, -2.0]),
    ];
    let router_probs = bf16_buffer(&device, &router_probs_values);
    let expert_indices = Buffer::new_zeroed(&device, config.expert_indices_bytes(shape));
    let expert_probs = Buffer::new_zeroed(&device, config.expert_probs_bytes(shape));
    let kernel = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        Buffers {
            router_probs: &router_probs,
            expert_indices: &expert_indices,
            expert_probs: &expert_probs,
        },
    ));
    let program = builder.build();
    let submitted = stream.submit_replay(&program);
    submitted.wait();

    let expected = moe_routing_from_bf16_probs_reference(&router_probs_values, 2, 4, 2, true);
    assert_eq!(expert_indices.read_typed::<u32>(0, 4), expected.expert_indices);
    let actual = expert_probs.read_typed::<f32>(0, 4);
    assert_close(&actual, &expected.expert_probs, 1.0e-3);
}

#[test]
fn test_no_topk_renorm() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 4,
        num_experts_per_token: 2,
        norm_topk_prob: false,
    };
    let shape = Shape { num_total_tokens: 1 };
    let router_probs_values = [
        softmax_prob(0.25, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(2.0, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(-1.0, &[0.25, 2.0, -1.0, 1.0]),
        softmax_prob(1.0, &[0.25, 2.0, -1.0, 1.0]),
    ];
    let router_probs = bf16_buffer(&device, &router_probs_values);
    let expert_indices = Buffer::new_zeroed(&device, config.expert_indices_bytes(shape));
    let expert_probs = Buffer::new_zeroed(&device, config.expert_probs_bytes(shape));
    let kernel = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        Buffers {
            router_probs: &router_probs,
            expert_indices: &expert_indices,
            expert_probs: &expert_probs,
        },
    ));
    let program = builder.build();
    let submitted = stream.submit_replay(&program);
    submitted.wait();

    let expected = moe_routing_from_bf16_probs_reference(&router_probs_values, 1, 4, 2, false);
    assert_eq!(expert_indices.read_typed::<u32>(0, 2), expected.expert_indices);
    let actual = expert_probs.read_typed::<f32>(0, 2);
    assert_close(&actual, &expected.expert_probs, 1.0e-6);
}

#[test]
fn test_bucketed_replay_reuses_capacity_with_both_norm_modes() {
    for norm_topk_prob in [false, true] {
        run_bucketed_replay_case(norm_topk_prob);
    }
}

#[test]
fn test_random() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 8,
        num_experts_per_token: 3,
        norm_topk_prob: true,
    };
    let shape = Shape { num_total_tokens: 5 };
    let random_seed = 0x91E4_63BA;
    let router_probs_values = generated_probs(
        shape.num_total_tokens as usize,
        config.num_experts as usize,
        random_seed,
    );
    let router_probs = bf16_buffer(&device, &router_probs_values);
    let expert_indices = Buffer::new_zeroed(&device, config.expert_indices_bytes(shape));
    let expert_probs = Buffer::new_zeroed(&device, config.expert_probs_bytes(shape));
    let kernel = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        Buffers {
            router_probs: &router_probs,
            expert_indices: &expert_indices,
            expert_probs: &expert_probs,
        },
    ));
    let program = builder.build();
    stream.submit_replay(&program).wait();

    let expected = moe_routing_from_bf16_probs_reference(
        &router_probs_values,
        shape.num_total_tokens as usize,
        config.num_experts as usize,
        config.num_experts_per_token as usize,
        config.norm_topk_prob,
    );
    let actual_probs = expert_probs.read_typed::<f32>(0, config.num_routes(shape));
    assert_eq!(
        expert_indices.read_typed::<u32>(0, config.num_routes(shape)),
        expected.expert_indices
    );
    assert_close(&actual_probs, &expected.expert_probs, 1.0e-3);
}

fn softmax_prob(logit: f32, all_logits: &[f32]) -> f32 {
    let all_logits: Vec<f32> = all_logits.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
    let max_logit = all_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let all_exp_sum: f32 = all_logits.iter().map(|value| (*value - max_logit).exp()).sum();
    bf16::from_f32((bf16::from_f32(logit).to_f32() - max_logit).exp() / all_exp_sum).to_f32()
}

fn generated_probs(num_tokens: usize, num_experts: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    let mut probs = Vec::with_capacity(num_tokens * num_experts);
    for _ in 0..num_tokens {
        let mut row = Vec::with_capacity(num_experts);
        let mut sum = 0.0f32;
        for _ in 0..num_experts {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = ((state >> 8) as f32 / 16_777_216.0) + 0.01;
            row.push(value);
            sum += value;
        }
        probs.extend(row.into_iter().map(|value| value / sum));
    }
    probs
}

fn run_bucketed_replay_case(norm_topk_prob: bool) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_experts: 8,
        num_experts_per_token: 3,
        norm_topk_prob,
    };
    let shape = Shape { num_total_tokens: 4 };
    let all_probs = generated_probs(4, config.num_experts as usize, 0x5A17_920B);
    let mut three_token_probs = all_probs.clone();
    three_token_probs[3 * config.num_experts as usize..].fill(f32::NAN);
    let router_probs = bf16_buffer(&device, &three_token_probs);
    let num_routes = config.num_routes(shape);
    let index_sentinel = 0xDEAD_BEEF_u32;
    let prob_sentinel = -777.0_f32;
    let expert_indices = Buffer::from_slice(&device, &vec![index_sentinel; num_routes]);
    let expert_probs = Buffer::from_slice(&device, &vec![prob_sentinel; num_routes]);
    let kernel = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        Buffers {
            router_probs: &router_probs,
            expert_indices: &expert_indices,
            expert_probs: &expert_probs,
        },
    ));
    let replay = builder.build();

    stream
        .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 3))
        .wait();
    let active_routes = 3 * config.num_experts_per_token as usize;
    let expected_three = moe_routing_from_bf16_probs_reference(
        &all_probs[..3 * config.num_experts as usize],
        3,
        config.num_experts as usize,
        config.num_experts_per_token as usize,
        norm_topk_prob,
    );
    let first_indices = expert_indices.read_typed::<u32>(0, num_routes);
    let first_probs = expert_probs.read_typed::<f32>(0, num_routes);
    assert_eq!(&first_indices[..active_routes], expected_three.expert_indices);
    assert_close(&first_probs[..active_routes], &expected_three.expert_probs, 1.0e-3);
    assert_eq!(
        &first_indices[active_routes..],
        &vec![index_sentinel; num_routes - active_routes]
    );
    assert_eq!(
        &first_probs[active_routes..],
        &vec![prob_sentinel; num_routes - active_routes]
    );

    write_bf16_values(&router_probs, &all_probs);
    stream
        .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 4))
        .wait();
    let expected_four = moe_routing_from_bf16_probs_reference(
        &all_probs,
        4,
        config.num_experts as usize,
        config.num_experts_per_token as usize,
        norm_topk_prob,
    );
    let full_indices = expert_indices.read_typed::<u32>(0, num_routes);
    let full_probs = expert_probs.read_typed::<f32>(0, num_routes);
    assert_eq!(full_indices, expected_four.expert_indices);
    assert_close(&full_probs, &expected_four.expert_probs, 1.0e-3);

    write_bf16_values(&router_probs, &three_token_probs);
    stream
        .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 3))
        .wait();
    let shrunk_indices = expert_indices.read_typed::<u32>(0, num_routes);
    let shrunk_probs = expert_probs.read_typed::<f32>(0, num_routes);
    assert_eq!(&shrunk_indices[..active_routes], expected_three.expert_indices);
    assert_close(&shrunk_probs[..active_routes], &expected_three.expert_probs, 1.0e-3);
    assert_eq!(&shrunk_indices[active_routes..], &full_indices[active_routes..]);
    assert_eq!(&shrunk_probs[active_routes..], &full_probs[active_routes..]);
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    Buffer::from_slice(device, &bits)
}

fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    buffer.write_typed(0, &bits);
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
