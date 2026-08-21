use half::bf16;
use inference_executor_core::mlp::moe::reference::moe_combine_with_shared_experts_bf16_reference;
use inference_executor_core::mlp::moe::reference::moe_combine_without_shared_experts_bf16_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.combine.num_active_tokens");

#[test]
fn test_constants_have_explicit_thread_block_scope() {
    let constants = KernelConstants::current();
    assert_eq!(constants.thread_block.required_threads, 256);
}

#[test]
#[should_panic(expected = "MoE combine output elements exceeds the shader u32 count domain")]
fn test_shape_rejects_shader_count_overflow() {
    Config::bf16(1, 4).validate_shape(Shape {
        num_total_tokens: 1 << 30,
    });
}

#[test]
fn test_without_shared_experts_fixed() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config::bf16(2, 3);
    let shape = Shape { num_total_tokens: 2 };
    let routed_hidden_values = [
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        -1.0, 0.5, 2.0, //
        0.25, -0.75, 1.5,
    ];
    let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
    let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
    let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
    let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
    let kernels = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_without_shared_experts(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        WithoutSharedExpertsBuffers {
            routed_hidden: &routed_hidden,
            routed_probs: &routed_probs,
            output: &output,
        },
    ));
    let program = builder.build();
    let submitted = stream.submit_replay(&program);
    submitted.wait();

    let actual = output.read_typed::<u16>(0, 6);
    let expected =
        moe_combine_without_shared_experts_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
    assert_close_bits(&actual, &expected, 1.0e-3);
}

#[test]
fn test_with_shared_experts_fixed() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config::bf16(2, 3);
    let shape = Shape { num_total_tokens: 2 };
    let routed_hidden_values = [
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        -1.0, 0.5, 2.0, //
        0.25, -0.75, 1.5,
    ];
    let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
    let shared_hidden_values = [0.5, 1.0, -2.0, 1.5, -0.5, 0.25];
    let shared_expert_gate_logits_values = [-1.0, 2.0];
    let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
    let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
    let shared_hidden = bf16_buffer(&device, &shared_hidden_values);
    let shared_expert_gate_logits = bf16_buffer(&device, &shared_expert_gate_logits_values);
    let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
    let kernels = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_shared_experts(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        WithSharedExpertsBuffers {
            routed_hidden: &routed_hidden,
            routed_probs: &routed_probs,
            shared_hidden: &shared_hidden,
            shared_expert_gate_logits: &shared_expert_gate_logits,
            output: &output,
        },
    ));
    let program = builder.build();
    let submitted = stream.submit_replay(&program);
    submitted.wait();

    let actual = output.read_typed::<u16>(0, 6);
    let routed =
        moe_combine_without_shared_experts_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
    let expected = moe_combine_with_shared_experts_bf16_reference(
        &routed,
        &shared_hidden_values,
        &shared_expert_gate_logits_values,
        2,
        3,
    );
    assert_close_bits(&actual, &expected, 1.0e-3);
}

#[test]
fn test_with_shared_experts_random() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config::bf16(3, 5);
    let shape = Shape { num_total_tokens: 3 };
    let random_seed = 0xC461_8E2B;
    let routed_hidden_values = generated_values(
        shape.num_total_tokens as usize * config.num_experts_per_token as usize * config.hidden_dim as usize,
        random_seed,
    );
    let routed_probs_values = generated_probs(
        shape.num_total_tokens as usize,
        config.num_experts_per_token as usize,
        random_seed.wrapping_add(1),
    );
    let shared_hidden_values = generated_values(
        shape.num_total_tokens as usize * config.hidden_dim as usize,
        random_seed.wrapping_add(2),
    );
    let shared_expert_gate_logits_values =
        generated_values(shape.num_total_tokens as usize, random_seed.wrapping_add(3));
    let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
    let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
    let shared_hidden = bf16_buffer(&device, &shared_hidden_values);
    let shared_expert_gate_logits = bf16_buffer(&device, &shared_expert_gate_logits_values);
    let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
    let kernels = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_shared_experts(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        WithSharedExpertsBuffers {
            routed_hidden: &routed_hidden,
            routed_probs: &routed_probs,
            shared_hidden: &shared_hidden,
            shared_expert_gate_logits: &shared_expert_gate_logits,
            output: &output,
        },
    ));
    let program = builder.build();
    stream.submit_replay(&program).wait();

    let routed = moe_combine_without_shared_experts_bf16_reference(
        &routed_hidden_values,
        &routed_probs_values,
        shape.num_total_tokens as usize,
        config.num_experts_per_token as usize,
        config.hidden_dim as usize,
    );
    let expected = moe_combine_with_shared_experts_bf16_reference(
        &routed,
        &shared_hidden_values,
        &shared_expert_gate_logits_values,
        shape.num_total_tokens as usize,
        config.hidden_dim as usize,
    );
    let actual = output.read_typed::<u16>(0, shape.num_total_tokens as usize * config.hidden_dim as usize);
    assert_close_bits(&actual, &expected, 1.0e-3);
}

#[test]
fn test_bucketed_capacity_is_reusable_with_and_without_shared_experts() {
    for with_shared_experts in [false, true] {
        run_bucketed_capacity_case(with_shared_experts);
    }
}

fn run_bucketed_capacity_case(with_shared_experts: bool) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config::bf16(2, 3);
    let shape = Shape { num_total_tokens: 4 };
    let num_total_tokens = shape.num_total_tokens as usize;
    let num_active_tokens = 3_usize;
    let topk = config.num_experts_per_token as usize;
    let hidden_dim = config.hidden_dim as usize;
    let all_routed_hidden = generated_values(num_total_tokens * topk * hidden_dim, 0x2148_937A);
    let all_routed_probs = generated_probs(num_total_tokens, topk, 0x672D_A9B4);
    let all_shared_hidden = generated_values(num_total_tokens * hidden_dim, 0x153F_72C8);
    let all_gate_logits = generated_values(num_total_tokens, 0xB307_4D16);
    let active_routes = num_active_tokens * topk;
    let active_routed_values = active_routes * hidden_dim;
    let active_output_values = num_active_tokens * hidden_dim;
    let mut active_routed_hidden = all_routed_hidden.clone();
    active_routed_hidden[active_routed_values..].fill(f32::NAN);
    let mut active_routed_probs = all_routed_probs.clone();
    active_routed_probs[active_routes..].fill(f32::NAN);
    let mut active_shared_hidden = all_shared_hidden.clone();
    active_shared_hidden[active_output_values..].fill(f32::NAN);
    let mut active_gate_logits = all_gate_logits.clone();
    active_gate_logits[num_active_tokens..].fill(f32::NAN);

    let routed_hidden = bf16_buffer(&device, &active_routed_hidden);
    let routed_probs = Buffer::from_slice(&device, &active_routed_probs);
    let shared_hidden = bf16_buffer(&device, &active_shared_hidden);
    let gate_logits = bf16_buffer(&device, &active_gate_logits);
    let output_sentinel = bf16::from_f32(91.0).to_bits();
    let output = Buffer::from_slice(&device, &vec![output_sentinel; num_total_tokens * hidden_dim]);
    let kernels = Compute::new(&device, config);
    let mut builder = stream.create_replay_program();
    if with_shared_experts {
        builder.record(kernels.invoke_with_shared_experts(
            shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            WithSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &gate_logits,
                output: &output,
            },
        ));
    } else {
        builder.record(kernels.invoke_without_shared_experts(
            shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            WithoutSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
    }
    let replay = builder.build();

    stream
        .submit_replay_with_arguments(
            &replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
        )
        .wait();
    let expected_active = expected_output(
        &all_routed_hidden,
        &all_routed_probs,
        &all_shared_hidden,
        &all_gate_logits,
        num_active_tokens,
        topk,
        hidden_dim,
        with_shared_experts,
    );
    let first = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
    assert_close_bits(&first[..active_output_values], &expected_active, 1.0e-3);
    assert_eq!(
        &first[active_output_values..],
        &vec![output_sentinel; hidden_dim],
        "inactive output tail must preserve its canary"
    );

    write_bf16_values(&routed_hidden, &all_routed_hidden);
    routed_probs.write_typed(0, &all_routed_probs);
    write_bf16_values(&shared_hidden, &all_shared_hidden);
    write_bf16_values(&gate_logits, &all_gate_logits);
    stream
        .submit_replay_with_arguments(
            &replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, shape.num_total_tokens),
        )
        .wait();
    let expected_full = expected_output(
        &all_routed_hidden,
        &all_routed_probs,
        &all_shared_hidden,
        &all_gate_logits,
        num_total_tokens,
        topk,
        hidden_dim,
        with_shared_experts,
    );
    let full = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
    assert_close_bits(&full, &expected_full, 1.0e-3);

    write_bf16_values(&routed_hidden, &active_routed_hidden);
    routed_probs.write_typed(0, &active_routed_probs);
    write_bf16_values(&shared_hidden, &active_shared_hidden);
    write_bf16_values(&gate_logits, &active_gate_logits);
    stream
        .submit_replay_with_arguments(
            &replay,
            &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
        )
        .wait();
    let shrunk = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
    assert_close_bits(&shrunk[..active_output_values], &expected_active, 1.0e-3);
    assert_eq!(
        &shrunk[active_output_values..],
        &full[active_output_values..],
        "shrinking the active prefix must not rewrite the previous full tail"
    );
}

#[allow(clippy::too_many_arguments)]
fn expected_output(
    routed_hidden: &[f32],
    routed_probs: &[f32],
    shared_hidden: &[f32],
    gate_logits: &[f32],
    num_tokens: usize,
    topk: usize,
    hidden_dim: usize,
    with_shared_experts: bool,
) -> Vec<u16> {
    let num_routes = num_tokens * topk;
    let routed = moe_combine_without_shared_experts_bf16_reference(
        &routed_hidden[..num_routes * hidden_dim],
        &routed_probs[..num_routes],
        num_tokens,
        topk,
        hidden_dim,
    );
    if with_shared_experts {
        moe_combine_with_shared_experts_bf16_reference(
            &routed,
            &shared_hidden[..num_tokens * hidden_dim],
            &gate_logits[..num_tokens],
            num_tokens,
            hidden_dim,
        )
    } else {
        routed
    }
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    Buffer::from_slice(device, &bits)
}

fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    buffer.write_typed(0, &bits);
}

fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 8_388_608.0) - 1.0
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

fn assert_close_bits(actual: &[u16], expected: &[u16], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let actual = bf16::from_bits(actual).to_f32();
        let expected = bf16::from_bits(expected).to_f32();
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
