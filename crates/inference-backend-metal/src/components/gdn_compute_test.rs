use std::mem::size_of;

use half::bf16;
use inference_executor_core::attn::gdn::GDNCore;
use inference_executor_core::attn::gdn::reference::GDNRecurrentReferenceInput;
use inference_executor_core::attn::gdn::reference::gdn_recurrent_reference;
use inference_executor_core::attn::gdn::reference::gdn_short_conv_reference;

use super::GDNCompute;
use super::GDNComputeBuffers;
use super::GDNComputeConfig;
use super::GDNComputeShape;
use super::selected_v_dim_tile_size;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::metal::Stream;

const NUM_ACTIVE_REQUESTS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_compute.num_active_requests");
const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_compute.num_active_tokens");

#[test]
fn test_recurrent_v_dim_tile_selection() {
    let mut config = fixture_config();
    assert_eq!(selected_v_dim_tile_size(config), 8);
    config.v_head_dim = 12;
    assert_eq!(selected_v_dim_tile_size(config), 4);
    config.v_head_dim = 6;
    assert_eq!(selected_v_dim_tile_size(config), 2);
    config.v_head_dim = 3;
    assert_eq!(selected_v_dim_tile_size(config), 1);
}

#[test]
#[should_panic(expected = "GDN convolution exceeds the shader u32 count domain")]
fn test_shape_rejects_shader_count_overflow() {
    let config = GDNComputeConfig {
        num_qk_heads: 1,
        qk_head_dim: 1,
        num_v_heads: 1,
        v_head_dim: 2,
        conv_kernel_size: 2,
        q_scale: 1.0,
        norm_eps: 1.0e-6,
    };
    let shape = GDNComputeShape {
        num_reqs: 1,
        num_tokens: 1 << 30,
    };
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = GDNCompute::new(&device, config);
    let buffer = Buffer::new_zeroed(&device, size_of::<f32>());
    let mut builder = stream.create_replay_program();

    builder.record(kernels.invoke(
        shape,
        GDNComputeBuffers {
            qkv: &buffer,
            a: &buffer,
            b: &buffer,
            z: &buffer,
            conv_weight: &buffer,
            norm_weight: &buffer,
            a_log: &buffer,
            dt_bias: &buffer,
            cu_tokens: &buffer,
            src_state_slots: &buffer,
            flat_materialized_state_slots: &buffer,
            conv_state: &buffer,
            conv_state_offset_bytes: 0,
            next_conv_state: &buffer,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &buffer,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &buffer,
            recurrent_output: &buffer,
            norm_gated_output: &buffer,
        },
    ));
}

#[test]
fn test_ragged_recurrent_fixed() {
    let shape = fixture_shape(1, 1);
    let cu_tokens = vec![0_u32, 1];
    let src_state_slots = vec![0_u32];
    let dst_slot_ids = vec![0_u32];
    let qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 3);
    let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 7);
    let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 11);
    let conv_weight = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        13,
    );
    let a = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        17,
    );
    let b = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        19,
    );
    let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 23);
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        2.0e-5,
    );
}

#[test]
fn test_ragged_recurrent_random() {
    let random_seed = 0x729B_40D6;
    let shape = fixture_shape(1, 3);
    let cu_tokens = vec![0_u32, 3];
    let src_state_slots = vec![0_u32];
    let dst_slot_ids = vec![0_u32];
    let qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
    let conv_state = generated_values(
        fixture_config().num_conv_state_values(shape),
        random_seed.wrapping_add(1),
    );
    let recurrent_state = generated_values(fixture_config().recurrent_state_stride(), random_seed.wrapping_add(2));
    let conv_weight = generated_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        random_seed.wrapping_add(3),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value * 0.125).to_f32())
    .collect::<Vec<_>>();
    let a = generated_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(4),
    );
    let b = generated_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(5),
    );
    let z = generated_values(
        fixture_config().num_recurrent_output_values(shape),
        random_seed.wrapping_add(6),
    );
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.125_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        4.0e-5,
    );
}

#[test]
fn test_ragged_multi_random() {
    let random_seed = 0xE13C_58A4;
    let shape = fixture_shape(2, 4);
    let cu_tokens = vec![0_u32, 1, 4];
    let src_state_slots = vec![0_u32, 1];
    let dst_slot_ids = vec![0_u32, 1];
    let qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
    let conv_state = generated_values(
        fixture_config().num_conv_state_values(shape),
        random_seed.wrapping_add(1),
    );
    let recurrent_state = generated_values(
        shape.num_reqs as usize * fixture_config().recurrent_state_stride(),
        random_seed.wrapping_add(2),
    );
    let conv_weight = generated_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        random_seed.wrapping_add(3),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value * 0.125).to_f32())
    .collect::<Vec<_>>();
    let a = generated_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(4),
    );
    let b = generated_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(5),
    );
    let z = generated_values(
        fixture_config().num_recurrent_output_values(shape),
        random_seed.wrapping_add(6),
    );
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.125_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        4.0e-5,
    );
}

#[test]
fn test_bucketed_candidate_replay_guards_poisoned_tails_and_optional_final_state() {
    const NUM_STATE_SLOTS: usize = 6;
    const CANARY: f32 = -777.0;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = fixture_config();
    let shape = fixture_shape(2, 2);
    let kernels = GDNCompute::new(&device, config);
    let qkv_values = fixture_values(config.num_qkv_values(shape), 0.03125, 3);
    let a_values = fixture_values(shape.num_tokens as usize * config.num_v_heads as usize, 0.0625, 5);
    let b_values = fixture_values(shape.num_tokens as usize * config.num_v_heads as usize, 0.0625, 7);
    let z_values = fixture_values(config.num_recurrent_output_values(shape), 0.03125, 11);
    let conv_weight_values = fixture_values(
        config.qkv_dim() as usize * config.conv_kernel_size as usize,
        0.00390625,
        13,
    );
    let norm_weight_values = vec![1.0_f32; config.v_head_dim as usize];
    let a_log_values = vec![-0.25_f32; config.num_v_heads as usize];
    let dt_bias_values = vec![0.125_f32; config.num_v_heads as usize];
    let conv_state_stride = config.qkv_dim() as usize * config.conv_state_len() as usize;
    let recurrent_state_stride = config.recurrent_state_stride();
    let source_conv_states = fixture_values(2 * conv_state_stride, 0.015625, 17);
    let source_recurrent_states = fixture_values(2 * recurrent_state_stride, 0.0078125, 19);

    let qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Float32);
    let a = Buffer::new_zeroed_elements(
        &device,
        shape.num_tokens as usize * config.num_v_heads as usize,
        Dtype::Float32,
    );
    let b = Buffer::new_zeroed_elements(
        &device,
        shape.num_tokens as usize * config.num_v_heads as usize,
        Dtype::Float32,
    );
    let z = Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Float32);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::new_zeroed_elements(&device, 3, Dtype::Uint32);
    let src_state_slots = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
    let flat_materialized_state_slots = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
    let conv_state_arena = Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * conv_state_stride, Dtype::Float32);
    let recurrent_state_arena =
        Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * recurrent_state_stride, Dtype::Float32);
    let conv_qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Float32);
    let recurrent_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Float32);
    let norm_gated_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Float32);
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_candidate_state_update_bucketed(
        shape,
        GDNComputeBuffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_state_slots: &src_state_slots,
            flat_materialized_state_slots: &flat_materialized_state_slots,
            conv_state: &conv_state_arena,
            conv_state_offset_bytes: 0,
            next_conv_state: &conv_state_arena,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        },
        ReplayU32::Parameter(NUM_ACTIVE_REQUESTS),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
    ));
    let replay = builder.build();
    assert_eq!(replay.stats().parameter_count, 2);

    for num_active in [1_usize, 2, 1] {
        let mut qkv_submission = qkv_values.clone();
        let mut a_submission = a_values.clone();
        let mut b_submission = b_values.clone();
        let mut z_submission = z_values.clone();
        if num_active == 1 {
            qkv_submission[config.qkv_dim() as usize..].fill(f32::NAN);
            a_submission[1..].fill(f32::NAN);
            b_submission[1..].fill(f32::NAN);
            z_submission[config.v_head_dim as usize..].fill(f32::NAN);
        }
        qkv.write_typed(0, &qkv_submission);
        a.write_typed(0, &a_submission);
        b.write_typed(0, &b_submission);
        z.write_typed(0, &z_submission);
        let metadata = if num_active == 1 {
            ([0_u32, 1, u32::MAX], [0_u32, u32::MAX], [u32::MAX, u32::MAX])
        } else {
            ([0_u32, 1, 2], [0_u32, 3], [2_u32, 5])
        };
        cu_tokens.write_typed(0, &metadata.0);
        src_state_slots.write_typed(0, &metadata.1);
        flat_materialized_state_slots.write_typed(0, &metadata.2);

        let mut conv_arena_values = vec![CANARY; NUM_STATE_SLOTS * conv_state_stride];
        conv_arena_values[..conv_state_stride].copy_from_slice(&source_conv_states[..conv_state_stride]);
        conv_arena_values[3 * conv_state_stride..4 * conv_state_stride]
            .copy_from_slice(&source_conv_states[conv_state_stride..]);
        conv_state_arena.write_typed(0, &conv_arena_values);
        let mut recurrent_arena_values = vec![CANARY; NUM_STATE_SLOTS * recurrent_state_stride];
        recurrent_arena_values[..recurrent_state_stride]
            .copy_from_slice(&source_recurrent_states[..recurrent_state_stride]);
        recurrent_arena_values[3 * recurrent_state_stride..4 * recurrent_state_stride]
            .copy_from_slice(&source_recurrent_states[recurrent_state_stride..]);
        recurrent_state_arena.write_typed(0, &recurrent_arena_values);
        conv_qkv.write_typed(0, &vec![CANARY; config.num_qkv_values(shape)]);
        recurrent_output.write_typed(0, &vec![CANARY; config.num_recurrent_output_values(shape)]);
        norm_gated_output.write_typed(0, &vec![CANARY; config.num_recurrent_output_values(shape)]);

        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_REQUESTS, num_active as u32)
                    .with_u32(NUM_ACTIVE_TOKENS, num_active as u32),
            )
            .wait();

        let active_shape = fixture_shape(num_active as u32, num_active as u32);
        let active_cu_tokens = (0..=num_active as u32).collect::<Vec<_>>();
        let active_qkv_values = &qkv_values[..config.num_qkv_values(active_shape)];
        let active_a_values = &a_values[..num_active * config.num_v_heads as usize];
        let active_b_values = &b_values[..num_active * config.num_v_heads as usize];
        let active_z_values = &z_values[..config.num_recurrent_output_values(active_shape)];
        let conv_reference = gdn_short_conv_reference(
            &fixture_core(active_shape),
            &active_cu_tokens,
            &source_conv_states[..num_active * conv_state_stride],
            active_qkv_values,
            &conv_weight_values,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &fixture_core(active_shape),
            GDNRecurrentReferenceInput {
                cu_tokens: &active_cu_tokens,
                source_recurrent_state: &source_recurrent_states[..num_active * recurrent_state_stride],
                conv_qkv: &conv_reference.conv_qkv,
                a: active_a_values,
                b: active_b_values,
                a_log: &a_log_values,
                dt_bias: &dt_bias_values,
            },
        );
        let norm_gate_reference = output_norm_gate_reference(
            &recurrent_reference.recurrent_output,
            active_z_values,
            &norm_weight_values,
            config,
        );
        assert_close(
            &conv_qkv.read_typed::<f32>(0, config.num_qkv_values(active_shape)),
            &conv_reference.conv_qkv,
            2.0e-5,
        );
        assert_close(
            &recurrent_output.read_typed::<f32>(0, config.num_recurrent_output_values(active_shape)),
            &recurrent_reference.recurrent_output,
            2.0e-5,
        );
        assert_close(
            &norm_gated_output.read_typed::<f32>(0, config.num_recurrent_output_values(active_shape)),
            &norm_gate_reference,
            2.0e-5,
        );
        let active_qkv_count = config.num_qkv_values(active_shape);
        assert_eq!(
            conv_qkv.read_typed::<f32>(active_qkv_count, config.num_qkv_values(shape) - active_qkv_count),
            vec![CANARY; config.num_qkv_values(shape) - active_qkv_count]
        );
        let active_output_count = config.num_recurrent_output_values(active_shape);
        for output in [&recurrent_output, &norm_gated_output] {
            assert_eq!(
                output.read_typed::<f32>(
                    active_output_count,
                    config.num_recurrent_output_values(shape) - active_output_count,
                ),
                vec![CANARY; config.num_recurrent_output_values(shape) - active_output_count]
            );
        }

        let conv_state_after = conv_state_arena.read_typed::<f32>(0, NUM_STATE_SLOTS * conv_state_stride);
        let recurrent_state_after =
            recurrent_state_arena.read_typed::<f32>(0, NUM_STATE_SLOTS * recurrent_state_stride);
        for req_index in 0..num_active {
            let dst_slot = [1_usize, 4][req_index];
            let candidate_slot = [2_usize, 5][req_index];
            let conv_expected =
                &conv_reference.next_conv_state[req_index * conv_state_stride..(req_index + 1) * conv_state_stride];
            let recurrent_expected = &recurrent_reference.next_recurrent_state
                [req_index * recurrent_state_stride..(req_index + 1) * recurrent_state_stride];
            if num_active == 1 {
                assert_eq!(
                    conv_state_slot(&conv_state_after, shape, dst_slot),
                    vec![CANARY; conv_state_stride]
                );
                assert_eq!(
                    recurrent_state_slot(&recurrent_state_after, shape, dst_slot),
                    vec![CANARY; recurrent_state_stride]
                );
                assert_eq!(
                    conv_state_slot(&conv_state_after, shape, candidate_slot),
                    vec![CANARY; conv_state_stride]
                );
                assert_eq!(
                    recurrent_state_slot(&recurrent_state_after, shape, candidate_slot),
                    vec![CANARY; recurrent_state_stride]
                );
            } else {
                assert_close(
                    conv_state_slot(&conv_state_after, shape, candidate_slot),
                    conv_expected,
                    2.0e-5,
                );
                assert_close(
                    recurrent_state_slot(&recurrent_state_after, shape, candidate_slot),
                    recurrent_expected,
                    2.0e-5,
                );
                assert_eq!(
                    conv_state_slot(&conv_state_after, shape, dst_slot),
                    vec![CANARY; conv_state_stride]
                );
                assert_eq!(
                    recurrent_state_slot(&recurrent_state_after, shape, dst_slot),
                    vec![CANARY; recurrent_state_stride]
                );
            }
        }
        if num_active == 1 {
            assert_eq!(
                conv_state_slot(&conv_state_after, shape, 4),
                vec![CANARY; conv_state_stride]
            );
            assert_eq!(
                conv_state_slot(&conv_state_after, shape, 5),
                vec![CANARY; conv_state_stride]
            );
            assert_eq!(
                recurrent_state_slot(&recurrent_state_after, shape, 4),
                vec![CANARY; recurrent_state_stride]
            );
            assert_eq!(
                recurrent_state_slot(&recurrent_state_after, shape, 5),
                vec![CANARY; recurrent_state_stride]
            );
        }
    }
}

#[test]
fn test_candidate_state_prefixes() {
    let shape = fixture_shape(1, 3);
    let cu_tokens = vec![0_u32, 3];
    let src_state_slots = vec![0_u32];
    let candidate_dst_slot_ids = vec![1_u32, 2, 3];
    let qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
    let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
    let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
    let conv_weight = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        41,
    );
    let a = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        43,
    );
    let b = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        47,
    );
    let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_forward_candidate_state(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &candidate_dst_slot_ids,
        5,
    );

    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual.full,
        2.0e-5,
    );
    let core = fixture_core(shape);
    for verified_tokens in 1..=shape.num_tokens as usize {
        let prefix_cu_tokens = [0_u32, verified_tokens as u32];
        let conv_reference = gdn_short_conv_reference(
            &core,
            &prefix_cu_tokens,
            &conv_state,
            &qkv[..verified_tokens * fixture_config().qkv_dim() as usize],
            &conv_weight,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &prefix_cu_tokens,
                source_recurrent_state: &recurrent_state,
                conv_qkv: &conv_reference.conv_qkv,
                a: &a[..verified_tokens * fixture_config().num_v_heads as usize],
                b: &b[..verified_tokens * fixture_config().num_v_heads as usize],
                a_log: &a_log,
                dt_bias: &dt_bias,
            },
        );
        let slot = candidate_dst_slot_ids[verified_tokens - 1] as usize;
        assert_close(
            conv_state_slot(&actual.next_conv_state_arena, shape, slot),
            &conv_reference.next_conv_state,
            2.0e-5,
        );
        assert_close(
            recurrent_state_slot(&actual.recurrent_state_arena, shape, slot),
            &recurrent_reference.next_recurrent_state,
            2.0e-5,
        );
    }
}

#[test]
fn test_candidate_states_above_u32_byte_offset() {
    let shape = fixture_shape(1, 3);
    let num_state_slots = 5usize;
    let cu_tokens_values = [0_u32, 3];
    let src_state_slot_values = [0_u32];
    let candidate_dst_slot_id_values = [1_u32, 2, 3];
    let qkv_values = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
    let conv_state_values = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
    let recurrent_state_values = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
    let conv_weight_values = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        41,
    );
    let a_values = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        43,
    );
    let b_values = fixture_values(
        shape.num_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        47,
    );
    let z_values = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
    let norm_weight_values = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log_values = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias_values = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = GDNCompute::new(&device, fixture_config());
    let qkv = Buffer::from_slice(&device, &qkv_values);
    let a = Buffer::from_slice(&device, &a_values);
    let b = Buffer::from_slice(&device, &b_values);
    let z = Buffer::from_slice(&device, &z_values);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, &cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, &src_state_slot_values);
    let candidate_dst_slot_ids = Buffer::from_slice(&device, &candidate_dst_slot_id_values);

    let high_base = u64::from(u32::MAX) + 1;
    let conv_state_offset_bytes = high_base;
    let next_conv_state_offset_bytes = high_base + 4096;
    let recurrent_state_offset_bytes = high_base + 8192;
    let recurrent_arena_bytes = u64::try_from(num_state_slots)
        .expect("test state-slot count must fit u64")
        .checked_mul(
            u64::try_from(
                fixture_config()
                    .recurrent_state_stride()
                    .checked_mul(size_of::<f32>())
                    .expect("test recurrent-state byte stride must fit usize"),
            )
            .expect("test recurrent-state byte stride must fit u64"),
        )
        .expect("test recurrent-state arena byte length must fit u64");
    let state_arena = Buffer::new_uninit(
        &device,
        recurrent_state_offset_bytes
            .checked_add(recurrent_arena_bytes)
            .expect("test state arena byte length must fit u64"),
    );
    state_arena.write_typed(
        usize::try_from(conv_state_offset_bytes / size_of::<f32>() as u64)
            .expect("test convolution state offset must fit usize"),
        &conv_state_values,
    );
    state_arena.write_typed(
        usize::try_from(recurrent_state_offset_bytes / size_of::<f32>() as u64)
            .expect("test recurrent state offset must fit usize"),
        &recurrent_state_values,
    );

    let conv_qkv = Buffer::new_zeroed_elements(&device, fixture_config().num_qkv_values(shape), Dtype::Float32);
    let recurrent_output = Buffer::new_zeroed_elements(
        &device,
        fixture_config().num_recurrent_output_values(shape),
        Dtype::Float32,
    );
    let norm_gated_output = Buffer::new_zeroed_elements(
        &device,
        fixture_config().num_recurrent_output_values(shape),
        Dtype::Float32,
    );
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_candidate_state_update(
        shape,
        GDNComputeBuffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_state_slots: &src_state_slots,
            flat_materialized_state_slots: &candidate_dst_slot_ids,
            conv_state: &state_arena,
            conv_state_offset_bytes,
            next_conv_state: &state_arena,
            next_conv_state_offset_bytes,
            recurrent_state_arena: &state_arena,
            recurrent_state_arena_offset_bytes: recurrent_state_offset_bytes,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        },
    ));
    stream.submit_replay(&builder.build()).wait();

    let next_conv_state_arena = state_arena.read_typed::<f32>(
        usize::try_from(next_conv_state_offset_bytes / size_of::<f32>() as u64)
            .expect("test next convolution state offset must fit usize"),
        num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
    );
    let recurrent_state_arena = state_arena.read_typed::<f32>(
        usize::try_from(recurrent_state_offset_bytes / size_of::<f32>() as u64)
            .expect("test recurrent state offset must fit usize"),
        num_state_slots * fixture_config().recurrent_state_stride(),
    );
    let core = fixture_core(shape);
    for verified_tokens in 1..=shape.num_tokens as usize {
        let prefix_cu_tokens = [0_u32, verified_tokens as u32];
        let conv_reference = gdn_short_conv_reference(
            &core,
            &prefix_cu_tokens,
            &conv_state_values,
            &qkv_values[..verified_tokens * fixture_config().qkv_dim() as usize],
            &conv_weight_values,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &prefix_cu_tokens,
                source_recurrent_state: &recurrent_state_values,
                conv_qkv: &conv_reference.conv_qkv,
                a: &a_values[..verified_tokens * fixture_config().num_v_heads as usize],
                b: &b_values[..verified_tokens * fixture_config().num_v_heads as usize],
                a_log: &a_log_values,
                dt_bias: &dt_bias_values,
            },
        );
        let slot = candidate_dst_slot_id_values[verified_tokens - 1] as usize;
        assert_close(
            conv_state_slot(&next_conv_state_arena, shape, slot),
            &conv_reference.next_conv_state,
            2.0e-5,
        );
        assert_close(
            recurrent_state_slot(&recurrent_state_arena, shape, slot),
            &recurrent_reference.next_recurrent_state,
            2.0e-5,
        );
    }
}

struct GDNCoreOutputs {
    conv_qkv: Vec<f32>,
    next_conv_state: Vec<f32>,
    recurrent_output: Vec<f32>,
    recurrent_state: Vec<f32>,
}

struct GDNForwardCandidateStateOutputs {
    full: GDNCoreOutputs,
    next_conv_state_arena: Vec<f32>,
    recurrent_state_arena: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_core(
    shape: GDNComputeShape,
    qkv_values: &[f32],
    conv_state_values: &[f32],
    recurrent_state_values: &[f32],
    conv_weight_values: &[f32],
    a_values: &[f32],
    b_values: &[f32],
    z_values: &[f32],
    norm_weight_values: &[f32],
    a_log_values: &[f32],
    dt_bias_values: &[f32],
    cu_tokens_values: &[u32],
    src_state_slot_values: &[u32],
    dst_slot_id_values: &[u32],
) -> GDNCoreOutputs {
    const STATE_PREFIX_VALUES: usize = 7;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = GDNCompute::new(&device, fixture_config());
    let qkv = Buffer::from_slice(&device, qkv_values);
    let a = Buffer::from_slice(&device, a_values);
    let b = Buffer::from_slice(&device, b_values);
    let z = Buffer::from_slice(&device, z_values);
    let conv_weight = bf16_buffer(&device, conv_weight_values);
    let norm_weight = bf16_buffer(&device, norm_weight_values);
    let a_log = bf16_buffer(&device, a_log_values);
    let dt_bias = bf16_buffer(&device, dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
    let mut flat_candidate_state_slot_values = vec![u32::MAX; shape.num_tokens as usize];
    for (req_index, &dst_state_slot) in dst_slot_id_values.iter().enumerate() {
        let flat_token_end = cu_tokens_values[req_index + 1] as usize;
        flat_candidate_state_slot_values[flat_token_end - 1] = dst_state_slot;
    }
    let flat_materialized_state_slots = Buffer::from_slice(&device, &flat_candidate_state_slot_values);
    let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    conv_state_values_with_prefix.extend_from_slice(conv_state_values);
    let conv_state = Buffer::from_slice(&device, &conv_state_values_with_prefix);
    let next_conv_state = Buffer::new_zeroed(
        &device,
        (STATE_PREFIX_VALUES + fixture_config().num_conv_state_values(shape)) * size_of::<f32>(),
    );
    let mut recurrent_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    recurrent_state_values_with_prefix.extend_from_slice(recurrent_state_values);
    let recurrent_state_arena = Buffer::from_slice(&device, &recurrent_state_values_with_prefix);
    let state_offset_bytes =
        u64::try_from(STATE_PREFIX_VALUES * size_of::<f32>()).expect("test GDN state offset must fit u64");
    let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<f32>());
    let recurrent_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
    );
    let norm_gated_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
    );
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke(
        shape,
        GDNComputeBuffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_state_slots: &src_state_slots,
            flat_materialized_state_slots: &flat_materialized_state_slots,
            conv_state: &conv_state,
            conv_state_offset_bytes: state_offset_bytes,
            next_conv_state: &next_conv_state,
            next_conv_state_offset_bytes: state_offset_bytes,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: state_offset_bytes,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    GDNCoreOutputs {
        conv_qkv: conv_qkv.read_typed::<f32>(0, fixture_config().num_qkv_values(shape)),
        next_conv_state: next_conv_state
            .read_typed::<f32>(STATE_PREFIX_VALUES, fixture_config().num_conv_state_values(shape)),
        recurrent_output: recurrent_output.read_typed::<f32>(0, fixture_config().num_recurrent_output_values(shape)),
        recurrent_state: recurrent_state_arena.read_typed::<f32>(
            STATE_PREFIX_VALUES,
            shape.num_reqs as usize * fixture_config().recurrent_state_stride(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_forward_candidate_state(
    shape: GDNComputeShape,
    qkv_values: &[f32],
    conv_state_values: &[f32],
    recurrent_state_values: &[f32],
    conv_weight_values: &[f32],
    a_values: &[f32],
    b_values: &[f32],
    z_values: &[f32],
    norm_weight_values: &[f32],
    a_log_values: &[f32],
    dt_bias_values: &[f32],
    cu_tokens_values: &[u32],
    src_state_slot_values: &[u32],
    candidate_dst_slot_id_values: &[u32],
    num_state_slots: usize,
) -> GDNForwardCandidateStateOutputs {
    const STATE_PREFIX_VALUES: usize = 7;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = GDNCompute::new(&device, fixture_config());
    let qkv = Buffer::from_slice(&device, qkv_values);
    let a = Buffer::from_slice(&device, a_values);
    let b = Buffer::from_slice(&device, b_values);
    let z = Buffer::from_slice(&device, z_values);
    let conv_weight = bf16_buffer(&device, conv_weight_values);
    let norm_weight = bf16_buffer(&device, norm_weight_values);
    let a_log = bf16_buffer(&device, a_log_values);
    let dt_bias = bf16_buffer(&device, dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
    let candidate_dst_slot_ids = Buffer::from_slice(&device, candidate_dst_slot_id_values);
    let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    conv_state_values_with_prefix.extend_from_slice(conv_state_values);
    let conv_state = Buffer::from_slice(&device, &conv_state_values_with_prefix);
    let next_conv_state = Buffer::new_zeroed(
        &device,
        (STATE_PREFIX_VALUES
            + num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize)
            * size_of::<f32>(),
    );
    let mut recurrent_state_arena_values =
        vec![0.0_f32; STATE_PREFIX_VALUES + num_state_slots * fixture_config().recurrent_state_stride()];
    recurrent_state_arena_values[STATE_PREFIX_VALUES..STATE_PREFIX_VALUES + fixture_config().recurrent_state_stride()]
        .copy_from_slice(recurrent_state_values);
    let recurrent_state_arena = Buffer::from_slice(&device, &recurrent_state_arena_values);
    let state_offset_bytes =
        u64::try_from(STATE_PREFIX_VALUES * size_of::<f32>()).expect("test GDN state offset must fit u64");
    let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<f32>());
    let recurrent_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
    );
    let norm_gated_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<f32>(),
    );

    let mut builder = stream.create_replay_program();
    let core = GDNComputeBuffers {
        qkv: &qkv,
        a: &a,
        b: &b,
        z: &z,
        conv_weight: &conv_weight,
        norm_weight: &norm_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        cu_tokens: &cu_tokens,
        src_state_slots: &src_state_slots,
        flat_materialized_state_slots: &candidate_dst_slot_ids,
        conv_state: &conv_state,
        conv_state_offset_bytes: state_offset_bytes,
        next_conv_state: &next_conv_state,
        next_conv_state_offset_bytes: state_offset_bytes,
        recurrent_state_arena: &recurrent_state_arena,
        recurrent_state_arena_offset_bytes: state_offset_bytes,
        conv_qkv: &conv_qkv,
        recurrent_output: &recurrent_output,
        norm_gated_output: &norm_gated_output,
    };
    builder.record(kernels.invoke_with_candidate_state_update(shape, core));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    GDNForwardCandidateStateOutputs {
        full: GDNCoreOutputs {
            conv_qkv: conv_qkv.read_typed::<f32>(0, fixture_config().num_qkv_values(shape)),
            next_conv_state: next_conv_state.read_typed::<f32>(
                STATE_PREFIX_VALUES
                    + candidate_dst_slot_id_values[candidate_dst_slot_id_values.len() - 1] as usize
                        * fixture_config().qkv_dim() as usize
                        * fixture_config().conv_state_len() as usize,
                fixture_config().num_conv_state_values(shape),
            ),
            recurrent_output: recurrent_output
                .read_typed::<f32>(0, fixture_config().num_recurrent_output_values(shape)),
            recurrent_state: recurrent_state_arena.read_typed::<f32>(
                STATE_PREFIX_VALUES
                    + candidate_dst_slot_id_values[candidate_dst_slot_id_values.len() - 1] as usize
                        * fixture_config().recurrent_state_stride(),
                fixture_config().recurrent_state_stride(),
            ),
        },
        next_conv_state_arena: next_conv_state.read_typed::<f32>(
            STATE_PREFIX_VALUES,
            num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
        ),
        recurrent_state_arena: recurrent_state_arena.read_typed::<f32>(
            STATE_PREFIX_VALUES,
            num_state_slots * fixture_config().recurrent_state_stride(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_gdn_reference_matches(
    shape: GDNComputeShape,
    cu_tokens: &[u32],
    qkv: &[f32],
    conv_state: &[f32],
    recurrent_state: &[f32],
    conv_weight: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    actual: &GDNCoreOutputs,
    tolerance: f32,
) {
    let core = fixture_core(shape);
    let conv_reference = gdn_short_conv_reference(&core, cu_tokens, conv_state, qkv, conv_weight);
    let recurrent_reference = gdn_recurrent_reference(
        &core,
        GDNRecurrentReferenceInput {
            cu_tokens,
            source_recurrent_state: recurrent_state,
            conv_qkv: &conv_reference.conv_qkv,
            a,
            b,
            a_log,
            dt_bias,
        },
    );

    assert_close(&actual.conv_qkv, &conv_reference.conv_qkv, tolerance);
    assert_close(&actual.next_conv_state, &conv_reference.next_conv_state, tolerance);
    assert_close(
        &actual.recurrent_output,
        &recurrent_reference.recurrent_output,
        tolerance,
    );
    assert_close(
        &actual.recurrent_state,
        &recurrent_reference.next_recurrent_state,
        tolerance,
    );
}

fn output_norm_gate_reference(
    recurrent_output: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    config: GDNComputeConfig,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(recurrent_output.len());
    for (recurrent_row, z_row) in recurrent_output
        .chunks_exact(config.v_head_dim as usize)
        .zip(z.chunks_exact(config.v_head_dim as usize))
    {
        let inv_rms = (recurrent_row.iter().map(|value| value * value).sum::<f32>() / config.v_head_dim as f32
            + config.norm_eps)
            .sqrt()
            .recip();
        output.extend(
            recurrent_row
                .iter()
                .zip(z_row)
                .zip(norm_weight)
                .map(|((&value, &gate), &weight)| value * inv_rms * weight * gate / (1.0 + (-gate).exp())),
        );
    }
    output
}

fn fixture_shape(num_reqs: u32, num_tokens: u32) -> GDNComputeShape {
    GDNComputeShape { num_reqs, num_tokens }
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(
        device,
        &values
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>(),
    )
}

fn fixture_config() -> GDNComputeConfig {
    GDNComputeConfig {
        num_qk_heads: 1,
        qk_head_dim: 4,
        num_v_heads: 1,
        v_head_dim: 8,
        conv_kernel_size: 3,
        q_scale: 1.0,
        norm_eps: 1.0e-6,
    }
}

fn fixture_core(_shape: GDNComputeShape) -> GDNCore {
    GDNCore {
        model_layer_index: 0,
        hidden_dim: fixture_config().num_v_heads as usize * fixture_config().v_head_dim as usize,
        num_qk_heads: fixture_config().num_qk_heads as usize,
        qk_head_dim: fixture_config().qk_head_dim as usize,
        num_v_heads: fixture_config().num_v_heads as usize,
        v_head_dim: fixture_config().v_head_dim as usize,
        conv_kernel_size: fixture_config().conv_kernel_size as usize,
        q_scale: 1.0,
    }
}

fn conv_state_slot(arena: &[f32], _shape: GDNComputeShape, state_slot: usize) -> &[f32] {
    let conv_state_stride = fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize;
    &arena[state_slot * conv_state_stride..(state_slot + 1) * conv_state_stride]
}

fn recurrent_state_slot(arena: &[f32], _shape: GDNComputeShape, state_slot: usize) -> &[f32] {
    let recurrent_state_stride = fixture_config().recurrent_state_stride();
    &arena[state_slot * recurrent_state_stride..(state_slot + 1) * recurrent_state_stride]
}

fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 17 + pattern_offset) % 29) as f32 * scale - 14.0 * scale)
        .collect()
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual_value - expected_value).abs();
        assert!(
            diff <= tolerance,
            "GDN reference mismatch at {index}: expected={expected_value} actual={actual_value} diff={diff} \
             tolerance={tolerance}"
        );
    }
}
