use half::bf16;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

// Reuse one recording for different accepted prefixes and active job counts.
// Compare state reconstruction with an F64 recurrence. Convolution history and
// inactive slots remain exact.
#[test]
fn test_commit_matches_reference_across_prefixes_and_active_jobs() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let active_key = ReplayParameterKey::new("test.gdn.commit.active_jobs");
    for (qk_head_dim, v_head_dim) in [(3, 4), (4, 4), (128, 128), (256, 128)] {
        let config = Config {
            num_gdn_layers: 2,
            num_state_slots: 5,
            max_replay_tokens: 80,
            num_qk_heads: 1,
            qk_head_dim,
            num_v_heads: 2,
            v_head_dim,
            conv_state_len: 3,
        };
        let commit = Commit::new(&device, config);
        let recurrent_stride = config.recurrent_state_values();
        let conv_stride = config.conv_state_values();
        let pattern = |index: usize| ((index * 17 + index / 31) % 127) as f32 / 512.0 - 0.125;
        let initial = |len| {
            (0..len)
                .map(|index| bf16::from_f32(pattern(index)).to_bits())
                .collect::<Vec<_>>()
        };
        let states_initial = initial(config.arena_bytes(recurrent_stride) / 2);
        let conv_initial = initial(config.arena_bytes(conv_stride) / 2);
        let states = Buffer::from_slice(&device, &states_initial);
        let conv = Buffer::from_slice(&device, &conv_initial);
        let alpha_values = (0..config.alpha_bytes() / 4)
            .map(|index| {
                if index % 19 == 0 {
                    0.0
                } else {
                    0.97 + pattern(index) * 0.2
                }
            })
            .collect::<Vec<_>>();
        let k_values = (0..config.k_bytes() / 4).map(pattern).collect::<Vec<_>>();
        let u_values = (0..config.u_bytes() / 4)
            .map(|index| pattern(index + 7))
            .collect::<Vec<_>>();
        let qkv_values = initial(config.qkv_bytes() / 2);
        let alpha = Buffer::from_slice(&device, &alpha_values);
        let k = Buffer::from_slice(&device, &k_values);
        let u = Buffer::from_slice(&device, &u_values);
        let qkv = Buffer::from_slice(&device, &qkv_values);
        let job_buffer = Buffer::new_zeroed(&device, 3 * size_of::<Job>());
        let mut recorder = stream.create_replay_program();
        recorder.record(commit.invoke(
            3,
            ReplayU32::Parameter(active_key),
            Buffers {
                recurrent_states: &states,
                conv_states: &conv,
                alpha: &alpha,
                k: &k,
                u: &u,
                qkv: &qkv,
                jobs: &job_buffer,
            },
        ));
        let replay = recorder.build();
        for (num_active_jobs, num_tokens) in [(3, 1), (1, 8), (0, 16), (2, 17), (3, 33), (1, 0)] {
            states.write_typed(0, &states_initial);
            conv.write_typed(0, &conv_initial);
            let jobs = [
                Job {
                    src_recurrent_state_slot: 0,
                    src_conv_state_slot: 0,
                    dst_recurrent_state_slot: 2,
                    dst_conv_state_slot: 2,
                    replay_token_begin: 0,
                    num_tokens,
                },
                Job {
                    src_recurrent_state_slot: 1,
                    src_conv_state_slot: 1,
                    dst_recurrent_state_slot: 3,
                    dst_conv_state_slot: 3,
                    replay_token_begin: 40,
                    num_tokens: 7,
                },
                Job {
                    src_recurrent_state_slot: 0,
                    src_conv_state_slot: 0,
                    dst_recurrent_state_slot: 4,
                    dst_conv_state_slot: 4,
                    replay_token_begin: 0,
                    num_tokens: 16,
                },
            ];
            write_jobs(&job_buffer, &jobs);
            stream
                .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(active_key, num_active_jobs))
                .wait();
            let mut expected = states_initial.clone();
            let mut expected_conv = conv_initial.clone();
            for layer in 0..2 {
                for job in &jobs[..num_active_jobs as usize] {
                    let token_begin = layer * config.max_replay_tokens as usize + job.replay_token_begin as usize;
                    let base = layer * config.num_state_slots as usize * recurrent_stride;
                    for index in 0..recurrent_stride {
                        let head = index / (v_head_dim * qk_head_dim) as usize;
                        let row = index / qk_head_dim as usize % v_head_dim as usize;
                        let col = index % qk_head_dim as usize;
                        let mut value = bf16::from_bits(
                            states_initial[base + job.src_recurrent_state_slot as usize * recurrent_stride + index],
                        )
                        .to_f32() as f64;
                        for token in token_begin..token_begin + job.num_tokens as usize {
                            value = value * alpha_values[token * 2 + head] as f64
                                + k_values[token * qk_head_dim as usize + col] as f64
                                    * u_values[(token * 2 + head) * v_head_dim as usize + row] as f64;
                        }
                        expected[base + job.dst_recurrent_state_slot as usize * recurrent_stride + index] =
                            bf16::from_f32(value as f32).to_bits();
                    }
                    let base = layer * config.num_state_slots as usize * conv_stride;
                    for channel in 0..config.qkv_dim() {
                        for history in 0..3 {
                            let input = job.num_tokens as i64 + history as i64 - 3;
                            let value = if input < 0 {
                                conv_initial[base
                                    + job.src_conv_state_slot as usize * conv_stride
                                    + channel * 3
                                    + history
                                    + job.num_tokens as usize]
                            } else {
                                qkv_values[(token_begin + input as usize) * config.qkv_dim() + channel]
                            };
                            expected_conv
                                [base + job.dst_conv_state_slot as usize * conv_stride + channel * 3 + history] = value;
                        }
                    }
                }
            }
            let actual = states.read_typed::<u16>(0, states_initial.len());
            let (error, norm) = actual
                .iter()
                .zip(&expected)
                .fold((0.0_f64, 0.0_f64), |(error, norm), (&a, &b)| {
                    let a = bf16::from_bits(a).to_f32() as f64;
                    let b = bf16::from_bits(b).to_f32() as f64;
                    assert!(a.is_finite());
                    (error + (a - b).powi(2), norm + b * b)
                });
            assert!((error / norm).sqrt() < 0.005, "GDN commit CPU reference mismatch");
            for layer in 0..2 {
                for slot in 0..5 {
                    if !jobs[..num_active_jobs as usize]
                        .iter()
                        .any(|job| job.dst_recurrent_state_slot == slot)
                    {
                        let begin = (layer * 5 + slot as usize) * recurrent_stride;
                        assert_eq!(
                            &actual[begin..begin + recurrent_stride],
                            &states_initial[begin..begin + recurrent_stride]
                        );
                    }
                }
            }
            assert_eq!(conv.read_typed::<u16>(0, conv_initial.len()), expected_conv);
        }
    }
}

#[test]
#[should_panic(expected = "GDN replay concatenated Q/K/V dimension must fit u32")]
fn test_qkv_log_sizing_rejects_shader_count_overflow() {
    Config {
        num_gdn_layers: 1,
        num_state_slots: 2,
        max_replay_tokens: 1,
        num_qk_heads: 2,
        qk_head_dim: 1 << 30,
        num_v_heads: 2,
        v_head_dim: 1,
        conv_state_len: 1,
    }
    .qkv_bytes();
}
