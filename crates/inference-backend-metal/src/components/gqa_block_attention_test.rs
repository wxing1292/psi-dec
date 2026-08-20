use half::bf16;

use super::*;
use crate::metal::Stream;

#[test]
fn test_constants_have_explicit_thread_block_scope() {
    let config = GQABlockSDPAConfig {
        block_size: 3,
        num_q_heads: 5,
        num_kv_heads: 1,
        head_dim: 32,
        scale: 32.0_f32.sqrt().recip(),
        dtype: Dtype::Bfloat16,
    };
    let constants = GQABlockSDPAKernelConstants::current(config);
    assert_eq!(constants.config, config);
    assert_eq!(constants.thread_block.required_threads, 32);
    assert_eq!(constants.thread_block.simdgroup_width, 32);
}

#[test]
#[should_panic(expected = "complete request blocks")]
fn test_block_shape_rejects_partial_request_block() {
    GQABlockSDPAShape {
        num_tokens: 10,
        num_total_sdpa_map_task_templates: 16,
    }
    .validate(GQABlockSDPAConfig {
        block_size: 9,
        num_q_heads: 40,
        num_kv_heads: 8,
        head_dim: 128,
        scale: 1.0,
        dtype: Dtype::Bfloat16,
    });
}

#[test]
fn test_f32_kernel_matches_request_block_bidirectional_reference() {
    assert_kernel_matches_request_block_bidirectional_reference(Dtype::Float32, 1.0e-5);
}

#[test]
fn test_bf16_kernel_matches_request_block_bidirectional_reference() {
    assert_kernel_matches_request_block_bidirectional_reference(Dtype::Bfloat16, 1.0e-2);
}

fn assert_kernel_matches_request_block_bidirectional_reference(dtype: Dtype, output_tolerance: f32) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = GQABlockSDPAConfig {
        block_size: 3,
        num_q_heads: 5,
        num_kv_heads: 1,
        head_dim: 32,
        scale: 32.0_f32.sqrt().recip(),
        dtype,
    };
    let shape = GQABlockSDPAShape {
        num_tokens: 6,
        num_total_sdpa_map_task_templates: 8,
    };
    let q_values = round_for_dtype(
        (0..config.q_elements(shape))
            .map(|index| index as f32 * 0.03125 - 0.4)
            .collect::<Vec<_>>(),
        dtype,
    );
    let k_values = round_for_dtype(
        (0..config.kv_elements(shape))
            .map(|index| index as f32 * -0.025 + 0.3)
            .collect::<Vec<_>>(),
        dtype,
    );
    let v_values = round_for_dtype(
        (0..config.kv_elements(shape))
            .map(|index| index as f32 * 0.05 - 0.2)
            .collect::<Vec<_>>(),
        dtype,
    );
    let q = buffer(&device, &q_values, dtype);
    let local_k = buffer(&device, &k_values, dtype);
    let local_v = buffer(&device, &v_values, dtype);
    let block_sdpa_map_task_template_indices = Buffer::from_slice(&device, &[0_u32, 1, 2, 3, 4, 5]);
    let partial_exp_sums =
        Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
    let partial_max_logits =
        Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
    let partial_output = Buffer::new_zeroed_elements(&device, config.partial_output_values(shape), dtype);
    let kernel = GQABlockSDPAKernel::new(&device, config);
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        GQABlockSDPABuffers {
            q: &q,
            local_k: &local_k,
            local_v: &local_v,
            block_sdpa_map_task_template_indices: &block_sdpa_map_task_template_indices,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            partial_output: &partial_output,
        },
    ));
    stream.submit_replay(&builder.build()).wait();

    let actual_partial_exp_sums = partial_exp_sums.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
    let actual_partial_max_logits = partial_max_logits.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
    let actual_output = read_values(&partial_output, config.partial_output_values(shape), dtype);
    for q_token_index in 0..shape.num_tokens as usize {
        let local_kv_token_begin = q_token_index / config.block_size as usize * config.block_size as usize;
        for q_head in 0..config.num_q_heads as usize {
            let q_start = (q_token_index * config.num_q_heads as usize + q_head) * config.head_dim as usize;
            let q_row = &q_values[q_start..q_start + config.head_dim as usize];
            let mut scores = Vec::new();
            for local_kv_offset in 0..config.block_size as usize {
                let kv_token_index = local_kv_token_begin + local_kv_offset;
                let kv_start = kv_token_index * config.head_dim as usize;
                let key = &k_values[kv_start..kv_start + config.head_dim as usize];
                scores.push(q_row.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>() * config.scale);
            }
            let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|&score| (score - max_logit).exp())
                .collect::<Vec<_>>();
            let exp_sum = weights.iter().sum::<f32>();
            let partial_output_index = q_token_index * config.num_q_heads as usize + q_head;
            assert!(
                (actual_partial_max_logits[partial_output_index] - max_logit).abs() < 1.0e-4,
                "actual={} expected={max_logit}",
                actual_partial_max_logits[partial_output_index]
            );
            assert!(
                (actual_partial_exp_sums[partial_output_index] - exp_sum).abs() < 1.0e-4,
                "actual={} expected={exp_sum}",
                actual_partial_exp_sums[partial_output_index]
            );
            for dim in 0..config.head_dim as usize {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(local_kv_offset, &weight)| {
                        let kv_token_index = local_kv_token_begin + local_kv_offset;
                        let kv_start = kv_token_index * config.head_dim as usize;
                        weight * v_values[kv_start + dim]
                    })
                    .sum::<f32>()
                    / exp_sum;
                let actual = actual_output[partial_output_index * config.head_dim as usize + dim];
                assert!(
                    (actual - expected).abs() < output_tolerance,
                    "actual={actual} expected={expected}"
                );
            }
        }
    }
}

fn round_for_dtype(values: Vec<f32>, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => values,
        Dtype::Bfloat16 => values.into_iter().map(|value| bf16::from_f32(value).to_f32()).collect(),
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}

fn buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, values),
        Dtype::Bfloat16 => {
            let bits = values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            Buffer::from_slice(device, &bits)
        },
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}

fn read_values(buffer: &Buffer, count: usize, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed::<f32>(0, count),
        Dtype::Bfloat16 => {
            buffer
                .read_typed::<u16>(0, count)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect()
        },
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}
