use half::bf16;
use inference_executor_core::attn::gqa::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.split_kv_tiled_q.num_active_tokens");
const NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey = ReplayParameterKey::new("test.gqa.tiled.num_active_q_token_tiles");
const NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.tiled.num_active_kv_splits");
const BF16_CANARY: u16 = 0x42B6;
const F32_CANARY: f32 = -777.0;

#[test]
fn test_bucketed_replay_matches_reference_and_preserves_inactive_tails() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let (config, shape) = tiled_workload(128, 8);
    let kernels = Compute::new(&device, config, shape);
    let q_values = generated_bf16_values(config.q_bytes(shape) as usize / size_of::<u16>(), 0x4751_4154);
    let kv_values_per_kind =
        config.num_tokens_per_page() as usize * config.num_kv_heads as usize * config.head_dim as usize;
    let k_values = generated_bf16_values(kv_values_per_kind, 0x4751_4155);
    let v_values = generated_bf16_values(kv_values_per_kind, 0x4751_4156);
    let q = bf16_buffer(&device, &q_values);
    let kv_pages = bf16_buffer(&device, &kv_page_values(config, &k_values, &v_values));
    let req_slots = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize, Dtype::Uint32);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let flat_token_indices = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize, Dtype::Uint32);
    let q_token_ranges = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
    let sdpa_map_task_templates = Buffer::new_zeroed_elements(&device, 3, Dtype::Uint32);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &[0_u32, 1]);
    let partial_output = Buffer::new_zeroed(&device, config.partial_output_bytes(shape));
    let partial_exp_sums = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(shape));
    let partial_max_logits = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(shape));
    let output = Buffer::new_zeroed(&device, config.q_bytes(shape));

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map_bucketed(
        MapBuffers {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &req_slots,
            page_ids: &page_ids,
            flat_token_indices: &flat_token_indices,
            q_token_ranges: &q_token_ranges,
            sdpa_map_task_templates: &sdpa_map_task_templates,
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
        ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce_bucketed(
        ReduceBuffers {
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
            output: &output,
        },
        ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
    ));
    let replay = builder.build();
    let q_values_per_token = config.num_q_heads as usize * config.head_dim as usize;
    let kv_values_per_token = config.num_kv_heads as usize * config.head_dim as usize;
    let partial_values_per_head = config.max_q_tokens as usize * config.head_dim as usize;
    let partial_stats_per_head = config.max_q_tokens as usize;
    let mut full_partial_output = None;
    let mut full_partial_exp_sums = None;
    let mut full_partial_max_logits = None;
    let mut full_output = None;

    for (case_index, num_active_tokens) in [5_usize, 8, 5].into_iter().enumerate() {
        let q_submission = active_bf16_values(&q_values, num_active_tokens * q_values_per_token);
        let k_submission = active_bf16_values(&k_values, num_active_tokens * kv_values_per_token);
        let v_submission = active_bf16_values(&v_values, num_active_tokens * kv_values_per_token);
        q.write_typed(0, &bf16_bits(&q_submission));
        kv_pages.write_typed(0, &bf16_bits(&kv_page_values(config, &k_submission, &v_submission)));
        req_slots.write_typed(
            0,
            &active_u32_values(&vec![0; shape.num_total_tokens as usize], num_active_tokens),
        );
        flat_token_indices.write_typed(
            0,
            &active_u32_values(&(0..shape.num_total_tokens).collect::<Vec<_>>(), num_active_tokens),
        );
        q_token_ranges.write_typed(0, &[0_u32, num_active_tokens as u32]);
        sdpa_map_task_templates.write_typed(0, &[0_u32, 0, num_active_tokens as u32]);
        if case_index == 0 {
            partial_output.write_typed(0, &vec![BF16_CANARY; config.partial_output_bytes(shape) as usize / 2]);
            partial_exp_sums.write_typed(
                0,
                &vec![F32_CANARY; config.partial_output_stats_bytes(shape) as usize / 4],
            );
            partial_max_logits.write_typed(
                0,
                &vec![F32_CANARY; config.partial_output_stats_bytes(shape) as usize / 4],
            );
            output.write_typed(0, &vec![BF16_CANARY; config.q_bytes(shape) as usize / 2]);
        }

        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32)
                    .with_u32(NUM_ACTIVE_Q_TOKEN_TILES, 1)
                    .with_u32(NUM_ACTIVE_KV_SPLITS, 1),
            )
            .wait();

        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, num_active_tokens as u32],
                token_indices: &[0],
                q: &q_values[..num_active_tokens * q_values_per_token],
                context_k_by_req: &[&k_values[..num_active_tokens * kv_values_per_token]],
                context_v_by_req: &[&v_values[..num_active_tokens * kv_values_per_token]],
            },
        );
        let actual = read_bf16_values(&output, num_active_tokens * q_values_per_token);
        assert_close(&actual, &expected, 2.0e-2);

        let partial_output_values =
            partial_output.read_typed::<u16>(0, config.partial_output_bytes(shape) as usize / size_of::<u16>());
        let partial_exp_sum_values =
            partial_exp_sums.read_typed::<f32>(0, config.partial_output_stats_bytes(shape) as usize / size_of::<f32>());
        let partial_max_logit_values = partial_max_logits
            .read_typed::<f32>(0, config.partial_output_stats_bytes(shape) as usize / size_of::<f32>());
        let output_values = output.read_typed::<u16>(0, config.q_bytes(shape) as usize / size_of::<u16>());
        if case_index == 0 {
            assert_head_tails(
                &partial_output_values,
                config.num_q_heads as usize,
                partial_values_per_head,
                num_active_tokens * config.head_dim as usize,
                BF16_CANARY,
            );
            assert_head_tails(
                &partial_exp_sum_values,
                config.num_q_heads as usize,
                partial_stats_per_head,
                num_active_tokens,
                F32_CANARY,
            );
            assert_head_tails(
                &partial_max_logit_values,
                config.num_q_heads as usize,
                partial_stats_per_head,
                num_active_tokens,
                F32_CANARY,
            );
            assert_eq!(
                &output_values[num_active_tokens * q_values_per_token..],
                &vec![BF16_CANARY; (shape.num_total_tokens as usize - num_active_tokens) * q_values_per_token]
            );
        } else if case_index == 1 {
            full_partial_output = Some(partial_output_values);
            full_partial_exp_sums = Some(partial_exp_sum_values);
            full_partial_max_logits = Some(partial_max_logit_values);
            full_output = Some(output_values);
        } else {
            assert_head_tail_matches(
                &partial_output_values,
                full_partial_output.as_ref().unwrap(),
                config.num_q_heads as usize,
                partial_values_per_head,
                num_active_tokens * config.head_dim as usize,
            );
            assert_head_tail_matches(
                &partial_exp_sum_values,
                full_partial_exp_sums.as_ref().unwrap(),
                config.num_q_heads as usize,
                partial_stats_per_head,
                num_active_tokens,
            );
            assert_head_tail_matches(
                &partial_max_logit_values,
                full_partial_max_logits.as_ref().unwrap(),
                config.num_q_heads as usize,
                partial_stats_per_head,
                num_active_tokens,
            );
            assert_eq!(
                &output_values[num_active_tokens * q_values_per_token..],
                &full_output.as_ref().unwrap()[num_active_tokens * q_values_per_token..]
            );
        }
    }
}

#[test]
#[should_panic(expected = "GQA SplitKV TiledQ supports only")]
fn test_shape_rejects_unsupported_profile() {
    let (config, shape) = tiled_workload(192, 8);
    shape.validate(config);
}

#[test]
#[should_panic(expected = "GQA SplitKV TiledQ Q-token-range metadata exceeds the shader u32 element-index domain")]
fn test_shape_rejects_shader_index_overflow() {
    let shape = Shape {
        num_total_tokens: u32::MAX,
        num_total_q_token_tiles: u32::MAX,
        num_total_sdpa_map_task_templates: u32::MAX,
    };
    let (config, _) = tiled_workload(256, 16);
    shape.validate(config);
}

fn tiled_workload(head_dim: u32, num_tokens_per_page: u32) -> (Config, Shape) {
    (
        Config {
            num_q_heads: 5,
            num_kv_heads: 1,
            head_dim,
            max_q_heads: 5,
            max_q_tokens: 8,
            kv_tokens_per_iteration: 16,
            scale: (head_dim as f32).sqrt().recip(),
            page_bytes: 2 * num_tokens_per_page * head_dim * Dtype::Bfloat16.item_size() as u32,
            dtype: Dtype::Bfloat16,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        },
        Shape {
            num_total_tokens: 8,
            num_total_q_token_tiles: 1,
            num_total_sdpa_map_task_templates: 1,
        },
    )
}

fn fixture_core(config: Config) -> GQACore {
    GQACore::new(
        0,
        config.num_q_heads as usize * config.head_dim as usize,
        config.head_dim as usize,
        config.num_q_heads as usize,
        config.num_kv_heads as usize,
        config.scale,
    )
}

fn kv_page_values(config: Config, k: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(k.len(), v.len());
    let values_per_token = config.num_kv_heads as usize * config.head_dim as usize;
    let num_tokens_per_page = config.num_tokens_per_page() as usize;
    assert_eq!(k.len(), num_tokens_per_page * values_per_token);
    let mut page = vec![f32::NAN; 2 * k.len()];
    for token in 0..num_tokens_per_page {
        for kv_head in 0..config.num_kv_heads as usize {
            for dim in 0..config.head_dim as usize {
                let source = (token * config.num_kv_heads as usize + kv_head) * config.head_dim as usize + dim;
                let k_target = (kv_head * num_tokens_per_page + token) * config.head_dim as usize + dim;
                let v_target = ((config.num_kv_heads as usize + kv_head) * num_tokens_per_page + token)
                    * config.head_dim as usize
                    + dim;
                page[k_target] = k[source];
                page[v_target] = v[source];
            }
        }
    }
    page
}

fn generated_bf16_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16::from_f32(((state >> 8) as f32 / 16_777_216.0) - 0.5).to_f32()
        })
        .collect()
}

fn active_bf16_values(values: &[f32], active_len: usize) -> Vec<f32> {
    let mut submission = values.to_vec();
    submission[active_len..].fill(f32::NAN);
    submission
}

fn active_u32_values(values: &[u32], active_len: usize) -> Vec<u32> {
    let mut submission = values.to_vec();
    submission[active_len..].fill(u32::MAX);
    submission
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bf16_bits(values))
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn read_bf16_values(buffer: &Buffer, count: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, count)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn assert_head_tails<T: Copy + std::fmt::Debug + PartialEq>(
    actual: &[T],
    num_heads: usize,
    values_per_head: usize,
    num_active_values_per_head: usize,
    expected: T,
) {
    for head in 0..num_heads {
        let tail_begin = head * values_per_head + num_active_values_per_head;
        let tail_end = (head + 1) * values_per_head;
        assert_eq!(&actual[tail_begin..tail_end], &vec![expected; tail_end - tail_begin]);
    }
}

fn assert_head_tail_matches<T: std::fmt::Debug + PartialEq>(
    actual: &[T],
    expected: &[T],
    num_heads: usize,
    values_per_head: usize,
    num_active_values_per_head: usize,
) {
    for head in 0..num_heads {
        let tail_begin = head * values_per_head + num_active_values_per_head;
        let tail_end = (head + 1) * values_per_head;
        assert_eq!(&actual[tail_begin..tail_end], &expected[tail_begin..tail_end]);
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
