use inference_executor_core::attn::gqa::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.split_kv_single_q.active_tokens");
const NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.paged.active_kv_splits");

#[test]
#[should_panic(expected = "GQA SDPA query/output exceeds the shader u32 count domain")]
fn test_sdpa_shape_rejects_shader_count_overflow() {
    let config = GQASplitKVSingleQConfig {
        num_q_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        scale: 1.0,
        page_bytes: 8,
        page_table_layout: GQAPageTableLayout {
            num_req_slots: 1,
            num_gqa_layers: 1,
            num_blocks: 1,
            num_page_ids_per_block: 1,
        },
        kv_tokens_per_iteration: 1,
        required_threads: 32,
        max_q_heads: 1,
        dtype: Dtype::Bfloat16,
    };
    GQASplitKVSingleQShape {
        num_total_tokens: 1 << 30,
        num_total_sdpa_map_task_templates: 1,
    }
    .validate(config);
}

#[test]
#[should_panic(expected = "GQA activation/gate exceeds the shader u32 count domain")]
fn test_activation_gate_shape_rejects_shader_count_overflow() {
    GQAActivationGateShape {
        num_total_tokens: 1 << 30,
    }
    .validate(GQAActivationGateConfig::bf16(2, 2));
}

#[test]
fn test_fixed() {
    let config = fixture_config();
    let shape = fixture_shape();
    let q = fixture_values(config.num_output_values(shape), 0.125, 3);
    let k = fixture_values(2 * config.num_kv_heads as usize * config.head_dim as usize, 0.0625, 5);
    let v = fixture_values(2 * config.num_kv_heads as usize * config.head_dim as usize, 0.25, 7);
    let kv_pages = kv_page_values(config, &[(&k, &v)]);
    let actual = run_gqa_split_kv_single_q(
        config,
        shape,
        GQASplitKVSingleQTestInput {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &[0, 0],
            page_ids: &[0],
            flat_token_indices: &[0, 1],
        },
    );
    let expected = projected_gqa_reference(
        &fixture_core(config),
        GQAReferenceInput {
            cu_tokens: &[0, 2],
            token_indices: &[0],
            q: &q,
            context_k_by_req: &[&k],
            context_v_by_req: &[&v],
        },
    );
    assert_close(&actual, &expected, 2.0e-5);
}

#[test]
fn test_bucketed_replay_matches_active_prefix_and_preserves_output_tail() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = fixture_config();
    let shape = GQASplitKVSingleQShape {
        num_total_tokens: 4,
        num_total_sdpa_map_task_templates: 4,
    };
    let q = generated_values(config.num_output_values(shape), 0x8B1D_5A73);
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let k = generated_values(4 * kv_stride, 0x8B1D_5A74);
    let v = generated_values(4 * kv_stride, 0x8B1D_5A75);
    let kv_pages = kv_page_values(config, &[(&k, &v)]);
    let q_buffer = Buffer::from_slice(&device, &q);
    let kv_pages_buffer = Buffer::from_slice(&device, &kv_pages);
    let req_slots = Buffer::from_slice(&device, &[0_u32; 4]);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let task_templates = Buffer::from_slice(&device, &[0_u32, 0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4]);
    let cu_partial_outputs = Buffer::from_slice(&device, &[0_u32, 1, 2, 3, 4]);
    let sentinel = -321.0_f32;
    let output = Buffer::from_slice(&device, &vec![sentinel; config.num_output_values(shape)]);
    let scratch = GQASplitKVSingleQScratch::new(&device, config, shape);
    let kernels = GQASplitKVSingleQKernels::new(&device, config, shape);
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map_bucketed(
        scratch.map_buffers(&q_buffer, &kv_pages_buffer, &req_slots, &page_ids, &task_templates),
        ReplayU32::Fixed(0),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce_bucketed(
        scratch.reduce_buffers(&cu_partial_outputs, &output),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
    ));
    let replay = builder.build();
    for num_active_tokens in [3_usize, 4] {
        output.write_typed(0, &vec![sentinel; config.num_output_values(shape)]);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32)
                    .with_u32(NUM_ACTIVE_KV_SPLITS, num_active_tokens as u32),
            )
            .wait();
        let num_active_values = num_active_tokens * config.num_q_heads as usize * config.head_dim as usize;
        let actual = output.read_typed::<f32>(0, num_active_values);
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, num_active_tokens as u32],
                token_indices: &[0],
                q: &q[..num_active_values],
                context_k_by_req: &[&k[..num_active_tokens * kv_stride]],
                context_v_by_req: &[&v[..num_active_tokens * kv_stride]],
            },
        );
        assert_close(&actual, &expected, 2.0e-5);
        assert_eq!(
            output.read_typed::<f32>(num_active_values, config.num_output_values(shape) - num_active_values,),
            vec![sentinel; config.num_output_values(shape) - num_active_values]
        );
    }
}

#[test]
fn test_random() {
    let random_seed = 0x4C8F_17D2;
    let config = fixture_config();
    let shape = fixture_shape();
    let q = generated_values(config.num_output_values(shape), random_seed);
    let k = generated_values(
        2 * config.num_kv_heads as usize * config.head_dim as usize,
        random_seed.wrapping_add(1),
    );
    let v = generated_values(
        2 * config.num_kv_heads as usize * config.head_dim as usize,
        random_seed.wrapping_add(2),
    );
    let kv_pages = kv_page_values(config, &[(&k, &v)]);
    let actual = run_gqa_split_kv_single_q(
        config,
        shape,
        GQASplitKVSingleQTestInput {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &[0, 0],
            page_ids: &[0],
            flat_token_indices: &[0, 1],
        },
    );
    let expected = projected_gqa_reference(
        &fixture_core(config),
        GQAReferenceInput {
            cu_tokens: &[0, 2],
            token_indices: &[0],
            q: &q,
            context_k_by_req: &[&k],
            context_v_by_req: &[&v],
        },
    );
    assert_close(&actual, &expected, 2.0e-5);
}

#[test]
fn test_ragged_random() {
    let random_seed = 0xD205_6AB9;
    let mut config = fixture_config();
    let mut shape = fixture_shape();
    shape.num_total_tokens = 3;
    shape.num_total_sdpa_map_task_templates = 4;
    config.page_table_layout.num_req_slots = 2;
    let q = generated_values(config.num_output_values(shape), random_seed);
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let req0_k = generated_values(2 * kv_stride, random_seed.wrapping_add(1));
    let req0_v = generated_values(2 * kv_stride, random_seed.wrapping_add(2));
    let req1_k = generated_values(2 * kv_stride, random_seed.wrapping_add(3));
    let req1_v = generated_values(2 * kv_stride, random_seed.wrapping_add(4));
    let kv_pages = kv_page_values(config, &[(&req0_k, &req0_v), (&req1_k, &req1_v)]);
    let actual = run_gqa_split_kv_single_q(
        config,
        shape,
        GQASplitKVSingleQTestInput {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &[0, 1, 1],
            page_ids: &[0, 1],
            flat_token_indices: &[1, 0, 1],
        },
    );
    let expected = projected_gqa_reference(
        &fixture_core(config),
        GQAReferenceInput {
            cu_tokens: &[0, 1, 3],
            token_indices: &[1, 0],
            q: &q,
            context_k_by_req: &[&req0_k, &req1_k],
            context_v_by_req: &[&req0_v, &req1_v],
        },
    );
    assert_close(&actual, &expected, 2.0e-5);
}

#[test]
fn test_multiple_page_ids_per_block() {
    let mut config = fixture_config();
    let mut shape = fixture_shape();
    shape.num_total_tokens = 1;
    shape.num_total_sdpa_map_task_templates = 1;
    config.page_table_layout.num_page_ids_per_block = 2;
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let q = fixture_values(config.num_output_values(shape), 0.125, 3);
    let k = fixture_values(8 * kv_stride, 0.0625, 5);
    let v = fixture_values(8 * kv_stride, 0.25, 7);
    let kv_pages = kv_page_values(
        config,
        &[
            (&k[..4 * kv_stride], &v[..4 * kv_stride]),
            (&k[4 * kv_stride..], &v[4 * kv_stride..]),
        ],
    );
    let actual = run_gqa_split_kv_single_q(
        config,
        shape,
        GQASplitKVSingleQTestInput {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &[0],
            page_ids: &[0, 1],
            flat_token_indices: &[7],
        },
    );
    let expected = projected_gqa_reference(
        &fixture_core(config),
        GQAReferenceInput {
            cu_tokens: &[0, 1],
            token_indices: &[7],
            q: &q,
            context_k_by_req: &[&k],
            context_v_by_req: &[&v],
        },
    );
    assert_close(&actual, &expected, 2.0e-5);
}

fn fixture_config() -> GQASplitKVSingleQConfig {
    let num_kv_heads = 2;
    let num_tokens_per_page = 4;
    let head_dim = 2;
    GQASplitKVSingleQConfig {
        num_q_heads: 4,
        num_kv_heads,
        head_dim,
        scale: 0.5,
        page_bytes: 2 * num_kv_heads * num_tokens_per_page * head_dim * Dtype::Float32.item_size() as u32,
        page_table_layout: GQAPageTableLayout {
            num_req_slots: 1,
            num_blocks: 1,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        },
        kv_tokens_per_iteration: 4,
        required_threads: 64,
        max_q_heads: 2,
        dtype: Dtype::Float32,
    }
}

fn fixture_shape() -> GQASplitKVSingleQShape {
    GQASplitKVSingleQShape {
        num_total_tokens: 2,
        num_total_sdpa_map_task_templates: 2,
    }
}

fn fixture_core(config: GQASplitKVSingleQConfig) -> GQACore {
    let q_dim = config.num_q_heads as usize * config.head_dim as usize;
    GQACore::new(
        0,
        q_dim,
        config.head_dim as usize,
        config.num_q_heads as usize,
        config.num_kv_heads as usize,
        config.scale,
    )
}

struct GQASplitKVSingleQTestInput<'a> {
    q: &'a [f32],
    kv_pages: &'a [f32],
    req_slots: &'a [u32],
    page_ids: &'a [u32],
    flat_token_indices: &'a [u32],
}

fn run_gqa_split_kv_single_q(
    config: GQASplitKVSingleQConfig,
    shape: GQASplitKVSingleQShape,
    input: GQASplitKVSingleQTestInput<'_>,
) -> Vec<f32> {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = GQASplitKVSingleQKernels::new(&device, config, shape);
    let q = Buffer::from_slice(&device, input.q);
    let kv_pages = Buffer::from_slice(&device, input.kv_pages);
    let req_slots = Buffer::from_slice(&device, input.req_slots);
    let page_ids = Buffer::from_slice(&device, input.page_ids);
    let (sdpa_map_task_template_values, cu_sdpa_partial_output_values) =
        sdpa_map_task_template_buffers(config, shape, input.flat_token_indices);
    let sdpa_map_task_templates = Buffer::from_slice(&device, &sdpa_map_task_template_values);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &cu_sdpa_partial_output_values);
    let output = Buffer::new_zeroed(&device, config.q_bytes(shape));
    let scratch = GQASplitKVSingleQScratch::new(&device, config, shape);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        scratch.map_buffers(&q, &kv_pages, &req_slots, &page_ids, &sdpa_map_task_templates),
        ReplayU32::Fixed(0),
    ));
    builder
        .record_with_barrier_before(kernels.invoke_reduce(scratch.reduce_buffers(&cu_sdpa_partial_outputs, &output)));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();
    output.read_typed::<f32>(0, config.num_output_values(shape))
}

fn sdpa_map_task_template_buffers(
    config: GQASplitKVSingleQConfig,
    shape: GQASplitKVSingleQShape,
    flat_token_indices: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let num_kv_iterations = flat_token_indices
        .iter()
        .map(|&token_index| (token_index + 1).div_ceil(config.kv_tokens_per_iteration) as usize)
        .collect::<Vec<_>>();
    let mut num_sdpa_map_task_templates_by_q_token_range = vec![1_usize; flat_token_indices.len()];
    let mut num_sdpa_map_task_templates = num_sdpa_map_task_templates_by_q_token_range.len();
    while num_sdpa_map_task_templates < shape.num_total_sdpa_map_task_templates as usize {
        let Some(q_token_range_index) = (0..num_kv_iterations.len())
            .filter(|&q_token_range_index| {
                num_sdpa_map_task_templates_by_q_token_range[q_token_range_index]
                    < num_kv_iterations[q_token_range_index]
            })
            .max_by_key(|&q_token_range_index| {
                num_kv_iterations[q_token_range_index]
                    .div_ceil(num_sdpa_map_task_templates_by_q_token_range[q_token_range_index])
            })
        else {
            break;
        };
        num_sdpa_map_task_templates_by_q_token_range[q_token_range_index] += 1;
        num_sdpa_map_task_templates += 1;
    }
    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = vec![0];
    for (q_token_range_index, &token_index) in flat_token_indices.iter().enumerate() {
        let context_len = token_index + 1;
        for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_by_q_token_range[q_token_range_index] {
            let kv_iteration_begin = num_kv_iterations[q_token_range_index] * sdpa_map_task_template_index
                / num_sdpa_map_task_templates_by_q_token_range[q_token_range_index];
            let kv_iteration_end = num_kv_iterations[q_token_range_index] * (sdpa_map_task_template_index + 1)
                / num_sdpa_map_task_templates_by_q_token_range[q_token_range_index];
            let kv_token_begin = kv_iteration_begin as u32 * config.kv_tokens_per_iteration;
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_range_index as u32,
                kv_token_begin,
                context_len.min(kv_iteration_end as u32 * config.kv_tokens_per_iteration),
            ]);
        }
        cu_sdpa_partial_outputs.push((sdpa_map_task_templates.len() / 3) as u32);
    }
    assert!(sdpa_map_task_templates.len() / 3 <= shape.num_total_sdpa_map_task_templates as usize);
    sdpa_map_task_templates.resize(shape.num_total_sdpa_map_task_templates as usize * 3, u32::MAX);
    (sdpa_map_task_templates, cu_sdpa_partial_outputs)
}

fn kv_page_values(config: GQASplitKVSingleQConfig, pages: &[(&[f32], &[f32])]) -> Vec<f32> {
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let page_f32_values = config.page_bytes as usize / size_of::<f32>();
    let mut v = vec![0.0_f32; pages.len() * page_f32_values];
    for (page_index, (k, page_v)) in pages.iter().enumerate() {
        assert_eq!(k.len(), page_v.len());
        assert_eq!(k.len() % kv_stride, 0);
        let num_tokens = k.len() / kv_stride;
        let num_tokens_per_page = config.num_tokens_per_page() as usize;
        assert!(num_tokens <= num_tokens_per_page);
        let page_base = page_index * page_f32_values;
        for token in 0..num_tokens {
            for kv_head in 0..config.num_kv_heads as usize {
                for dim in 0..config.head_dim as usize {
                    let source = (token * config.num_kv_heads as usize + kv_head) * config.head_dim as usize + dim;
                    let k_target = page_base + (kv_head * num_tokens_per_page + token) * config.head_dim as usize + dim;
                    let v_target = page_base
                        + ((config.num_kv_heads as usize + kv_head) * num_tokens_per_page + token)
                            * config.head_dim as usize
                        + dim;
                    v[k_target] = k[source];
                    v[v_target] = page_v[source];
                }
            }
        }
    }
    v
}

fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 11 + pattern_offset) % 23) as f32 * scale - 11.0 * scale)
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
            "GQA reference mismatch at {index}: expected={expected_value} actual={actual_value} diff={diff} \
             tolerance={tolerance}"
        );
    }
}
