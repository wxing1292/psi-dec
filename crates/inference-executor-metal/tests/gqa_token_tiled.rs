use std::mem::size_of;

use half::bf16;
use inference_backend_metal::components::GQAPageTableLayout;
use inference_backend_metal::components::GQATiledSDPAKernels;
use inference_backend_metal::components::GQATiledSDPAMapBuffers;
use inference_backend_metal::components::GQATiledSDPAReduceBuffers;
use inference_backend_metal::components::GQATiledSDPAShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

const HEAD_DIM: usize = 256;
const NUM_TOKENS_PER_PAGE: usize = 16;
const Q_TOKEN_TILE_SIZE: u32 = 8;
const KV_TOKEN_TILE_SIZE: u32 = 16;
const START_TOKEN_INDEX: usize = 17;

#[test]
fn test_fixed() {
    run_case(&[16], 8);
}

#[test]
fn test_ragged() {
    run_case(&[1, 2, 4, 6, 8, 11, 16, 16], 6);
}

#[test]
fn test_multiple_tiles() {
    run_case(&[32, 8], 8);
}

fn run_case(num_tokens_per_req: &[usize], num_q_heads: usize) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let num_kv_heads = 1;
    let num_tokens = num_tokens_per_req.iter().sum::<usize>();
    let num_blocks =
        (START_TOKEN_INDEX + num_tokens_per_req.iter().copied().max().unwrap()).div_ceil(NUM_TOKENS_PER_PAGE);
    let page_bytes = 2 * num_kv_heads * NUM_TOKENS_PER_PAGE * HEAD_DIM * Dtype::Bfloat16.item_size();
    let q_values = pattern(num_tokens * num_q_heads * HEAD_DIM, 29, 0.015625);
    let mut context_k_values_by_req = Vec::new();
    let mut context_v_values_by_req = Vec::new();
    let mut kv_page_values = Vec::new();
    for req_index in 0..num_tokens_per_req.len() {
        let num_kv_slots = num_blocks * NUM_TOKENS_PER_PAGE * HEAD_DIM;
        let k = pattern(num_kv_slots, 31 + req_index, 0.015625);
        let v = pattern(num_kv_slots, 37 + req_index, 0.03125);
        for block_index in 0..num_blocks {
            let start = block_index * NUM_TOKENS_PER_PAGE * HEAD_DIM;
            let end = start + NUM_TOKENS_PER_PAGE * HEAD_DIM;
            kv_page_values.extend(k[start..end].iter().map(|value| bf16::from_f32(*value).to_bits()));
            kv_page_values.extend(v[start..end].iter().map(|value| bf16::from_f32(*value).to_bits()));
        }
        context_k_values_by_req.push(k);
        context_v_values_by_req.push(v);
    }
    let context_k_by_req = context_k_values_by_req
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();
    let context_v_by_req = context_v_values_by_req
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();

    let mut req_slot_values = Vec::new();
    let mut flat_token_index_values = Vec::new();
    let mut q_token_tile_values = Vec::new();
    let mut sdpa_map_task_template_values = Vec::new();
    let mut cu_sdpa_partial_output_values = vec![0u32];
    let mut cu_tokens = vec![0u32];
    let mut flat_token_start = 0u32;
    for (req_slot, &num_req_tokens) in num_tokens_per_req.iter().enumerate() {
        req_slot_values.extend(std::iter::repeat_n(req_slot as u32, num_req_tokens));
        flat_token_index_values.extend(START_TOKEN_INDEX as u32..START_TOKEN_INDEX as u32 + num_req_tokens as u32);
        let flat_token_end = flat_token_start + num_req_tokens as u32;
        let mut q_token_begin = flat_token_start;
        while q_token_begin < flat_token_end {
            let q_token_end = (q_token_begin + Q_TOKEN_TILE_SIZE).min(flat_token_end);
            q_token_tile_values.extend_from_slice(&[q_token_begin, q_token_end]);
            let q_token_tile_index = q_token_tile_values.len() / 2 - 1;
            let context_len = START_TOKEN_INDEX as u32 + q_token_end - flat_token_start;
            let mut kv_token_begin = 0;
            while kv_token_begin < context_len {
                let kv_token_end = (kv_token_begin + KV_TOKEN_TILE_SIZE).min(context_len);
                sdpa_map_task_template_values.extend_from_slice(&[
                    q_token_tile_index as u32,
                    kv_token_begin,
                    kv_token_end,
                ]);
                kv_token_begin = kv_token_end;
            }
            cu_sdpa_partial_output_values.push((sdpa_map_task_template_values.len() / 3) as u32);
            q_token_begin = q_token_end;
        }
        cu_tokens.push(flat_token_end);
        flat_token_start = flat_token_end;
    }
    let num_q_token_tiles = q_token_tile_values.len() / 2;
    let num_sdpa_map_task_templates = sdpa_map_task_template_values.len() / 3;
    let total_sdpa_map_task_templates = num_sdpa_map_task_templates.next_power_of_two();
    sdpa_map_task_template_values.resize(total_sdpa_map_task_templates * 3, u32::MAX);
    let shape = GQATiledSDPAShape {
        num_tokens: num_tokens.try_into().unwrap(),
        num_q_token_tiles: num_q_token_tiles.try_into().unwrap(),
        total_sdpa_map_task_templates: total_sdpa_map_task_templates.try_into().unwrap(),
        num_q_heads: num_q_heads.try_into().unwrap(),
        num_kv_heads: num_kv_heads.try_into().unwrap(),
        head_dim: HEAD_DIM.try_into().unwrap(),
        q_head_tile_size: num_q_heads.try_into().unwrap(),
        q_token_tile_size: Q_TOKEN_TILE_SIZE,
        kv_token_tile_size: KV_TOKEN_TILE_SIZE,
        scale: (HEAD_DIM as f32).sqrt().recip(),
        page_bytes: page_bytes.try_into().unwrap(),
        dtype: Dtype::Bfloat16,
        page_table_layout: GQAPageTableLayout {
            num_req_slots: num_tokens_per_req.len().try_into().unwrap(),
            num_blocks: num_blocks.try_into().unwrap(),
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        },
        gqa_layer_index: 0,
    };

    let q_bf16 = q_values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let q_f32 = q_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    let context_k_f32_by_req = context_k_by_req
        .iter()
        .map(|k| {
            k.iter()
                .map(|value| bf16::from_f32(*value).to_f32())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let context_v_f32_by_req = context_v_by_req
        .iter()
        .map(|v| {
            v.iter()
                .map(|value| bf16::from_f32(*value).to_f32())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let context_k_refs = context_k_f32_by_req
        .iter()
        .zip(num_tokens_per_req)
        .map(|(k, &num_req_tokens)| &k[..(START_TOKEN_INDEX + num_req_tokens) * HEAD_DIM])
        .collect::<Vec<_>>();
    let context_v_refs = context_v_f32_by_req
        .iter()
        .zip(num_tokens_per_req)
        .map(|(v, &num_req_tokens)| &v[..(START_TOKEN_INDEX + num_req_tokens) * HEAD_DIM])
        .collect::<Vec<_>>();

    let expected = projected_gqa_reference(
        &GQACore::new(
            0,
            num_q_heads * HEAD_DIM,
            HEAD_DIM,
            num_q_heads,
            num_kv_heads,
            shape.scale,
        ),
        GQAReferenceInput {
            cu_tokens: &cu_tokens,
            token_indices: &vec![START_TOKEN_INDEX as u32; num_tokens_per_req.len()],
            q: &q_f32,
            context_k_by_req: &context_k_refs,
            context_v_by_req: &context_v_refs,
        },
    );

    let q = Buffer::from_slice(&device, &q_bf16);
    let kv_pages = Buffer::from_slice(&device, &kv_page_values);
    let req_slots = Buffer::from_slice(&device, &req_slot_values);
    let page_ids = Buffer::from_slice(
        &device,
        &(0..(num_tokens_per_req.len() * num_blocks) as u32).collect::<Vec<_>>(),
    );
    let flat_token_indices = Buffer::from_slice(&device, &flat_token_index_values);
    let q_token_tiles = Buffer::from_slice(&device, &q_token_tile_values);
    let sdpa_map_task_templates = Buffer::from_slice(&device, &sdpa_map_task_template_values);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &cu_sdpa_partial_output_values);
    let num_sdpa_partial_output_tokens = total_sdpa_map_task_templates * Q_TOKEN_TILE_SIZE as usize;
    let partial_output = Buffer::new_zeroed(
        &device,
        num_sdpa_partial_output_tokens * num_q_heads * HEAD_DIM * Dtype::Bfloat16.item_size(),
    );
    let partial_exp_sums = Buffer::new_zeroed(&device, num_sdpa_partial_output_tokens * num_q_heads * size_of::<f32>());
    let partial_max_logits =
        Buffer::new_zeroed(&device, num_sdpa_partial_output_tokens * num_q_heads * size_of::<f32>());
    let output = Buffer::new_zeroed(
        &device,
        num_tokens * num_q_heads * HEAD_DIM * Dtype::Bfloat16.item_size(),
    );
    let kernels = GQATiledSDPAKernels::new(&device);
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        shape,
        GQATiledSDPAMapBuffers {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &req_slots,
            page_ids: &page_ids,
            flat_token_indices: &flat_token_indices,
            q_token_tiles: &q_token_tiles,
            sdpa_map_task_templates: &sdpa_map_task_templates,
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
        },
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce(
        shape,
        GQATiledSDPAReduceBuffers {
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            q_token_tiles: &q_token_tiles,
            cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
            output: &output,
        },
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();
    let actual = output
        .read_typed::<u16>(0, expected.len())
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();

    let max_abs_diff = actual
        .iter()
        .zip(&expected)
        .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 0.02, "max_abs_diff={max_abs_diff}");
}

fn pattern(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (index % modulus) as f32 * scale - (modulus as f32 * scale * 0.5))
        .collect()
}
