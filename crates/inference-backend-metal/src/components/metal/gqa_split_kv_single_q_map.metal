// HND SplitKV SingleQ SDPA map. One MapThreadBlockTask maps 1:1 to one
// threadblock. Its fields are sourced as follows:
//
// Map task template {     // materialized; three u32 fields
//   q_token_range_index,  // sdpa_map_task_templates[map_task_template_index, 0]
//   kv_token_begin,       // sdpa_map_task_templates[map_task_template_index, 1]
//   kv_token_end,         // sdpa_map_task_templates[map_task_template_index, 2]
// }
// MapThreadBlockTask {
//   q_token_range_index,  // from the Map task template
//   kv_token_begin,       // from the Map task template
//   kv_token_end,         // from the Map task template
//   kv_head_index,        // grid-derived
//   q_head_range_index,   // grid-derived
// }
//
// The task walks consecutive KV iterations and writes one SDPAPartialOutput
// per active Q head. A sentinel Map task template returns without writing any
// partial output or statistics.
//
// q              : [num_tokens, num_q_heads, kv_head_dim]
// kv_pages       : [num_pages, K/V, num_kv_heads, num_tokens, kv_head_dim]
// req_slots      : [num_tokens]
// page_ids       : [num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]
// sdpa_map_task_templates: [num_total_kv_splits, q_token_range_index/kv_token_begin/kv_token_end]
//
// partial_exp_sums       : [num_total_kv_splits, num_q_heads]
// partial_max_logits     : [num_total_kv_splits, num_q_heads]
// partial_output         : [num_total_kv_splits, num_q_heads, kv_head_dim]

uint global_thread_index = thread_position_in_grid.x;
uint thread_index_in_threadblock = global_thread_index % (uint)REQUIRED_THREADS;
uint threadblock_linear_index = global_thread_index / (uint)REQUIRED_THREADS;

constexpr uint num_q_heads = uint(NUM_Q_HEADS);
if (threadblock_linear_index >=
    (uint)(TOTAL_KV_SPLITS * NUM_KV_HEADS * NUM_Q_HEAD_RANGES_PER_KV_HEAD)) {
    return;
}

uint map_task_template_index = threadblock_linear_index % (uint)TOTAL_KV_SPLITS;
if (map_task_template_index >= num_active_kv_splits) {
    return;
}
uint head_group_index = threadblock_linear_index / (uint)TOTAL_KV_SPLITS;
uint q_head_range_index = head_group_index % (uint)NUM_Q_HEAD_RANGES_PER_KV_HEAD;
uint kv_head_index = head_group_index / (uint)NUM_Q_HEAD_RANGES_PER_KV_HEAD;
uint q_token_range_index = sdpa_map_task_templates[map_task_template_index * 3];
// Invalid Map task templates are replay padding or slots intentionally
// populated by another attention task before the shared partial-output reduce.
if (q_token_range_index >= num_active_tokens) {
    return;
}

uint kv_token_begin = sdpa_map_task_templates[map_task_template_index * 3 + 1];
uint kv_token_end = sdpa_map_task_templates[map_task_template_index * 3 + 2];
uint q_head_range_begin = q_head_range_index * uint(MAX_Q_HEADS);
uint num_active_q_heads = metal::min(
    uint(MAX_Q_HEADS), uint(Q_HEADS_PER_KV_HEAD) - q_head_range_begin);
uint q_head_base = kv_head_index * uint(Q_HEADS_PER_KV_HEAD) + q_head_range_begin;

threadgroup float logits[MAX_Q_HEADS * KV_TOKENS_PER_ITERATION];
threadgroup float reduce_scratch[REQUIRED_THREADS];

const ulong q_head_range_offset =
    ((ulong)q_token_range_index * (ulong)num_q_heads + (ulong)q_head_base) * (ulong)KV_HEAD_DIM;
const device T* q_head_range_ptr = q + q_head_range_offset;
uint req_slot = req_slots[q_token_range_index];

float running_max[MAX_Q_HEADS];
float running_exp_sum[MAX_Q_HEADS];
for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
    running_max[local_q_head] = -INFINITY;
    running_exp_sum[local_q_head] = 0.0f;
}

#define NUM_DIMS_PER_THREAD ((KV_HEAD_DIM + REQUIRED_THREADS - 1) / REQUIRED_THREADS)
float running_output[MAX_Q_HEADS][NUM_DIMS_PER_THREAD];
for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
    for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
        running_output[local_q_head][dim_slot] = 0.0f;
    }
}

for (uint kv_iteration_begin = kv_token_begin; kv_iteration_begin < kv_token_end;
     kv_iteration_begin += uint(KV_TOKENS_PER_ITERATION)) {
    uint kv_iteration_end = metal::min(kv_iteration_begin + uint(KV_TOKENS_PER_ITERATION), kv_token_end);
    uint num_kv_tokens_in_iteration = kv_iteration_end - kv_iteration_begin;
    float local_max[MAX_Q_HEADS];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        local_max[local_q_head] = -INFINITY;
    }

    for (uint kv_token_index = kv_iteration_begin + thread_index_in_threadblock; kv_token_index < kv_iteration_end;
         kv_token_index += uint(REQUIRED_THREADS)) {
        uint block_index = kv_token_index / uint(NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
        uint page_id_index = (kv_token_index / uint(NUM_TOKENS)) % uint(NUM_PAGE_IDS_PER_BLOCK);
        uint page_token_index = kv_token_index % uint(NUM_TOKENS);
        ulong page_id_table_index =
            ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)gqa_layer_index)
              * (ulong)NUM_BLOCKS
              + (ulong)block_index)
             * (ulong)NUM_PAGE_IDS_PER_BLOCK)
            + (ulong)page_id_index;
        ulong page_id = (ulong)page_ids[page_id_table_index];
        uint token_offset = kv_token_index - kv_iteration_begin;
        const device KV_T* k_ptr =
            kv_pages + (page_id * ((ulong)PAGE_BYTES / sizeof(KV_T))
                        + (ulong)(((0 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS
                                   + page_token_index)
                                  * KV_HEAD_DIM));

        float scores[MAX_Q_HEADS];
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            scores[local_q_head] = 0.0f;
        }
        for (uint d = 0; d < uint(KV_HEAD_DIM); ++d) {
            const float k = fp8_e4m3_to_bf16(k_ptr[d]);
            for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
                const device T* q_ptr = q_head_range_ptr + local_q_head * KV_HEAD_DIM;
                scores[local_q_head] += static_cast<float>(q_ptr[d]) * k;
            }
        }
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            float score = scores[local_q_head] * ATTENTION_SCALE;
            logits[local_q_head * KV_TOKENS_PER_ITERATION + token_offset] = score;
            local_max[local_q_head] = metal::max(local_max[local_q_head], score);
        }
    }

    float iteration_max[MAX_Q_HEADS];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        reduce_scratch[thread_index_in_threadblock] = local_max[local_q_head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = uint(REQUIRED_THREADS / 2); stride > 0; stride >>= 1) {
            if (thread_index_in_threadblock < stride) {
                reduce_scratch[thread_index_in_threadblock] = metal::max(reduce_scratch[thread_index_in_threadblock], reduce_scratch[thread_index_in_threadblock + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        iteration_max[local_q_head] = reduce_scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_exp_sum[MAX_Q_HEADS];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        local_exp_sum[local_q_head] = 0.0f;
    }
    for (uint token_offset = thread_index_in_threadblock; token_offset < num_kv_tokens_in_iteration;
         token_offset += uint(REQUIRED_THREADS)) {
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            uint logits_index = local_q_head * uint(KV_TOKENS_PER_ITERATION) + token_offset;
            float weight = metal::exp(logits[logits_index] - iteration_max[local_q_head]);
            logits[logits_index] = weight;
            local_exp_sum[local_q_head] += weight;
        }
    }

    float iteration_exp_sum[MAX_Q_HEADS];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        reduce_scratch[thread_index_in_threadblock] = local_exp_sum[local_q_head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = uint(REQUIRED_THREADS / 2); stride > 0; stride >>= 1) {
            if (thread_index_in_threadblock < stride) {
                reduce_scratch[thread_index_in_threadblock] += reduce_scratch[thread_index_in_threadblock + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        iteration_exp_sum[local_q_head] = reduce_scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float old_scale[MAX_Q_HEADS];
    float iteration_scale[MAX_Q_HEADS];
    float next_max[MAX_Q_HEADS];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        next_max[local_q_head] = metal::max(running_max[local_q_head], iteration_max[local_q_head]);
        old_scale[local_q_head] = isfinite(running_max[local_q_head])
            ? metal::exp(running_max[local_q_head] - next_max[local_q_head])
            : 0.0f;
        iteration_scale[local_q_head] = metal::exp(iteration_max[local_q_head] - next_max[local_q_head]);
    }

    for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
        uint d = thread_index_in_threadblock + dim_slot * uint(REQUIRED_THREADS);
        if (d >= uint(KV_HEAD_DIM)) {
            continue;
        }
        float iteration_output[MAX_Q_HEADS];
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            iteration_output[local_q_head] = 0.0f;
        }
        for (uint token_offset = 0; token_offset < num_kv_tokens_in_iteration; ++token_offset) {
            uint kv_token_index = kv_iteration_begin + token_offset;
            uint block_index = kv_token_index / uint(NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
            uint page_id_index = (kv_token_index / uint(NUM_TOKENS)) % uint(NUM_PAGE_IDS_PER_BLOCK);
            uint page_token_index = kv_token_index % uint(NUM_TOKENS);
            ulong page_id_table_index =
                ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)gqa_layer_index)
                  * (ulong)NUM_BLOCKS
                  + (ulong)block_index)
                 * (ulong)NUM_PAGE_IDS_PER_BLOCK)
                + (ulong)page_id_index;
            ulong page_id = (ulong)page_ids[page_id_table_index];
            const device KV_T* v_ptr =
                kv_pages + (page_id * ((ulong)PAGE_BYTES / sizeof(KV_T))
                            + (ulong)(((1 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS
                                       + page_token_index)
                                      * KV_HEAD_DIM));
            float v = fp8_e4m3_to_bf16(v_ptr[d]);
            for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
                iteration_output[local_q_head] +=
                    logits[local_q_head * KV_TOKENS_PER_ITERATION + token_offset] * v;
            }
        }
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            running_output[local_q_head][dim_slot] =
                running_output[local_q_head][dim_slot] * old_scale[local_q_head]
                + iteration_output[local_q_head] * iteration_scale[local_q_head];
        }
    }

    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        running_exp_sum[local_q_head] =
            running_exp_sum[local_q_head] * old_scale[local_q_head]
            + iteration_exp_sum[local_q_head] * iteration_scale[local_q_head];
        running_max[local_q_head] = next_max[local_q_head];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (thread_index_in_threadblock == 0) {
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        uint q_head_index = q_head_base + local_q_head;
        ulong partial_output_index = (ulong)map_task_template_index * (ulong)num_q_heads + (ulong)q_head_index;
        partial_exp_sums[partial_output_index] = running_exp_sum[local_q_head];
        partial_max_logits[partial_output_index] = running_max[local_q_head];
    }
}
for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
    uint d = thread_index_in_threadblock + dim_slot * uint(REQUIRED_THREADS);
    if (d >= uint(KV_HEAD_DIM)) {
        continue;
    }
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        uint q_head_index = q_head_base + local_q_head;
        ulong partial_output_index = (ulong)map_task_template_index * (ulong)num_q_heads + (ulong)q_head_index;
        float inv_exp_sum = running_exp_sum[local_q_head] > 0.0f
            ? 1.0f / running_exp_sum[local_q_head]
            : 0.0f;
        partial_output[partial_output_index * KV_HEAD_DIM + d] =
            static_cast<T>(running_output[local_q_head][dim_slot] * inv_exp_sum);
    }
}
