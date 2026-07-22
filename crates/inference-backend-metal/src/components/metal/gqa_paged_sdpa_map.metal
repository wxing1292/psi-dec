// HND paged context-parallel SDPA map. One logical SDPAMapTask maps 1:1 to one
// threadblock. Its fields are sourced as follows:
//
// SDPAMapTaskTemplate {  // materialized; three u32 fields
//   q_token_tile_index,  // sdpa_map_task_templates[sdpa_map_task_template_index, 0]
//   kv_token_begin,      // sdpa_map_task_templates[sdpa_map_task_template_index, 1]
//   kv_token_end,        // sdpa_map_task_templates[sdpa_map_task_template_index, 2]
// }
// SDPAMapTask {
//   q_token_tile_index,  // from TaskTemplate
//   kv_token_begin,      // from TaskTemplate
//   kv_token_end,        // from TaskTemplate
//   kv_head_index,       // grid-derived
//   q_head_tile_index,   // grid-derived
// }
//
// The Task walks consecutive SDPAMapTiles along Tkv and writes one
// SDPAPartialOutput per active Q head. A sentinel TaskTemplate returns without
// writing any partial output or statistics.
//
// q              : [num_tokens, num_q_heads, kv_head_dim]
// kv_pages       : [num_pages, K/V, num_kv_heads, num_tokens, kv_head_dim]
// req_slots      : [num_tokens]
// page_ids       : [num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]
// sdpa_map_task_templates : [total_sdpa_map_task_templates, q_token_tile_index/kv_token_begin/kv_token_end]
//
// partial_exp_sums       : [total_sdpa_map_task_templates, num_q_heads]
// partial_max_logits     : [total_sdpa_map_task_templates, num_q_heads]
// partial_output        : [total_sdpa_map_task_templates, num_q_heads, kv_head_dim]

uint global_thread_index = thread_position_in_grid.x;
uint thread_index_in_threadblock = global_thread_index % (uint)NUM_THREADS_PER_THREADBLOCK;
uint threadblock_linear_index = global_thread_index / (uint)NUM_THREADS_PER_THREADBLOCK;

constexpr uint num_active_tokens = uint(NUM_ACTIVE_TOKENS);
constexpr uint num_q_heads = uint(NUM_Q_HEADS);
if (threadblock_linear_index >=
    (uint)(TOTAL_SDPA_MAP_TASK_TEMPLATES * NUM_KV_HEADS * NUM_Q_HEAD_TILES_PER_KV_HEAD)) {
    return;
}

uint sdpa_map_task_template_index = threadblock_linear_index % (uint)TOTAL_SDPA_MAP_TASK_TEMPLATES;
uint head_group_index = threadblock_linear_index / (uint)TOTAL_SDPA_MAP_TASK_TEMPLATES;
uint q_head_tile_index = head_group_index % (uint)NUM_Q_HEAD_TILES_PER_KV_HEAD;
uint kv_head_index = head_group_index / (uint)NUM_Q_HEAD_TILES_PER_KV_HEAD;
uint q_token_tile_index = sdpa_map_task_templates[sdpa_map_task_template_index * 3];
// Invalid TaskTemplates are either replay padding or slots intentionally
// populated by another attention task before the shared partial-output reduce.
if (q_token_tile_index >= num_active_tokens) {
    return;
}

uint kv_token_begin = sdpa_map_task_templates[sdpa_map_task_template_index * 3 + 1];
uint kv_token_end = sdpa_map_task_templates[sdpa_map_task_template_index * 3 + 2];
uint q_head_tile_base = q_head_tile_index * uint(Q_HEAD_TILE_SIZE);
uint num_active_q_heads = metal::min(
    uint(Q_HEAD_TILE_SIZE), uint(Q_HEADS_PER_KV_HEAD) - q_head_tile_base);
uint q_head_base = kv_head_index * uint(Q_HEADS_PER_KV_HEAD) + q_head_tile_base;

threadgroup float logits[Q_HEAD_TILE_SIZE * KV_TOKEN_TILE_SIZE];
threadgroup float reduce_scratch[NUM_THREADS_PER_THREADBLOCK];

const ulong q_tile_offset =
    ((ulong)q_token_tile_index * (ulong)num_q_heads + (ulong)q_head_base) * (ulong)KV_HEAD_DIM;
const device T* q_tile_ptr = q + q_tile_offset;
uint req_slot = req_slots[q_token_tile_index];

float running_max[Q_HEAD_TILE_SIZE];
float running_exp_sum[Q_HEAD_TILE_SIZE];
for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
    running_max[local_q_head] = -INFINITY;
    running_exp_sum[local_q_head] = 0.0f;
}

#define NUM_DIMS_PER_THREAD ((KV_HEAD_DIM + NUM_THREADS_PER_THREADBLOCK - 1) / NUM_THREADS_PER_THREADBLOCK)
float running_output[Q_HEAD_TILE_SIZE][NUM_DIMS_PER_THREAD];
for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
    for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
        running_output[local_q_head][dim_slot] = 0.0f;
    }
}

for (uint kv_token_tile_begin = kv_token_begin; kv_token_tile_begin < kv_token_end;
     kv_token_tile_begin += uint(KV_TOKEN_TILE_SIZE)) {
    uint kv_token_tile_end = metal::min(kv_token_tile_begin + uint(KV_TOKEN_TILE_SIZE), kv_token_end);
    uint kv_token_tile_len = kv_token_tile_end - kv_token_tile_begin;
    float local_max[Q_HEAD_TILE_SIZE];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        local_max[local_q_head] = -INFINITY;
    }

    for (uint kv_token_index = kv_token_tile_begin + thread_index_in_threadblock; kv_token_index < kv_token_tile_end;
         kv_token_index += uint(NUM_THREADS_PER_THREADBLOCK)) {
        uint block_index = kv_token_index / uint(NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
        uint page_id_index = (kv_token_index / uint(NUM_TOKENS)) % uint(NUM_PAGE_IDS_PER_BLOCK);
        uint page_token_index = kv_token_index % uint(NUM_TOKENS);
        ulong page_id_table_index =
            ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)GQA_LAYER_INDEX)
              * (ulong)NUM_BLOCKS
              + (ulong)block_index)
             * (ulong)NUM_PAGE_IDS_PER_BLOCK)
            + (ulong)page_id_index;
        ulong page_id = (ulong)page_ids[page_id_table_index];
        uint token_offset = kv_token_index - kv_token_tile_begin;
        const device KV_T* k_ptr =
            kv_pages + (page_id * ((ulong)PAGE_BYTES / sizeof(KV_T))
                        + (ulong)(((0 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS
                                   + page_token_index)
                                  * KV_HEAD_DIM));

        float scores[Q_HEAD_TILE_SIZE];
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            scores[local_q_head] = 0.0f;
        }
        for (uint d = 0; d < uint(KV_HEAD_DIM); ++d) {
            float k = static_cast<float>(k_ptr[d]);
            for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
                const device T* q_ptr = q_tile_ptr + local_q_head * KV_HEAD_DIM;
                scores[local_q_head] += static_cast<float>(q_ptr[d]) * k;
            }
        }
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            float score = scores[local_q_head] * ATTENTION_SCALE;
            logits[local_q_head * KV_TOKEN_TILE_SIZE + token_offset] = score;
            local_max[local_q_head] = metal::max(local_max[local_q_head], score);
        }
    }

    float tile_max[Q_HEAD_TILE_SIZE];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        reduce_scratch[thread_index_in_threadblock] = local_max[local_q_head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = uint(NUM_THREADS_PER_THREADBLOCK / 2); stride > 0; stride >>= 1) {
            if (thread_index_in_threadblock < stride) {
                reduce_scratch[thread_index_in_threadblock] = metal::max(reduce_scratch[thread_index_in_threadblock], reduce_scratch[thread_index_in_threadblock + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tile_max[local_q_head] = reduce_scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_exp_sum[Q_HEAD_TILE_SIZE];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        local_exp_sum[local_q_head] = 0.0f;
    }
    for (uint token_offset = thread_index_in_threadblock; token_offset < kv_token_tile_len;
         token_offset += uint(NUM_THREADS_PER_THREADBLOCK)) {
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            uint logits_index = local_q_head * uint(KV_TOKEN_TILE_SIZE) + token_offset;
            float weight = metal::exp(logits[logits_index] - tile_max[local_q_head]);
            logits[logits_index] = weight;
            local_exp_sum[local_q_head] += weight;
        }
    }

    float tile_exp_sum[Q_HEAD_TILE_SIZE];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        reduce_scratch[thread_index_in_threadblock] = local_exp_sum[local_q_head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = uint(NUM_THREADS_PER_THREADBLOCK / 2); stride > 0; stride >>= 1) {
            if (thread_index_in_threadblock < stride) {
                reduce_scratch[thread_index_in_threadblock] += reduce_scratch[thread_index_in_threadblock + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tile_exp_sum[local_q_head] = reduce_scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float old_scale[Q_HEAD_TILE_SIZE];
    float tile_scale[Q_HEAD_TILE_SIZE];
    float next_max[Q_HEAD_TILE_SIZE];
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        next_max[local_q_head] = metal::max(running_max[local_q_head], tile_max[local_q_head]);
        old_scale[local_q_head] = isfinite(running_max[local_q_head])
            ? metal::exp(running_max[local_q_head] - next_max[local_q_head])
            : 0.0f;
        tile_scale[local_q_head] = metal::exp(tile_max[local_q_head] - next_max[local_q_head]);
    }

    for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
        uint d = thread_index_in_threadblock + dim_slot * uint(NUM_THREADS_PER_THREADBLOCK);
        if (d >= uint(KV_HEAD_DIM)) {
            continue;
        }
        float tile_output[Q_HEAD_TILE_SIZE];
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            tile_output[local_q_head] = 0.0f;
        }
        for (uint token_offset = 0; token_offset < kv_token_tile_len; ++token_offset) {
            uint kv_token_index = kv_token_tile_begin + token_offset;
            uint block_index = kv_token_index / uint(NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
            uint page_id_index = (kv_token_index / uint(NUM_TOKENS)) % uint(NUM_PAGE_IDS_PER_BLOCK);
            uint page_token_index = kv_token_index % uint(NUM_TOKENS);
            ulong page_id_table_index =
                ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)GQA_LAYER_INDEX)
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
            float v = static_cast<float>(v_ptr[d]);
            for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
                tile_output[local_q_head] += logits[local_q_head * KV_TOKEN_TILE_SIZE + token_offset] * v;
            }
        }
        for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
            running_output[local_q_head][dim_slot] =
                running_output[local_q_head][dim_slot] * old_scale[local_q_head]
                + tile_output[local_q_head] * tile_scale[local_q_head];
        }
    }

    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        running_exp_sum[local_q_head] =
            running_exp_sum[local_q_head] * old_scale[local_q_head]
            + tile_exp_sum[local_q_head] * tile_scale[local_q_head];
        running_max[local_q_head] = next_max[local_q_head];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (thread_index_in_threadblock == 0) {
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        uint q_head_index = q_head_base + local_q_head;
        ulong partial_output_index = (ulong)sdpa_map_task_template_index * (ulong)num_q_heads + (ulong)q_head_index;
        partial_exp_sums[partial_output_index] = running_exp_sum[local_q_head];
        partial_max_logits[partial_output_index] = running_max[local_q_head];
    }
}
for (uint dim_slot = 0; dim_slot < uint(NUM_DIMS_PER_THREAD); ++dim_slot) {
    uint d = thread_index_in_threadblock + dim_slot * uint(NUM_THREADS_PER_THREADBLOCK);
    if (d >= uint(KV_HEAD_DIM)) {
        continue;
    }
    for (uint local_q_head = 0; local_q_head < num_active_q_heads; ++local_q_head) {
        uint q_head_index = q_head_base + local_q_head;
        ulong partial_output_index = (ulong)sdpa_map_task_template_index * (ulong)num_q_heads + (ulong)q_head_index;
        float inv_exp_sum = running_exp_sum[local_q_head] > 0.0f
            ? 1.0f / running_exp_sum[local_q_head]
            : 0.0f;
        partial_output[partial_output_index * KV_HEAD_DIM + d] =
            static_cast<T>(running_output[local_q_head][dim_slot] * inv_exp_sum);
    }
}
