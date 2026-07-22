#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// Tiled SDPA map. One logical SDPAMapTask maps 1:1 to one threadblock. Its
// fields are sourced as follows:
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
//   kv_head_index,       // grid-derived from threadblock_position.x
//   q_head_tile_index,   // grid-derived from threadblock_position.x
// }
//
// A sentinel TaskTemplate returns without writing any partial output or
// statistics.

template <int NBYTES> struct LoadUnit;
template <> struct LoadUnit<16> { using type = uint4; };

inline ushort2 frag_coord(ushort lane_id) {
    const ushort qid = lane_id / 4;
    const ushort row = (qid & 4) + (lane_id / 2) % 4;
    const ushort col = (qid & 2) * 2 + (lane_id % 2) * 2;
    return ushort2{col, row};
}

struct FragMax {
    static inline float apply(float a, float b) { return max(a, b); }
};

struct FragSum {
    static inline float apply(float a, float b) { return a + b; }
};

template <typename Op>
inline float frag_row_reduce(float v) {
    v = Op::apply(v, simd_shuffle_xor(v, ushort(1)));
    v = Op::apply(v, simd_shuffle_xor(v, ushort(8)));
    return v;
}

kernel void gqa_tiled_sdpa_map(
    device const bfloat16_t* q [[buffer(0)]],
    device const bfloat16_t* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* flat_token_indices [[buffer(4)]],
    device const uint* q_token_tiles [[buffer(5)]],
    device const uint* sdpa_map_task_templates [[buffer(6)]],
    device bfloat16_t* partial_output [[buffer(7)]],
    device float* partial_exp_sums [[buffer(8)]],
    device float* partial_max_logits [[buffer(9)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int NUM_SIMD_LANES = 32;
    constexpr int NUM_SIMDGROUPS = NUM_THREADS_PER_THREADBLOCK / NUM_SIMD_LANES;
    constexpr int NUM_SIMDGROUPS_PER_Q_HEAD = Q_TOKEN_TILE_SIZE / 8;
    constexpr int NUM_HEAD_FRAGMENTS = HEAD_DIM / 8;
    constexpr int NUM_KV_TOKEN_FRAGMENTS = KV_TOKEN_TILE_SIZE / 8;
    constexpr int Q_HEADS_PER_KV_HEAD = NUM_Q_HEADS / NUM_KV_HEADS;
    static_assert(NUM_SIMDGROUPS == NUM_SIMDGROUPS_PER_Q_HEAD * Q_HEAD_TILE_SIZE);

    const uint sdpa_map_task_template_index = threadblock_position.y;
    const uint q_token_tile_index = sdpa_map_task_templates[sdpa_map_task_template_index * 3];
    if (q_token_tile_index >= uint(NUM_Q_TOKEN_TILES)) {
        return;
    }
    const uint flat_token_start = q_token_tiles[q_token_tile_index * 2];
    const uint flat_token_end = q_token_tiles[q_token_tile_index * 2 + 1];
    const uint num_tile_tokens = flat_token_end - flat_token_start;
    const uint kv_token_begin = sdpa_map_task_templates[sdpa_map_task_template_index * 3 + 1];
    const uint kv_token_end = sdpa_map_task_templates[sdpa_map_task_template_index * 3 + 2];
    const uint req_slot = req_slots[flat_token_start];

    const uint head_group_index = threadblock_position.x;
    const uint q_head_tile_index = head_group_index % uint(NUM_Q_HEAD_TILES_PER_KV_HEAD);
    const uint kv_head_index = head_group_index / uint(NUM_Q_HEAD_TILES_PER_KV_HEAD);
    const uint q_head_tile_base = q_head_tile_index * uint(Q_HEAD_TILE_SIZE);
    const uint num_active_q_heads = min(
        uint(Q_HEAD_TILE_SIZE), uint(Q_HEADS_PER_KV_HEAD) - q_head_tile_base);
    const uint local_q_head_index = simdgroup_index / uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const uint token_fragment_index = simdgroup_index % uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const bool active_q_head = local_q_head_index < num_active_q_heads;
    const uint q_head_index = min(
        kv_head_index * uint(Q_HEADS_PER_KV_HEAD) + q_head_tile_base + local_q_head_index,
        uint(NUM_Q_HEADS - 1));

    constexpr int PAD = 16 / int(sizeof(bfloat16_t));
    constexpr int LEADING_DIM = HEAD_DIM + PAD;
    constexpr int KV_TOKEN_VALUES = KV_TOKEN_TILE_SIZE * LEADING_DIM;
    threadgroup bfloat16_t* k_shared = reinterpret_cast<threadgroup bfloat16_t*>(shared_mem);
    threadgroup bfloat16_t* v_shared = k_shared + KV_TOKEN_VALUES;

    const ushort2 coordinate = frag_coord(ushort(lane));
    const ushort token_fragment_offset = coordinate.y;
    const ushort fragment_col = coordinate.x;
    const uint token_offset = token_fragment_index * 8 + uint(token_fragment_offset);
    const bool inactive_token = !active_q_head || token_offset >= num_tile_tokens;
    const bool full_token_fragment = token_fragment_index * 8 + 8 <= num_tile_tokens;
    const uint causal_token_index = inactive_token ? 0 : flat_token_indices[flat_token_start + token_offset];

    using Vec2BF16 = vec<bfloat16_t, 2>;
    using Vec2F32 = vec<float, 2>;
    Vec2BF16 q_fragments[NUM_HEAD_FRAGMENTS];
    #pragma unroll
    for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
        q_fragments[dim_fragment] = Vec2BF16(0);
        if (active_q_head && full_token_fragment) {
            simdgroup_matrix<bfloat16_t, 8, 8> q_fragment;
            const ulong q_base_offset =
                ((ulong)flat_token_start * NUM_Q_HEADS + (ulong)q_head_index) * HEAD_DIM
                + (ulong)token_fragment_index * 8 * NUM_Q_HEADS * HEAD_DIM;
            const device bfloat16_t* q_base = q + q_base_offset;
            simdgroup_load(
                q_fragment,
                q_base + dim_fragment * 8,
                NUM_Q_HEADS * HEAD_DIM);
            q_fragments[dim_fragment] =
                reinterpret_cast<thread Vec2BF16&>(q_fragment.thread_elements());
        } else if (!inactive_token) {
            const ulong q_base_offset =
                ((ulong)(flat_token_start + token_offset) * NUM_Q_HEADS + (ulong)q_head_index) * HEAD_DIM;
            const device bfloat16_t* q_base = q + q_base_offset;
            const int dim = dim_fragment * 8 + int(fragment_col);
            q_fragments[dim_fragment] = Vec2BF16(q_base[dim], q_base[dim + 1]);
        }
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    Vec2F32 output_fragments[NUM_HEAD_FRAGMENTS];
    #pragma unroll
    for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
        output_fragments[dim_fragment] = Vec2F32(0.0f);
    }

    const float scale_log2 = ATTENTION_SCALE * M_LOG2E_F;
    const uint num_kv_token_tiles =
        (kv_token_end - kv_token_begin + uint(KV_TOKEN_TILE_SIZE - 1)) / uint(KV_TOKEN_TILE_SIZE);
    for (uint kv_token_tile_index = 0; kv_token_tile_index < num_kv_token_tiles; ++kv_token_tile_index) {
        const uint kv_token_tile_begin = kv_token_begin + kv_token_tile_index * uint(KV_TOKEN_TILE_SIZE);
        constexpr int LOAD_BYTES = 16;
        constexpr int LOAD_VALUES = LOAD_BYTES / int(sizeof(bfloat16_t));
        using Load = typename LoadUnit<LOAD_BYTES>::type;
        static_assert(LEADING_DIM % LOAD_VALUES == 0);
        for (uint kv_token_offset = simdgroup_index; kv_token_offset < uint(KV_TOKEN_TILE_SIZE);
             kv_token_offset += uint(NUM_SIMDGROUPS)) {
            const uint kv_token_index = kv_token_tile_begin + kv_token_offset;
            if (kv_token_index < kv_token_end) {
                const uint block_index = kv_token_index / uint(NUM_TOKENS_PER_PAGE * NUM_PAGE_IDS_PER_BLOCK);
                const uint page_id_index =
                    (kv_token_index / uint(NUM_TOKENS_PER_PAGE)) % uint(NUM_PAGE_IDS_PER_BLOCK);
                const uint page_token_index = kv_token_index % uint(NUM_TOKENS_PER_PAGE);
                const ulong page_table_index =
                    ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)GQA_LAYER_INDEX)
                      * (ulong)NUM_BLOCKS + (ulong)block_index)
                     * (ulong)NUM_PAGE_IDS_PER_BLOCK) + (ulong)page_id_index;
                const ulong page_id = (ulong)page_ids[page_table_index];
                const ulong page_base = page_id * ((ulong)PAGE_BYTES / sizeof(bfloat16_t));
                const device bfloat16_t* k = kv_pages + page_base
                    + (ulong)(((0 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                const device bfloat16_t* v = kv_pages + page_base
                    + (ulong)(((1 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                #pragma unroll
                for (uint dim = lane * uint(LOAD_VALUES); dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES * LOAD_VALUES)) {
                    *((threadgroup Load*)&k_shared[kv_token_offset * LEADING_DIM + dim]) =
                        *((const device Load*)(k + dim));
                    *((threadgroup Load*)&v_shared[kv_token_offset * LEADING_DIM + dim]) =
                        *((const device Load*)(v + dim));
                }
            } else {
                const Load zero = {};
                #pragma unroll
                for (uint dim = lane * uint(LOAD_VALUES); dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES * LOAD_VALUES)) {
                    *((threadgroup Load*)&k_shared[kv_token_offset * LEADING_DIM + dim]) = zero;
                    *((threadgroup Load*)&v_shared[kv_token_offset * LEADING_DIM + dim]) = zero;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        Vec2F32 score_fragments[NUM_KV_TOKEN_FRAGMENTS];
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            score_fragments[kv_token_fragment] = Vec2F32(0.0f);
        }
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            simdgroup_matrix<bfloat16_t, 8, 8> q_fragment;
            reinterpret_cast<thread Vec2BF16&>(q_fragment.thread_elements()) = q_fragments[dim_fragment];
            #pragma unroll
            for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
                simdgroup_matrix<bfloat16_t, 8, 8> k_fragment;
                simdgroup_load(
                    k_fragment,
                    k_shared + kv_token_fragment * 8 * LEADING_DIM + dim_fragment * 8,
                    LEADING_DIM,
                    ulong2(0),
                    true);
                simdgroup_matrix<float, 8, 8> score_fragment;
                reinterpret_cast<thread Vec2F32&>(score_fragment.thread_elements()) =
                    score_fragments[kv_token_fragment];
                simdgroup_multiply_accumulate(score_fragment, q_fragment, k_fragment, score_fragment);
                score_fragments[kv_token_fragment] =
                    reinterpret_cast<thread Vec2F32&>(score_fragment.thread_elements());
            }
        }

        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            #pragma unroll
            for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                const uint kv_token_index = kv_token_tile_begin + uint(kv_token_fragment * 8)
                    + uint(fragment_col) + uint(fragment_slot);
                const float score = score_fragments[kv_token_fragment][fragment_slot] * scale_log2;
                score_fragments[kv_token_fragment][fragment_slot] =
                    inactive_token || kv_token_index > causal_token_index || kv_token_index >= kv_token_end
                    ? -INFINITY
                    : score;
            }
        }

        float local_max = -INFINITY;
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            local_max = max(local_max, max(score_fragments[kv_token_fragment][0], score_fragments[kv_token_fragment][1]));
        }
        const float tile_max = frag_row_reduce<FragMax>(local_max);
        float next_max = max(running_max, tile_max);
        const float running_scale = running_max == -INFINITY ? 0.0f : fast::exp2(running_max - next_max);
        if (running_max == -INFINITY && next_max == -INFINITY) {
            next_max = 0.0f;
        }
        running_max = next_max;

        float local_sum = 0.0f;
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            #pragma unroll
            for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                const float probability = score_fragments[kv_token_fragment][fragment_slot] == -INFINITY
                    ? 0.0f
                    : fast::exp2(score_fragments[kv_token_fragment][fragment_slot] - next_max);
                score_fragments[kv_token_fragment][fragment_slot] = probability;
                local_sum += probability;
            }
        }
        running_sum = running_sum * running_scale + frag_row_reduce<FragSum>(local_sum);
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            output_fragments[dim_fragment] *= running_scale;
        }

        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            simdgroup_matrix<float, 8, 8> probability_fragment;
            reinterpret_cast<thread Vec2F32&>(probability_fragment.thread_elements()) =
                score_fragments[kv_token_fragment];
            #pragma unroll
            for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
                simdgroup_matrix<bfloat16_t, 8, 8> v_fragment;
                simdgroup_load(
                    v_fragment,
                    v_shared + kv_token_fragment * 8 * LEADING_DIM + dim_fragment * 8,
                    LEADING_DIM);
                simdgroup_matrix<float, 8, 8> output_fragment;
                reinterpret_cast<thread Vec2F32&>(output_fragment.thread_elements()) =
                    output_fragments[dim_fragment];
                simdgroup_multiply_accumulate(output_fragment, probability_fragment, v_fragment, output_fragment);
                output_fragments[dim_fragment] =
                    reinterpret_cast<thread Vec2F32&>(output_fragment.thread_elements());
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!inactive_token) {
        const ulong partial_output_index =
            ((ulong)sdpa_map_task_template_index * NUM_Q_HEADS + (ulong)q_head_index) * Q_TOKEN_TILE_SIZE
            + (ulong)token_offset;
        if (fragment_col == 0) {
            partial_exp_sums[partial_output_index] = running_sum;
            partial_max_logits[partial_output_index] = running_max;
        }
        const float inverse_sum = 1.0f / (running_sum + 1.0e-6f);
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            #pragma unroll
            for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                const int dim = dim_fragment * 8 + int(fragment_col) + fragment_slot;
                partial_output[partial_output_index * HEAD_DIM + (ulong)dim] =
                    bfloat16_t(output_fragments[dim_fragment][fragment_slot] * inverse_sum);
            }
        }
    }
}

// For one Q-token-tile/Q-head output coordinate, adjacent
// cu_sdpa_partial_outputs values select the leading partial-output dimension to
// merge. The cumulative values do not count scalar tensor elements.
kernel void gqa_tiled_sdpa_reduce(
    device const bfloat16_t* partial_output [[buffer(0)]],
    device const float* partial_exp_sums [[buffer(1)]],
    device const float* partial_max_logits [[buffer(2)]],
    device const uint* q_token_tiles [[buffer(3)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    const uint q_head_index = threadblock_position.x;
    const uint q_token_tile_index = threadblock_position.y;
    const uint flat_token_start = q_token_tiles[q_token_tile_index * 2];
    const uint flat_token_end = q_token_tiles[q_token_tile_index * 2 + 1];
    const uint num_tile_tokens = flat_token_end - flat_token_start;
    const uint partial_output_begin = cu_sdpa_partial_outputs[q_token_tile_index];
    const uint partial_output_end = cu_sdpa_partial_outputs[q_token_tile_index + 1];

    for (uint local_index = thread_index; local_index < num_tile_tokens * uint(HEAD_DIM);
         local_index += uint(NUM_THREADS_PER_THREADBLOCK)) {
        const uint local_token_index = local_index / uint(HEAD_DIM);
        const uint dim = local_index % uint(HEAD_DIM);
        float global_max = -INFINITY;
        for (uint partial_output_index = partial_output_begin;
             partial_output_index < partial_output_end;
             ++partial_output_index) {
            const ulong partial_output_stats_index =
                ((ulong)partial_output_index * NUM_Q_HEADS + (ulong)q_head_index) * Q_TOKEN_TILE_SIZE
                + (ulong)local_token_index;
            global_max = max(global_max, partial_max_logits[partial_output_stats_index]);
        }
        float global_sum = 0.0f;
        float v = 0.0f;
        for (uint partial_output_index = partial_output_begin;
             partial_output_index < partial_output_end;
             ++partial_output_index) {
            const ulong partial_output_stats_index =
                ((ulong)partial_output_index * NUM_Q_HEADS + (ulong)q_head_index) * Q_TOKEN_TILE_SIZE
                + (ulong)local_token_index;
            const float weight = exp2(partial_max_logits[partial_output_stats_index] - global_max)
                * partial_exp_sums[partial_output_stats_index];
            const ulong partial_output_value_index = partial_output_stats_index * HEAD_DIM + (ulong)dim;
            global_sum += weight;
            v += weight * float(partial_output[partial_output_value_index]);
        }
        const ulong output_index =
            ((ulong)(flat_token_start + local_token_index) * NUM_Q_HEADS + (ulong)q_head_index) * HEAD_DIM
            + (ulong)dim;
        output[output_index] = bfloat16_t(v / (global_sum + 1.0e-6f));
    }
}
