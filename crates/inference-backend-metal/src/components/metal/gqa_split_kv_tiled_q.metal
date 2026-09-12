#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// SplitKV TiledQ SDPA map. One MapThreadBlockTask maps 1:1 to one threadblock. Its
// fields are sourced as follows:
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
//   kv_head_index,        // grid-derived from threadblock_position.x
//   q_head_range_index,   // grid-derived from threadblock_position.x
// }
//
// visible_kv_token_ranges stores one request-local half-open
// [kv_token_begin, kv_token_end) range for each flat Q token. Each row computes
// the intersection of this range and its Map task-template range.
//
// A sentinel Map task template returns without writing any partial output or
// statistics.

// For one Q-token-range/Q-head output coordinate, adjacent
// cu_sdpa_partial_outputs values select the leading partial-output dimension to
// merge. The cumulative values do not count scalar tensor elements.
kernel void gqa_split_kv_tiled_q_reduce(
    device const bfloat16_t* partial_output [[buffer(0)]],
    device const float* partial_exp_sums [[buffer(1)]],
    device const float* partial_max_logits [[buffer(2)]],
    device const uint* q_token_ranges [[buffer(3)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    constant uint& num_active_q_token_tiles [[buffer(6)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    const uint q_head_index = threadblock_position.x;
    const uint q_token_range_index = threadblock_position.y;
    if (q_token_range_index >= num_active_q_token_tiles) {
        return;
    }
    const uint flat_token_start = q_token_ranges[q_token_range_index * 2];
    const uint flat_token_end = q_token_ranges[q_token_range_index * 2 + 1];
    const uint num_q_tokens_in_range = flat_token_end - flat_token_start;
    const uint partial_output_begin = cu_sdpa_partial_outputs[q_token_range_index];
    const uint partial_output_end = cu_sdpa_partial_outputs[q_token_range_index + 1];

    for (uint local_index = thread_index; local_index < num_q_tokens_in_range * uint(HEAD_DIM);
         local_index += uint(REDUCE_REQUIRED_THREADS)) {
        const uint local_token_index = local_index / uint(HEAD_DIM);
        const uint dim = local_index % uint(HEAD_DIM);
        float global_max = -INFINITY;
        for (uint partial_output_index = partial_output_begin;
             partial_output_index < partial_output_end;
             ++partial_output_index) {
            const ulong partial_output_stats_index =
                ((ulong)partial_output_index * NUM_Q_HEADS + (ulong)q_head_index) * MAX_Q_TOKENS
                + (ulong)local_token_index;
            global_max = max(global_max, partial_max_logits[partial_output_stats_index]);
        }
        float global_sum = 0.0f;
        float v = 0.0f;
        for (uint partial_output_index = partial_output_begin;
             partial_output_index < partial_output_end;
             ++partial_output_index) {
            const ulong partial_output_stats_index =
                ((ulong)partial_output_index * NUM_Q_HEADS + (ulong)q_head_index) * MAX_Q_TOKENS
                + (ulong)local_token_index;
            const float partial_exp_sum = partial_exp_sums[partial_output_stats_index];
            const float weight = partial_exp_sum == 0.0f
                ? 0.0f
                : metal::exp(partial_max_logits[partial_output_stats_index] - global_max) * partial_exp_sum;
            const ulong partial_output_value_index = partial_output_stats_index * HEAD_DIM + (ulong)dim;
            global_sum += weight;
            v += weight * float(partial_output[partial_output_value_index]);
        }
        const ulong output_index =
            ((ulong)(flat_token_start + local_token_index) * NUM_Q_HEADS + (ulong)q_head_index) * HEAD_DIM
            + (ulong)dim;
        output[output_index] = bfloat16_t(global_sum > 0.0f ? v / global_sum : 0.0f);
    }
}
