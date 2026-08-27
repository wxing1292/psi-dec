#include <metal_stdlib>
using namespace metal;

// One 32-thread threadgroup owns one Decode request. This is one Apple
// SIMDgroup on current targets, but the kernel does not use SIMDgroup intrinsics.
// Thread 0 writes request scalars. All threads stride the fixed Spec block
// and each history-Task quota. Acceptance changes endpoints, not work or the
// dispatched grid.
kernel void rejection_to_spec_decode_input(
    device const uint* num_accepted_tokens [[buffer(0)]],
    device const int* sampled_token_ids [[buffer(1)]],
    device const uint* anchor_indices [[buffer(2)]],
    device int* anchor_token_ids [[buffer(3)]],
    device uint* sample_positions [[buffer(4)]],
    device int* block_token_ids [[buffer(5)]],
    device uint* flat_query_token_indices [[buffer(6)]],
    device uint2* visible_history_token_ranges [[buffer(7)]],
    device const uint2* q_token_ranges [[buffer(8)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(9)]],
    device uint* sdpa_map_task_templates [[buffer(10)]],
    constant uint& num_active_requests [[buffer(11)]],
    constant uint& spec_block_size [[buffer(12)]],
    constant uint& num_q_ranges_per_request [[buffer(13)]],
    constant uint& kv_tokens_per_iteration [[buffer(14)]],
    constant uint& history_window [[buffer(15)]],
    constant int& mask_token_id [[buffer(16)]],
    uint request_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    if (request_index >= num_active_requests) {
        return;
    }

    const uint anchor_index = anchor_indices[request_index] + num_accepted_tokens[request_index];
    const int anchor_token_id = sampled_token_ids[request_index];
    if (thread_index == 0) {
        anchor_token_ids[request_index] = anchor_token_id;
        sample_positions[request_index] = anchor_index + 1;
    }

    const uint spec_block_begin = request_index * spec_block_size;
    for (uint block_offset = thread_index; block_offset < spec_block_size; block_offset += 32) {
        const uint query_index = spec_block_begin + block_offset;
        const uint query_position = anchor_index + block_offset;
        block_token_ids[query_index] = block_offset == 0 ? anchor_token_id : mask_token_id;
        flat_query_token_indices[query_index] = query_position;
        visible_history_token_ranges[query_index] = uint2(
            query_position + 1 > history_window ? query_position + 1 - history_window : 0,
            anchor_index
        );
    }

    const uint q_token_range_begin = request_index * num_q_ranges_per_request;
    for (uint local_range_index = 0; local_range_index < num_q_ranges_per_request; ++local_range_index) {
        const uint q_token_range_index = q_token_range_begin + local_range_index;
        const uint local_q_token_begin = q_token_ranges[q_token_range_index].x - spec_block_begin;
        const uint first_query_position = anchor_index + local_q_token_begin;
        const uint history_begin =
            first_query_position + 1 > history_window ? first_query_position + 1 - history_window : 0;
        const uint task_begin = cu_sdpa_partial_outputs[q_token_range_index];
        const uint task_end = cu_sdpa_partial_outputs[q_token_range_index + 1] - 1;
        const uint num_tasks = task_end - task_begin;
        const uint history_tokens = anchor_index - history_begin;
        const uint num_iterations = history_tokens / kv_tokens_per_iteration
            + (history_tokens % kv_tokens_per_iteration == 0 ? 0 : 1);
        // The threadgroup cooperatively handles one Q-token range at a time.
        // Threads stride its fixed history-Task quota.
        for (uint task_offset = thread_index; task_offset < num_tasks; task_offset += 32) {
            const uint iteration_begin = num_iterations * task_offset / num_tasks;
            const uint iteration_end = num_iterations * (task_offset + 1) / num_tasks;
            const uint token_begin = history_begin + iteration_begin * kv_tokens_per_iteration;
            const uint token_end = metal::min(
                anchor_index,
                history_begin + iteration_end * kv_tokens_per_iteration
            );
            const uint task_template_index = task_begin + task_offset;
            sdpa_map_task_templates[task_template_index * 3] = q_token_range_index;
            sdpa_map_task_templates[task_template_index * 3 + 1] = token_begin;
            sdpa_map_task_templates[task_template_index * 3 + 2] = token_end;
        }
    }
}
