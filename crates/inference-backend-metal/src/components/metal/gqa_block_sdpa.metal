#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// One BlockSDPAThreadBlockTask maps 1:1 to one threadblock. The grid supplies
// q_head_index, q_token_range_index, and q_token_offset through x, y, and z.
// q_token_ranges derives q_token_index. block_size derives the local K/V block.
// cu_sdpa_partial_outputs derives the selected block partial-output slot.

template <typename T>
void gqa_block_sdpa_impl(
    device const T* q,
    device const T* local_k,
    device const T* local_v,
    device const uint* q_token_ranges,
    device const uint* cu_sdpa_partial_outputs,
    device float* partial_exp_sums,
    device float* partial_max_logits,
    device T* partial_output,
    threadgroup float* logits,
    uint3 group_index,
    uint simd_lane
) {
    const uint q_head_index = group_index.x;
    const uint q_token_range_index = group_index.y;
    const uint q_token_offset = group_index.z;
    const uint flat_q_token_begin = q_token_ranges[q_token_range_index * 2];
    const uint flat_q_token_end = q_token_ranges[q_token_range_index * 2 + 1];
    if (q_token_offset >= flat_q_token_end - flat_q_token_begin) return;
    const uint q_token_index = flat_q_token_begin + q_token_offset;
    const uint q_heads_per_kv_head = num_q_heads / num_kv_heads;
    const uint kv_head_index = q_head_index / q_heads_per_kv_head;
    const uint local_kv_token_begin = (q_token_index / block_size) * block_size;
    const uint partial_output_slot = cu_sdpa_partial_outputs[q_token_range_index + 1] - 1;
    const ulong q_offset = ((ulong)q_token_index * num_q_heads + q_head_index) * head_dim;
    const device T* q_ptr = q + q_offset;
    thread float q_values[q_values_per_thread];
    for (uint value_index = 0; value_index < q_values_per_thread; ++value_index) {
        q_values[value_index] = float(q_ptr[value_index * simd_width + simd_lane]);
    }

    for (uint local_kv_offset = 0; local_kv_offset < block_size; ++local_kv_offset) {
        const uint kv_token_index = local_kv_token_begin + local_kv_offset;
        const ulong k_offset = ((ulong)kv_token_index * num_kv_heads + kv_head_index) * head_dim;
        const device T* k_ptr = local_k + k_offset;
        float score = 0.0f;
        for (uint value_index = 0; value_index < q_values_per_thread; ++value_index) {
            score += q_values[value_index] * float(k_ptr[value_index * simd_width + simd_lane]);
        }
        score = simd_sum(score) * attention_scale;
        if (simd_lane == 0) logits[local_kv_offset] = score;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float block_max = -INFINITY;
    float block_exp_sum = 0.0f;
    if (simd_lane == 0) {
        for (uint local_kv_offset = 0; local_kv_offset < block_size; ++local_kv_offset) {
            block_max = metal::max(block_max, logits[local_kv_offset]);
        }
        for (uint local_kv_offset = 0; local_kv_offset < block_size; ++local_kv_offset) {
            const float weight = metal::exp(logits[local_kv_offset] - block_max);
            logits[local_kv_offset] = weight;
            block_exp_sum += weight;
        }
        const ulong partial_output_index =
            ((ulong)partial_output_slot * num_q_heads + q_head_index) * max_q_tokens + q_token_offset;
        partial_exp_sums[partial_output_index] = block_exp_sum;
        partial_max_logits[partial_output_index] = block_max;
    }
    block_exp_sum = simd_broadcast_first(block_exp_sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const ulong partial_output_index =
        ((ulong)partial_output_slot * num_q_heads + q_head_index) * max_q_tokens + q_token_offset;
    for (uint dim = simd_lane; dim < head_dim; dim += simd_width) {
        float output = 0.0f;
        for (uint local_kv_offset = 0; local_kv_offset < block_size; ++local_kv_offset) {
            const uint kv_token_index = local_kv_token_begin + local_kv_offset;
            const ulong v_offset = ((ulong)kv_token_index * num_kv_heads + kv_head_index) * head_dim + dim;
            output += logits[local_kv_offset] * float(local_v[v_offset]);
        }
        partial_output[(partial_output_index * head_dim) + dim] =
            T(block_exp_sum > 0.0f ? output / block_exp_sum : 0.0f);
    }
}

kernel void gqa_block_sdpa_f32(
    device const float* q [[buffer(0)]],
    device const float* local_k [[buffer(1)]],
    device const float* local_v [[buffer(2)]],
    device const uint* q_token_ranges [[buffer(3)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device float* partial_output [[buffer(7)]],
    uint3 group_index [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup float logits[block_size];
    gqa_block_sdpa_impl<float>(
        q, local_k, local_v, q_token_ranges, cu_sdpa_partial_outputs, partial_exp_sums, partial_max_logits,
        partial_output, logits, group_index, simd_lane);
}

kernel void gqa_block_sdpa_bf16(
    device const bfloat16_t* q [[buffer(0)]],
    device const bfloat16_t* local_k [[buffer(1)]],
    device const bfloat16_t* local_v [[buffer(2)]],
    device const uint* q_token_ranges [[buffer(3)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device bfloat16_t* partial_output [[buffer(7)]],
    uint3 group_index [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup float logits[block_size];
    gqa_block_sdpa_impl<bfloat16_t>(
        q, local_k, local_v, q_token_ranges, cu_sdpa_partial_outputs, partial_exp_sums, partial_max_logits,
        partial_output, logits, group_index, simd_lane);
}
