#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void gqa_local_sdpa_impl(
    device const T* q,
    device const T* local_k,
    device const T* local_v,
    device const uint* local_sdpa_map_task_template_indices,
    device float* partial_exp_sums,
    device float* partial_max_logits,
    device T* partial_output,
    constant uint& num_tokens,
    threadgroup float* logits,
    threadgroup float* reduction,
    uint group_index,
    uint lane
) {
    const uint q_token_index = group_index % num_tokens;
    const uint q_head_index = group_index / num_tokens;
    if (q_head_index >= num_q_heads) return;

    const uint q_heads_per_kv_head = num_q_heads / num_kv_heads;
    const uint kv_head_index = q_head_index / q_heads_per_kv_head;
    const uint local_kv_token_begin = (q_token_index / local_block_size) * local_block_size;
    const uint sdpa_map_task_template_index = local_sdpa_map_task_template_indices[q_token_index];
    const ulong q_offset = ((ulong)q_token_index * num_q_heads + q_head_index) * head_dim;
    const device T* q_ptr = q + q_offset;

    float local_max = -INFINITY;
    for (uint local_kv_offset = lane; local_kv_offset < local_block_size;
         local_kv_offset += num_threads_per_threadblock) {
        const uint kv_token_index = local_kv_token_begin + local_kv_offset;
        const ulong k_offset = ((ulong)kv_token_index * num_kv_heads + kv_head_index) * head_dim;
        const device T* k_ptr = local_k + k_offset;
        float score = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            score += float(q_ptr[dim]) * float(k_ptr[dim]);
        }
        score *= attention_scale;
        logits[local_kv_offset] = score;
        local_max = metal::max(local_max, score);
    }

    reduction[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = num_threads_per_threadblock / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = metal::max(reduction[lane], reduction[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float block_max = reduction[0];

    float local_exp_sum = 0.0f;
    for (uint local_kv_offset = lane; local_kv_offset < local_block_size;
         local_kv_offset += num_threads_per_threadblock) {
        const float weight = metal::exp(logits[local_kv_offset] - block_max);
        logits[local_kv_offset] = weight;
        local_exp_sum += weight;
    }
    reduction[lane] = local_exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = num_threads_per_threadblock / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float block_exp_sum = reduction[0];
    const ulong partial_output_index = (ulong)sdpa_map_task_template_index * num_q_heads + q_head_index;
    if (lane == 0) {
        partial_exp_sums[partial_output_index] = block_exp_sum;
        partial_max_logits[partial_output_index] = block_max;
    }

    for (uint dim = lane; dim < head_dim; dim += num_threads_per_threadblock) {
        float output = 0.0f;
        for (uint local_kv_offset = 0; local_kv_offset < local_block_size; ++local_kv_offset) {
            const uint kv_token_index = local_kv_token_begin + local_kv_offset;
            const ulong v_offset = ((ulong)kv_token_index * num_kv_heads + kv_head_index) * head_dim + dim;
            output += logits[local_kv_offset] * float(local_v[v_offset]);
        }
        partial_output[(partial_output_index * head_dim) + dim] =
            T(block_exp_sum > 0.0f ? output / block_exp_sum : 0.0f);
    }
}

kernel void gqa_local_sdpa_f32(
    device const float* q [[buffer(0)]],
    device const float* local_k [[buffer(1)]],
    device const float* local_v [[buffer(2)]],
    device const uint* local_sdpa_map_task_template_indices [[buffer(3)]],
    device float* partial_exp_sums [[buffer(4)]],
    device float* partial_max_logits [[buffer(5)]],
    device float* partial_output [[buffer(6)]],
    constant uint& num_tokens [[buffer(7)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float logits[local_block_size];
    threadgroup float reduction[num_threads_per_threadblock];
    gqa_local_sdpa_impl<float>(
        q, local_k, local_v, local_sdpa_map_task_template_indices, partial_exp_sums, partial_max_logits, partial_output,
        num_tokens, logits, reduction, group_index, lane);
}

kernel void gqa_local_sdpa_bf16(
    device const bfloat16_t* q [[buffer(0)]],
    device const bfloat16_t* local_k [[buffer(1)]],
    device const bfloat16_t* local_v [[buffer(2)]],
    device const uint* local_sdpa_map_task_template_indices [[buffer(3)]],
    device float* partial_exp_sums [[buffer(4)]],
    device float* partial_max_logits [[buffer(5)]],
    device bfloat16_t* partial_output [[buffer(6)]],
    constant uint& num_tokens [[buffer(7)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float logits[local_block_size];
    threadgroup float reduction[num_threads_per_threadblock];
    gqa_local_sdpa_impl<bfloat16_t>(
        q, local_k, local_v, local_sdpa_map_task_template_indices, partial_exp_sums, partial_max_logits, partial_output,
        num_tokens, logits, reduction, group_index, lane);
}
