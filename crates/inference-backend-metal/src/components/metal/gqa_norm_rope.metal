
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
constant int RMS_N_READS = 4;

template <typename T>
void gqa_norm_rope_impl(
    device const T* input,
    device const float* norm_weight,
    device const uint* flat_token_indices,
    device T* output,
    constant uint& num_tokens,
    threadgroup float* local_inv_mean,
    threadgroup float* local_sums,
    uint gid,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id
) {
    const uint row = gid;
    const uint total_rows = num_tokens * num_heads;
    if (row >= total_rows) return;

    const uint token_index = row / num_heads;
    const uint row_base = row * head_dim;

    constexpr int SIMD_SIZE = 32;
    float acc = 0.0f;
    const device T* row_input = input + row_base + lid * RMS_N_READS;
    for (uint r = 0; r < head_dim; r += lsize * RMS_N_READS) {
        if (r + lid * RMS_N_READS + RMS_N_READS <= head_dim) {
            for (int i = 0; i < RMS_N_READS; i++) {
                float x = float(row_input[i + r]);
                acc += x * x;
            }
        } else {
            for (int i = 0; i < RMS_N_READS; i++) {
                if (r + lid * RMS_N_READS + i < head_dim) {
                    float x = float(row_input[i + r]);
                    acc += x * x;
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(head_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rope_half = rope_dim / 2;
    const float log2_base = log2(rope_theta);
    const float position = rope_scale * float(flat_token_indices[token_index]);

    for (uint d = lid; d < head_dim; d += lsize) {
        if (d < rope_half) {
            const float x1 = float(T(norm_weight[d]) * T(float(input[row_base + d]) * local_inv_mean[0]));
            const uint d2 = d + rope_half;
            const float x2 = float(T(norm_weight[d2]) * T(float(input[row_base + d2]) * local_inv_mean[0]));
            const float freq_index = float(d) / float(rope_half);
            const float inv_freq = exp2(-freq_index * log2_base);
            const float theta = position * inv_freq;
            const float c = metal::fast::cos(theta);
            const float s = metal::fast::sin(theta);
            output[row_base + d] = T(x1 * c - x2 * s);
            output[row_base + d2] = T(x1 * s + x2 * c);
        } else if (d >= rope_dim) {
            output[row_base + d] = T(norm_weight[d]) * T(float(input[row_base + d]) * local_inv_mean[0]);
        }
    }
}

kernel void gqa_norm_rope_f32(
    device const float* input [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const uint* flat_token_indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    gqa_norm_rope_impl<float>(
        input, norm_weight, flat_token_indices, output, num_tokens,
        local_inv_mean, local_sums, gid, lid, lsize, simd_lane_id, simd_group_id);
}

kernel void gqa_norm_rope_bf16(
    device const bfloat16_t* input [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const uint* flat_token_indices [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    gqa_norm_rope_impl<bfloat16_t>(
        input, norm_weight, flat_token_indices, output, num_tokens,
        local_inv_mean, local_sums, gid, lid, lsize, simd_lane_id, simd_group_id);
}
