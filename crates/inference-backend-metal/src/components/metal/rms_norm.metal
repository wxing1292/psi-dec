
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint RMS_NORM_F32_VALUES_PER_THREAD = 4;

void rms_norm_f32_impl(
    device const float* input,
    device const float* weight,
    device float* output,
    constant uint& num_tokens,
    constant uint& hidden_dim,
    constant float& eps,
    threadgroup float* local_inv_mean,
    threadgroup float* local_sums,
    uint gid,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id
) {
    const uint row = gid;
    if (row >= num_tokens) return;

    float acc = 0.0f;
    const device float* row_input = input + row * size_t(hidden_dim) + lid * RMS_NORM_F32_VALUES_PER_THREAD;
    for (uint r = 0; r < hidden_dim; r += lsize * RMS_NORM_F32_VALUES_PER_THREAD) {
        if (r + lid * RMS_NORM_F32_VALUES_PER_THREAD + RMS_NORM_F32_VALUES_PER_THREAD <= hidden_dim) {
            for (uint i = 0; i < RMS_NORM_F32_VALUES_PER_THREAD; i++) {
                float x = row_input[i + r];
                acc += x * x;
            }
        } else {
            for (uint i = 0; i < RMS_NORM_F32_VALUES_PER_THREAD; i++) {
                if (r + lid * RMS_NORM_F32_VALUES_PER_THREAD + i < hidden_dim) {
                    float x = row_input[i + r];
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
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(hidden_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* row_output = output + row * size_t(hidden_dim) + lid * RMS_NORM_F32_VALUES_PER_THREAD;
    const device float* row_weight = weight + lid * RMS_NORM_F32_VALUES_PER_THREAD;
    for (uint r = 0; r < hidden_dim; r += lsize * RMS_NORM_F32_VALUES_PER_THREAD) {
        if (r + lid * RMS_NORM_F32_VALUES_PER_THREAD + RMS_NORM_F32_VALUES_PER_THREAD <= hidden_dim) {
            for (uint i = 0; i < RMS_NORM_F32_VALUES_PER_THREAD; i++) {
                row_output[i + r] = row_weight[i + r] * (row_input[i + r] * local_inv_mean[0]);
            }
        } else {
            for (uint i = 0; i < RMS_NORM_F32_VALUES_PER_THREAD; i++) {
                if (r + lid * RMS_NORM_F32_VALUES_PER_THREAD + i < hidden_dim) {
                    row_output[i + r] = row_weight[i + r] * (row_input[i + r] * local_inv_mean[0]);
                }
            }
        }
    }
}

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    rms_norm_f32_impl(
        input, weight, output, num_tokens, hidden_dim, eps, local_inv_mean, local_sums, gid, lid, lsize,
        simd_lane_id, simd_group_id);
}

kernel void rms_norm_bf16_vec4(
    device const bfloat4* input [[buffer(0)]],
    device const bfloat4* weight [[buffer(1)]],
    device bfloat4* output [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid;
    if (row >= num_tokens) return;

    // One thread processes one bfloat4 at a time. Adjacent threads process adjacent vectors.
    const uint hidden_dim_vec = hidden_dim / 4;
    float acc = 0.0f;
    const device bfloat4* row_input = input + row * size_t(hidden_dim_vec);
    for (uint vector_index = lid; vector_index < hidden_dim_vec; vector_index += lsize) {
        float4 x = float4(row_input[vector_index]);
        acc += dot(x, x);
    }
    acc = simd_sum(acc);
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
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
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(hidden_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device bfloat4* row_output = output + row * size_t(hidden_dim_vec);
    for (uint vector_index = lid; vector_index < hidden_dim_vec; vector_index += lsize) {
        row_output[vector_index] =
            weight[vector_index] * bfloat4(float4(row_input[vector_index]) * local_inv_mean[0]);
    }
}
