#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;

constant int RESIDUAL_RMS_N_READS = 4;

template <typename T, bool DUPLICATE_RESIDUAL>
void residual_rms_norm_impl(
    device const T* lhs,
    device const T* rhs,
    device const T* weight,
    device T* residual_output,
    device T* duplicate_residual_output,
    device T* norm_output,
    constant uint& num_tokens,
    constant uint& hidden_dim,
    uint duplicate_row_stride,
    uint duplicate_column_offset,
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

    constexpr int SIMD_SIZE = 32;
    float acc = 0.0f;
    const device T* row_lhs = lhs + row * size_t(hidden_dim) + lid * RESIDUAL_RMS_N_READS;
    const device T* row_rhs = rhs + row * size_t(hidden_dim) + lid * RESIDUAL_RMS_N_READS;
    device T* row_residual_output = residual_output + row * size_t(hidden_dim) + lid * RESIDUAL_RMS_N_READS;
    device T* row_duplicate_residual_output =
        duplicate_residual_output + row * size_t(duplicate_row_stride) + duplicate_column_offset +
        lid * RESIDUAL_RMS_N_READS;
    for (uint r = 0; r < hidden_dim; r += lsize * RESIDUAL_RMS_N_READS) {
        if (r + lid * RESIDUAL_RMS_N_READS + RESIDUAL_RMS_N_READS <= hidden_dim) {
            for (int i = 0; i < RESIDUAL_RMS_N_READS; i++) {
                T residual = T(float(row_lhs[i + r]) + float(row_rhs[i + r]));
                row_residual_output[i + r] = residual;
                if constexpr (DUPLICATE_RESIDUAL) {
                    row_duplicate_residual_output[i + r] = residual;
                }
                float x = float(residual);
                acc += x * x;
            }
        } else {
            for (int i = 0; i < RESIDUAL_RMS_N_READS; i++) {
                if (r + lid * RESIDUAL_RMS_N_READS + i < hidden_dim) {
                    T residual = T(float(row_lhs[i + r]) + float(row_rhs[i + r]));
                    row_residual_output[i + r] = residual;
                    if constexpr (DUPLICATE_RESIDUAL) {
                        row_duplicate_residual_output[i + r] = residual;
                    }
                    float x = float(residual);
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

    device T* row_norm_output = norm_output + row * size_t(hidden_dim) + lid * RESIDUAL_RMS_N_READS;
    const device T* row_residual_input = residual_output + row * size_t(hidden_dim) + lid * RESIDUAL_RMS_N_READS;
    const device T* row_weight = weight + lid * RESIDUAL_RMS_N_READS;
    for (uint r = 0; r < hidden_dim; r += lsize * RESIDUAL_RMS_N_READS) {
        if (r + lid * RESIDUAL_RMS_N_READS + RESIDUAL_RMS_N_READS <= hidden_dim) {
            for (int i = 0; i < RESIDUAL_RMS_N_READS; i++) {
                T residual = row_residual_input[i + r];
                row_norm_output[i + r] = row_weight[i + r] * T(residual * local_inv_mean[0]);
            }
        } else {
            for (int i = 0; i < RESIDUAL_RMS_N_READS; i++) {
                if (r + lid * RESIDUAL_RMS_N_READS + i < hidden_dim) {
                    T residual = row_residual_input[i + r];
                    row_norm_output[i + r] = row_weight[i + r] * T(residual * local_inv_mean[0]);
                }
            }
        }
    }
}

kernel void residual_rms_norm_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* residual_output [[buffer(3)]],
    device float* norm_output [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    residual_rms_norm_impl<float, false>(
        lhs, rhs, weight, residual_output, residual_output, norm_output, num_tokens, hidden_dim, hidden_dim, 0, eps,
        local_inv_mean, local_sums, gid, lid, lsize, simd_lane_id, simd_group_id);
}

kernel void residual_rms_norm_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device const bfloat16_t* weight [[buffer(2)]],
    device bfloat16_t* residual_output [[buffer(3)]],
    device bfloat16_t* norm_output [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    residual_rms_norm_impl<bfloat16_t, false>(
        lhs, rhs, weight, residual_output, residual_output, norm_output, num_tokens, hidden_dim, hidden_dim, 0, eps,
        local_inv_mean, local_sums, gid, lid, lsize, simd_lane_id, simd_group_id);
}

kernel void duplicate_residual_rms_norm_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device const bfloat16_t* weight [[buffer(2)]],
    device bfloat16_t* residual_output [[buffer(3)]],
    device bfloat16_t* duplicate_residual_output [[buffer(4)]],
    device bfloat16_t* norm_output [[buffer(5)]],
    constant uint& num_tokens [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant uint& duplicate_row_stride [[buffer(8)]],
    constant uint& duplicate_column_offset [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    residual_rms_norm_impl<bfloat16_t, true>(
        lhs, rhs, weight, residual_output, duplicate_residual_output, norm_output, num_tokens, hidden_dim,
        duplicate_row_stride, duplicate_column_offset, eps, local_inv_mean, local_sums, gid, lid, lsize, simd_lane_id,
        simd_group_id);
}

kernel void residual_rms_norm_bf16_vec4(
    device const bfloat4* lhs [[buffer(0)]],
    device const bfloat4* rhs [[buffer(1)]],
    device const bfloat4* weight [[buffer(2)]],
    device bfloat4* residual_output [[buffer(3)]],
    device bfloat4* norm_output [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid;
    if (row >= num_tokens) return;

    const uint hidden_dim_vec = hidden_dim / 4;
    float acc = 0.0f;
    const device bfloat4* row_lhs = lhs + row * size_t(hidden_dim_vec) + lid;
    const device bfloat4* row_rhs = rhs + row * size_t(hidden_dim_vec) + lid;
    device bfloat4* row_residual_output = residual_output + row * size_t(hidden_dim_vec) + lid;
    for (uint r = 0; r < hidden_dim_vec; r += lsize) {
        if (r + lid < hidden_dim_vec) {
            bfloat4 residual = bfloat4(float4(row_lhs[r]) + float4(row_rhs[r]));
            row_residual_output[r] = residual;
            float4 x = float4(residual);
            acc += dot(x, x);
        }
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

    const device bfloat4* row_residual_input = residual_output + row * size_t(hidden_dim_vec) + lid;
    const device bfloat4* row_weight = weight + lid;
    device bfloat4* row_norm_output = norm_output + row * size_t(hidden_dim_vec) + lid;
    for (uint r = 0; r < hidden_dim_vec; r += lsize) {
        if (r + lid < hidden_dim_vec) {
            bfloat4 residual = row_residual_input[r];
            row_norm_output[r] = row_weight[r] * bfloat4(float4(residual) * local_inv_mean[0]);
        }
    }
}

kernel void duplicate_residual_rms_norm_bf16_vec4(
    device const bfloat4* lhs [[buffer(0)]],
    device const bfloat4* rhs [[buffer(1)]],
    device const bfloat4* weight [[buffer(2)]],
    device bfloat4* residual_output [[buffer(3)]],
    device bfloat4* duplicate_residual_output [[buffer(4)]],
    device bfloat4* norm_output [[buffer(5)]],
    constant uint& num_tokens [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant uint& duplicate_row_stride_vec [[buffer(8)]],
    constant uint& duplicate_column_offset_vec [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid;
    if (row >= num_tokens) return;

    const uint hidden_dim_vec = hidden_dim / 4;
    float acc = 0.0f;
    const device bfloat4* row_lhs = lhs + row * size_t(hidden_dim_vec) + lid;
    const device bfloat4* row_rhs = rhs + row * size_t(hidden_dim_vec) + lid;
    device bfloat4* row_residual_output = residual_output + row * size_t(hidden_dim_vec) + lid;
    device bfloat4* row_duplicate_residual_output =
        duplicate_residual_output + row * size_t(duplicate_row_stride_vec) + duplicate_column_offset_vec + lid;
    for (uint r = 0; r < hidden_dim_vec; r += lsize) {
        if (r + lid < hidden_dim_vec) {
            bfloat4 residual = bfloat4(float4(row_lhs[r]) + float4(row_rhs[r]));
            row_residual_output[r] = residual;
            row_duplicate_residual_output[r] = residual;
            float4 x = float4(residual);
            acc += dot(x, x);
        }
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

    const device bfloat4* row_residual_input = residual_output + row * size_t(hidden_dim_vec) + lid;
    const device bfloat4* row_weight = weight + lid;
    device bfloat4* row_norm_output = norm_output + row * size_t(hidden_dim_vec) + lid;
    for (uint r = 0; r < hidden_dim_vec; r += lsize) {
        if (r + lid < hidden_dim_vec) {
            bfloat4 residual = row_residual_input[r];
            row_norm_output[r] = row_weight[r] * bfloat4(float4(residual) * local_inv_mean[0]);
        }
    }
}
