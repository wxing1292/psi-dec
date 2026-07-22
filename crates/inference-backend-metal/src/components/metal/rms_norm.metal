
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;

constant int RMS_N_READS = 4;

template <typename T>
void rms_norm_impl(
    device const T* input,
    device const T* weight,
    device T* output,
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

    constexpr int SIMD_SIZE = 32;
    float acc = 0.0f;
    const device T* row_input = input + row * size_t(hidden_dim) + lid * RMS_N_READS;
    for (uint r = 0; r < hidden_dim; r += lsize * RMS_N_READS) {
        if (r + lid * RMS_N_READS + RMS_N_READS <= hidden_dim) {
            for (int i = 0; i < RMS_N_READS; i++) {
                float x = float(row_input[i + r]);
                acc += x * x;
            }
        } else {
            for (int i = 0; i < RMS_N_READS; i++) {
                if (r + lid * RMS_N_READS + i < hidden_dim) {
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
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(hidden_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device T* row_output = output + row * size_t(hidden_dim) + lid * RMS_N_READS;
    const device T* row_weight = weight + lid * RMS_N_READS;
    for (uint r = 0; r < hidden_dim; r += lsize * RMS_N_READS) {
        if (r + lid * RMS_N_READS + RMS_N_READS <= hidden_dim) {
            for (int i = 0; i < RMS_N_READS; i++) {
                row_output[i + r] = row_weight[i + r] * T(row_input[i + r] * local_inv_mean[0]);
            }
        } else {
            for (int i = 0; i < RMS_N_READS; i++) {
                if (r + lid * RMS_N_READS + i < hidden_dim) {
                    row_output[i + r] = row_weight[i + r] * T(row_input[i + r] * local_inv_mean[0]);
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
    rms_norm_impl<float>(
        input, weight, output, num_tokens, hidden_dim, eps, local_inv_mean, local_sums, gid, lid, lsize,
        simd_lane_id, simd_group_id);
}

kernel void rms_norm_bf16(
    device const bfloat16_t* input [[buffer(0)]],
    device const bfloat16_t* weight [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
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
    rms_norm_impl<bfloat16_t>(
        input, weight, output, num_tokens, hidden_dim, eps, local_inv_mean, local_sums, gid, lid, lsize,
        simd_lane_id, simd_group_id);
}
