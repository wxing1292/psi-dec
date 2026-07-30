
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void residual_add_impl(
    device const T* lhs,
    device const T* rhs,
    device T* output,
    constant uint& num_values,
    uint gid
) {
    if (gid >= num_values) return;
    output[gid] = T(float(lhs[gid]) + float(rhs[gid]));
}

kernel void residual_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    residual_add_impl<float>(lhs, rhs, output, num_values, gid);
}

kernel void residual_add_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    residual_add_impl<bfloat16_t>(lhs, rhs, output, num_values, gid);
}

kernel void residual_add_bf16_f32_to_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_values) return;
    output[gid] = bfloat16_t(float(lhs[gid]) + rhs[gid]);
}
