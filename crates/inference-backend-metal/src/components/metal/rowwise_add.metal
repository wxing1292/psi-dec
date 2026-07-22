#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void rowwise_add_impl(
    device const T* lhs,
    device const T* rhs,
    device T* output,
    constant uint& num_rows,
    constant uint& lhs_row_offset,
    uint gid
) {
    const uint num_values = num_rows * row_width;
    if (gid >= num_values) return;
    const ulong lhs_offset = (ulong)lhs_row_offset * row_width + gid;
    output[gid] = T(float(lhs[lhs_offset]) + float(rhs[gid]));
}

kernel void rowwise_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& lhs_row_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    rowwise_add_impl<float>(lhs, rhs, output, num_rows, lhs_row_offset, gid);
}

kernel void rowwise_add_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& lhs_row_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    rowwise_add_impl<bfloat16_t>(lhs, rhs, output, num_rows, lhs_row_offset, gid);
}
