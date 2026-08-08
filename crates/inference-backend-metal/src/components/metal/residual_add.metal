
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename LhsT, typename RhsT, typename OutputT>
void residual_add_impl(
    device const LhsT* lhs,
    device const RhsT* rhs,
    device OutputT* output,
    constant uint& num_active_rows,
    constant uint& num_columns,
    uint gid
) {
    const ulong num_active_values = ulong(num_active_rows) * ulong(num_columns);
    if (ulong(gid) >= num_active_values) return;
    output[gid] = OutputT(float(lhs[gid]) + float(rhs[gid]));
}

kernel void residual_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_active_rows [[buffer(3)]],
    constant uint& num_columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    residual_add_impl<float, float, float>(lhs, rhs, output, num_active_rows, num_columns, gid);
}

kernel void residual_add_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_active_rows [[buffer(3)]],
    constant uint& num_columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    residual_add_impl<bfloat16_t, bfloat16_t, bfloat16_t>(lhs, rhs, output, num_active_rows, num_columns, gid);
}

kernel void residual_add_bf16_f32_to_bf16(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_active_rows [[buffer(3)]],
    constant uint& num_columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    residual_add_impl<bfloat16_t, float, bfloat16_t>(lhs, rhs, output, num_active_rows, num_columns, gid);
}

kernel void residual_add_capture_bf16_vec4(
    device const bfloat4* lhs [[buffer(0)]],
    device const bfloat4* rhs [[buffer(1)]],
    device bfloat4* output [[buffer(2)]],
    device bfloat4* capture_output [[buffer(3)]],
    constant uint& num_active_rows [[buffer(4)]],
    constant uint& num_columns_vec [[buffer(5)]],
    constant uint& capture_num_columns_vec [[buffer(6)]],
    constant uint& capture_column_start_vec [[buffer(7)]],
    uint vector_index [[thread_position_in_grid]]
) {
    const uint row = vector_index / num_columns_vec;
    if (row >= num_active_rows) return;
    const uint column = vector_index - row * num_columns_vec;
    const bfloat4 residual = bfloat4(float4(lhs[vector_index]) + float4(rhs[vector_index]));
    output[vector_index] = residual;
    capture_output[
        size_t(row) * size_t(capture_num_columns_vec) + size_t(capture_column_start_vec) + size_t(column)] = residual;
}
