
#include <metal_stdlib>
using namespace metal;

// qkv_dim is Cqkv = 2 * Hqk * Dqk + Hv * Dv, the concatenated Q/K/V
// channel width. It is not a head axis, head width, or convolution-kernel
// extent.

kernel void gdn_projection_split_f32(
    device const float* qkvabz [[buffer(0)]],
    device float* projected_qkv [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device float* z [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& qkv_dim [[buffer(6)]],
    constant uint& num_v_heads [[buffer(7)]],
    constant uint& v_dim [[buffer(8)]],
    uint global_linear_index [[thread_position_in_grid]]
) {
    uint qkvabz_row_stride = qkv_dim + num_v_heads * 2 + v_dim;
    uint num_qkvabz_values = num_tokens * qkvabz_row_stride;
    if (global_linear_index >= num_qkvabz_values) {
        return;
    }

    uint flat_token_index = global_linear_index / qkvabz_row_stride;
    uint qkvabz_dim_index = global_linear_index - flat_token_index * qkvabz_row_stride;
    float value = qkvabz[global_linear_index];
    if (qkvabz_dim_index < qkv_dim) {
        projected_qkv[flat_token_index * qkv_dim + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= qkv_dim;
    if (qkvabz_dim_index < num_v_heads) {
        a[flat_token_index * num_v_heads + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= num_v_heads;
    if (qkvabz_dim_index < num_v_heads) {
        b[flat_token_index * num_v_heads + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= num_v_heads;
    z[flat_token_index * v_dim + qkvabz_dim_index] = value;
}

kernel void gdn_projection_split_bf16_to_f32(
    device const ushort* qkvabz [[buffer(0)]],
    device float* projected_qkv [[buffer(1)]],
    device float* a [[buffer(2)]],
    device float* b [[buffer(3)]],
    device float* z [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& qkv_dim [[buffer(6)]],
    constant uint& num_v_heads [[buffer(7)]],
    constant uint& v_dim [[buffer(8)]],
    uint global_linear_index [[thread_position_in_grid]]
) {
    uint qkvabz_row_stride = qkv_dim + num_v_heads * 2 + v_dim;
    uint num_qkvabz_values = num_tokens * qkvabz_row_stride;
    if (global_linear_index >= num_qkvabz_values) {
        return;
    }

    uint flat_token_index = global_linear_index / qkvabz_row_stride;
    uint qkvabz_dim_index = global_linear_index - flat_token_index * qkvabz_row_stride;
    float value = as_type<float>(uint(qkvabz[global_linear_index]) << 16);
    if (qkvabz_dim_index < qkv_dim) {
        projected_qkv[flat_token_index * qkv_dim + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= qkv_dim;
    if (qkvabz_dim_index < num_v_heads) {
        a[flat_token_index * num_v_heads + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= num_v_heads;
    if (qkvabz_dim_index < num_v_heads) {
        b[flat_token_index * num_v_heads + qkvabz_dim_index] = value;
        return;
    }
    qkvabz_dim_index -= num_v_heads;
    z[flat_token_index * v_dim + qkvabz_dim_index] = value;
}
