#include <metal_stdlib>
using namespace metal;

kernel void dynamic_grouped_conv_bf16_f32(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* projected_coefficients [[buffer(1)]],
    device const float* base [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& num_active_query_blocks [[buffer(4)]],
    constant uint& query_block_size [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& side [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const ulong num_active_tokens = (ulong)num_active_query_blocks * (ulong)query_block_size;
    const ulong num_active_values = num_active_tokens * (ulong)hidden_dim;
    if ((ulong)gid >= num_active_values) {
        return;
    }
    const uint hidden_index = gid % hidden_dim;
    const uint token = gid / hidden_dim;
    const uint row = token % query_block_size;
    const uint num_groups = hidden_dim / group_size;
    const uint group = hidden_index / group_size;
    float value = 0.0f;
    const uint visible_taps = metal::min(kernel_size, row + 1u);
    for (uint tap = 0u; tap < visible_taps; ++tap) {
        const uint source_token = token - tap;
        const ulong coefficient_index =
            (((ulong)token * 2ul + (ulong)side) * (ulong)kernel_size + (ulong)tap) * (ulong)num_groups
            + (ulong)group;
        const ulong base_index =
            ((ulong)side * (ulong)kernel_size + (ulong)tap) * (ulong)hidden_dim + (ulong)hidden_index;
        const float coefficient = base[base_index] + float(projected_coefficients[coefficient_index]);
        value += coefficient * float(hidden[(ulong)source_token * (ulong)hidden_dim + (ulong)hidden_index]);
    }
    output[gid] = bfloat(value);
}
