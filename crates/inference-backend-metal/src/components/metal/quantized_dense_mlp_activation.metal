
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void dense_mlp_activation_impl(
    device const T* gate_up,
    device T* output,
    constant uint& total_tokens,
    constant uint& intermediate_dim,
    uint gid
) {
    const uint num_values = total_tokens * intermediate_dim;
    if (gid >= num_values) return;
    const uint row = gid / intermediate_dim;
    const uint col = gid - row * intermediate_dim;
    const uint row_base = row * intermediate_dim * 2;
    const float gate = float(gate_up[row_base + col]);
    const float up = float(gate_up[row_base + intermediate_dim + col]);
    output[gid] = T((gate / (1.0f + exp(-gate))) * up);
}

kernel void dense_mlp_activation_f32(
    device const float* gate_up [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& total_tokens [[buffer(2)]],
    constant uint& intermediate_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dense_mlp_activation_impl<float>(gate_up, output, total_tokens, intermediate_dim, gid);
}

kernel void dense_mlp_activation_bf16(
    device const bfloat16_t* gate_up [[buffer(0)]],
    device bfloat16_t* output [[buffer(1)]],
    constant uint& total_tokens [[buffer(2)]],
    constant uint& intermediate_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dense_mlp_activation_impl<bfloat16_t>(gate_up, output, total_tokens, intermediate_dim, gid);
}
