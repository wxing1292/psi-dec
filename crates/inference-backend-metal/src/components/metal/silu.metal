
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void silu_mul_impl(
    device const T* gate,
    device const T* up,
    device T* output,
    constant uint& num_values,
    uint gid
) {
    if (gid >= num_values) return;
    const float g = float(gate[gid]);
    output[gid] = T((g / (1.0f + exp(-g))) * float(up[gid]));
}

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    silu_mul_impl<float>(gate, up, output, num_values, gid);
}

kernel void silu_mul_bf16(
    device const bfloat16_t* gate [[buffer(0)]],
    device const bfloat16_t* up [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    silu_mul_impl<bfloat16_t>(gate, up, output, num_values, gid);
}
