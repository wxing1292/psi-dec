#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

kernel void bf16_to_f32(
    device const bfloat16_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_values [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_values) {
        return;
    }
    output[gid] = float(input[gid]);
}

kernel void f32_to_bf16(
    device const float* input [[buffer(0)]],
    device bfloat16_t* output [[buffer(1)]],
    constant uint& num_values [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_values) {
        return;
    }
    output[gid] = bfloat(input[gid]);
}
