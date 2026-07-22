#include <metal_stdlib>
using namespace metal;

kernel void f32_buffer_copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_values [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_values) return;
    output[gid] = input[gid];
}

kernel void u32_buffer_copy(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& num_values [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_values) return;
    output[gid] = input[gid];
}
