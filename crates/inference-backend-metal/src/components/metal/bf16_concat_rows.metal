#include <metal_stdlib>
using namespace metal;

typedef bfloat bfloat16_t;

kernel void bf16_concat_rows(
    device const bfloat16_t* lhs [[buffer(0)]],
    device const bfloat16_t* rhs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& num_cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = num_rows * num_cols * 2;
    if (gid >= total) {
        return;
    }
    const uint row = gid / (num_cols * 2);
    const uint col = gid - row * num_cols * 2;
    if (col < num_cols) {
        output[gid] = lhs[row * num_cols + col];
    } else {
        output[gid] = rhs[row * num_cols + (col - num_cols)];
    }
}
