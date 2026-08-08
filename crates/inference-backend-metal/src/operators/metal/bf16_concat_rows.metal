// Row-wise BF16 matrix concatenation with four BF16 values per thread.
#include <metal_stdlib>
using namespace metal;

kernel void bf16_concat_rows_bfloat4(
    device const bfloat4* lhs [[buffer(0)]],
    device const bfloat4* rhs [[buffer(1)]],
    device bfloat4* output [[buffer(2)]],
    constant uint& num_active_rows [[buffer(3)]],
    constant uint& num_columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint num_column_vectors = num_columns / 4;
    const uint num_output_vectors_per_row = num_column_vectors * 2;
    const uint num_active_vectors = num_active_rows * num_output_vectors_per_row;
    if (gid >= num_active_vectors) {
        return;
    }

    const uint row = gid / num_output_vectors_per_row;
    const uint vector_column = gid - row * num_output_vectors_per_row;
    if (vector_column < num_column_vectors) {
        output[gid] = lhs[row * num_column_vectors + vector_column];
    } else {
        output[gid] = rhs[row * num_column_vectors + (vector_column - num_column_vectors)];
    }
}
