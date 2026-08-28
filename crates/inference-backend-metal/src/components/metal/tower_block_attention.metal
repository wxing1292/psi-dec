#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void tower_block_attention_bf16(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* key [[buffer(1)]],
    device const bfloat* value [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& num_rows [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& window_size [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint dim = thread_index.x;
  const uint head = group.x;
  const uint row = group.y;
  const bool active_dim = dim < head_dim;
  const uint window_start = (row / window_size) * window_size;
  const uint window_end = min(window_start + window_size, num_rows);
  const ulong query_index = ((ulong)row * num_heads + head) * head_dim + dim;
  const float q = active_dim ? float(query[query_index]) : 0.0f;
  threadgroup float partials[4];
  const uint num_simd_groups = (head_dim + 31u) / 32u;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (uint key_row = window_start; key_row < window_end; ++key_row) {
    const ulong item_index = ((ulong)key_row * num_heads + head) * head_dim + dim;
    const float partial = simd_sum(active_dim ? q * float(key[item_index]) : 0.0f);
    if (lane == 0u) partials[simd_group] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dim == 0u) {
      float score = 0.0f;
      for (uint group_index = 0; group_index < num_simd_groups; ++group_index) {
        score += partials[group_index];
      }
      partials[0] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float score = partials[0];
    if (score > maximum) {
      const float correction = maximum == -INFINITY ? 0.0f : precise::exp(maximum - score);
      accumulator = accumulator * correction + (active_dim ? float(value[item_index]) : 0.0f);
      denominator = denominator * correction + 1.0f;
      maximum = score;
    } else {
      const float probability = precise::exp(score - maximum);
      accumulator += probability * (active_dim ? float(value[item_index]) : 0.0f);
      denominator += probability;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active_dim) output[query_index] = bfloat(accumulator / denominator);
}
