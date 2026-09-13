template <typename InT, typename ParamT, typename OutT, int group_size,
          int bits, int BN, bool aligned>
[[kernel]] void affine_qmv_gate_up_swiglu(
    const device uint32_t *w [[buffer(0)]],
    const device ParamT *scales [[buffer(1)]],
    const device ParamT *biases [[buffer(2)]],
    const device InT *x [[buffer(3)]], device OutT *y [[buffer(4)]],
    const constant int &K [[buffer(5)]], const constant int &N [[buffer(6)]],
    const constant uint &num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.x >= num_active_rows)
    return;
  constexpr int columns = BN;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = bits == 3 || bits == 6 ? 3 : 4;
  constexpr int values_per_thread = 2 * pack_factor;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step = group_size / values_per_thread;
  const int col = tid.y * columns;
  if (!aligned && col >= N)
    return;
  // One SIMDgroup owns gate; the other owns up for the same output columns.
  const long row = long(simd_gid) * N + col;
  const long weight_stride = long(K) * bytes_per_pack / pack_factor;
  const int affine_stride = K / group_size;
  const device uint8_t *ws = reinterpret_cast<const device uint8_t *>(w) +
                             row * weight_stride +
                             simd_lid * 2 * bytes_per_pack;
  scales += row * affine_stride + simd_lid / scale_step;
  biases += row * affine_stride + simd_lid / scale_step;
  x += long(tid.x) * K + simd_lid * values_per_thread;
  thread float input[values_per_thread];
  thread float result[columns] = {0};
  threadgroup OutT projections[2 * columns];
  int k = 0;
  for (; k < (aligned ? K : K - block_size + 1); k += block_size) {
    const float sum =
        load_vector<InT, float, values_per_thread, bits>(x, input);
    for (int n = 0; n < columns; ++n) {
      if (aligned || col + n < N)
        result[n] += qdot<float, values_per_thread, bits>(
            ws + n * weight_stride, input, scales[n * affine_stride],
            biases[n * affine_stride], sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
  if constexpr (!aligned) {
    const int remaining =
        clamp(K - k - int(simd_lid) * values_per_thread, 0, values_per_thread);
    if (remaining > 0) {
      const float sum = load_vector_safe<InT, float, values_per_thread, bits>(
          x, input, remaining);
      for (int n = 0; n < columns; ++n) {
        if (col + n < N)
          result[n] += qdot_safe<float, values_per_thread, bits>(
              ws + n * weight_stride, input, scales[n * affine_stride],
              biases[n * affine_stride], sum, remaining);
      }
    }
  }
  for (int n = 0; n < columns; ++n) {
    const float value = simd_sum(result[n]);
    if (simd_lid == 0)
      projections[simd_gid * columns + n] = OutT(value);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = simd_gid * SIMD_SIZE + simd_lid;
  if (n < columns && (aligned || col + int(n) < N)) {
    const float gate = float(projections[n]);
    const float up = float(projections[columns + n]);
    y[long(tid.x) * N + col + n] = OutT((gate / (1.0f + exp(-gate))) * up);
  }
}
