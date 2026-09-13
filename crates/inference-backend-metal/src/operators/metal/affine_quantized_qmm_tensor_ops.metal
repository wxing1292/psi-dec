template <typename InT, typename ParamT, typename OutT, int group_size, int bits, int BM, int BK>
[[kernel]] void
affine_qmm_tensor_ops(const device uint32_t *w [[buffer(0)]], const device ParamT *scales [[buffer(1)]],
                      const device ParamT *biases [[buffer(2)]], const device InT *x [[buffer(3)]],
                      device OutT *y [[buffer(4)]], const constant int &K [[buffer(5)]],
                      const constant int &N [[buffer(6)]], const constant uint &num_active_rows [[buffer(7)]],
                      uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
                      uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.y * BM >= num_active_rows) {
    return;
  }

  // Same-dtype dequantization rounds to that dtype; mixed operands use F32.
  constexpr bool same_dtype = is_same_v<InT, ParamT> && is_same_v<InT, OutT>;
  using T = conditional_t<same_dtype, InT, float>;
  constexpr int LD = BK + 16 / sizeof(T);
  constexpr int WM = BM == 32 ? 2 : 1;
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[32 * LD];
  const int y_row = tid.y * BM;
  const int y_col = tid.x * 32;
  const short num_rows = min(BM, int(num_active_rows) - y_row);
  const short num_cols = min(32, N - y_col);
  const long weight_row_bytes = long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  auto result = affine_qmm_tile<InT, ParamT, T, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(w) + y_col * weight_row_bytes, scales + long(y_col) * (K / group_size),
      biases + long(y_col) * (K / group_size), x + long(y_row) * K, Xs, Ws, K, num_rows, num_cols, simd_gid, simd_lid);
  const int row = (simd_gid / 2) * (BM / WM);
  const int col = (simd_gid % 2) * 16;
  for (auto it = result.begin(); it != result.end(); ++it) {
    const auto coordinate = it.get_multidimensional_index();
    const int m = row + coordinate[1];
    const int n = col + coordinate[0];
    if (m < num_rows && n < num_cols) {
      y[long(y_row + m) * N + y_col + n] = OutT(*it);
    }
  }
}
