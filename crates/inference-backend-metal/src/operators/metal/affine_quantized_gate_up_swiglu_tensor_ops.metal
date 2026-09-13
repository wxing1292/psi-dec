template <typename InT, typename ParamT, typename OutT, int group_size,
          int bits, int BM, int BK>
[[kernel]] void affine_qmm_gate_up_swiglu(
    const device uint32_t *w [[buffer(0)]],
    const device ParamT *scales [[buffer(1)]],
    const device ParamT *biases [[buffer(2)]],
    const device InT *x [[buffer(3)]], device OutT *y [[buffer(4)]],
    const constant int &K [[buffer(5)]], const constant int &N [[buffer(6)]],
    const constant uint &num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.y * BM >= num_active_rows)
    return;
  constexpr bool same_dtype = is_same_v<InT, ParamT> && is_same_v<InT, OutT>;
  using T = conditional_t<same_dtype, InT, float>;
  constexpr int LD = BK + 16 / sizeof(T);
  constexpr int WM = BM == 32 ? 2 : 1;
  constexpr int SM = BM / WM;
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[32 * LD];
  static_assert(BM * 32 * sizeof(OutT) <= 32 * LD * sizeof(T));
  threadgroup OutT *projections = reinterpret_cast<threadgroup OutT *>(Ws);
  const int y_row = tid.y * BM;
  const int y_col = tid.x * 16;
  const short num_rows = min(BM, int(num_active_rows) - y_row);
  const short num_cols = min(16, N - y_col);
  const long weight_row_bytes =
      long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  using loader_x_t =
      PsiDecMixedBlockLoader<InT, T, BM, BK, LD, 1, WM * 2 * SIMD_SIZE>;
  using loader_w_t =
      PsiDecMixedQuantizedBlockLoader<ParamT, T, 16, BK, LD, 1, WM * SIMD_SIZE,
                                      group_size, bits>;
  loader_x_t loader_x(x + long(y_row) * K, K, Xs, simd_gid, simd_lid);
  const long weight_row = long(simd_gid % 2) * N + y_col;
  loader_w_t loader_w(reinterpret_cast<const device uint8_t *>(w) +
                          weight_row * weight_row_bytes,
                      scales + weight_row * (K / group_size),
                      biases + weight_row * (K / group_size), K,
                      Ws + (simd_gid % 2) * 16 * LD, simd_gid / 2, simd_lid);
  const int row = (simd_gid / 2) * SM;
  const int col = (simd_gid % 2) * 16;
  tensor<threadgroup T, extents<int, BK, SM>, tensor_inline> a(
      Xs + row * LD, extents<int, BK, SM>{}, array<int, 2>{1, LD});
  tensor<threadgroup T, extents<int, BK, 16>, tensor_inline> b(
      Ws + col * LD, extents<int, BK, 16>{}, array<int, 2>{1, LD});
  constexpr auto descriptor =
      matmul2d_descriptor(SM, 16, BK, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  auto result =
      op.template get_destination_cooperative_tensor<decltype(a), decltype(b),
                                                     float>();
  for (uint i = 0; i < result.get_capacity(); ++i)
    result[i] = 0.0f;
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, num_rows));
    if (num_cols == 16) {
      loader_w.load_unsafe();
    } else {
      loader_w.load_safe(short2(BK, num_cols));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(a, b, result);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (auto it = result.begin(); it != result.end(); ++it) {
    auto coordinate = it.get_multidimensional_index();
    projections[(row + coordinate[1]) * 32 + col + coordinate[0]] = OutT(*it);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = simd_gid * SIMD_SIZE + simd_lid; index < BM * 16;
       index += WM * 2 * SIMD_SIZE) {
    const uint m = index / 16;
    const uint n = index % 16;
    if (m < uint(num_rows) && n < uint(num_cols)) {
      const float gate = projections[m * 32 + n];
      const float up = projections[m * 32 + 16 + n];
      y[long(y_row + m) * N + y_col + n] =
          OutT((gate / (1.0f + exp(-gate))) * up);
    }
  }
}
