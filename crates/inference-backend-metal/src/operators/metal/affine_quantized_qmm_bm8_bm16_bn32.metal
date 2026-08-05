template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM>
[[kernel]] void qmm_t_bm8_bm16_bn32(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.y * BM >= num_active_rows) {
    return;
  }

  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int WM = 1;
  constexpr int WN = 2;
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const device uint8_t* wl = reinterpret_cast<const device uint8_t*>(w) + y_col * K_w;
  x += y_row * static_cast<int64_t>(K);
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, static_cast<int>(num_active_rows) - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, num_els));
    if (!aligned_N && num_outs < BN) {
      loader_w.load_safe(short2(BK, num_outs));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result_safe(y, N, short2(num_outs, num_els));
}
