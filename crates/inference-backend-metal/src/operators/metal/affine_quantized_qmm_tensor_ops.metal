#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

template <typename InT, typename ParamT, typename OutT,
          int group_size, int bits, bool aligned_N, int BM, int BK>
[[kernel]] void affine_qmm_tensor_ops(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.y * BM >= num_active_rows) {
    return;
  }

  // Preserve the affine contract: same-dtype weights round to that dtype;
  // mixed-dtype weights and inputs use F32 before multiplication.
  constexpr bool same_dtype = is_same_v<InT, ParamT> && is_same_v<InT, OutT>;
  using T = conditional_t<same_dtype, InT, float>;
  constexpr int BN = 32;
  constexpr int WM = BM == 32 ? 2 : 1;
  constexpr int WN = 2;
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr int LD = BK + 16 / sizeof(T);
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * LD];
  threadgroup T Ws[BN * LD];
  using loader_x_t = PsiDecMixedBlockLoader<InT, T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = PsiDecMixedQuantizedBlockLoader<ParamT, T, BN, BK, LD, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const short num_rows = min(BM, int(num_active_rows) - y_row);
  const short num_cols = min(BN, N - y_col);
  const device uint8_t* wl = reinterpret_cast<const device uint8_t*>(w) + long(y_col) * (K * bytes_per_pack / pack_factor);
  loader_x_t loader_x(x + long(y_row) * K, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales + long(y_col) * (K / group_size),
                    biases + long(y_col) * (K / group_size), K, Ws, simd_gid, simd_lid);

  const int row = (simd_gid / WN) * SM;
  const int col = (simd_gid % WN) * SN;
  tensor<threadgroup T, extents<int, BK, SM>, tensor_inline> a(
      Xs + row * LD, extents<int, BK, SM>{}, array<int, 2>{1, LD});
  tensor<threadgroup T, extents<int, BK, SN>, tensor_inline> b(
      Ws + col * LD, extents<int, BK, SN>{}, array<int, 2>{1, LD});
  constexpr auto descriptor = matmul2d_descriptor(
      SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  auto result = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (uint i = 0; i < result.get_capacity(); ++i) {
    result[i] = 0.0f;
  }
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, num_rows));
    if (!aligned_N && num_cols < BN) {
      loader_w.load_safe(short2(BK, num_cols));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(a, b, result);
    loader_x.next();
    loader_w.next();
  }
  for (auto it = result.begin(); it != result.end(); ++it) {
    const auto coordinate = it.get_multidimensional_index();
    const int m = row + coordinate[1];
    const int n = col + coordinate[0];
    if (m < num_rows && n < num_cols) {
      y[long(y_row + m) * N + y_col + n] = OutT(*it);
    }
  }
}
