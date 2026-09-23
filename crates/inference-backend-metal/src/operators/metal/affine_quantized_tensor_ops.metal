#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
using namespace mpp::tensor_ops;

// One dequantized weight tile serves all BM input rows. The caller owns scratch
// and the output epilogue; this operation owns loading and F32 accumulation.
template <typename InT, typename ParamT, typename T, int group_size, int bits, int BM, int BK>
METAL_FUNC auto affine_qmm_tile(const device uint8_t *w, const device ParamT *scales, const device ParamT *biases,
                                const device InT *x, threadgroup T *Xs, threadgroup T *Ws, int K, short num_rows,
                                short num_cols, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int WM = BM == 32 ? 2 : 1;
  constexpr int WN = 2;
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr int LD = BK + 16 / sizeof(T);

  using loader_x_t = PsiDecMixedBlockLoader<InT, T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = PsiDecMixedQuantizedBlockLoader<ParamT, T, BN, BK, LD, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(w, scales, biases, K, Ws, simd_gid, simd_lid);

  const int row = (simd_gid / WN) * SM;
  const int col = (simd_gid % WN) * SN;
  tensor<threadgroup T, extents<int, BK, SM>, tensor_inline> a(Xs + row * LD, extents<int, BK, SM>{},
                                                               array<int, 2>{1, LD});
  tensor<threadgroup T, extents<int, BK, SN>, tensor_inline> b(Ws + col * LD, extents<int, BK, SN>{},
                                                               array<int, 2>{1, LD});
  constexpr auto descriptor =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  auto result = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (uint i = 0; i < result.get_capacity(); ++i) {
    if (result.is_valid_element(i)) result[i] = 0.0f;
  }
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, num_rows));
    if (num_cols < BN) {
      loader_w.load_safe(short2(BK, num_cols));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(a, b, result);
    loader_x.next();
    loader_w.next();
  }
  return result;
}
