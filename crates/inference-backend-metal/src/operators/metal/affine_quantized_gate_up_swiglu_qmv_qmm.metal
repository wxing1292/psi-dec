template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM = 32, const int BK = 32, const int BN = 32>
[[kernel]] void qmm_t_gate_up_swiglu(
    const device uint32_t* w_gate [[buffer(0)]],
    const device T* scales_gate [[buffer(1)]],
    const device T* biases_gate [[buffer(2)]],
    const device uint32_t* w_up [[buffer(3)]],
    const device T* scales_up [[buffer(4)]],
    const device T* biases_up [[buffer(5)]],
    const device T* x [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& M [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  const device uint8_t* gate_wl = (const device uint8_t*)w_gate + y_col * K_w;
  const device uint8_t* up_wl = (const device uint8_t*)w_up + y_col * K_w;
  scales_gate += y_col * K_g;
  biases_gate += y_col * K_g;
  scales_up += y_col * K_g;
  biases_up += y_col * K_g;
  x += y_row * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_gate(gate_wl, scales_gate, biases_gate, K, Ws, simd_gid, simd_lid);
  loader_w_t loader_up(up_wl, scales_up, biases_up, K, Ws, simd_gid, simd_lid);
  mma_t mma_gate(simd_gid, simd_lid);
  mma_t mma_up(simd_gid, simd_lid);

  const bool x_safe = num_els < BM;
  const bool w_safe = !aligned_N && num_outs < BN;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (x_safe) {
      loader_x.load_safe(short2(BK, num_els));
    } else {
      loader_x.load_unsafe();
    }
    if (w_safe) {
      loader_gate.load_safe(short2(BK, num_outs));
    } else {
      loader_gate.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_gate.mma(Xs, Ws);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w_safe) {
      loader_up.load_safe(short2(BK, num_outs));
    } else {
      loader_up.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_up.mma(Xs, Ws);

    loader_x.next();
    loader_gate.next();
    loader_up.next();
  }

  for (short i = 0; i < decltype(mma_gate.Ctile)::kElemsPerTile; ++i) {
    T gate = static_cast<T>(mma_gate.Ctile.elems()[i]);
    T up = static_cast<T>(mma_up.Ctile.elems()[i]);
    T sigmoid = static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(gate))));
    T silu = static_cast<T>(static_cast<float>(gate) * static_cast<float>(sigmoid));
    mma_gate.Ctile.elems()[i] = static_cast<T>(static_cast<float>(silu) * static_cast<float>(up));
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_gate.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_gate.store_result(y, N);
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U value = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(value);
      }
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_gate_up_swiglu(
    const device uint32_t* gate_w,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_w,
    const device T* up_scales,
    const device T* up_biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* gate_ws = (const device uint8_t*)gate_w;
  const device uint8_t* up_ws = (const device uint8_t*)up_w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U gate_result[results_per_simdgroup] = {0};
  thread U up_result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  gate_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  up_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  gate_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  gate_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum);
        up_result[row] += qdot<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum);
      }
    }

    gate_ws += block_size * bytes_per_pack / pack_factor;
    up_ws += block_size * bytes_per_pack / pack_factor;
    gate_scales += block_size / group_size;
    gate_biases += block_size / group_size;
    up_scales += block_size / group_size;
    up_biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot_safe<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum, remaining);
        up_result[row] += qdot_safe<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U gate = simd_sum(gate_result[row]);
      U up = simd_sum(up_result[row]);
      if (simd_lid == 0) {
        T gate_t = static_cast<T>(gate);
        T up_t = static_cast<T>(up);
        T sigmoid = static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(gate_t))));
        T silu = static_cast<T>(static_cast<float>(gate_t) * static_cast<float>(sigmoid));
        y[row] = static_cast<T>(static_cast<float>(silu) * static_cast<float>(up_t));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void dense_gate_up_swiglu(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint32_t* gate_w = w;
  const device uint32_t* up_w = (const device uint32_t*)((const device uint8_t*)w + out_vec_size * in_vec_size_w);
  const device T* gate_scales = scales;
  const device T* gate_biases = biases;
  const device T* up_scales = scales + out_vec_size * in_vec_size_g;
  const device T* up_biases = biases + out_vec_size * in_vec_size_g;
  qmv_gate_up_swiglu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + tid.x * in_vec_size,
      y + tid.x * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void split_gate_up_swiglu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_gate_up_swiglu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + tid.x * in_vec_size,
      y + tid.x * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void token_major_gate_up_swiglu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* lhs_indices [[buffer(7)]],
    const device uint32_t* rhs_indices [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant int& num_experts [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.z;
  const uint input_row = lhs_indices[route];
  const uint expert = rhs_indices[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;
  qmv_gate_up_swiglu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + input_row * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_down_matmul(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* experts_by_route [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant int& num_experts [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint expert = experts_by_route[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;

  qmv_impl<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)w + expert * expert_weight_stride),
      scales + expert * expert_affine_stride,
      biases + expert * expert_affine_stride,
      x + route * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_gate_up_swiglu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* experts_by_route [[buffer(7)]],
    device T* y [[buffer(8)]],
    const constant int& in_vec_size [[buffer(9)]],
    const constant int& out_vec_size [[buffer(10)]],
    const constant int& num_experts [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint expert = experts_by_route[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const int in_vec_size_w = in_vec_size * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const int expert_weight_stride = out_vec_size * in_vec_size_w;
  const int expert_affine_stride = out_vec_size * in_vec_size_g;

  qmv_gate_up_swiglu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + route * in_vec_size,
      y + route * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}
