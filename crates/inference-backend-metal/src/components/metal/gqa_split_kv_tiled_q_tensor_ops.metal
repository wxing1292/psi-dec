#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// The paged Map/Reduce ABI retains per-query visibility and partial statistics.
// TensorOps owns register layouts. Iterator coordinates and row mappings do
// not assume the lane layout of one GPU generation or precision mode.
kernel void gqa_split_kv_tiled_q_map(
    device const bfloat16_t* q [[buffer(0)]],
    device const uchar* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* visible_kv_token_ranges [[buffer(4)]],
    device const uint* q_token_ranges [[buffer(5)]],
    device const uint* sdpa_map_task_templates [[buffer(6)]],
    device bfloat16_t* partial_output [[buffer(7)]],
    device float* partial_exp_sums [[buffer(8)]],
    device float* partial_max_logits [[buffer(9)]],
    constant uint& gqa_layer_index [[buffer(10)]],
    constant uint& num_active_q_token_tiles [[buffer(11)]],
    constant uint& num_active_kv_splits [[buffer(12)]],
    constant uint& num_active_tokens [[buffer(13)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int NUM_SIMD_LANES = 32;
    constexpr int NUM_SIMDGROUPS = MAP_REQUIRED_THREADS / NUM_SIMD_LANES;
    constexpr int NUM_SIMDGROUPS_PER_Q_HEAD = MAX_Q_TOKENS / 8;
    constexpr int Q_HEADS_PER_KV_HEAD = NUM_Q_HEADS / NUM_KV_HEADS;
    static_assert(NUM_SIMDGROUPS == NUM_SIMDGROUPS_PER_Q_HEAD * MAX_Q_HEADS);

    const uint map_task_template_index = threadblock_position.y;
    if (map_task_template_index >= num_active_kv_splits) {
        return;
    }
    const uint q_token_range_index = sdpa_map_task_templates[map_task_template_index * 3];
    if (q_token_range_index >= num_active_q_token_tiles) {
        return;
    }
    const uint flat_token_start = q_token_ranges[q_token_range_index * 2];
    const uint flat_token_end = q_token_ranges[q_token_range_index * 2 + 1];
    if (flat_token_start >= num_active_tokens || flat_token_end > num_active_tokens) {
        return;
    }
    const uint num_q_tokens_in_range = flat_token_end - flat_token_start;
    const uint kv_token_begin = sdpa_map_task_templates[map_task_template_index * 3 + 1];
    const uint kv_token_end = sdpa_map_task_templates[map_task_template_index * 3 + 2];
    const uint req_slot = req_slots[flat_token_start];

    const uint head_group_index = threadblock_position.x;
    const uint q_head_range_index = head_group_index % uint(NUM_Q_HEAD_RANGES_PER_KV_HEAD);
    const uint kv_head_index = head_group_index / uint(NUM_Q_HEAD_RANGES_PER_KV_HEAD);
    const uint q_head_range_begin = q_head_range_index * uint(MAX_Q_HEADS);
    const uint num_active_q_heads = min(
        uint(MAX_Q_HEADS), uint(Q_HEADS_PER_KV_HEAD) - q_head_range_begin);
    const uint local_q_head_index = simdgroup_index / uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const uint token_fragment_index = simdgroup_index % uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const bool active_q_head = local_q_head_index < num_active_q_heads;
    const uint q_head_index = min(
        kv_head_index * uint(Q_HEADS_PER_KV_HEAD) + q_head_range_begin + local_q_head_index,
        uint(NUM_Q_HEADS - 1));

    constexpr int PAD = 16 / int(sizeof(bfloat16_t));
    constexpr int LEADING_DIM = HEAD_DIM + PAD;
    constexpr int TENSOR_KV_TOKENS = 16;
    constexpr int KV_TOKEN_VALUES = TENSOR_KV_TOKENS * LEADING_DIM;
    threadgroup bfloat16_t* k_shared = reinterpret_cast<threadgroup bfloat16_t*>(shared_mem);
    threadgroup bfloat16_t* v_shared = k_shared + KV_TOKEN_VALUES;

    static_assert(KV_TOKENS_PER_ITERATION <= TENSOR_KV_TOKENS);
    constexpr auto qk_descriptor = matmul2d_descriptor(8, TENSOR_KV_TOKENS, HEAD_DIM, false, true, false);
    // Bound the PV tile width; D=256 uses two independent accumulators.
    constexpr int OUTPUT_TILE_DIM = 128;
    constexpr auto pv_descriptor = matmul2d_descriptor(
        8, OUTPUT_TILE_DIM, TENSOR_KV_TOKENS, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_descriptor, execution_simdgroup> qk_op;
    matmul2d<pv_descriptor, execution_simdgroup> pv_op;

    // A zero-length view pads an inactive head or token fragment.
    const uint token_fragment_begin = token_fragment_index * 8;
    const uint q_rows = active_q_head && token_fragment_begin < num_q_tokens_in_range
        ? min(8u, num_q_tokens_in_range - token_fragment_begin)
        : 0;
    const uint q_token_begin = flat_token_start + min(token_fragment_begin, num_q_tokens_in_range);
    tensor<device bfloat16_t, extents<int, HEAD_DIM, dynamic_extent>, tensor_inline> q_tile(
        const_cast<device bfloat16_t*>(q + ((ulong)q_token_begin * NUM_Q_HEADS + q_head_index) * HEAD_DIM),
        extents<int, HEAD_DIM, dynamic_extent>(int(q_rows)),
        array<int, 2>{1, NUM_Q_HEADS * HEAD_DIM});
    tensor<threadgroup bfloat16_t, extents<int, HEAD_DIM, TENSOR_KV_TOKENS>, tensor_inline> k_tile(
        k_shared, extents<int, HEAD_DIM, TENSOR_KV_TOKENS>{}, array<int, 2>{1, LEADING_DIM});
    tensor<threadgroup bfloat16_t, extents<int, HEAD_DIM, TENSOR_KV_TOKENS>, tensor_inline> v_tile(
        v_shared, extents<int, HEAD_DIM, TENSOR_KV_TOKENS>{}, array<int, 2>{1, LEADING_DIM});

    // Preserve F32 softmax probabilities. This also supports the Metal 4.0
    // tensor API without relying on cooperative-input layout conversions.
    threadgroup float probabilities[NUM_SIMDGROUPS * 8 * TENSOR_KV_TOKENS];
    threadgroup float row_stats[NUM_SIMDGROUPS * 8 * 3];
    threadgroup float* row_sum = row_stats + simdgroup_index * 8 * 3;
    threadgroup float* row_max = row_sum + 8;
    threadgroup float* row_scale = row_max + 8;
    if (lane < 8) {
        row_sum[lane] = 0.0f;
        row_max[lane] = -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup float, extents<int, TENSOR_KV_TOKENS, 8>, tensor_inline> p_tile(
        probabilities + simdgroup_index * 8 * TENSOR_KV_TOKENS,
        extents<int, TENSOR_KV_TOKENS, 8>{});
    auto output_tile = pv_op.get_destination_cooperative_tensor<decltype(p_tile), decltype(v_tile), float>();
    for (uint i = 0; i < output_tile.get_capacity(); ++i) {
        if (output_tile.is_valid_element(i)) output_tile[i] = 0.0f;
    }
    auto output_high = output_tile;
    auto running_max = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(q_tile), decltype(k_tile), float>();
    for (uint i = 0; i < running_max.get_capacity(); ++i) {
        if (running_max.is_valid_element(i)) running_max[i] = -INFINITY;
    }
    auto running_sum = running_max;
    auto running_scale = running_max;
    for (uint i = 0; i < running_sum.get_capacity(); ++i) {
        if (running_sum.is_valid_element(i)) running_sum[i] = 0.0f;
    }
    const uint num_kv_iterations =
        (kv_token_end - kv_token_begin + uint(KV_TOKENS_PER_ITERATION - 1)) / uint(KV_TOKENS_PER_ITERATION);
    for (uint kv_iteration_index = 0; kv_iteration_index < num_kv_iterations; ++kv_iteration_index) {
        const uint kv_iteration_begin = kv_token_begin + kv_iteration_index * uint(KV_TOKENS_PER_ITERATION);
        for (uint kv_token_offset = simdgroup_index; kv_token_offset < uint(TENSOR_KV_TOKENS);
             kv_token_offset += uint(NUM_SIMDGROUPS)) {
            const uint kv_token_index = kv_iteration_begin + kv_token_offset;
            if (kv_token_offset < uint(KV_TOKENS_PER_ITERATION) && kv_token_index < kv_token_end) {
                const uint block_index = kv_token_index / uint(NUM_TOKENS_PER_PAGE * NUM_PAGE_IDS_PER_BLOCK);
                const uint page_id_index =
                    (kv_token_index / uint(NUM_TOKENS_PER_PAGE)) % uint(NUM_PAGE_IDS_PER_BLOCK);
                const uint page_token_index = kv_token_index % uint(NUM_TOKENS_PER_PAGE);
                const ulong page_table_index =
                    ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)gqa_layer_index)
                      * (ulong)NUM_BLOCKS + (ulong)block_index)
                     * (ulong)NUM_PAGE_IDS_PER_BLOCK) + (ulong)page_id_index;
                const ulong page_id = (ulong)page_ids[page_table_index];
                const ulong page_base = page_id * (ulong)PAGE_BYTES;
                const device uchar* k = kv_pages + page_base
                    + (ulong)(((0 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                const device uchar* v = kv_pages + page_base
                    + (ulong)(((1 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    const uint2 k_fp8 = *reinterpret_cast<device const uint2*>(k + dim);
                    const uint2 v_fp8 = *reinterpret_cast<device const uint2*>(v + dim);
                    *reinterpret_cast<threadgroup uint4*>(
                        k_shared + kv_token_offset * LEADING_DIM + dim) = fp8_e4m3x8_to_bf16x8(k_fp8);
                    *reinterpret_cast<threadgroup uint4*>(
                        v_shared + kv_token_offset * LEADING_DIM + dim) = fp8_e4m3x8_to_bf16x8(v_fp8);
                }
            } else {
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    *reinterpret_cast<threadgroup uint4*>(
                        k_shared + kv_token_offset * LEADING_DIM + dim) = uint4(0);
                    *reinterpret_cast<threadgroup uint4*>(
                        v_shared + kv_token_offset * LEADING_DIM + dim) = uint4(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto scores = qk_op.get_destination_cooperative_tensor<decltype(q_tile), decltype(k_tile), float>();
        qk_op.run(q_tile, k_tile, scores);
        for (auto it = scores.begin(); it != scores.end(); ++it) {
            if (!scores.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const uint token_offset = token_fragment_index * 8 + uint(coordinate[1]);
            const uint kv_token_index = kv_iteration_begin + uint(coordinate[0]);
            const bool active_token = active_q_head && token_offset < num_q_tokens_in_range;
            const uint flat_q_token_index = flat_token_start + token_offset;
            const uint visible_begin = active_token ? visible_kv_token_ranges[flat_q_token_index * 2] : 0;
            const uint visible_end = active_token ? visible_kv_token_ranges[flat_q_token_index * 2 + 1] : 0;
            *it = active_token && kv_token_index >= visible_begin && kv_token_index < visible_end
                    && uint(coordinate[0]) < uint(KV_TOKENS_PER_ITERATION) && kv_token_index < kv_token_end
                ? *it * ATTENTION_SCALE
                : -INFINITY;
        }
        auto iteration_max = running_max;
        reduce_rows(scores, iteration_max, reduction_operation::max, -INFINITY);
        for (uint i = 0; i < running_max.get_capacity(); ++i) {
            if (!running_max.is_valid_element(i)) continue;
            const float next_max = max(running_max[i], iteration_max[i]);
            running_scale[i] = running_max[i] == -INFINITY ? 0.0f : metal::exp(running_max[i] - next_max);
            running_max[i] = next_max;
        }
        for (auto it = scores.begin(); it != scores.end(); ++it) {
            if (!scores.is_valid_element(it)) continue;
            *it = *it == -INFINITY ? 0.0f : metal::exp(*it - *running_max.map_iterator(it));
        }
        auto iteration_sum = running_sum;
        reduce_rows(scores, iteration_sum, reduction_operation::sum, 0.0f);
        for (uint i = 0; i < running_sum.get_capacity(); ++i) {
            if (!running_sum.is_valid_element(i)) continue;
            running_sum[i] = running_sum[i] * running_scale[i] + iteration_sum[i];
        }
        // QK and PV can use different cooperative layouts. Exchange row
        // statistics by logical coordinates instead of mapping across ops.
        // Valid score elements have row coordinates in the eight-row tile.
        for (auto it = scores.begin(); it != scores.end(); ++it) {
            if (!scores.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            if (coordinate[0] == 0) {
                row_sum[coordinate[1]] = *running_sum.map_iterator(it);
                row_max[coordinate[1]] = *running_max.map_iterator(it);
                row_scale[coordinate[1]] = *running_scale.map_iterator(it);
            }
        }
        scores.store(p_tile);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto high = output_high.begin();
        for (auto it = output_tile.begin(); it != output_tile.end(); ++it, ++high) {
            if (!output_tile.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const float scale = row_scale[coordinate[1]];
            *it *= scale;
            if constexpr (HEAD_DIM == 256) {
                *high *= scale;
            }
        }
        auto v_low = v_tile.slice<OUTPUT_TILE_DIM, TENSOR_KV_TOKENS>(0, 0);
        pv_op.run(p_tile, v_low, output_tile);
        if constexpr (HEAD_DIM == 256) {
            auto v_high = v_tile.slice<OUTPUT_TILE_DIM, TENSOR_KV_TOKENS>(OUTPUT_TILE_DIM, 0);
            pv_op.run(p_tile, v_high, output_high);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto high = output_high.begin();
    for (auto it = output_tile.begin(); it != output_tile.end(); ++it, ++high) {
        if (!output_tile.is_valid_element(it)) continue;
        const auto coordinate = it.get_multidimensional_index();
        const uint token_offset = token_fragment_index * 8 + uint(coordinate[1]);
        if (active_q_head && token_offset < num_q_tokens_in_range) {
            const ulong partial_index =
                ((ulong)map_task_template_index * NUM_Q_HEADS + q_head_index) * MAX_Q_TOKENS + token_offset;
            const float sum = row_sum[coordinate[1]];
            partial_output[partial_index * HEAD_DIM + ulong(coordinate[0])] =
                bfloat16_t(sum > 0.0f ? *it / sum : 0.0f);
            if constexpr (HEAD_DIM == 256) {
                partial_output[partial_index * HEAD_DIM + OUTPUT_TILE_DIM + ulong(coordinate[0])] =
                    bfloat16_t(sum > 0.0f ? *high / sum : 0.0f);
            }
            if (coordinate[0] == 0) {
                partial_exp_sums[partial_index] = sum;
                partial_max_logits[partial_index] = sum == 0.0f ? -INFINITY : row_max[coordinate[1]];
            }
        }
    }
}
