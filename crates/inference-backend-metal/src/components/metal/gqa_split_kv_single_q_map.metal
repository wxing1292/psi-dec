#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Each SIMDgroup stages one 16-token x 64-dimension K or V tile. K and V
// reuse this storage; the final cross-SIMDgroup reduction reuses it as F32.
inline void load_kv_tile(
    device const uchar* pages, device const uint* page_ids,
    uint req_slot, uint layer_index, uint kv_head_index,
    uint token_begin, uint token_end, uint dim_begin, uint value_plane,
    threadgroup bfloat16_t* tile, uint lane)
{
    for (uint index = lane * 8; index < 16 * 64; index += 32 * 8) {
        const ulong token = (ulong)token_begin + index / 64;
        const uint dim = dim_begin + index % 64;
        if (token < token_end && dim < KV_HEAD_DIM) {
            const uint block = token / (NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
            const uint page_index = token / NUM_TOKENS % NUM_PAGE_IDS_PER_BLOCK;
            const ulong address = ((((ulong)req_slot * NUM_GQA_LAYERS + layer_index) * NUM_BLOCKS + block)
                                   * NUM_PAGE_IDS_PER_BLOCK) + page_index;
            const ulong page = page_ids[address];
            const device uchar* source = pages + page * PAGE_BYTES
                + ((ulong)(value_plane * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS + token % NUM_TOKENS) * KV_HEAD_DIM + dim;
            if constexpr (KV_HEAD_DIM % 8 == 0) {
                *reinterpret_cast<threadgroup uint4*>(tile + index) =
                    fp8_e4m3x8_to_bf16x8(*reinterpret_cast<device const uint2*>(source));
            } else {
                for (uint part = 0; part < 8; ++part) {
                    tile[index + part] = dim + part < KV_HEAD_DIM
                        ? as_type<bfloat16_t>(fp8_e4m3_to_bf16_bits(source[part])) : bfloat16_t(0.0f);
                }
            }
        } else {
            *reinterpret_cast<threadgroup uint4*>(tile + index) = uint4(0);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void gqa_split_kv_single_q_map(
    device const T* q [[buffer(0)]],
    device const uchar* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* sdpa_map_task_templates [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device T* partial_output [[buffer(7)]],
    constant uint& gqa_layer_index [[buffer(8)]],
    constant uint& num_active_tokens [[buffer(9)]],
    constant uint& num_active_kv_splits [[buffer(10)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint num_simdgroups = REQUIRED_THREADS / 32;
    constexpr uint num_output_tiles = (KV_HEAD_DIM + 63) / 64;
    const uint task = block.x;
    if (task >= num_active_kv_splits) return;
    const uint q_token = sdpa_map_task_templates[task * 3];
    if (q_token >= num_active_tokens) return;
    const uint head_group = block.y;
    const uint head_range = head_group % NUM_Q_HEAD_RANGES_PER_KV_HEAD;
    const uint kv_head = head_group / NUM_Q_HEAD_RANGES_PER_KV_HEAD;
    const uint head_begin = kv_head * Q_HEADS_PER_KV_HEAD + head_range * MAX_Q_HEADS;
    const uint num_heads = min(uint(MAX_Q_HEADS), uint(Q_HEADS_PER_KV_HEAD) - head_range * MAX_Q_HEADS);
    const uint kv_begin = sdpa_map_task_templates[task * 3 + 1];
    const uint kv_end = sdpa_map_task_templates[task * 3 + 2];
    const uint req_slot = req_slots[q_token];

    threadgroup float workspace[num_simdgroups * 8 * 64];
    threadgroup float probabilities[num_simdgroups * 8 * 16];
    threadgroup float stats[num_simdgroups * 8 * 3];
    threadgroup float global_stats[8 * 2];
    threadgroup bfloat16_t* kv = reinterpret_cast<threadgroup bfloat16_t*>(workspace + simdgroup_index * 8 * 64);
    threadgroup float* row_sum = stats + simdgroup_index * 8 * 3;
    threadgroup float* row_max = row_sum + 8;
    threadgroup float* row_scale = row_max + 8;
    if (lane < 8) { row_sum[lane] = 0.0f; row_max[lane] = -INFINITY; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    using Query = tensor<device T, extents<int, dynamic_extent, dynamic_extent>, tensor_inline>;
    using KV = tensor<threadgroup bfloat16_t, extents<int, 64, 16>, tensor_inline>;
    using Probability = tensor<threadgroup float, extents<int, 16, 8>, tensor_inline>;
    constexpr auto qk_desc = matmul2d_descriptor(8, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc = matmul2d_descriptor(8, 64, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    matmul2d<pv_desc, execution_simdgroup> pv_op;
    KV kv_tile(kv, extents<int, 64, 16>{});
    Probability p_tile(probabilities + simdgroup_index * 8 * 16, extents<int, 16, 8>{});
    using Output = decltype(pv_op.get_destination_cooperative_tensor<Probability, KV, float>());
    GQA_DECLARE_OUTPUT_TILES
    for (uint tile = 0; tile < num_output_tiles; ++tile) {
        for (auto it = outputs[tile]->begin(); it != outputs[tile]->end(); ++it) *it = 0.0f;
    }
    auto running_max = qk_op.get_row_reduction_destination_cooperative_tensor<Query, KV, float>();
    for (uint i = 0; i < running_max.get_capacity(); ++i) running_max[i] = -INFINITY;
    auto running_sum = running_max;
    auto running_scale = running_max;
    for (uint i = 0; i < running_sum.get_capacity(); ++i) running_sum[i] = 0.0f;
    for (ulong begin = (ulong)kv_begin + simdgroup_index * 16; begin < kv_end; begin += num_simdgroups * 16) {
        auto scores = qk_op.get_destination_cooperative_tensor<Query, KV, float>();
        for (auto it = scores.begin(); it != scores.end(); ++it) *it = 0.0f;
        for (uint tile = 0; tile < num_output_tiles; ++tile) {
            const uint dim_begin = tile * 64;
            load_kv_tile(kv_pages, page_ids, req_slot, gqa_layer_index, kv_head, uint(begin), kv_end, dim_begin, 0, kv, lane);
            Query query(const_cast<device T*>(q + ((ulong)q_token * NUM_Q_HEADS + head_begin) * KV_HEAD_DIM + dim_begin),
                        extents<int, dynamic_extent, dynamic_extent>(int(min(64u, uint(KV_HEAD_DIM) - dim_begin)), int(num_heads)),
                        array<int, 2>{1, KV_HEAD_DIM});
            qk_op.run(query, kv_tile, scores);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (auto it = scores.begin(); it != scores.end(); ++it) {
            const auto coord = it.get_multidimensional_index();
            *it = uint(coord[1]) < num_heads && begin + uint(coord[0]) < kv_end ? *it * ATTENTION_SCALE : -INFINITY;
        }
        auto iteration_max = running_max;
        reduce_rows(scores, iteration_max, reduction_operation::max, -INFINITY);
        for (uint i = 0; i < running_max.get_capacity(); ++i) {
            const float next = max(running_max[i], iteration_max[i]);
            running_scale[i] = running_max[i] == -INFINITY ? 0.0f : metal::exp(running_max[i] - next);
            running_max[i] = next;
        }
        for (auto it = scores.begin(); it != scores.end(); ++it)
            *it = *it == -INFINITY ? 0.0f : metal::exp(*it - *running_max.map_iterator(it));
        auto iteration_sum = running_sum;
        reduce_rows(scores, iteration_sum, reduction_operation::sum, 0.0f);
        for (uint i = 0; i < running_sum.get_capacity(); ++i)
            running_sum[i] = running_sum[i] * running_scale[i] + iteration_sum[i];
        for (auto it = scores.begin(); it != scores.end(); ++it) {
            const auto coord = it.get_multidimensional_index();
            if (coord[0] == 0) {
                row_sum[coord[1]] = *running_sum.map_iterator(it);
                row_max[coord[1]] = *running_max.map_iterator(it);
                row_scale[coord[1]] = *running_scale.map_iterator(it);
            }
        }
        scores.store(p_tile);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tile = 0; tile < num_output_tiles; ++tile) {
            load_kv_tile(kv_pages, page_ids, req_slot, gqa_layer_index, kv_head, uint(begin), kv_end, tile * 64, 1, kv, lane);
            for (auto it = outputs[tile]->begin(); it != outputs[tile]->end(); ++it) *it *= row_scale[it.get_multidimensional_index()[1]];
            pv_op.run(p_tile, kv_tile, *outputs[tile]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index == 0 && lane < num_heads) {
        float maximum = -INFINITY;
        for (uint sg = 0; sg < num_simdgroups; ++sg) maximum = max(maximum, stats[sg * 24 + 8 + lane]);
        float sum = 0.0f;
        for (uint sg = 0; sg < num_simdgroups; ++sg) {
            const float local_sum = stats[sg * 24 + lane];
            if (local_sum > 0.0f) sum += local_sum * metal::exp(stats[sg * 24 + 8 + lane] - maximum);
        }
        global_stats[lane] = sum;
        global_stats[8 + lane] = maximum;
        const ulong partial = (ulong)task * NUM_Q_HEADS + head_begin + lane;
        partial_exp_sums[partial] = sum;
        partial_max_logits[partial] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < num_output_tiles; ++tile) {
        for (auto it = outputs[tile]->begin(); it != outputs[tile]->end(); ++it) {
            const auto coord = it.get_multidimensional_index();
            workspace[simdgroup_index * 8 * 64 + uint(coord[1]) * 64 + uint(coord[0])] = *it;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup_index == 0) {
            for (auto it = outputs[tile]->begin(); it != outputs[tile]->end(); ++it) {
                const auto coord = it.get_multidimensional_index();
                const uint row = uint(coord[1]);
                const uint col = uint(coord[0]);
                if (row < num_heads && tile * 64 + col < KV_HEAD_DIM) {
                    float value = 0.0f;
                    for (uint sg = 0; sg < num_simdgroups; ++sg) {
                        const float scale = stats[sg * 24 + row] > 0.0f ? metal::exp(stats[sg * 24 + 8 + row] - global_stats[8 + row]) : 0.0f;
                        value += workspace[sg * 8 * 64 + row * 64 + col] * scale;
                    }
                    const ulong partial = (ulong)task * NUM_Q_HEADS + head_begin + row;
                    partial_output[partial * KV_HEAD_DIM + tile * 64 + col] = T(global_stats[row] > 0.0f ? value / global_stats[row] : 0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
