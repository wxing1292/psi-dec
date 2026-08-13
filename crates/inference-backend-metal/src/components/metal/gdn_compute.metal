
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

constant uint GDN_INVALID_STATE_SLOT_ID = 0xffffffffu;

// Tensor axes: T = flat tokens, Hqk/Dqk = Q/K heads and head width,
// Hv/Dv = V heads and head width, and
// Cqkv = 2 * Hqk * Dqk + Hv * Dv = concatenated Q/K/V channel width.
// Short convolution operates independently along Cqkv; conv_kernel_size is
// its temporal kernel extent, not a tensor-channel dimension.

kernel void gdn_compute_short_conv_f32(
    device float* conv_qkv [[buffer(0)]],
    device float* next_conv_state [[buffer(1)]],
    device const float* qkv [[buffer(2)]],
    device const float* conv_state [[buffer(3)]],
    device const bfloat16_t* conv_weight [[buffer(4)]],
    device const uint* src_conv_state_slots [[buffer(5)]],
    device const uint* flat_materialized_conv_state_slots [[buffer(6)]],
    device const uint* cu_tokens [[buffer(7)]],
    constant uint& num_active_reqs [[buffer(8)]],
    constant uint& num_active_tokens [[buffer(9)]],
    constant ulong& conv_state_offset_bytes [[buffer(10)]],
    constant ulong& next_conv_state_offset_bytes [[buffer(11)]],
    constant uint& write_final_state [[buffer(12)]],
    uint global_linear_index [[thread_position_in_grid]]
) {
    const ulong conv_state_base = conv_state_offset_bytes / sizeof(float);
    const ulong next_conv_state_base = next_conv_state_offset_bytes / sizeof(float);
    const uint num_conv_qkv_values = num_active_tokens * qkv_dim;
    const uint num_next_conv_state_values = num_active_reqs * qkv_dim * conv_state_len;

    if (global_linear_index < num_conv_qkv_values) {
        const uint channel_index = global_linear_index % qkv_dim;
        const uint flat_token_index = global_linear_index / qkv_dim;
        uint req_index = 0;
        for (uint candidate_req_index = 0; candidate_req_index < num_active_reqs; ++candidate_req_index) {
            if (flat_token_index < cu_tokens[candidate_req_index + 1]) {
                req_index = candidate_req_index;
                break;
            }
        }
        const uint flat_token_begin = cu_tokens[req_index];
        const uint token_index_in_req = flat_token_index - flat_token_begin;
        const uint src_state_slot = src_conv_state_slots[req_index];

        float acc = 0.0f;
        for (uint kernel_index = 0; kernel_index < conv_kernel_size; ++kernel_index) {
            const long sequence_index =
                (long)token_index_in_req + (long)kernel_index - (long)conv_state_len;
            float x = 0.0f;
            if (sequence_index < 0) {
                const uint state_index = uint(sequence_index + (long)conv_state_len);
                const uint state_offset = (src_state_slot * qkv_dim + channel_index) * conv_state_len + state_index;
                x = conv_state[conv_state_base + (ulong)state_offset];
            } else {
                const uint input_offset = (flat_token_begin + uint(sequence_index)) * qkv_dim + channel_index;
                x = qkv[input_offset];
            }
            const uint weight_offset = channel_index * conv_kernel_size + kernel_index;
            acc += x * float(conv_weight[weight_offset]);
        }
        conv_qkv[global_linear_index] = acc / (1.0f + metal::exp(-acc));
    }

    if (global_linear_index < num_next_conv_state_values) {
        const uint state_index = global_linear_index % conv_state_len;
        const uint channel_group = global_linear_index / conv_state_len;
        const uint channel_index = channel_group % qkv_dim;
        const uint req_index = channel_group / qkv_dim;
        const uint flat_token_begin = cu_tokens[req_index];
        const uint flat_token_end = cu_tokens[req_index + 1];
        const uint num_req_tokens = flat_token_end - flat_token_begin;
        const uint src_state_slot = src_conv_state_slots[req_index];
        // An invalid slot keeps the row output but does not materialize its state.
        const uint state_slot = flat_materialized_conv_state_slots[flat_token_end - 1];
        const long sequence_index = (long)num_req_tokens + (long)state_index - (long)conv_state_len;
        float x = 0.0f;
        if (sequence_index < 0) {
            const uint src_state_index = state_index + num_req_tokens;
            const uint state_offset =
                (src_state_slot * qkv_dim + channel_index) * conv_state_len + src_state_index;
            x = conv_state[conv_state_base + (ulong)state_offset];
        } else {
            const uint input_offset = (flat_token_begin + uint(sequence_index)) * qkv_dim + channel_index;
            x = qkv[input_offset];
        }
        if (write_final_state != 0 && state_slot != GDN_INVALID_STATE_SLOT_ID) {
            const uint dst_offset = (state_slot * qkv_dim + channel_index) * conv_state_len + state_index;
            next_conv_state[next_conv_state_base + (ulong)dst_offset] = x;
        }
    }
}

kernel void gdn_compute_forward_conv_candidate_state_f32(
    device float* next_conv_state [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const float* conv_state [[buffer(2)]],
    device const uint* src_conv_state_slots [[buffer(3)]],
    device const uint* flat_materialized_conv_state_slots [[buffer(4)]],
    device const uint* cu_tokens [[buffer(5)]],
    constant uint& num_active_reqs [[buffer(6)]],
    constant uint& num_active_tokens [[buffer(7)]],
    constant ulong& conv_state_offset_bytes [[buffer(8)]],
    constant ulong& next_conv_state_offset_bytes [[buffer(9)]],
    uint global_linear_index [[thread_position_in_grid]]
) {
    const ulong conv_state_base = conv_state_offset_bytes / sizeof(float);
    const ulong next_conv_state_base = next_conv_state_offset_bytes / sizeof(float);
    const uint state_index = global_linear_index % conv_state_len;
    uint coordinate_linear_index = global_linear_index / conv_state_len;
    const uint channel_index = coordinate_linear_index % qkv_dim;
    const uint flat_token_index = coordinate_linear_index / qkv_dim;
    if (flat_token_index >= num_active_tokens) {
        return;
    }

    uint req_index = 0;
    for (uint candidate_req_index = 0; candidate_req_index < num_active_reqs; ++candidate_req_index) {
        if (flat_token_index < cu_tokens[candidate_req_index + 1]) {
            req_index = candidate_req_index;
            break;
        }
    }

    const uint flat_token_begin = cu_tokens[req_index];
    const uint num_verified_req_tokens = flat_token_index - flat_token_begin + 1;
    const uint src_state_slot = src_conv_state_slots[req_index];
    const uint state_slot = flat_materialized_conv_state_slots[flat_token_index];
    if (state_slot == GDN_INVALID_STATE_SLOT_ID) {
        return;
    }
    const long sequence_index =
        (long)num_verified_req_tokens + (long)state_index - (long)conv_state_len;
    float x = 0.0f;
    if (sequence_index < 0) {
        const uint src_state_index = state_index + num_verified_req_tokens;
        const uint state_offset = (src_state_slot * qkv_dim + channel_index) * conv_state_len + src_state_index;
        x = conv_state[conv_state_base + (ulong)state_offset];
    } else {
        const uint input_offset = (flat_token_begin + uint(sequence_index)) * qkv_dim + channel_index;
        x = qkv[input_offset];
    }
    const uint dst_offset = (state_slot * qkv_dim + channel_index) * conv_state_len + state_index;
    next_conv_state[next_conv_state_base + (ulong)dst_offset] = x;
}

// One logical GDNRaggedRecurrentTask maps 1:1 to one threadblock, owns one
// GDNRecurrentStateTile [Dv_tile, Dqk], and advances it once per request token.
// No Task value, TaskTemplate, or ABI buffer is materialized:
//
// GDNRaggedRecurrentTask {
//   req_index,          // grid-derived from threadblock_position.y / Hv
//   v_head_index,       // grid-derived from threadblock_position.y % Hv
//   v_dim_tile_index,   // grid-derived from threadblock_position.x
//   flat_token_begin,   // derived from cu_tokens[req_index]
//   flat_token_end,     // derived from cu_tokens[req_index + 1]
// }
kernel void gdn_compute_ragged_recurrent_f32(
    device float* recurrent_output [[buffer(0)]],
    device float* recurrent_state_arena [[buffer(1)]],
    device const float* conv_qkv [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device const uint* flat_materialized_recurrent_state_slots [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_reqs [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadblock [[thread_position_in_threadgroup]]
) {
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(float);
    const uint qk_dim_thread_index = thread_position_in_threadblock.x;
    const uint v_dim_index_in_tile = thread_position_in_threadblock.y;
    const uint num_qk_dim_threads = 32;
    const uint num_state_fragments = (qk_head_dim + num_qk_dim_threads - 1) / num_qk_dim_threads;
    const uint v_dim_tile_index = threadblock_position.x;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    const uint v_dim_index = v_dim_tile_index * v_dim_tile_size + v_dim_index_in_tile;
    if (req_index >= num_active_reqs) {
        return;
    }

    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const uint recurrent_state_stride = num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    // An invalid slot keeps the row output but does not materialize its state.
    const uint state_slot = flat_materialized_recurrent_state_slots[flat_token_end - 1];

    threadgroup float q_inv_norm_shared;
    threadgroup float k_inv_norm_shared;
    threadgroup float decay_shared;
    threadgroup float beta_shared;

    if (v_dim_index >= v_head_dim) {
        return;
    }

    // Each thread owns one strided Dqk fragment. Keep it thread-local across
    // the request's ordered token loop and publish the final tile only once.
    const uint state_row_offset = (v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
    thread float state_fragments[num_state_fragments];
    for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
        const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
        state_fragments[state_fragment_index] =
            qk_dim_index < qk_head_dim
                ? recurrent_state_arena[
                      recurrent_state_base
                      + (ulong)(src_state_slot * recurrent_state_stride + state_row_offset + qk_dim_index)]
                : 0.0f;
    }

    for (uint flat_token_index = flat_token_begin; flat_token_index < flat_token_end; ++flat_token_index) {
        if (v_dim_index_in_tile == 0) {
            float q_square_sum_partial = 0.0f;
            float k_square_sum_partial = 0.0f;
            for (uint qk_dim_index = qk_dim_thread_index; qk_dim_index < qk_head_dim; qk_dim_index += num_qk_dim_threads) {
                const uint q_value_index = flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q = conv_qkv[q_value_index];
                const float k = conv_qkv[k_value_index];
                q_square_sum_partial += q * q;
                k_square_sum_partial += k * k;
            }
            const float q_square_sum = simd_sum(q_square_sum_partial);
            const float k_square_sum = simd_sum(k_square_sum_partial);
            if (qk_dim_thread_index == 0) {
                const uint gate_index = flat_token_index * num_v_heads + v_head_index;
                const float beta_t = 1.0f / (1.0f + metal::exp(-b[gate_index]));
                const float dt = a[gate_index] + float(dt_bias[v_head_index]);
                const float sp = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                const float decay_rate = -metal::exp(float(a_log[v_head_index]));
                q_inv_norm_shared = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
                k_inv_norm_shared = metal::rsqrt(k_square_sum + 1.0e-6f);
                beta_shared = beta_t;
                decay_shared = metal::exp(decay_rate * sp);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_t = conv_qkv[flat_token_index * qkv_dim + v_base + v_head_index * v_head_dim + v_dim_index];
        float state_k_partial = 0.0f;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index >= qk_head_dim) {
                continue;
            }
            const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
            const float k_norm = conv_qkv[k_value_index] * k_inv_norm_shared;
            const float decayed_state = state_fragments[state_fragment_index] * decay_shared;
            state_fragments[state_fragment_index] = decayed_state;
            state_k_partial += decayed_state * k_norm;
        }
        const float state_k_dot = simd_sum(state_k_partial);
        const float delta = (v_t - simd_broadcast(state_k_dot, 0)) * beta_shared;

        float state_q_partial = 0.0f;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index >= qk_head_dim) {
                continue;
            }
            const uint q_value_index = flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
            const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
            const float k_norm = conv_qkv[k_value_index] * k_inv_norm_shared;
            const float q_norm = conv_qkv[q_value_index] * q_inv_norm_shared;
            const float updated_state = state_fragments[state_fragment_index] + k_norm * delta;
            state_fragments[state_fragment_index] = updated_state;
            state_q_partial += updated_state * q_norm;
        }
        const float recurrent_output_value = simd_sum(state_q_partial);
        if (qk_dim_thread_index == 0) {
            recurrent_output[(flat_token_index * num_v_heads + v_head_index) * v_head_dim + v_dim_index] = recurrent_output_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (state_slot != GDN_INVALID_STATE_SLOT_ID) {
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index < qk_head_dim) {
                recurrent_state_arena[
                    recurrent_state_base
                    + (ulong)(state_slot * recurrent_state_stride + state_row_offset + qk_dim_index)] =
                    state_fragments[state_fragment_index];
            }
        }
    }
}

// Candidate recurrent Task/grid contract:
// - grid.x selects one threadblock V tile;
// - grid.y derives req_index and v_head_index;
// - one threadblock owns
//   [candidate_v_dim_tile_size * candidate_simdgroups_per_threadgroup, Dqk];
// - local SIMDgroup y selects its [candidate_v_dim_tile_size, Dqk] register subtile;
// - cu_tokens[req_index..req_index + 1] selects the ordered token segment.
// The grid and metadata derive these values. No Task value or TaskTemplate is materialized.
kernel void gdn_compute_ragged_recurrent_forward_candidate_state_f32(
    device float* recurrent_output [[buffer(0)]],
    device float* recurrent_state_arena [[buffer(1)]],
    device const float* conv_qkv [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device const uint* flat_materialized_recurrent_state_slots [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_reqs [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadblock [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_q[qk_head_dim];
    threadgroup float shared_k[qk_head_dim];
    threadgroup float shared_gate[2];

    const uint qk_dim_thread_index = thread_position_in_threadblock.x;
    const uint local_simdgroup_index = thread_position_in_threadblock.y;
    const uint v_dim_tile_index =
        threadblock_position.x * candidate_simdgroups_per_threadgroup + local_simdgroup_index;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    const uint v_dim_base = v_dim_tile_index * candidate_v_dim_tile_size;
    if (req_index >= num_active_reqs || v_dim_base + candidate_v_dim_tile_size > v_head_dim) {
        return;
    }

    const uint num_qk_dim_threads = 32;
    const uint num_state_fragments = (qk_head_dim + num_qk_dim_threads - 1) / num_qk_dim_threads;
    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const uint recurrent_state_stride = num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(float);

    thread float state_fragments[candidate_v_dim_tile_size * num_state_fragments];
    for (uint v_dim_index_in_tile = 0; v_dim_index_in_tile < candidate_v_dim_tile_size;
         ++v_dim_index_in_tile) {
        const uint v_dim_index = v_dim_base + v_dim_index_in_tile;
        const ulong state_row_offset = ((ulong)v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
        const ulong source_state_base =
            recurrent_state_base + (ulong)src_state_slot * recurrent_state_stride + state_row_offset;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            state_fragments[v_dim_index_in_tile * num_state_fragments + state_fragment_index] =
                qk_dim_index < qk_head_dim ? recurrent_state_arena[source_state_base + qk_dim_index] : 0.0f;
        }
    }

    for (uint flat_token_index = flat_token_begin; flat_token_index < flat_token_end; ++flat_token_index) {
        float q_square_sum_partial = 0.0f;
        float k_square_sum_partial = 0.0f;
        thread float q_fragments[num_state_fragments];
        thread float k_fragments[num_state_fragments];
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            q_fragments[state_fragment_index] = 0.0f;
            k_fragments[state_fragment_index] = 0.0f;
        }
        if (local_simdgroup_index == 0) {
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q_value = conv_qkv[q_value_index];
                const float k_value = conv_qkv[k_value_index];
                q_fragments[state_fragment_index] = q_value;
                k_fragments[state_fragment_index] = k_value;
                q_square_sum_partial += q_value * q_value;
                k_square_sum_partial += k_value * k_value;
            }
            const float q_square_sum = simd_broadcast(simd_sum(q_square_sum_partial), 0);
            const float k_square_sum = simd_broadcast(simd_sum(k_square_sum_partial), 0);
            const float q_inv_norm = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
            const float k_inv_norm = metal::rsqrt(k_square_sum + 1.0e-6f);
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                q_fragments[state_fragment_index] *= q_inv_norm;
                k_fragments[state_fragment_index] *= k_inv_norm;
                if (qk_dim_index < qk_head_dim) {
                    shared_q[qk_dim_index] = q_fragments[state_fragment_index];
                    shared_k[qk_dim_index] = k_fragments[state_fragment_index];
                }
            }
        }

        if (local_simdgroup_index == 0 && qk_dim_thread_index == 0) {
            const ulong gate_index = (ulong)flat_token_index * num_v_heads + v_head_index;
            const float beta = 1.0f / (1.0f + metal::exp(-b[gate_index]));
            const float dt = a[gate_index] + float(dt_bias[v_head_index]);
            const float sp = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
            const float decay_rate = -metal::exp(float(a_log[v_head_index]));
            shared_gate[0] = beta;
            shared_gate[1] = metal::exp(decay_rate * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            q_fragments[state_fragment_index] = shared_q[qk_dim_index];
            k_fragments[state_fragment_index] = shared_k[qk_dim_index];
        }
        const float beta = shared_gate[0];
        const float decay = shared_gate[1];

        float v_lane = 0.0f;
        if (qk_dim_thread_index < candidate_v_dim_tile_size) {
            const uint v_dim_index = v_dim_base + qk_dim_thread_index;
            v_lane = conv_qkv[(ulong)flat_token_index * qkv_dim + v_base
                + (ulong)v_head_index * v_head_dim + v_dim_index];
        }
        const uint candidate_state_slot = flat_materialized_recurrent_state_slots[flat_token_index];
        for (uint v_dim_index_in_tile = 0; v_dim_index_in_tile < candidate_v_dim_tile_size;
             ++v_dim_index_in_tile) {
            const uint v_dim_index = v_dim_base + v_dim_index_in_tile;
            float state_k_partial = 0.0f;
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint fragment_offset = v_dim_index_in_tile * num_state_fragments + state_fragment_index;
                const float decayed_state = state_fragments[fragment_offset] * decay;
                state_fragments[fragment_offset] = decayed_state;
                state_k_partial += decayed_state * k_fragments[state_fragment_index];
            }
            const float v_value = simd_broadcast(v_lane, v_dim_index_in_tile);
            const float state_k_dot = simd_broadcast(simd_sum(state_k_partial), 0);
            const float delta = (v_value - state_k_dot) * beta;

            float state_q_partial = 0.0f;
            const ulong state_row_offset = ((ulong)v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                const uint fragment_offset = v_dim_index_in_tile * num_state_fragments + state_fragment_index;
                const float updated_state =
                    state_fragments[fragment_offset] + k_fragments[state_fragment_index] * delta;
                state_fragments[fragment_offset] = updated_state;
                if (candidate_state_slot != GDN_INVALID_STATE_SLOT_ID && qk_dim_index < qk_head_dim) {
                    recurrent_state_arena[
                        recurrent_state_base + (ulong)candidate_state_slot * recurrent_state_stride
                        + state_row_offset + qk_dim_index] = updated_state;
                }
                state_q_partial += updated_state * q_fragments[state_fragment_index];
            }
            const float recurrent_output_value = simd_broadcast(simd_sum(state_q_partial), 0);
            if (qk_dim_thread_index == v_dim_index_in_tile) {
                recurrent_output[
                    ((ulong)flat_token_index * num_v_heads + v_head_index) * v_head_dim + v_dim_index] =
                    recurrent_output_value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One logical GDNOutputNormGateTask maps 1:1 to one 128-thread threadblock. It
// RMS-normalizes and gates one [Dv] recurrent-output vector. No Task value,
// TaskTemplate, or ABI buffer is materialized:
//
// GDNOutputNormGateTask {
//   flat_token_index,  // grid-derived from threadblock linear index / Hv
//   v_head_index,      // grid-derived from threadblock linear index % Hv
// }
kernel void gdn_compute_output_norm_gate_f32(
    device float* norm_gated_output [[buffer(0)]],
    device const float* recurrent_output [[buffer(1)]],
    device const float* z [[buffer(2)]],
    device const bfloat16_t* norm_weight [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    uint global_thread_index [[thread_position_in_grid]]
) {
    const uint reduction_thread_index = global_thread_index % 128;
    const uint token_head_index = global_thread_index / 128;
    const uint num_token_heads = num_active_tokens * num_v_heads;
    if (token_head_index >= num_token_heads) {
        return;
    }
    const uint flat_token_index = token_head_index / num_v_heads;
    const uint v_head_index = token_head_index % num_v_heads;
    const uint token_head_base = flat_token_index * num_v_heads * v_head_dim + v_head_index * v_head_dim;
    threadgroup float square_sum_partials[128];

    float square_sum_partial = 0.0f;
    for (uint v_dim_index = reduction_thread_index; v_dim_index < v_head_dim; v_dim_index += 128) {
        const float x = recurrent_output[token_head_base + v_dim_index];
        square_sum_partial += x * x;
    }
    square_sum_partials[reduction_thread_index] = square_sum_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (reduction_thread_index < stride) {
            square_sum_partials[reduction_thread_index] += square_sum_partials[reduction_thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = metal::rsqrt(square_sum_partials[0] / float(v_head_dim) + eps);
    for (uint v_dim_index = reduction_thread_index; v_dim_index < v_head_dim; v_dim_index += 128) {
        const uint output_index = token_head_base + v_dim_index;
        const float z_value = z[output_index];
        const float silu_z = z_value / (1.0f + metal::exp(-z_value));
        const float normalized_value = recurrent_output[output_index] * inv_rms * float(norm_weight[v_dim_index]);
        norm_gated_output[output_index] = normalized_value * silu_z;
    }
}
