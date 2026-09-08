inline ushort2 gdn_fragment_coordinate(ushort lane_id) {
    const ushort quad_id = lane_id / 4;
    const ushort row = (quad_id & 4) + (lane_id / 2) % 4;
    const ushort col = (quad_id & 2) * 2 + (lane_id % 2) * 2;
    return ushort2(col, row);
}

// One threadblock retains a state tile in F32 SIMDgroup fragments and
// advances sequential chunks of at most eight tokens with the gated WY form.
// All intermediates remain in registers or threadgroup memory. The request
// prefix is selected at submission time; an empty prefix exits before access.
kernel void gdn_compute_chunkwise_state_bf16(
    device bfloat16_t* recurrent_output [[buffer(0)]],
    device bfloat16_t* recurrent_state_arena [[buffer(1)]],
    device const bfloat16_t* conv_qkv [[buffer(2)]],
    device const bfloat16_t* a [[buffer(3)]],
    device const bfloat16_t* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device const uint* flat_recurrent_state_write_slots [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_prefill_requests [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    constant uint& write_candidate_states [[buffer(13)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint matrix_size = 8;
    const uint num_qk_fragments = qk_head_dim / matrix_size;
    const uint num_simdgroups = chunkwise_state_num_simdgroups;
    const uint num_threads = chunkwise_state_num_qk_dim_threads * num_simdgroups;
    const uint v_tile_storage_rows = num_simdgroups * matrix_size;
    const uint v_row_range_index = threadblock_position.x;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    if (req_index >= num_active_prefill_requests) {
        return;
    }

    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const ulong recurrent_state_stride = (ulong)num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);
    const uint v_dim_base = v_row_range_index * chunkwise_state_num_v_rows;
    const uint simdgroup_v_dim_base = v_dim_base + simdgroup_index * matrix_size;
    const ulong source_state_base = recurrent_state_base
        + (ulong)src_state_slot * recurrent_state_stride
        + ((ulong)v_head_index * v_head_dim + simdgroup_v_dim_base) * qk_head_dim;

    threadgroup float normalized_k[chunkwise_token_chunk_size * qk_head_dim];
    threadgroup float transformed_vectors[chunkwise_token_chunk_size * qk_head_dim];
    threadgroup float transform[chunkwise_token_chunk_size * chunkwise_token_chunk_size];
    threadgroup float weighted_transform[chunkwise_token_chunk_size * chunkwise_token_chunk_size];
    threadgroup float shared_v[chunkwise_token_chunk_size * chunkwise_state_num_simdgroups * matrix_size];
    threadgroup float q_inv_norm[chunkwise_token_chunk_size];
    threadgroup float cumulative_log_decay[chunkwise_token_chunk_size];
    threadgroup float beta[chunkwise_token_chunk_size];

    const ushort2 fragment_coordinate = gdn_fragment_coordinate(ushort(lane));
    thread simdgroup_matrix<float, 8, 8> state_fragments[num_qk_fragments];
    for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments; ++qk_fragment_index) {
        float2 state_elements = float2(0.0f);
        if (simdgroup_v_dim_base + fragment_coordinate.y < v_dim_base + chunkwise_state_num_v_rows) {
            const ulong state_row_base = source_state_base + (ulong)fragment_coordinate.y * qk_head_dim;
            state_elements[0] = float(recurrent_state_arena[
                state_row_base + qk_fragment_index * matrix_size + fragment_coordinate.x]);
            state_elements[1] = float(recurrent_state_arena[
                state_row_base + qk_fragment_index * matrix_size + fragment_coordinate.x + 1]);
        }
        reinterpret_cast<thread float2&>(state_fragments[qk_fragment_index].thread_elements()) = state_elements;
    }

    for (uint chunk_start = flat_token_begin; chunk_start < flat_token_end;) {
        uint num_chunk_tokens = min(chunkwise_token_chunk_size, flat_token_end - chunk_start);
        // A materialized row ends the chunk. The next chunk retains the F32
        // register state and does not reload the rounded BF16 checkpoint.
        if (write_candidate_states != 0) {
            for (uint token_index = 0; token_index < num_chunk_tokens; ++token_index) {
                if (flat_recurrent_state_write_slots[chunk_start + token_index] != GDN_INVALID_STATE_SLOT_ID) {
                    num_chunk_tokens = token_index + 1;
                    break;
                }
            }
        }

        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint token_index_in_chunk = value_index / qk_head_dim;
            if (token_index_in_chunk >= num_chunk_tokens) {
                normalized_k[value_index] = 0.0f;
            }
        }
        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * v_tile_storage_rows;
             value_index += num_threads) {
            const uint token_index_in_chunk = value_index / v_tile_storage_rows;
            const uint v_row_index = value_index - token_index_in_chunk * v_tile_storage_rows;
            shared_v[value_index] = token_index_in_chunk < num_chunk_tokens
                    && v_row_index < chunkwise_state_num_v_rows
                ? float(conv_qkv[
                      (ulong)(chunk_start + token_index_in_chunk) * qkv_dim + v_base
                      + v_head_index * v_head_dim + v_dim_base + v_row_index])
                : 0.0f;
        }
        for (uint token_index_in_chunk = simdgroup_index;
             token_index_in_chunk < num_chunk_tokens;
             token_index_in_chunk += num_simdgroups) {
            const uint flat_token_index = chunk_start + token_index_in_chunk;
            float q_square_sum_partial = 0.0f;
            float k_square_sum_partial = 0.0f;
            for (uint qk_dim_index = lane; qk_dim_index < qk_head_dim;
                 qk_dim_index += chunkwise_state_num_qk_dim_threads) {
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q_value = float(conv_qkv[q_value_index]);
                const float k_value = float(conv_qkv[k_value_index]);
                q_square_sum_partial += q_value * q_value;
                k_square_sum_partial += k_value * k_value;
            }
            const float q_square_sum = simd_sum(q_square_sum_partial);
            const float k_square_sum = simd_sum(k_square_sum_partial);
            const float k_inverse_norm = metal::rsqrt(k_square_sum + 1.0e-6f);
            for (uint qk_dim_index = lane; qk_dim_index < qk_head_dim;
                 qk_dim_index += chunkwise_state_num_qk_dim_threads) {
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                normalized_k[token_index_in_chunk * qk_head_dim + qk_dim_index] =
                    float(conv_qkv[k_value_index]) * k_inverse_norm;
            }
            if (lane == 0) {
                const ulong gate_index = (ulong)flat_token_index * num_v_heads + v_head_index;
                const float dt = float(a[gate_index]) + float(dt_bias[v_head_index]);
                const float softplus = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                const float decay_rate = -metal::exp(float(a_log[v_head_index]));
                q_inv_norm[token_index_in_chunk] = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
                cumulative_log_decay[token_index_in_chunk] = decay_rate * softplus;
                beta[token_index_in_chunk] = 1.0f / (1.0f + metal::exp(-float(b[gate_index])));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index == 0) {
            float log_decay_sum = 0.0f;
            for (uint token_index_in_chunk = 0; token_index_in_chunk < num_chunk_tokens;
                 ++token_index_in_chunk) {
                log_decay_sum += cumulative_log_decay[token_index_in_chunk];
                cumulative_log_decay[token_index_in_chunk] = log_decay_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index == 0) {
            simdgroup_matrix<float, 8, 8> kkt;
            reinterpret_cast<thread float2&>(kkt.thread_elements()) = float2(0.0f);
            for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments;
                 ++qk_fragment_index) {
                simdgroup_matrix<float, 8, 8> k_fragment;
                simdgroup_matrix<float, 8, 8> transposed_k_fragment;
                simdgroup_load(
                    k_fragment,
                    normalized_k + qk_fragment_index * matrix_size,
                    qk_head_dim);
                simdgroup_load(
                    transposed_k_fragment,
                    normalized_k + qk_fragment_index * matrix_size,
                    qk_head_dim,
                    ulong2(0),
                    true);
                simdgroup_multiply_accumulate(kkt, k_fragment, transposed_k_fragment, kkt);
            }
            simdgroup_store(kkt, transform, chunkwise_token_chunk_size);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            if (row >= num_chunk_tokens || col >= num_chunk_tokens || row < col) {
                transform[matrix_index] = 0.0f;
            } else if (row == col) {
                transform[matrix_index] = 1.0f;
            } else {
                transform[matrix_index] = -beta[row]
                    * metal::exp(cumulative_log_decay[row] - cumulative_log_decay[col])
                    * transform[matrix_index];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index == 0) {
            for (uint row = 1; row < num_chunk_tokens; ++row) {
                for (uint col = 0; col < row; ++col) {
                    float value = transform[row * chunkwise_token_chunk_size + col];
                    for (uint inner = col + 1; inner < row; ++inner) {
                        value += transform[row * chunkwise_token_chunk_size + inner]
                            * transform[inner * chunkwise_token_chunk_size + col];
                    }
                    transform[row * chunkwise_token_chunk_size + col] = value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            weighted_transform[matrix_index] = row < num_chunk_tokens && col <= row
                ? transform[matrix_index] * beta[col]
                : 0.0f;
        }
        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint row = value_index / qk_head_dim;
            transformed_vectors[value_index] = row < num_chunk_tokens
                ? metal::exp(cumulative_log_decay[row]) * normalized_k[value_index]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint qk_fragment_index = simdgroup_index;
             qk_fragment_index < num_qk_fragments;
             qk_fragment_index += num_simdgroups) {
            simdgroup_matrix<float, 8, 8> weighted_transform_fragment;
            simdgroup_matrix<float, 8, 8> decayed_k_fragment;
            simdgroup_matrix<float, 8, 8> w_fragment;
            simdgroup_load(
                weighted_transform_fragment,
                weighted_transform,
                chunkwise_token_chunk_size);
            simdgroup_load(
                decayed_k_fragment,
                transformed_vectors + qk_fragment_index * matrix_size,
                qk_head_dim);
            reinterpret_cast<thread float2&>(w_fragment.thread_elements()) = float2(0.0f);
            simdgroup_multiply_accumulate(
                w_fragment,
                weighted_transform_fragment,
                decayed_k_fragment,
                w_fragment);
            simdgroup_store(
                w_fragment,
                transformed_vectors + qk_fragment_index * matrix_size,
                qk_head_dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> state_w;
        reinterpret_cast<thread float2&>(state_w.thread_elements()) = float2(0.0f);
        for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments;
             ++qk_fragment_index) {
            simdgroup_matrix<float, 8, 8> transposed_w_fragment;
            simdgroup_load(
                transposed_w_fragment,
                transformed_vectors + qk_fragment_index * matrix_size,
                qk_head_dim,
                ulong2(0),
                true);
            simdgroup_multiply_accumulate(
                state_w,
                state_fragments[qk_fragment_index],
                transposed_w_fragment,
                state_w);
        }
        simdgroup_matrix<float, 8, 8> transposed_v_fragment;
        simdgroup_matrix<float, 8, 8> transposed_weighted_transform_fragment;
        simdgroup_matrix<float, 8, 8> transposed_u;
        simdgroup_load(
            transposed_v_fragment,
            shared_v + simdgroup_index * matrix_size,
            v_tile_storage_rows,
            ulong2(0),
            true);
        simdgroup_load(
            transposed_weighted_transform_fragment,
            weighted_transform,
            chunkwise_token_chunk_size,
            ulong2(0),
            true);
        reinterpret_cast<thread float2&>(transposed_u.thread_elements()) = float2(0.0f);
        simdgroup_multiply_accumulate(
            transposed_u,
            transposed_v_fragment,
            transposed_weighted_transform_fragment,
            transposed_u);
        simdgroup_matrix<float, 8, 8> delta_fragment;
        reinterpret_cast<thread float2&>(delta_fragment.thread_elements()) =
            reinterpret_cast<thread float2&>(transposed_u.thread_elements())
            - reinterpret_cast<thread float2&>(state_w.thread_elements());
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint row = value_index / qk_head_dim;
            const uint qk_dim_index = value_index - row * qk_head_dim;
            if (row < num_chunk_tokens) {
                const uint flat_token_index = chunk_start + row;
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                transformed_vectors[value_index] = float(conv_qkv[q_value_index]) * q_inv_norm[row];
            } else {
                transformed_vectors[value_index] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index == 0) {
            simdgroup_matrix<float, 8, 8> qk;
            reinterpret_cast<thread float2&>(qk.thread_elements()) = float2(0.0f);
            for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments;
                 ++qk_fragment_index) {
                simdgroup_matrix<float, 8, 8> q_fragment;
                simdgroup_matrix<float, 8, 8> transposed_k_fragment;
                simdgroup_load(
                    q_fragment,
                    transformed_vectors + qk_fragment_index * matrix_size,
                    qk_head_dim);
                simdgroup_load(
                    transposed_k_fragment,
                    normalized_k + qk_fragment_index * matrix_size,
                    qk_head_dim,
                    ulong2(0),
                    true);
                simdgroup_multiply_accumulate(qk, q_fragment, transposed_k_fragment, qk);
            }
            simdgroup_store(qk, transform, chunkwise_token_chunk_size);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            transform[matrix_index] = row < num_chunk_tokens && col <= row
                ? metal::exp(cumulative_log_decay[row] - cumulative_log_decay[col])
                    * transform[matrix_index]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> state_q;
        reinterpret_cast<thread float2&>(state_q.thread_elements()) = float2(0.0f);
        for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments;
             ++qk_fragment_index) {
            simdgroup_matrix<float, 8, 8> transposed_q_fragment;
            simdgroup_load(
                transposed_q_fragment,
                transformed_vectors + qk_fragment_index * matrix_size,
                qk_head_dim,
                ulong2(0),
                true);
            simdgroup_multiply_accumulate(
                state_q,
                state_fragments[qk_fragment_index],
                transposed_q_fragment,
                state_q);
        }
        thread float2 state_q_elements =
            reinterpret_cast<thread float2&>(state_q.thread_elements());
        state_q_elements[0] *= fragment_coordinate.x < num_chunk_tokens
            ? metal::exp(cumulative_log_decay[fragment_coordinate.x])
            : 0.0f;
        state_q_elements[1] *= fragment_coordinate.x + 1 < num_chunk_tokens
            ? metal::exp(cumulative_log_decay[fragment_coordinate.x + 1])
            : 0.0f;
        reinterpret_cast<thread float2&>(state_q.thread_elements()) = state_q_elements;

        simdgroup_matrix<float, 8, 8> transposed_transform_fragment;
        simdgroup_matrix<float, 8, 8> local_output;
        simdgroup_load(
            transposed_transform_fragment,
            transform,
            chunkwise_token_chunk_size,
            ulong2(0),
            true);
        reinterpret_cast<thread float2&>(local_output.thread_elements()) = float2(0.0f);
        simdgroup_multiply_accumulate(
            local_output,
            delta_fragment,
            transposed_transform_fragment,
            local_output);
        const float2 output_elements = state_q_elements
            + reinterpret_cast<thread float2&>(local_output.thread_elements());
        for (uint element_index = 0; element_index < 2; ++element_index) {
            const uint token_index_in_chunk = fragment_coordinate.x + element_index;
            if (token_index_in_chunk < num_chunk_tokens
                && simdgroup_v_dim_base + fragment_coordinate.y < v_dim_base + chunkwise_state_num_v_rows) {
                const uint v_dim_index = simdgroup_v_dim_base + fragment_coordinate.y;
                recurrent_output[
                    ((ulong)(chunk_start + token_index_in_chunk) * num_v_heads + v_head_index) * v_head_dim
                    + v_dim_index] = bfloat16_t(output_elements[element_index]);
            }
        }

        const float final_decay = metal::exp(cumulative_log_decay[num_chunk_tokens - 1]);
        for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments;
             ++qk_fragment_index) {
            simdgroup_matrix<float, 8, 8> weighted_k_fragment;
            simdgroup_load(
                weighted_k_fragment,
                normalized_k + qk_fragment_index * matrix_size,
                qk_head_dim);
            thread float2 weighted_k_elements =
                reinterpret_cast<thread float2&>(weighted_k_fragment.thread_elements());
            const float k_scale = fragment_coordinate.y < num_chunk_tokens
                ? metal::exp(
                      cumulative_log_decay[num_chunk_tokens - 1]
                      - cumulative_log_decay[fragment_coordinate.y])
                : 0.0f;
            weighted_k_elements *= k_scale;
            reinterpret_cast<thread float2&>(weighted_k_fragment.thread_elements()) = weighted_k_elements;
            reinterpret_cast<thread float2&>(state_fragments[qk_fragment_index].thread_elements()) *=
                final_decay;
            simdgroup_multiply_accumulate(
                state_fragments[qk_fragment_index],
                delta_fragment,
                weighted_k_fragment,
                state_fragments[qk_fragment_index]);
        }
        const uint state_slot = write_candidate_states != 0 || chunk_start + num_chunk_tokens == flat_token_end
            ? flat_recurrent_state_write_slots[chunk_start + num_chunk_tokens - 1]
            : GDN_INVALID_STATE_SLOT_ID;
        if (state_slot != GDN_INVALID_STATE_SLOT_ID
            && simdgroup_v_dim_base + fragment_coordinate.y < v_dim_base + chunkwise_state_num_v_rows) {
            const ulong state_row_base = recurrent_state_base
                + (ulong)state_slot * recurrent_state_stride
                + ((ulong)v_head_index * v_head_dim + simdgroup_v_dim_base + fragment_coordinate.y) * qk_head_dim;
            for (uint qk_fragment_index = 0; qk_fragment_index < num_qk_fragments; ++qk_fragment_index) {
                const float2 state_elements = reinterpret_cast<thread float2&>(
                    state_fragments[qk_fragment_index].thread_elements());
                recurrent_state_arena[
                    state_row_base + qk_fragment_index * matrix_size + fragment_coordinate.x] =
                    bfloat16_t(state_elements[0]);
                recurrent_state_arena[
                    state_row_base + qk_fragment_index * matrix_size + fragment_coordinate.x + 1] =
                    bfloat16_t(state_elements[1]);
            }
        }
        chunk_start += num_chunk_tokens;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
