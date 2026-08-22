#include <metal_stdlib>
using namespace metal;

#define THREADBLOCK_SIZE DSPARK_MARKOV_THREADBLOCK_SIZE
#define SIMD_SIZE DSPARK_MARKOV_SIMD_WIDTH
#define NUM_SIMDGROUPS (THREADBLOCK_SIZE / SIMD_SIZE)
#define RESULTS_PER_SIMDGROUP DSPARK_MARKOV_RESULTS_PER_SIMDGROUP
#define RESULTS_PER_WAVE (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP)
#define VALUES_PER_SIMD_LANE ((DSPARK_MARKOV_RANK + SIMD_SIZE - 1u) / SIMD_SIZE)
#define TOP_K_MAX 256u
#define NEG_INF -3.4028234663852886e38f

static inline bool rank_before(float left_logit, int left_token, float right_logit, int right_token) {
    if (left_token < 0) {
        return false;
    }
    if (right_token < 0) {
        return true;
    }
    return left_logit > right_logit || (left_logit == right_logit && left_token < right_token);
}

static inline uint unpack_affine_value(
    device const uchar* weight,
    ulong row_base_bytes,
    uint index,
    uint bits
) {
    if (bits == 2u) {
        uint byte = weight[row_base_bytes + index / 4u];
        return (byte >> ((index & 3u) * 2u)) & 0x03u;
    }
    if (bits == 3u) {
        uint pack = index / 8u;
        uint lane = index - pack * 8u;
        device const uchar* w = weight + row_base_bytes + pack * 3u;
        if (lane == 0u) return w[0] & 0x07u;
        if (lane == 1u) return (w[0] & 0x38u) >> 3u;
        if (lane == 2u) return ((w[0] & 0xc0u) >> 6u) + ((w[1] & 0x01u) << 2u);
        if (lane == 3u) return (w[1] & 0x0eu) >> 1u;
        if (lane == 4u) return (w[1] & 0x70u) >> 4u;
        if (lane == 5u) return ((w[1] & 0x80u) >> 7u) + ((w[2] & 0x03u) << 1u);
        if (lane == 6u) return (w[2] & 0x1cu) >> 2u;
        return (w[2] & 0xe0u) >> 5u;
    }
    if (bits == 4u) {
        uint byte = weight[row_base_bytes + index / 2u];
        return (byte >> ((index & 1u) * 4u)) & 0x0fu;
    }
    if (bits == 6u) {
        uint pack = index / 4u;
        uint lane = index - pack * 4u;
        device const uchar* w = weight + row_base_bytes + pack * 3u;
        if (lane == 0u) return w[0] & 0x3fu;
        if (lane == 1u) return ((w[0] >> 6u) & 0x03u) + ((w[1] & 0x0fu) << 2u);
        if (lane == 2u) return ((w[1] >> 4u) & 0x0fu) + ((w[2] & 0x03u) << 4u);
        return (w[2] >> 2u) & 0x3fu;
    }
    return weight[row_base_bytes + index];
}

static inline float dequantized_value(
    device const uchar* weight,
    device const bfloat* scales,
    device const bfloat* biases,
    uint row,
    uint index,
    uint rank,
    uint group_size,
    uint bits
) {
    const ulong row_bytes = (ulong)rank * (ulong)bits / 8ul;
    const ulong row_base_bytes = (ulong)row * row_bytes;
    const ulong affine_index =
        (ulong)row * ((ulong)rank / (ulong)group_size) + (ulong)(index / group_size);
    return float(unpack_affine_value(weight, row_base_bytes, index, bits)) * float(scales[affine_index])
        + float(biases[affine_index]);
}

kernel void dspark_markov_top_k_map(
    device const int* input_token_ids [[buffer(0)]],
    device const bfloat* base_logits [[buffer(1)]],
    device const uchar* w1_weight [[buffer(2)]],
    device const bfloat* w1_scales [[buffer(3)]],
    device const bfloat* w1_biases [[buffer(4)]],
    device const uchar* w2_weight [[buffer(5)]],
    device const bfloat* w2_scales [[buffer(6)]],
    device const bfloat* w2_biases [[buffer(7)]],
    device int* tile_token_ids [[buffer(8)]],
    device float* tile_logits [[buffer(9)]],
    constant uint& num_active_threads [[buffer(10)]],
    constant uint& top_k [[buffer(11)]],
    constant uint& num_tiles [[buffer(12)]],
    constant uint& base_logits_row_offset [[buffer(13)]],
    device const bfloat* confidence_hidden [[buffer(14)]],
    device const bfloat* confidence_weight [[buffer(15)]],
    device const bfloat* confidence_bias [[buffer(16)]],
    device float* confidence_output [[buffer(17)]],
    uint global_thread_id [[thread_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint group = global_thread_id / THREADBLOCK_SIZE;
    const uint group_first_thread = group * THREADBLOCK_SIZE;
    if (group_first_thread >= num_active_threads || top_k == 0u || top_k > TOP_K_MAX) {
        return;
    }
    const uint request = group / num_tiles;
    const uint tile = group - request * num_tiles;
    const uint tile_start = tile * DSPARK_MARKOV_VOCAB_TILE_SIZE;
    const int input_token_id = input_token_ids[request];

    threadgroup bfloat latent[DSPARK_MARKOV_RANK];
    threadgroup float values[DSPARK_MARKOV_VOCAB_TILE_SIZE];
    threadgroup int tokens[DSPARK_MARKOV_VOCAB_TILE_SIZE];

    for (uint rank_index = thread_index; rank_index < DSPARK_MARKOV_RANK; rank_index += THREADBLOCK_SIZE) {
        float value = 0.0f;
        if (input_token_id >= 0 && uint(input_token_id) < DSPARK_MARKOV_VOCAB_SIZE) {
            value = dequantized_value(
                w1_weight,
                w1_scales,
                w1_biases,
                uint(input_token_id),
                rank_index,
                DSPARK_MARKOV_RANK,
                DSPARK_MARKOV_W1_GROUP_SIZE,
                DSPARK_MARKOV_W1_BITS);
        }
        latent[rank_index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    thread float latent_values[VALUES_PER_SIMD_LANE];
    float latent_sum = 0.0f;
    const uint lane_rank_begin = simd_lane * VALUES_PER_SIMD_LANE;
    for (uint value_index = 0u; value_index < VALUES_PER_SIMD_LANE; ++value_index) {
        const uint rank_index = lane_rank_begin + value_index;
        const float value = rank_index < DSPARK_MARKOV_RANK ? float(latent[rank_index]) : 0.0f;
        latent_values[value_index] = value;
        latent_sum += value;
    }

    if (tile == 0u) {
        const ulong row = (ulong)base_logits_row_offset + (ulong)request;
        const ulong hidden_base = row * (ulong)DSPARK_CONFIDENCE_HIDDEN_DIM;
        float confidence_partial = 0.0f;
        for (uint hidden_index = thread_index;
             hidden_index < DSPARK_CONFIDENCE_HIDDEN_DIM;
             hidden_index += THREADBLOCK_SIZE) {
            confidence_partial += float(confidence_hidden[hidden_base + (ulong)hidden_index])
                * float(confidence_weight[hidden_index]);
        }
        for (uint rank_index = thread_index;
             rank_index < DSPARK_MARKOV_RANK;
             rank_index += THREADBLOCK_SIZE) {
            confidence_partial += float(latent[rank_index])
                * float(confidence_weight[DSPARK_CONFIDENCE_HIDDEN_DIM + rank_index]);
        }
        const float confidence_simd_sum = simd_sum(confidence_partial);
        if (simd_lane == 0u) {
            values[simd_group] = confidence_simd_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0u) {
            float confidence_raw = float(confidence_bias[0]);
            for (uint group_index = 0u; group_index < NUM_SIMDGROUPS; ++group_index) {
                confidence_raw += values[group_index];
            }
            confidence_output[row] = 1.0f / (1.0f + exp(-confidence_raw));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint wave = 0u; wave < DSPARK_MARKOV_VOCAB_TILE_SIZE / RESULTS_PER_WAVE; ++wave) {
        for (uint result_index = 0u; result_index < RESULTS_PER_SIMDGROUP; ++result_index) {
            const uint tile_index =
                wave * RESULTS_PER_WAVE + simd_group * RESULTS_PER_SIMDGROUP + result_index;
            const uint token_id = tile_start + tile_index;
            float partial = 0.0f;
            if (token_id < DSPARK_MARKOV_VOCAB_SIZE) {
#if DSPARK_MARKOV_W2_LANE_GROUP_ALIGNED
                float quantized_dot = 0.0f;
                const ulong row_bytes =
                    (ulong)DSPARK_MARKOV_RANK * (ulong)DSPARK_MARKOV_W2_BITS / 8ul;
                const ulong row_base_bytes = (ulong)token_id * row_bytes;
                for (uint value_index = 0u; value_index < VALUES_PER_SIMD_LANE; ++value_index) {
                    const uint rank_index = lane_rank_begin + value_index;
                    if (rank_index < DSPARK_MARKOV_RANK) {
                        quantized_dot += latent_values[value_index] * float(unpack_affine_value(
                            w2_weight, row_base_bytes, rank_index, DSPARK_MARKOV_W2_BITS));
                    }
                }
                const ulong affine_index =
                    (ulong)token_id * ((ulong)DSPARK_MARKOV_RANK / (ulong)DSPARK_MARKOV_W2_GROUP_SIZE)
                    + (ulong)(lane_rank_begin / DSPARK_MARKOV_W2_GROUP_SIZE);
                partial = quantized_dot * float(w2_scales[affine_index])
                    + latent_sum * float(w2_biases[affine_index]);
#else
                for (uint rank_index = simd_lane; rank_index < DSPARK_MARKOV_RANK; rank_index += SIMD_SIZE) {
                    partial += float(latent[rank_index]) * dequantized_value(
                        w2_weight,
                        w2_scales,
                        w2_biases,
                        token_id,
                        rank_index,
                        DSPARK_MARKOV_RANK,
                        DSPARK_MARKOV_W2_GROUP_SIZE,
                        DSPARK_MARKOV_W2_BITS);
                }
#endif
            }
            const float correction_sum = simd_sum(partial);
            if (simd_lane == 0u) {
                if (token_id < DSPARK_MARKOV_VOCAB_SIZE) {
                    const bfloat correction = bfloat(correction_sum);
                    const ulong base_index =
                        ((ulong)base_logits_row_offset + (ulong)request) * (ulong)DSPARK_MARKOV_VOCAB_SIZE
                        + (ulong)token_id;
                    const bfloat corrected = bfloat(float(base_logits[base_index]) + float(correction));
                    const float logit = float(corrected);
                    values[tile_index] = logit;
                    tokens[tile_index] = int(token_id);
                } else {
                    values[tile_index] = NEG_INF;
                    tokens[tile_index] = -1;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= DSPARK_MARKOV_VOCAB_TILE_SIZE; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            const uint other = thread_index ^ j;
            if (thread_index < DSPARK_MARKOV_VOCAB_TILE_SIZE && other > thread_index) {
                const float left_value = values[thread_index];
                const int left_token = tokens[thread_index];
                const float right_value = values[other];
                const int right_token = tokens[other];
                const bool descending = (thread_index & k) == 0u;
                const bool should_swap = descending
                    ? rank_before(right_value, right_token, left_value, left_token)
                    : rank_before(left_value, left_token, right_value, right_token);
                if (should_swap) {
                    values[thread_index] = right_value;
                    tokens[thread_index] = right_token;
                    values[other] = left_value;
                    tokens[other] = left_token;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const ulong output_base =
        ((ulong)request * (ulong)num_tiles + (ulong)tile) * (ulong)top_k;
    for (uint slot = thread_index; slot < top_k; slot += THREADBLOCK_SIZE) {
        if (slot < DSPARK_MARKOV_VOCAB_TILE_SIZE) {
            tile_token_ids[output_base + slot] = tokens[slot];
            tile_logits[output_base + slot] = values[slot];
        } else {
            tile_token_ids[output_base + slot] = -1;
            tile_logits[output_base + slot] = NEG_INF;
        }
    }
}
