
#include <metal_stdlib>
using namespace metal;

static inline ushort bf16_round(float value) {
    uint bits = as_type<uint>(value);
    uint lsb = (bits >> 16) & 1u;
    uint rounding_bias = 0x7fffu + lsb;
    return ushort((bits + rounding_bias) >> 16);
}

static inline uint unpack_affine_value(
    device const uchar* weight,
    ulong row_base_bytes,
    uint hidden_index,
    uint bits
) {
    if (bits == 2u) {
        uint byte = weight[row_base_bytes + hidden_index / 4u];
        return (byte >> ((hidden_index & 3u) * 2u)) & 0x03u;
    }
    if (bits == 3u) {
        uint pack = hidden_index / 8u;
        uint lane = hidden_index - pack * 8u;
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
        uint byte = weight[row_base_bytes + hidden_index / 2u];
        return (byte >> ((hidden_index & 1u) * 4u)) & 0x0fu;
    }
    if (bits == 6u) {
        uint pack = hidden_index / 4u;
        uint lane = hidden_index - pack * 4u;
        device const uchar* w = weight + row_base_bytes + pack * 3u;
        if (lane == 0u) return w[0] & 0x3fu;
        if (lane == 1u) return ((w[0] >> 6u) & 0x03u) + ((w[1] & 0x0fu) << 2u);
        if (lane == 2u) return ((w[1] >> 4u) & 0x0fu) + ((w[2] & 0x03u) << 4u);
        return (w[2] >> 2u) & 0x3fu;
    }
    return weight[row_base_bytes + hidden_index];
}

template <typename T>
static inline float affine_param_to_float(T value) {
    return float(value);
}

template <>
inline float affine_param_to_float<bfloat>(bfloat value) {
    return float(value);
}

template <typename T>
static inline void quantized_embedding_impl(
    device const int* token_ids,
    device const uchar* weight,
    device const T* scales,
    device const T* biases,
    device ushort* output,
    uint num_tokens,
    uint vocab_size,
    uint hidden_dim,
    uint group_size,
    uint bits,
    uint gid
) {
    const uint total = num_tokens * hidden_dim;
    if (gid >= total) {
        return;
    }
    const uint token_index = gid / hidden_dim;
    const uint hidden_index = gid - token_index * hidden_dim;
    const int token_id = token_ids[token_index];
    if (token_id < 0 || uint(token_id) >= vocab_size) {
        output[gid] = 0;
        return;
    }

    const ulong packed_cols = (ulong)hidden_dim * (ulong)bits / 32ul;
    const ulong row_base_bytes = (ulong)uint(token_id) * packed_cols * 4ul;
    const uint group_index = hidden_index / group_size;
    const ulong affine_index = (ulong)uint(token_id) * ((ulong)hidden_dim / (ulong)group_size) + (ulong)group_index;
    float scale = affine_param_to_float(scales[affine_index]);
    float bias = affine_param_to_float(biases[affine_index]);
    float value = float(unpack_affine_value(weight, row_base_bytes, hidden_index, bits)) * scale + bias;
    output[gid] = bf16_round(value);
}

kernel void quantized_embedding_f32_to_bf16(
    device const int* token_ids [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    quantized_embedding_impl<float>(
        token_ids, weight, scales, biases, output, num_tokens, vocab_size, hidden_dim, group_size, bits, gid);
}

kernel void quantized_embedding_bf16_to_bf16(
    device const int* token_ids [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    quantized_embedding_impl<bfloat>(
        token_ids, weight, scales, biases, output, num_tokens, vocab_size, hidden_dim, group_size, bits, gid);
}
