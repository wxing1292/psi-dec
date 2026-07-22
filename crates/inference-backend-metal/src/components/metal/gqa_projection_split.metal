
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void gqa_projection_split_impl(
    device const T* qgkv,
    device T* q,
    device T* g,
    device T* k,
    device T* v,
    constant uint& total_tokens,
    uint gid
) {
    const uint q_slots = num_q_heads * head_dim;
    const uint kv_slots = num_kv_heads * head_dim;
    const uint token_width = 2 * q_slots + 2 * kv_slots;
    const uint total = total_tokens * token_width;
    if (gid >= total) return;

    const uint token = gid / token_width;
    uint slot_index = gid - token * token_width;
    const T qgkv_slot = qgkv[gid];

    if (slot_index < 2 * q_slots) {
        const uint qg_pair = slot_index / (2 * head_dim);
        const uint in_pair = slot_index - qg_pair * 2 * head_dim;
        const uint dst = (token * num_q_heads + qg_pair) * head_dim + (in_pair % head_dim);
        if (in_pair < head_dim) {
            q[dst] = qgkv_slot;
        } else {
            g[dst] = qgkv_slot;
        }
        return;
    }
    slot_index -= 2 * q_slots;
    if (slot_index < kv_slots) {
        k[token * kv_slots + slot_index] = qgkv_slot;
        return;
    }
    slot_index -= kv_slots;
    v[token * kv_slots + slot_index] = qgkv_slot;
}

kernel void gqa_projection_split_f32(
    device const float* qgkv [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* g [[buffer(2)]],
    device float* k [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant uint& total_tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_projection_split_impl<float>(
        qgkv, q, g, k, v, total_tokens, gid);
}

kernel void gqa_projection_split_bf16(
    device const bfloat16_t* qgkv [[buffer(0)]],
    device bfloat16_t* q [[buffer(1)]],
    device bfloat16_t* g [[buffer(2)]],
    device bfloat16_t* k [[buffer(3)]],
    device bfloat16_t* v [[buffer(4)]],
    constant uint& total_tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_projection_split_impl<bfloat16_t>(
        qgkv, q, g, k, v, total_tokens, gid);
}
