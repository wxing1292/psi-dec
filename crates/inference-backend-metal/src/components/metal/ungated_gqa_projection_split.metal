#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void ungated_gqa_projection_split_impl(
    device const T* qkv,
    device T* q,
    device T* k,
    device T* v,
    constant uint& total_tokens,
    uint gid
) {
    const uint q_slots = num_q_heads * head_dim;
    const uint kv_slots = num_kv_heads * head_dim;
    const uint token_width = q_slots + 2 * kv_slots;
    const uint total = total_tokens * token_width;
    if (gid >= total) return;

    const uint token = gid / token_width;
    uint slot_index = gid - token * token_width;
    const T qkv_slot = qkv[gid];

    if (slot_index < q_slots) {
        q[token * q_slots + slot_index] = qkv_slot;
        return;
    }
    slot_index -= q_slots;
    if (slot_index < kv_slots) {
        k[token * kv_slots + slot_index] = qkv_slot;
        return;
    }
    slot_index -= kv_slots;
    v[token * kv_slots + slot_index] = qkv_slot;
}

kernel void ungated_gqa_projection_split_f32(
    device const float* qkv [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* k [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& total_tokens [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    ungated_gqa_projection_split_impl<float>(
        qkv, q, k, v, total_tokens, gid);
}

kernel void ungated_gqa_projection_split_bf16(
    device const bfloat16_t* qkv [[buffer(0)]],
    device bfloat16_t* q [[buffer(1)]],
    device bfloat16_t* k [[buffer(2)]],
    device bfloat16_t* v [[buffer(3)]],
    constant uint& total_tokens [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    ungated_gqa_projection_split_impl<bfloat16_t>(
        qkv, q, k, v, total_tokens, gid);
}
