#include <metal_stdlib>
using namespace metal;

struct GDNStateReplayJob {
    uint src_recurrent_state_slot;
    uint src_conv_state_slot;
    uint dst_recurrent_state_slot;
    uint dst_conv_state_slot;
    uint replay_token_begin;
    uint num_tokens;
};

// Each thread owns one state value. Jobs choose accepted prefixes; layers are independent.
kernel void gdn_state_replay_recurrent_bf16(
    device bfloat* states [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* u [[buffer(3)]],
    device const GDNStateReplayJob* jobs [[buffer(4)]],
    constant uint& num_active_jobs [[buffer(5)]],
    constant uint& num_state_slots [[buffer(6)]],
    constant uint& max_replay_tokens [[buffer(7)]],
    uint3 position [[thread_position_in_grid]]
) {
    const uint state_stride = num_v_heads * v_head_dim * qk_head_dim;
    if (position.x >= state_stride || position.y >= num_active_jobs) {
        return;
    }
    const GDNStateReplayJob job = jobs[position.y];
    const uint qk_dim_index = position.x % qk_head_dim;
    const uint v_dim_index = (position.x / qk_head_dim) % v_head_dim;
    const uint v_head_index = position.x / (qk_head_dim * v_head_dim);
    const uint qk_head_index = v_head_index / (num_v_heads / num_qk_heads);
    const ulong state_layer_base = (ulong)position.z * num_state_slots * state_stride;
    float state = states[state_layer_base + (ulong)job.src_recurrent_state_slot * state_stride + position.x];
    const ulong token_begin = (ulong)position.z * max_replay_tokens + job.replay_token_begin;
    for (uint token_index = 0; token_index < job.num_tokens; ++token_index) {
        const ulong token = token_begin + token_index;
        const float decay = alpha[token * num_v_heads + v_head_index];
        const float normalized_k = k[(token * num_qk_heads + qk_head_index) * qk_head_dim + qk_dim_index];
        const float delta = u[(token * num_v_heads + v_head_index) * v_head_dim + v_dim_index];
        const float decayed_state = state * decay;
        state = decayed_state + normalized_k * delta;
    }
    states[state_layer_base + (ulong)job.dst_recurrent_state_slot * state_stride + position.x] = bfloat(state);
}

kernel void gdn_state_replay_conv_bf16(
    device bfloat* states [[buffer(0)]],
    device const bfloat* qkv [[buffer(1)]],
    device const GDNStateReplayJob* jobs [[buffer(2)]],
    constant uint& num_active_jobs [[buffer(3)]],
    constant uint& num_state_slots [[buffer(4)]],
    constant uint& max_replay_tokens [[buffer(5)]],
    uint3 position [[thread_position_in_grid]]
) {
    const uint state_stride = qkv_dim * conv_state_len;
    if (position.x >= state_stride || position.y >= num_active_jobs) {
        return;
    }
    const GDNStateReplayJob job = jobs[position.y];
    const uint state_index = position.x % conv_state_len;
    const uint channel_index = position.x / conv_state_len;
    const ulong state_layer_base = (ulong)position.z * num_state_slots * state_stride;
    const long input_index = (long)job.num_tokens + state_index - conv_state_len;
    bfloat value;
    if (input_index < 0) {
        value = states[state_layer_base + (ulong)job.src_conv_state_slot * state_stride
            + channel_index * conv_state_len + state_index + job.num_tokens];
    } else {
        const ulong token = (ulong)position.z * max_replay_tokens + job.replay_token_begin + input_index;
        value = qkv[token * qkv_dim + channel_index];
    }
    states[state_layer_base + (ulong)job.dst_conv_state_slot * state_stride + position.x] = value;
}
