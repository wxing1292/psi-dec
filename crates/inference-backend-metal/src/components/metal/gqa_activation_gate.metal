
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

METAL_FUNC bfloat16_t gqa_bf16_exp(bfloat16_t x) {
    return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__));
}

METAL_FUNC bfloat16_t gqa_bf16_abs(bfloat16_t x) {
    return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__));
}

METAL_FUNC float gqa_sigmoid(float x) {
    const float y = 1.0f / (1.0f + metal::exp(-metal::abs(x)));
    return (x < 0.0f) ? 1.0f - y : y;
}

METAL_FUNC bfloat16_t gqa_sigmoid(bfloat16_t x) {
    const bfloat16_t abs_x = gqa_bf16_abs(x);
    const bfloat16_t neg_abs_x = bfloat16_t(-static_cast<float>(abs_x));
    const bfloat16_t exp_x = gqa_bf16_exp(neg_abs_x);
    const bfloat16_t denom = bfloat16_t(1.0f + static_cast<float>(exp_x));
    const bfloat16_t y = bfloat16_t(1.0f / static_cast<float>(denom));
    return (x < bfloat16_t(0.0f)) ? bfloat16_t(1.0f - static_cast<float>(y)) : y;
}

template <typename T>
void gqa_activation_gate_impl(
    device const T* attention_output,
    device const T* gates,
    device T* output,
    constant uint& total_tokens,
    uint gid
) {
    const uint total = total_tokens * num_q_heads * head_dim;
    if (gid >= total) return;
    output[gid] = T(attention_output[gid] * gqa_sigmoid(gates[gid]));
}

kernel void gqa_activation_gate_f32(
    device const float* attention_output [[buffer(0)]],
    device const float* gates [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& total_tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_activation_gate_impl<float>(attention_output, gates, output, total_tokens, gid);
}

kernel void gqa_activation_gate_bf16(
    device const bfloat16_t* attention_output [[buffer(0)]],
    device const bfloat16_t* gates [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& total_tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_activation_gate_impl<bfloat16_t>(attention_output, gates, output, total_tokens, gid);
}
