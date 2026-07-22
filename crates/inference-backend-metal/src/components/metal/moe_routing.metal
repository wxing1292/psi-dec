
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

static inline float read_bf16(device const bfloat16_t* values, uint index) {
    return float(values[index]);
}

static inline void write_bf16(device bfloat16_t* values, uint index, float value) {
    values[index] = bfloat16_t(value);
}

kernel void moe_route_topk(
    device const bfloat16_t* router_probs [[buffer(0)]],
    device uint* expert_indices [[buffer(1)]],
    device float* expert_probs [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& num_experts_per_token [[buffer(5)]],
    constant uint& norm_topk_prob [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (token >= num_tokens) return;
    const uint base = token * num_experts;
    threadgroup float candidate_probs[256];
    threadgroup uint candidate_experts[256];
    threadgroup float selected_probs[16];
    threadgroup uint selected_experts[16];
    for (uint slot = 0; slot < num_experts_per_token; ++slot) {
        float prob = -1.0f;
        uint expert_id = UINT_MAX;
        if (tid < num_experts) {
            bool already_selected = false;
            for (uint prev = 0; prev < slot; ++prev) {
                if (selected_experts[prev] == tid) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected) {
                prob = read_bf16(router_probs, base + tid);
                expert_id = tid;
            }
        }
        candidate_probs[tid] = prob;
        candidate_experts[tid] = expert_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float rhs_prob = candidate_probs[tid + stride];
                const uint rhs_expert = candidate_experts[tid + stride];
                const float lhs_prob = candidate_probs[tid];
                const uint lhs_expert = candidate_experts[tid];
                if (rhs_prob > lhs_prob || (rhs_prob == lhs_prob && rhs_expert < lhs_expert)) {
                    candidate_probs[tid] = rhs_prob;
                    candidate_experts[tid] = rhs_expert;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            selected_probs[slot] = candidate_probs[0];
            selected_experts[slot] = candidate_experts[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float topk_sum = 0.0f;
        for (uint slot = 0; slot < num_experts_per_token; ++slot) {
            topk_sum = float(bfloat16_t(topk_sum + selected_probs[slot]));
        }

        const uint out_base = token * num_experts_per_token;
        for (uint slot = 0; slot < num_experts_per_token; ++slot) {
            expert_indices[out_base + slot] = selected_experts[slot];
            float prob = selected_probs[slot];
            if (norm_topk_prob != 0 && num_experts_per_token > 1 && topk_sum > 0.0f) {
                prob = float(bfloat16_t(prob / topk_sum));
            }
            expert_probs[out_base + slot] = float(bfloat16_t(prob));
        }
    }
}
