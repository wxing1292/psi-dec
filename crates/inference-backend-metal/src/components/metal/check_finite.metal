#include <metal_stdlib>
using namespace metal;

template <typename T>
static inline void check_value(device const T* input, uint rows, uint width,
                               device atomic_uint* status, uint index) {
    if (index >= rows * width) {
        return;
    }
    float value = float(input[index]);
    if (isfinite(value)) {
        return;
    }
    uint expected = 0;
    while (expected == 0) {
        if (atomic_compare_exchange_weak_explicit(status, &expected, 1,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(status + 1, index / width, memory_order_relaxed);
            atomic_store_explicit(status + 2, index % width, memory_order_relaxed);
            atomic_store_explicit(status + 3, as_type<uint>(value), memory_order_relaxed);
            return;
        }
    }
}

kernel void check_finite_f32(device const float* input [[buffer(0)]],
                             constant uint& rows [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             device atomic_uint* status [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    check_value(input, rows, width, status, index);
}

kernel void check_finite_bf16(device const bfloat* input [[buffer(0)]],
                              constant uint& rows [[buffer(1)]],
                              constant uint& width [[buffer(2)]],
                              device atomic_uint* status [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    check_value(input, rows, width, status, index);
}
