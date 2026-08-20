
#include <metal_stdlib>
using namespace metal;

// One logical GDNStatePageWriteTask maps 1:1 to one threadblock and copies one
// state-slot page into page storage. No Task value, TaskTemplate, or ABI buffer
// is materialized:
//
// GDNStatePageWriteTask {
//   state_io_request_index,  // grid-derived
//   gdn_layer_index,       // grid-derived
//   state_kind,            // grid-derived: recurrent or convolution
//   page_index_in_state,   // grid-derived
// }
//
// page_id, recurrent_state_slot, and conv_state_slot are data inputs, not Task
// coordinates. state_kind selects the applicable physical slot.
kernel void gdn_state_page_batch_write_f32(
    device uchar* pages [[buffer(0)]],
    device const uchar* recurrent_states [[buffer(1)]],
    device const uchar* conv_states [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* recurrent_state_slots [[buffer(4)]],
    device const uint* conv_state_slots [[buffer(5)]],
    constant uint& num_gdn_layers [[buffer(6)]],
    constant uint& num_state_slots [[buffer(7)]],
    constant uint& num_state_io_requests [[buffer(8)]],
    constant uint& num_recurrent_pages_per_state_slot [[buffer(9)]],
    constant uint& recurrent_state_bytes [[buffer(10)]],
    constant uint& num_conv_pages_per_state_slot [[buffer(11)]],
    constant uint& conv_state_bytes [[buffer(12)]],
    constant uint& page_bytes [[buffer(13)]],
    uint state_page_threadblock_index [[threadgroup_position_in_grid]],
    uint thread_index_in_threadblock [[thread_position_in_threadgroup]],
    uint num_threads [[threads_per_threadgroup]]
) {
    const uint pages_per_layer = num_recurrent_pages_per_state_slot + num_conv_pages_per_state_slot;
    const uint pages_per_state_io_request = num_gdn_layers * pages_per_layer;
    const uint total_pages = num_state_io_requests * pages_per_state_io_request;
    if (state_page_threadblock_index >= total_pages) return;

    const uint state_io_request_index = state_page_threadblock_index / pages_per_state_io_request;
    const uint page_index_in_state_io_request = state_page_threadblock_index - state_io_request_index * pages_per_state_io_request;
    const uint gdn_layer_index = page_index_in_state_io_request / pages_per_layer;
    const uint page_index_in_layer = page_index_in_state_io_request - gdn_layer_index * pages_per_layer;
    const bool is_recurrent_state = page_index_in_layer < num_recurrent_pages_per_state_slot;
    const uint page_index_in_state =
        is_recurrent_state ? page_index_in_layer : page_index_in_layer - num_recurrent_pages_per_state_slot;
    const uint state_bytes = is_recurrent_state ? recurrent_state_bytes : conv_state_bytes;
    const ulong page_id = (ulong)page_ids[state_page_threadblock_index];
    const ulong state_slot = is_recurrent_state
        ? (ulong)recurrent_state_slots[state_io_request_index]
        : (ulong)conv_state_slots[state_io_request_index];
    device const uchar* states = is_recurrent_state ? recurrent_states : conv_states;
    const ulong page_offset_bytes = page_id * (ulong)page_bytes;
    const ulong state_slot_offset_bytes =
        ((ulong)gdn_layer_index * (ulong)num_state_slots + state_slot) * (ulong)state_bytes;

    for (uint byte_offset_in_page = thread_index_in_threadblock * sizeof(float4);
         byte_offset_in_page < page_bytes;
         byte_offset_in_page += num_threads * sizeof(float4)) {
        const ulong state_byte_offset =
            (ulong)page_index_in_state * (ulong)page_bytes + (ulong)byte_offset_in_page;
        const float4 value = state_byte_offset < (ulong)state_bytes
            ? *reinterpret_cast<device const float4*>(states + state_slot_offset_bytes + state_byte_offset)
            : float4(0.0f);
        *reinterpret_cast<device float4*>(pages + page_offset_bytes + byte_offset_in_page) = value;
    }
}
