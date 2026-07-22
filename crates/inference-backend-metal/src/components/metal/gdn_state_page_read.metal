
#include <metal_stdlib>
using namespace metal;

kernel void gdn_state_page_read_f32(
    device const float* pages [[buffer(0)]],
    device float* flat_state [[buffer(1)]],
    device const uint* page_ids [[buffer(2)]],
    constant uint& page_id_start_index [[buffer(3)]],
    constant uint& num_pages [[buffer(4)]],
    constant uint& state_bytes [[buffer(5)]],
    constant uint& page_bytes [[buffer(6)]],
    uint state_value_index [[thread_position_in_grid]]
) {
    const ulong state_byte_offset = (ulong)state_value_index * sizeof(float);
    if (state_byte_offset >= (ulong)state_bytes) return;
    const uint page_index = (uint)(state_byte_offset / (ulong)page_bytes);
    if (page_index >= num_pages) return;
    const ulong value_index_in_page =
        (state_byte_offset - (ulong)page_index * (ulong)page_bytes) / sizeof(float);
    const ulong page_id = (ulong)page_ids[page_id_start_index + page_index];
    flat_state[state_value_index] = pages[page_id * ((ulong)page_bytes / sizeof(float)) + value_index_in_page];
}

// One logical GDNStatePageReadTask maps 1:1 to one threadblock and copies one
// page from page storage into a state slot. No Task value, TaskTemplate, or ABI
// buffer is materialized:
//
// GDNStatePageReadTask {
//   state_io_request_index,  // grid-derived
//   gdn_layer_index,       // grid-derived
//   state_kind,            // grid-derived: recurrent or convolution
//   page_index_in_state,   // grid-derived
// }
//
// page_id and state_slot are data inputs, not Task coordinates.
kernel void gdn_state_page_batch_read_f32(
    device const uchar* pages [[buffer(0)]],
    device uchar* recurrent_states [[buffer(1)]],
    device uchar* conv_states [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* state_slots [[buffer(4)]],
    constant uint& num_gdn_layers [[buffer(5)]],
    constant uint& num_state_slots [[buffer(6)]],
    constant uint& num_state_io_requests [[buffer(7)]],
    constant uint& num_recurrent_pages_per_state_slot [[buffer(8)]],
    constant uint& recurrent_state_bytes [[buffer(9)]],
    constant uint& num_conv_pages_per_state_slot [[buffer(10)]],
    constant uint& conv_state_bytes [[buffer(11)]],
    constant uint& page_bytes [[buffer(12)]],
    uint state_page_threadblock_index [[threadgroup_position_in_grid]],
    uint thread_index_in_threadblock [[thread_position_in_threadgroup]],
    uint num_threads_per_threadblock [[threads_per_threadgroup]]
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
    const ulong state_slot = (ulong)state_slots[state_io_request_index];
    device uchar* states = is_recurrent_state ? recurrent_states : conv_states;
    const ulong page_offset_bytes = page_id * (ulong)page_bytes;
    const ulong state_slot_offset_bytes =
        ((ulong)gdn_layer_index * (ulong)num_state_slots + state_slot) * (ulong)state_bytes;

    for (uint byte_offset_in_page = thread_index_in_threadblock * sizeof(float4);
         byte_offset_in_page < page_bytes;
         byte_offset_in_page += num_threads_per_threadblock * sizeof(float4)) {
        const ulong state_byte_offset =
            (ulong)page_index_in_state * (ulong)page_bytes + (ulong)byte_offset_in_page;
        if (state_byte_offset >= (ulong)state_bytes) break;
        *reinterpret_cast<device float4*>(states + state_slot_offset_bytes + state_byte_offset) =
            *reinterpret_cast<device const float4*>(pages + page_offset_bytes + byte_offset_in_page);
    }
}
