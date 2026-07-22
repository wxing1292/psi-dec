
#include <metal_stdlib>
using namespace metal;

template <typename T>
void gqa_kv_page_update_impl(
    device T* pages,
    device const T* flat_k,
    device const T* flat_v,
    device const uint* req_slots,
    device const uint* flat_token_indices,
    device const uint* page_ids,
    constant uint& num_token_writes,
    constant uint& gqa_layer_index,
    constant uint& num_gqa_layers,
    constant uint& num_blocks,
    constant uint& num_page_ids_per_block,
    uint gid
) {
    const uint slots_per_write = num_kv_heads * head_dim;
    const uint total = num_token_writes * slots_per_write;
    if (gid >= total) return;

    const uint write_index = gid / slots_per_write;
    const uint slot_index = gid - write_index * slots_per_write;
    const uint head_index = slot_index / head_dim;
    const uint dim_index = slot_index - head_index * head_dim;

    const uint req_slot = req_slots[write_index];
    const uint token_index = flat_token_indices[write_index];
    const uint block_index = token_index / (num_tokens_per_page * num_page_ids_per_block);
    const uint page_id_index = (token_index / num_tokens_per_page) % num_page_ids_per_block;
    const ulong page_id_table_index =
        ((((ulong)req_slot * (ulong)num_gqa_layers + (ulong)gqa_layer_index)
          * (ulong)num_blocks
          + (ulong)block_index)
         * (ulong)num_page_ids_per_block)
        + (ulong)page_id_index;
    const ulong page_id = (ulong)page_ids[page_id_table_index];
    const uint page_token_index = token_index % num_tokens_per_page;

    const ulong num_kv_slots_per_page = (ulong)num_kv_heads * (ulong)num_tokens_per_page * (ulong)head_dim;
    const ulong page_base = page_id * ((ulong)page_bytes / sizeof(T));
    const ulong v_region_offset = num_kv_slots_per_page;
    const ulong page_token_offset =
        ((ulong)head_index * (ulong)num_tokens_per_page + (ulong)page_token_index) * (ulong)head_dim
        + (ulong)dim_index;
    const ulong flat_offset =
        ((ulong)write_index * (ulong)num_kv_heads + (ulong)head_index) * (ulong)head_dim + (ulong)dim_index;

    pages[page_base + page_token_offset] = flat_k[flat_offset];
    pages[page_base + v_region_offset + page_token_offset] = flat_v[flat_offset];
}

kernel void gqa_kv_page_update_u16(
    device ushort* pages [[buffer(0)]],
    device const ushort* flat_k [[buffer(1)]],
    device const ushort* flat_v [[buffer(2)]],
    device const uint* req_slots [[buffer(3)]],
    device const uint* flat_token_indices [[buffer(4)]],
    device const uint* page_ids [[buffer(5)]],
    constant uint& num_token_writes [[buffer(6)]],
    constant uint& gqa_layer_index [[buffer(7)]],
    constant uint& num_gqa_layers [[buffer(8)]],
    constant uint& num_blocks [[buffer(9)]],
    constant uint& num_page_ids_per_block [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_kv_page_update_impl(
        pages,
        flat_k,
        flat_v,
        req_slots,
        flat_token_indices,
        page_ids,
        num_token_writes,
        gqa_layer_index,
        num_gqa_layers,
        num_blocks,
        num_page_ids_per_block,
        gid
    );
}

kernel void gqa_kv_page_update_f32(
    device float* pages [[buffer(0)]],
    device const float* flat_k [[buffer(1)]],
    device const float* flat_v [[buffer(2)]],
    device const uint* req_slots [[buffer(3)]],
    device const uint* flat_token_indices [[buffer(4)]],
    device const uint* page_ids [[buffer(5)]],
    constant uint& num_token_writes [[buffer(6)]],
    constant uint& gqa_layer_index [[buffer(7)]],
    constant uint& num_gqa_layers [[buffer(8)]],
    constant uint& num_blocks [[buffer(9)]],
    constant uint& num_page_ids_per_block [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_kv_page_update_impl(
        pages,
        flat_k,
        flat_v,
        req_slots,
        flat_token_indices,
        page_ids,
        num_token_writes,
        gqa_layer_index,
        num_gqa_layers,
        num_blocks,
        num_page_ids_per_block,
        gid
    );
}
