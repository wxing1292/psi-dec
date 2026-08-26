#include <metal_stdlib>
using namespace metal;

constant uint RESOURCE_EMBED_NUM_U32S_PER_MAPPING = 3u;

kernel void resource_embed_bf16(
    device const uchar* resource_arena [[buffer(0)]],
    device const uint* mappings [[buffer(1)]],
    device bfloat* hidden [[buffer(2)]],
    constant uint& num_active_mappings [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    uint value_index [[thread_position_in_grid]]
) {
    const uint mapping_index = value_index / hidden_dim;
    if (mapping_index >= num_active_mappings) return;

    const uint hidden_index = value_index - mapping_index * hidden_dim;
    const uint mapping_u32_index = mapping_index * RESOURCE_EMBED_NUM_U32S_PER_MAPPING;
    const uint destination_row = mappings[mapping_u32_index];
    const ulong source_offset_bytes =
        (ulong)mappings[mapping_u32_index + 1u] | ((ulong)mappings[mapping_u32_index + 2u] << 32u);
    device const bfloat* source = reinterpret_cast<device const bfloat*>(resource_arena + source_offset_bytes);
    hidden[(ulong)destination_row * (ulong)hidden_dim + (ulong)hidden_index] = source[hidden_index];
}
