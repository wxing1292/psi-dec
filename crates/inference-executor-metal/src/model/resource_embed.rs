use inference_backend_metal::components::resource_embed::Config;
use inference_backend_metal::components::resource_embed::Mapping;
use inference_backend_metal::components::resource_embed::MappingTable;
use inference_runtime_core::compute::DeviceRequest;

pub fn build_mapping_table(config: Config, requests: &[DeviceRequest]) -> Option<MappingTable> {
    let hidden_dim_bytes = config.hidden_dim_bytes();
    let mut flat_token_start = 0usize;
    let mut mappings = vec![];
    for request in requests {
        let query_token_start = request.decoder_query_tokens.token_index();
        let num_query_tokens = request.decoder_query_tokens.token_consumption();
        let query_token_end = query_token_start + num_query_tokens;
        for resource in &request.resource_placements {
            for &(placement_token_start, resource_index, num_resource_tokens) in resource.placements() {
                let placement_token_end = placement_token_start + num_resource_tokens;
                let active_token_start = query_token_start.max(placement_token_start);
                let active_token_end = query_token_end.min(placement_token_end);
                if active_token_start >= active_token_end {
                    continue;
                }

                let active_resource_start = resource_index + active_token_start - placement_token_start;
                let num_active_resource_tokens = active_token_end - active_token_start;
                let active_resource_end = active_resource_start + num_active_resource_tokens;
                let relative_source_start_bytes = u64::try_from(active_resource_start)
                    .expect("resource index must fit the u64 byte-address domain")
                    .checked_mul(hidden_dim_bytes)
                    .expect("resource embedding source byte offset must fit u64");
                let relative_source_end_bytes = u64::try_from(active_resource_end)
                    .expect("resource index must fit the u64 byte-address domain")
                    .checked_mul(hidden_dim_bytes)
                    .expect("resource embedding source byte end must fit u64");
                assert!(
                    relative_source_end_bytes <= resource.arena_len_bytes(),
                    "resource embedding source range exceeds its allocation"
                );
                resource
                    .arena_offset_bytes()
                    .checked_add(relative_source_end_bytes)
                    .expect("resource embedding arena byte range must fit u64");
                let mut source_offset_bytes = resource
                    .arena_offset_bytes()
                    .checked_add(relative_source_start_bytes)
                    .expect("resource embedding arena byte offset must fit u64");
                let destination_row_start = flat_token_start + active_token_start - query_token_start;
                for token_offset in 0..num_active_resource_tokens {
                    let destination_row = destination_row_start + token_offset;
                    mappings.push(Mapping {
                        destination_row: u32::try_from(destination_row)
                            .expect("resource embedding destination row must fit the shader u32 domain"),
                        source_offset_bytes,
                    });
                    source_offset_bytes += hidden_dim_bytes;
                }
            }
        }
        flat_token_start += num_query_tokens;
    }

    (!mappings.is_empty()).then(|| MappingTable::new(config, mappings))
}

#[cfg(test)]
#[path = "resource_embed_test.rs"]
mod tests;
