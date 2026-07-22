use std::cell::Cell;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAReplayShape;

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;

struct Qwen35DSparkBlockMetadata {
    req_slots: Vec<u32>,
    flat_token_indices: Vec<u32>,
    sdpa_map_task_templates: Vec<u32>,
    cu_sdpa_partial_outputs: Vec<u32>,
    local_sdpa_map_task_template_indices: Vec<u32>,
    replay_shape: GQAReplayShape,
}

pub struct Qwen35DSparkBlockRequest {
    local_block_size: u32,
    max_tokens: usize,
    max_sdpa_map_task_templates: usize,
    req_slots: Buffer,
    flat_token_indices: Buffer,
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    local_sdpa_map_task_template_indices: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
}

impl Qwen35DSparkBlockRequest {
    pub fn new(
        device: &Device,
        max_requests: usize,
        local_block_size: usize,
        max_sdpa_map_task_templates: usize,
    ) -> Self {
        assert!(max_requests > 0, "DSpark block request requires request capacity");
        assert!(
            local_block_size > 1,
            "DSpark block request requires anchor plus MASK rows"
        );
        let max_tokens = max_requests
            .checked_mul(local_block_size)
            .expect("DSpark block token capacity must fit usize");
        let min_sdpa_map_task_templates = max_tokens
            .checked_mul(2)
            .expect("DSpark block map TaskTemplate capacity must fit usize");
        assert!(
            u32::try_from(max_tokens).is_ok(),
            "DSpark block token capacity must fit u32"
        );
        assert!(
            max_sdpa_map_task_templates.is_power_of_two() && max_sdpa_map_task_templates >= min_sdpa_map_task_templates,
            "DSpark block map TaskTemplate capacity must be a power of two with persistent plus local TaskTemplates"
        );
        assert!(
            u32::try_from(max_sdpa_map_task_templates).is_ok(),
            "DSpark block map TaskTemplate capacity must fit u32"
        );
        Self {
            local_block_size: local_block_size
                .try_into()
                .expect("DSpark local block size must fit u32"),
            max_tokens,
            max_sdpa_map_task_templates,
            req_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(
                device,
                max_sdpa_map_task_templates
                    .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                    .expect("DSpark block map TaskTemplate metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_add(1)
                    .expect("DSpark cumulative partial-output capacity must fit usize"),
                Dtype::Uint32,
            ),
            local_sdpa_map_task_template_indices: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            replay_shape: Cell::new(None),
        }
    }

    pub fn update(&self, req_slots: &[u32], anchor_positions: &[u32], kv_token_tile_size: u32) -> GQAReplayShape {
        let metadata = build_block_metadata(
            req_slots,
            anchor_positions,
            self.local_block_size,
            kv_token_tile_size,
            self.max_sdpa_map_task_templates,
        );
        assert!(metadata.req_slots.len() <= self.max_tokens);
        self.req_slots.write_typed(0, &metadata.req_slots);
        self.flat_token_indices.write_typed(0, &metadata.flat_token_indices);
        self.sdpa_map_task_templates
            .write_typed(0, &metadata.sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs
            .write_typed(0, &metadata.cu_sdpa_partial_outputs);
        self.local_sdpa_map_task_template_indices
            .write_typed(0, &metadata.local_sdpa_map_task_template_indices);
        self.replay_shape.set(Some(metadata.replay_shape));
        metadata.replay_shape
    }

    pub fn local_block_size(&self) -> u32 {
        self.local_block_size
    }

    pub fn req_slots(&self) -> &Buffer {
        &self.req_slots
    }

    pub fn flat_token_indices(&self) -> &Buffer {
        &self.flat_token_indices
    }

    pub fn sdpa_map_task_templates(&self) -> &Buffer {
        &self.sdpa_map_task_templates
    }

    pub fn cu_sdpa_partial_outputs(&self) -> &Buffer {
        &self.cu_sdpa_partial_outputs
    }

    pub fn local_sdpa_map_task_template_indices(&self) -> &Buffer {
        &self.local_sdpa_map_task_template_indices
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
            .get()
            .expect("DSpark block request must be updated before recording")
    }
}

fn build_block_metadata(
    req_slots: &[u32],
    anchor_positions: &[u32],
    local_block_size: u32,
    kv_token_tile_size: u32,
    max_sdpa_map_task_templates: usize,
) -> Qwen35DSparkBlockMetadata {
    assert!(!req_slots.is_empty(), "DSpark block request requires requests");
    assert_eq!(req_slots.len(), anchor_positions.len());
    assert!(local_block_size > 1);
    assert!(kv_token_tile_size > 0);
    let num_tokens = req_slots
        .len()
        .checked_mul(local_block_size as usize)
        .expect("DSpark block token count must fit usize");
    assert!(
        max_sdpa_map_task_templates
            >= num_tokens
                .checked_mul(2)
                .expect("DSpark block map TaskTemplate count must fit usize")
    );
    assert!(max_sdpa_map_task_templates.is_power_of_two());
    assert!(anchor_positions.iter().all(|&position| position > 0));

    let mut req_slots_by_token = Vec::with_capacity(num_tokens);
    let mut flat_token_indices = Vec::with_capacity(num_tokens);
    let mut persistent_context_lens = Vec::with_capacity(num_tokens);
    for (&req_slot, &anchor_position) in req_slots.iter().zip(anchor_positions) {
        for local_index in 0..local_block_size {
            req_slots_by_token.push(req_slot);
            flat_token_indices.push(
                anchor_position
                    .checked_add(local_index)
                    .expect("DSpark block token position must fit u32"),
            );
            persistent_context_lens.push(anchor_position);
        }
    }

    let max_persistent_map_task_templates = max_sdpa_map_task_templates - num_tokens;
    let num_kv_token_tiles = persistent_context_lens
        .iter()
        .map(|&context_len| context_len.div_ceil(kv_token_tile_size) as usize)
        .collect::<Vec<_>>();
    let mut num_persistent_map_task_templates = vec![1usize; num_tokens];
    let mut total_persistent_map_task_templates = num_tokens;
    while total_persistent_map_task_templates < max_persistent_map_task_templates {
        let candidate = num_kv_token_tiles
            .iter()
            .zip(&num_persistent_map_task_templates)
            .enumerate()
            .filter(|&(_, (&tiles, &map_task_templates))| map_task_templates < tiles)
            .max_by_key(|&(_, (&tiles, &map_task_templates))| tiles.div_ceil(map_task_templates))
            .map(|(token_index, _)| token_index);
        let Some(token_index) = candidate else {
            break;
        };
        num_persistent_map_task_templates[token_index] += 1;
        total_persistent_map_task_templates += 1;
    }

    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(num_tokens + 1);
    let mut local_sdpa_map_task_template_indices = Vec::with_capacity(num_tokens);
    cu_sdpa_partial_outputs.push(0);
    for (
        q_token_tile_index,
        ((&context_len, &num_kv_token_tiles_for_q_token), &num_sdpa_map_task_templates_for_q_token),
    ) in persistent_context_lens
        .iter()
        .zip(&num_kv_token_tiles)
        .zip(&num_persistent_map_task_templates)
        .enumerate()
    {
        for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_for_q_token {
            let kv_token_tile_begin =
                num_kv_token_tiles_for_q_token * sdpa_map_task_template_index / num_sdpa_map_task_templates_for_q_token;
            let kv_token_tile_end = num_kv_token_tiles_for_q_token * (sdpa_map_task_template_index + 1)
                / num_sdpa_map_task_templates_for_q_token;
            let kv_token_begin = (kv_token_tile_begin as u64 * kv_token_tile_size as u64)
                .try_into()
                .expect("DSpark map TaskTemplate KV-token begin must fit u32");
            let kv_token_end = context_len.min(
                (kv_token_tile_end as u64 * kv_token_tile_size as u64)
                    .try_into()
                    .expect("DSpark map TaskTemplate KV-token end must fit u32"),
            );
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_tile_index
                    .try_into()
                    .expect("DSpark Q-token tile index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }
        // The paged map treats an invalid Q-token-tile index as inactive and
        // returns before writing this reserved TaskTemplate's partial output.
        // The local SDPA component fills that exact partial-output slot before
        // the shared partial-output reducer runs.
        local_sdpa_map_task_template_indices.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("DSpark local SDPA map TaskTemplate index must fit u32"),
        );
        sdpa_map_task_templates.extend_from_slice(&[u32::MAX, u32::MAX, u32::MAX]);
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("DSpark cumulative partial-output count must fit u32"),
        );
    }
    let num_sdpa_map_task_templates = sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS;
    let total_sdpa_map_task_templates = num_sdpa_map_task_templates
        .checked_next_power_of_two()
        .expect("DSpark total SDPA map TaskTemplate count must fit usize");
    assert!(total_sdpa_map_task_templates <= max_sdpa_map_task_templates);
    sdpa_map_task_templates.resize(
        total_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
        u32::MAX,
    );
    let num_tokens_u32 = num_tokens.try_into().expect("DSpark block token count must fit u32");
    let replay_shape = GQAReplayShape {
        num_tokens: num_tokens_u32,
        num_q_token_tiles: num_tokens_u32,
        total_sdpa_map_task_templates: total_sdpa_map_task_templates
            .try_into()
            .expect("DSpark total SDPA map TaskTemplate count must fit u32"),
        reduce_sdpa_partial_outputs: true,
    };
    replay_shape.validate();
    Qwen35DSparkBlockMetadata {
        req_slots: req_slots_by_token,
        flat_token_indices,
        sdpa_map_task_templates,
        cu_sdpa_partial_outputs,
        local_sdpa_map_task_template_indices,
        replay_shape,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_metadata_preserves_target_history_and_reserves_bidirectional_local_partial() {
        let metadata = build_block_metadata(&[3], &[11], 6, 4, 32);
        assert_eq!(metadata.req_slots, [3; 6]);
        assert_eq!(metadata.flat_token_indices, [11, 12, 13, 14, 15, 16]);
        assert_eq!(metadata.cu_sdpa_partial_outputs.len(), 7);
        assert_eq!(metadata.local_sdpa_map_task_template_indices.len(), 6);
        for q_token_index in 0..6 {
            let map_task_template_begin = metadata.cu_sdpa_partial_outputs[q_token_index] as usize;
            let map_task_template_end = metadata.cu_sdpa_partial_outputs[q_token_index + 1] as usize;
            assert!(map_task_template_end > map_task_template_begin + 1);
            let local_map_task_template = metadata.local_sdpa_map_task_template_indices[q_token_index] as usize;
            assert_eq!(local_map_task_template, map_task_template_end - 1);
            assert_eq!(
                metadata.sdpa_map_task_templates[local_map_task_template * 3..local_map_task_template * 3 + 3],
                [u32::MAX; 3]
            );
            for map_task_template in map_task_template_begin..local_map_task_template {
                assert_eq!(
                    metadata.sdpa_map_task_templates[map_task_template * 3],
                    q_token_index as u32
                );
                assert!(metadata.sdpa_map_task_templates[map_task_template * 3 + 2] <= 11);
            }
        }
        assert_eq!(metadata.replay_shape.num_tokens, 6);
        assert_eq!(metadata.replay_shape.total_sdpa_map_task_templates, 32);
        assert!(metadata.replay_shape.reduce_sdpa_partial_outputs);
    }

    #[test]
    fn block_metadata_keeps_requests_in_separate_local_blocks() {
        let metadata = build_block_metadata(&[2, 7], &[5, 20], 3, 8, 16);
        assert_eq!(metadata.req_slots, [2, 2, 2, 7, 7, 7]);
        assert_eq!(metadata.flat_token_indices, [5, 6, 7, 20, 21, 22]);
        assert!(
            metadata.sdpa_map_task_templates[..metadata.local_sdpa_map_task_template_indices[0] as usize * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .all(|map_task_template| map_task_template[2] <= 5)
        );
        let request_one_start = metadata.cu_sdpa_partial_outputs[3] as usize;
        let request_one_local = metadata.local_sdpa_map_task_template_indices[3] as usize;
        assert!(
            metadata.sdpa_map_task_templates[request_one_start * 3..request_one_local * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .all(|map_task_template| map_task_template[2] <= 20)
        );
    }

    #[test]
    fn bounded_map_task_templates_cover_complete_long_history() {
        let anchor_position = 262_144;
        let metadata = build_block_metadata(&[1], &[anchor_position], 6, 256, 16);
        for q_token_index in 0..6 {
            let map_task_template_begin = metadata.cu_sdpa_partial_outputs[q_token_index] as usize;
            let local_map_task_template = metadata.local_sdpa_map_task_template_indices[q_token_index] as usize;
            let persistent = metadata.sdpa_map_task_templates[map_task_template_begin * 3..local_map_task_template * 3]
                .as_chunks::<3>()
                .0;
            assert_eq!(persistent.first().expect("persistent map TaskTemplate")[1], 0);
            assert_eq!(
                persistent.last().expect("persistent map TaskTemplate")[2],
                anchor_position
            );
            assert!(persistent.windows(2).all(|pair| pair[0][2] == pair[1][1]));
        }
        assert_eq!(metadata.replay_shape.total_sdpa_map_task_templates, 16);
    }
}
