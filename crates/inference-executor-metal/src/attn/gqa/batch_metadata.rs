use std::cell::Cell;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAReplayShape;

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;
const NUM_Q_TOKEN_TILE_FIELDS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GQAMetadataBuffersPath {
    ContextParallel {
        kv_token_tile_size: u32,
    },
    Tiled {
        q_token_tile_size: u32,
        kv_token_tile_size: u32,
    },
}

/// Capacity-sized GPU metadata and replay shape refreshed from each model
/// microbatch and shared by all GQA layers.
pub struct GQAMetadataBuffers {
    req_slots: Buffer,
    // Indexed by flat token order; each index is request-absolute, not a
    // batch-flat gather index. For request starts [10, 20] and cu_tokens
    // [0, 3, 6], this is [10, 11, 12, 20, 21, 22].
    flat_token_indices: Buffer,
    // Compact Q-token tile `[flat_token_start, flat_token_end]` entries. A tile never
    // crosses a request boundary.
    q_token_tiles: Buffer,
    // Compact `SDPAMapTaskTemplate` entries with the materialized ABI
    // `[q_token_tile_index, kv_token_begin, kv_token_end]`. `kv_head_index` and
    // `q_head_tile_index` are derived from the regular dispatch grid. TaskTemplates
    // for one Q-token tile are contiguous; adjacent `cu_sdpa_partial_outputs`
    // values select that tile's partial outputs.
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
}

impl GQAMetadataBuffers {
    pub fn new(device: &Device, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "GQA batch metadata requires tokens");
        assert!(u32::try_from(max_tokens).is_ok(), "GQA token capacity must fit u32");
        Self {
            req_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            q_token_tiles: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_mul(NUM_Q_TOKEN_TILE_FIELDS)
                    .expect("GQA token-tile metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                    .expect("GQA SDPA map TaskTemplate metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_add(1)
                    .expect("GQA SDPA partial-output cumulative-count capacity must fit usize"),
                Dtype::Uint32,
            ),
            replay_shape: Cell::new(None),
        }
    }

    pub fn update_context_parallel(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        kv_token_tile_size: u32,
    ) -> GQAReplayShape {
        self.update(
            req_slots,
            token_indices,
            cu_tokens,
            GQAMetadataBuffersPath::ContextParallel { kv_token_tile_size },
        )
    }

    pub fn update_tiled(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        q_token_tile_size: u32,
        kv_token_tile_size: u32,
    ) -> GQAReplayShape {
        self.update(
            req_slots,
            token_indices,
            cu_tokens,
            GQAMetadataBuffersPath::Tiled {
                q_token_tile_size,
                kv_token_tile_size,
            },
        )
    }

    fn update(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        path: GQAMetadataBuffersPath,
    ) -> GQAReplayShape {
        let tiled = matches!(path, GQAMetadataBuffersPath::Tiled { .. });
        assert_eq!(req_slots.len(), token_indices.len());
        assert_eq!(cu_tokens.len(), req_slots.len() + 1);
        let num_tokens = cu_tokens.last().copied().unwrap_or_default() as usize;
        assert!(num_tokens > 0, "GQA batch metadata requires tokens");
        assert!(num_tokens <= self.req_slots.len_bytes() / size_of::<u32>());

        let mut req_slots_by_token = Vec::with_capacity(num_tokens);
        let mut flat_token_indices = Vec::with_capacity(num_tokens);
        for req_index in 0..req_slots.len() {
            let flat_token_begin = cu_tokens[req_index] as usize;
            let flat_token_end = cu_tokens[req_index + 1] as usize;
            assert!(
                flat_token_begin <= flat_token_end,
                "GQA batch cu_tokens must be nondecreasing"
            );
            for token_index_in_req in 0..(flat_token_end - flat_token_begin) {
                let token_index = token_indices[req_index]
                    .checked_add(
                        token_index_in_req
                            .try_into()
                            .expect("GQA request-local token index must fit u32"),
                    )
                    .expect("GQA token index overflow");
                req_slots_by_token.push(req_slots[req_index]);
                flat_token_indices.push(token_index);
            }
        }
        assert_eq!(req_slots_by_token.len(), num_tokens);
        self.req_slots.write_typed(0, &req_slots_by_token);
        self.flat_token_indices.write_typed(0, &flat_token_indices);

        let max_sdpa_map_task_templates =
            self.sdpa_map_task_templates.len_bytes() / (NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS * size_of::<u32>());
        let (q_token_tiles, mut sdpa_map_task_templates, cu_sdpa_partial_outputs) = match path {
            GQAMetadataBuffersPath::ContextParallel { kv_token_tile_size } => {
                assert!(kv_token_tile_size > 0, "GQA KV-token tile size must be positive");
                let (sdpa_map_task_templates, cu_sdpa_partial_outputs) = build_context_parallel_map_task_templates(
                    &flat_token_indices,
                    kv_token_tile_size,
                    max_sdpa_map_task_templates,
                );
                (Vec::new(), sdpa_map_task_templates, cu_sdpa_partial_outputs)
            },
            GQAMetadataBuffersPath::Tiled {
                q_token_tile_size,
                kv_token_tile_size,
            } => {
                build_tiled_map_task_templates(
                    token_indices,
                    cu_tokens,
                    q_token_tile_size,
                    kv_token_tile_size,
                    self.req_slots.len_bytes() / size_of::<u32>(),
                )
            },
        };
        let num_q_token_tiles = if q_token_tiles.is_empty() {
            num_tokens
        } else {
            q_token_tiles.len() / NUM_Q_TOKEN_TILE_FIELDS
        };
        let num_sdpa_map_task_templates = sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS;
        let total_sdpa_map_task_templates = num_sdpa_map_task_templates
            .checked_next_power_of_two()
            .unwrap_or(max_sdpa_map_task_templates)
            .min(max_sdpa_map_task_templates);
        assert!(total_sdpa_map_task_templates >= num_sdpa_map_task_templates);
        sdpa_map_task_templates.resize(
            total_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
            u32::MAX,
        );
        if !q_token_tiles.is_empty() {
            self.q_token_tiles.write_typed(0, &q_token_tiles);
        }
        self.sdpa_map_task_templates.write_typed(0, &sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs.write_typed(0, &cu_sdpa_partial_outputs);

        let replay_shape = GQAReplayShape {
            num_tokens: num_tokens.try_into().expect("GQA batch tokens must fit u32"),
            num_q_token_tiles: num_q_token_tiles.try_into().expect("GQA token tile count must fit u32"),
            total_sdpa_map_task_templates: total_sdpa_map_task_templates
                .try_into()
                .expect("GQA total SDPA map TaskTemplate count must fit u32"),
            reduce_sdpa_partial_outputs: tiled || num_sdpa_map_task_templates > num_tokens,
        };
        replay_shape.validate();
        self.replay_shape.set(Some(replay_shape));
        replay_shape
    }

    pub fn req_slots(&self) -> &Buffer {
        &self.req_slots
    }

    pub fn flat_token_indices(&self) -> &Buffer {
        &self.flat_token_indices
    }

    pub fn q_token_tiles(&self) -> &Buffer {
        &self.q_token_tiles
    }

    pub fn sdpa_map_task_templates(&self) -> &Buffer {
        &self.sdpa_map_task_templates
    }

    pub fn cu_sdpa_partial_outputs(&self) -> &Buffer {
        &self.cu_sdpa_partial_outputs
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
            .get()
            .expect("GQA batch metadata must be updated before recording")
    }
}

fn build_context_parallel_map_task_templates(
    flat_token_indices: &[u32],
    kv_token_tile_size: u32,
    max_sdpa_map_task_templates: usize,
) -> (Vec<u32>, Vec<u32>) {
    assert!(!flat_token_indices.is_empty());
    assert!(max_sdpa_map_task_templates >= flat_token_indices.len());

    let context_lens = flat_token_indices
        .iter()
        .map(|&token_index| token_index.checked_add(1).expect("GQA context length overflow"))
        .collect::<Vec<_>>();
    let num_kv_token_tiles = context_lens
        .iter()
        .map(|&context_len| context_len.div_ceil(kv_token_tile_size) as usize)
        .collect::<Vec<_>>();
    let mut num_sdpa_map_task_templates_by_q_token_tile = vec![1_usize; flat_token_indices.len()];
    let mut num_sdpa_map_task_templates = num_sdpa_map_task_templates_by_q_token_tile.len();
    while num_sdpa_map_task_templates < max_sdpa_map_task_templates {
        let mut split_candidate = None;
        for q_token_tile_index in 0..num_kv_token_tiles.len() {
            if num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] < num_kv_token_tiles[q_token_tile_index]
            {
                let num_kv_token_tiles_per_task_template = num_kv_token_tiles[q_token_tile_index]
                    .div_ceil(num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index]);
                if split_candidate
                    .is_none_or(|(_, best_tile_count)| num_kv_token_tiles_per_task_template > best_tile_count)
                {
                    split_candidate = Some((q_token_tile_index, num_kv_token_tiles_per_task_template));
                }
            }
        }
        let Some((q_token_tile_index, _)) = split_candidate else {
            break;
        };
        num_sdpa_map_task_templates_by_q_token_tile[q_token_tile_index] += 1;
        num_sdpa_map_task_templates += 1;
    }

    let mut sdpa_map_task_templates =
        Vec::with_capacity(num_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS);
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(flat_token_indices.len() + 1);
    cu_sdpa_partial_outputs.push(0);
    for (
        q_token_tile_index,
        ((&context_len, &num_q_tile_kv_token_tiles), &num_sdpa_map_task_templates_for_q_token_tile),
    ) in context_lens
        .iter()
        .zip(&num_kv_token_tiles)
        .zip(&num_sdpa_map_task_templates_by_q_token_tile)
        .enumerate()
    {
        for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_for_q_token_tile {
            let kv_token_tile_begin =
                num_q_tile_kv_token_tiles * sdpa_map_task_template_index / num_sdpa_map_task_templates_for_q_token_tile;
            let kv_token_tile_end = num_q_tile_kv_token_tiles * (sdpa_map_task_template_index + 1)
                / num_sdpa_map_task_templates_for_q_token_tile;
            let kv_token_begin = (kv_token_tile_begin as u64 * kv_token_tile_size as u64)
                .try_into()
                .expect("GQA map TaskTemplate KV-token begin must fit u32");
            let kv_token_end = context_len.min(
                (kv_token_tile_end as u64 * kv_token_tile_size as u64)
                    .try_into()
                    .expect("GQA map TaskTemplate KV-token end must fit u32"),
            );
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_tile_index
                    .try_into()
                    .expect("GQA Q-token tile index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("GQA map TaskTemplate count must fit u32"),
        );
    }
    (sdpa_map_task_templates, cu_sdpa_partial_outputs)
}

struct QTokenTile {
    flat_token_start: u32,
    flat_token_end: u32,
    context_len: u32,
    num_sdpa_map_task_templates_for_q_token_tile: usize,
}

fn build_tiled_map_task_templates(
    token_indices: &[u32],
    cu_tokens: &[u32],
    q_token_tile_size: u32,
    kv_token_tile_size: u32,
    max_sdpa_partial_output_tokens: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    assert!(q_token_tile_size > 0, "GQA Q-token tile size must be positive");
    assert!(kv_token_tile_size > 0, "GQA KV-token tile size must be positive");
    let mut tiles = Vec::new();
    for (req_index, &token_index) in token_indices.iter().enumerate() {
        let flat_req_start = cu_tokens[req_index];
        let flat_req_end = cu_tokens[req_index + 1];
        let mut flat_token_start = flat_req_start;
        while flat_token_start < flat_req_end {
            let flat_token_end = flat_token_start.saturating_add(q_token_tile_size).min(flat_req_end);
            tiles.push(QTokenTile {
                flat_token_start,
                flat_token_end,
                context_len: token_index
                    .checked_add(flat_token_end - flat_req_start)
                    .expect("GQA tiled context length overflow"),
                num_sdpa_map_task_templates_for_q_token_tile: 1,
            });
            flat_token_start = flat_token_end;
        }
    }
    assert!(!tiles.is_empty());

    let mut num_sdpa_partial_output_tokens = tiles
        .iter()
        .map(|tile| (tile.flat_token_end - tile.flat_token_start) as usize)
        .sum::<usize>();
    assert!(num_sdpa_partial_output_tokens <= max_sdpa_partial_output_tokens);
    while num_sdpa_partial_output_tokens < max_sdpa_partial_output_tokens {
        let mut split_candidate = None;
        for (q_token_tile_index, tile) in tiles.iter().enumerate() {
            let num_tile_tokens = (tile.flat_token_end - tile.flat_token_start) as usize;
            if num_sdpa_partial_output_tokens + num_tile_tokens > max_sdpa_partial_output_tokens {
                continue;
            }
            let num_kv_token_tiles = tile.context_len.div_ceil(kv_token_tile_size) as usize;
            if tile.num_sdpa_map_task_templates_for_q_token_tile >= num_kv_token_tiles {
                continue;
            }
            let num_kv_token_tiles_per_task_template =
                num_kv_token_tiles.div_ceil(tile.num_sdpa_map_task_templates_for_q_token_tile);
            if split_candidate.is_none_or(|(_, best_tile_count)| num_kv_token_tiles_per_task_template > best_tile_count)
            {
                split_candidate = Some((q_token_tile_index, num_kv_token_tiles_per_task_template));
            }
        }
        let Some((q_token_tile_index, _)) = split_candidate else {
            break;
        };
        num_sdpa_partial_output_tokens +=
            (tiles[q_token_tile_index].flat_token_end - tiles[q_token_tile_index].flat_token_start) as usize;
        tiles[q_token_tile_index].num_sdpa_map_task_templates_for_q_token_tile += 1;
    }

    let mut q_token_tiles = Vec::with_capacity(tiles.len() * NUM_Q_TOKEN_TILE_FIELDS);
    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(tiles.len() + 1);
    cu_sdpa_partial_outputs.push(0);
    for (q_token_tile_index, tile) in tiles.iter().enumerate() {
        q_token_tiles.extend_from_slice(&[tile.flat_token_start, tile.flat_token_end]);
        let num_kv_token_tiles = tile.context_len.div_ceil(kv_token_tile_size) as usize;
        for sdpa_map_task_template_index in 0..tile.num_sdpa_map_task_templates_for_q_token_tile {
            let kv_token_tile_begin =
                num_kv_token_tiles * sdpa_map_task_template_index / tile.num_sdpa_map_task_templates_for_q_token_tile;
            let kv_token_tile_end = num_kv_token_tiles * (sdpa_map_task_template_index + 1)
                / tile.num_sdpa_map_task_templates_for_q_token_tile;
            let kv_token_begin = (kv_token_tile_begin as u64 * kv_token_tile_size as u64)
                .try_into()
                .expect("GQA tiled map TaskTemplate KV-token begin must fit u32");
            let kv_token_end = tile.context_len.min(
                (kv_token_tile_end as u64 * kv_token_tile_size as u64)
                    .try_into()
                    .expect("GQA tiled map TaskTemplate KV-token end must fit u32"),
            );
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_tile_index
                    .try_into()
                    .expect("GQA Q-token tile index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("GQA tiled map TaskTemplate count must fit u32"),
        );
    }
    (q_token_tiles, sdpa_map_task_templates, cu_sdpa_partial_outputs)
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GQAReplayShape;

    use super::GQAMetadataBuffers;

    #[test]
    fn test_fixed() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 8);
        let replay_shape = metadata.update_context_parallel(&[2, 5], &[7, 20], &[0, 2, 5], 8);

        assert_eq!(metadata.req_slots().read_typed::<u32>(0, 5), vec![2, 2, 5, 5, 5]);
        assert_eq!(
            metadata.flat_token_indices().read_typed::<u32>(0, 5),
            vec![7, 8, 20, 21, 22]
        );
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(0, 24),
            vec![
                0, 0, 8, 1, 0, 9, 2, 0, 8, 2, 8, 21, 3, 0, 8, 3, 8, 22, 4, 0, 8, 4, 8, 23,
            ]
        );
        assert_eq!(
            metadata.cu_sdpa_partial_outputs().read_typed::<u32>(0, 6),
            vec![0, 1, 2, 4, 6, 8]
        );
        assert_eq!(
            replay_shape,
            GQAReplayShape {
                num_tokens: 5,
                num_q_token_tiles: 5,
                total_sdpa_map_task_templates: 8,
                reduce_sdpa_partial_outputs: true,
            }
        );
        assert_eq!(replay_shape, metadata.replay_shape());
    }

    #[test]
    fn test_tiled() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 8);
        let replay_shape = metadata.update_tiled(&[2, 5], &[7, 20], &[0, 2, 5], 8, 4);

        assert_eq!(metadata.q_token_tiles().read_typed::<u32>(0, 4), vec![0, 2, 2, 5]);
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(0, 12),
            vec![0, 0, 9, 1, 0, 12, 1, 12, 23, u32::MAX, u32::MAX, u32::MAX]
        );
        assert_eq!(
            metadata.cu_sdpa_partial_outputs().read_typed::<u32>(0, 3),
            vec![0, 1, 3]
        );
        assert_eq!(
            replay_shape,
            GQAReplayShape {
                num_tokens: 5,
                num_q_token_tiles: 2,
                total_sdpa_map_task_templates: 4,
                reduce_sdpa_partial_outputs: true,
            }
        );
        assert_eq!(replay_shape, metadata.replay_shape());
    }
}
