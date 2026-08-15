use inference_backend_metal::components::GQASplitKV;
use inference_backend_metal::components::GQASplitKVConfig;
use inference_backend_metal::components::GQASplitKVVariant;

use self::backend::GQAMetalConfig;
use self::batch_metadata::GQAReplayBucketPolicy;

pub mod batch_metadata;
pub mod backend;
pub mod request_page_table;
pub mod scratch;
pub mod ungated_backend;
pub mod ungated_scratch;

const D256_PAGE8_MIN_COMPLETE_PLAN_WORK: u64 = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SplitKVSelectionWork {
    num_kv_splits: u64,
    scheduled_map_work: u64,
    active_map_work: u64,
    map_blocks_per_kv_head: u64,
    map_simdgroup_waves_per_kv_head: u64,
    active_partial_states: u64,
    padded_replay_work: u64,
}

impl SplitKVSelectionWork {
    fn bookkeeping_work(self) -> u64 {
        self.map_blocks_per_kv_head
            .checked_add(self.map_simdgroup_waves_per_kv_head)
            .and_then(|work| work.checked_add(self.active_partial_states))
            .and_then(|work| work.checked_add(self.padded_replay_work))
            .expect("GQA SplitKV selection bookkeeping work must fit u64")
    }
}

#[derive(Clone, Copy)]
struct SplitKVSelectionTile {
    active_q_tokens: u32,
    num_kv_token_tiles: u32,
    num_kv_splits: u32,
}

fn gqa_split_kv_config(
    config: GQAMetalConfig,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> GQASplitKVConfig {
    GQASplitKVConfig {
        io_dtype: config.io_dtype,
        page_bytes: config.page_bytes,
        num_q_heads: num_q_heads.try_into().expect("GQA Q-head count must fit u32"),
        num_kv_heads: num_kv_heads.try_into().expect("GQA KV-head count must fit u32"),
        head_dim: head_dim.try_into().expect("GQA head_dim must fit u32"),
    }
}

fn uses_d256_page8_selector(config: GQASplitKVConfig) -> bool {
    config.head_dim == 256 && config.num_tokens_per_page() == 8
}

fn select_split_kv_variant(
    split_kv: GQASplitKV,
    use_d256_page8_selector: bool,
    max_tokens: usize,
    replay_policy: Option<&GQAReplayBucketPolicy>,
    token_indices: &[u32],
    cu_tokens: &[u32],
) -> GQASplitKVVariant {
    assert_eq!(cu_tokens.len(), token_indices.len() + 1);
    assert_eq!(cu_tokens.first().copied(), Some(0));
    let num_tokens = cu_tokens.last().copied().unwrap_or_default();
    assert!(num_tokens > 0);
    let num_q_token_tiles = cu_tokens
        .windows(2)
        .map(|cu| {
            assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
            (cu[1] - cu[0]).div_ceil(split_kv.tiled_q_token_tile_size())
        })
        .sum();
    let candidate = split_kv.select(num_tokens, num_q_token_tiles);
    if !use_d256_page8_selector || !matches!(candidate, GQASplitKVVariant::TiledQ { .. }) {
        return candidate;
    }

    let single_q = split_kv.select(num_tokens, num_tokens);
    if prefer_d256_page8_tiled_q(single_q, candidate, max_tokens, replay_policy, token_indices, cu_tokens) {
        candidate
    } else {
        single_q
    }
}

fn prefer_d256_page8_tiled_q(
    single_q: GQASplitKVVariant,
    tiled_q: GQASplitKVVariant,
    max_tokens: usize,
    replay_policy: Option<&GQAReplayBucketPolicy>,
    token_indices: &[u32],
    cu_tokens: &[u32],
) -> bool {
    let single_q_work = single_q_selection_work(single_q, max_tokens, replay_policy, token_indices, cu_tokens);
    let GQASplitKVVariant::SingleQ { q_head_tile_size, .. } = single_q else {
        unreachable!()
    };
    let tiled_q_work = tiled_q_selection_work(
        tiled_q,
        q_head_tile_size,
        max_tokens,
        replay_policy,
        token_indices,
        cu_tokens,
    );
    let logical_query_key_pairs = logical_query_key_pairs(token_indices, cu_tokens);
    let complete_plan_work = logical_query_key_pairs
        .checked_add(single_q_work.bookkeeping_work().min(tiled_q_work.bookkeeping_work()))
        .expect("GQA SplitKV complete plan work must fit u64");
    let tiled_q_has_sufficient_map_utilization = tiled_q_work
        .active_map_work
        .checked_mul(2)
        .is_some_and(|active_work| active_work >= tiled_q_work.scheduled_map_work);

    tiled_q_has_sufficient_map_utilization && complete_plan_work >= D256_PAGE8_MIN_COMPLETE_PLAN_WORK
}

fn single_q_selection_work(
    variant: GQASplitKVVariant,
    max_tokens: usize,
    replay_policy: Option<&GQAReplayBucketPolicy>,
    token_indices: &[u32],
    cu_tokens: &[u32],
) -> SplitKVSelectionWork {
    let GQASplitKVVariant::SingleQ {
        kv_token_tile_size,
        num_threads_per_threadblock,
        q_head_tile_size,
    } = variant
    else {
        panic!("GQA SplitKV SingleQ selection work requires a SingleQ variant");
    };
    let tiles = flat_context_lens(token_indices, cu_tokens)
        .into_iter()
        .map(|context_len| {
            SplitKVSelectionTile {
                active_q_tokens: 1,
                num_kv_token_tiles: context_len.div_ceil(kv_token_tile_size),
                num_kv_splits: 1,
            }
        })
        .collect::<Vec<_>>();
    selection_work(
        tiles,
        kv_token_tile_size,
        1,
        1,
        num_threads_per_threadblock / 32,
        max_tokens,
        replay_policy,
    )
}

fn tiled_q_selection_work(
    variant: GQASplitKVVariant,
    q_heads_per_kv_head: u32,
    max_tokens: usize,
    replay_policy: Option<&GQAReplayBucketPolicy>,
    token_indices: &[u32],
    cu_tokens: &[u32],
) -> SplitKVSelectionWork {
    let GQASplitKVVariant::TiledQ {
        q_token_tile_size,
        kv_token_tile_size,
        q_head_tile_size,
    } = variant
    else {
        panic!("GQA SplitKV TiledQ selection work requires a TiledQ variant");
    };
    let mut tiles = Vec::new();
    for (request_index, &token_index) in token_indices.iter().enumerate() {
        let flat_request_start = cu_tokens[request_index];
        let flat_request_end = cu_tokens[request_index + 1];
        let mut flat_token_start = flat_request_start;
        while flat_token_start < flat_request_end {
            let active_q_tokens = (flat_request_end - flat_token_start).min(q_token_tile_size);
            let flat_token_end = flat_token_start + active_q_tokens;
            let context_len = token_index
                .checked_add(flat_token_end - flat_request_start)
                .expect("GQA request context length must fit u32");
            tiles.push(SplitKVSelectionTile {
                active_q_tokens,
                num_kv_token_tiles: context_len.div_ceil(kv_token_tile_size),
                num_kv_splits: 1,
            });
            flat_token_start = flat_token_end;
        }
    }
    selection_work(
        tiles,
        kv_token_tile_size,
        q_token_tile_size,
        q_heads_per_kv_head.div_ceil(q_head_tile_size),
        q_token_tile_size / 8 * q_head_tile_size,
        max_tokens,
        replay_policy,
    )
}

fn selection_work(
    mut tiles: Vec<SplitKVSelectionTile>,
    kv_token_tile_size: u32,
    scheduled_q_tokens: u32,
    map_blocks_per_split: u32,
    simdgroups_per_map_block: u32,
    max_tokens: usize,
    replay_policy: Option<&GQAReplayBucketPolicy>,
) -> SplitKVSelectionWork {
    assert!(!tiles.is_empty());
    let mut active_partial_states = tiles.iter().map(|tile| tile.active_q_tokens as usize).sum::<usize>();
    assert!(active_partial_states <= max_tokens);
    while active_partial_states < max_tokens {
        let split_candidate = tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| {
                active_partial_states + tile.active_q_tokens as usize <= max_tokens
                    && tile.num_kv_splits < tile.num_kv_token_tiles
            })
            .map(|(index, tile)| (index, tile.num_kv_token_tiles.div_ceil(tile.num_kv_splits)))
            .max_by_key(|&(index, tiles_per_split)| (tiles_per_split, std::cmp::Reverse(index)));
        let Some((index, _)) = split_candidate else {
            break;
        };
        active_partial_states += tiles[index].active_q_tokens as usize;
        tiles[index].num_kv_splits += 1;
    }

    let num_kv_splits = tiles.iter().map(|tile| u64::from(tile.num_kv_splits)).sum::<u64>();
    let scheduled_map_work = tiles
        .iter()
        .map(|tile| u64::from(tile.num_kv_token_tiles) * u64::from(kv_token_tile_size * scheduled_q_tokens))
        .sum();
    let active_map_work = tiles
        .iter()
        .map(|tile| u64::from(tile.num_kv_token_tiles) * u64::from(kv_token_tile_size * tile.active_q_tokens))
        .sum();
    let map_blocks_per_kv_head = num_kv_splits * u64::from(map_blocks_per_split);
    let map_simdgroup_waves_per_kv_head = map_blocks_per_kv_head * u64::from(simdgroups_per_map_block);
    SplitKVSelectionWork {
        num_kv_splits,
        scheduled_map_work,
        active_map_work,
        map_blocks_per_kv_head,
        map_simdgroup_waves_per_kv_head,
        active_partial_states: active_partial_states as u64,
        padded_replay_work: padded_kv_splits(num_kv_splits, max_tokens, replay_policy) * u64::from(scheduled_q_tokens),
    }
}

fn flat_context_lens(token_indices: &[u32], cu_tokens: &[u32]) -> Vec<u32> {
    let num_tokens = cu_tokens.last().copied().unwrap_or_default() as usize;
    let mut context_lens = Vec::with_capacity(num_tokens);
    for (request_index, &token_index) in token_indices.iter().enumerate() {
        let num_request_tokens = cu_tokens[request_index + 1] - cu_tokens[request_index];
        for token_offset in 0..num_request_tokens {
            context_lens.push(
                token_index
                    .checked_add(token_offset + 1)
                    .expect("GQA request context length must fit u32"),
            );
        }
    }
    context_lens
}

fn logical_query_key_pairs(token_indices: &[u32], cu_tokens: &[u32]) -> u64 {
    flat_context_lens(token_indices, cu_tokens)
        .into_iter()
        .map(u64::from)
        .sum()
}

fn padded_kv_splits(num_kv_splits: u64, max_tokens: usize, replay_policy: Option<&GQAReplayBucketPolicy>) -> u64 {
    if let Some(policy) = replay_policy {
        return u64::from(
            policy.kv_split_capacity(
                num_kv_splits
                    .try_into()
                    .expect("GQA active KV-split count must fit u32"),
            ),
        );
    }
    num_kv_splits
        .checked_next_power_of_two()
        .unwrap_or(max_tokens as u64)
        .min(max_tokens as u64)
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::GQASplitKV;
    use inference_backend_metal::components::GQASplitKVConfig;
    use inference_backend_metal::components::GQASplitKVVariant;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;

    use super::select_split_kv_variant;
    use super::tiled_q_selection_work;
    use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
    use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;

    fn split_kv() -> GQASplitKV {
        GQASplitKV::new(GQASplitKVConfig {
            io_dtype: Dtype::Bfloat16,
            page_bytes: 32 * 1024,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
        })
    }

    fn select(token_indices: &[u32], request_tokens: &[u32]) -> GQASplitKVVariant {
        select_with_policy(token_indices, request_tokens, None)
    }

    fn select_bucketed(token_indices: &[u32], request_tokens: &[u32]) -> GQASplitKVVariant {
        let policy = GQAReplayBucketPolicy::new(128, &[]);
        select_with_policy(token_indices, request_tokens, Some(&policy))
    }

    fn select_with_policy(
        token_indices: &[u32],
        request_tokens: &[u32],
        replay_policy: Option<&GQAReplayBucketPolicy>,
    ) -> GQASplitKVVariant {
        let mut cu_tokens = Vec::with_capacity(request_tokens.len() + 1);
        cu_tokens.push(0u32);
        for &count in request_tokens {
            cu_tokens.push(
                cu_tokens
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(count)
                    .expect("test GQA token count must fit u32"),
            );
        }
        select_split_kv_variant(split_kv(), true, 128, replay_policy, token_indices, &cu_tokens)
    }

    fn materialized_active_partial_states(metadata: &GQAMetadataBuffers) -> u64 {
        let shape = metadata.replay_shape();
        let q_token_tiles = metadata
            .q_token_tiles()
            .read_typed::<u32>(0, shape.num_q_token_tiles as usize * 2);
        let cu_kv_splits = metadata
            .cu_kv_splits()
            .read_typed::<u32>(0, shape.num_q_token_tiles as usize + 1);
        q_token_tiles
            .as_chunks::<2>()
            .0
            .iter()
            .zip(cu_kv_splits.windows(2))
            .map(|(tile, splits)| u64::from(tile[1] - tile[0]) * u64::from(splits[1] - splits[0]))
            .sum()
    }

    #[test]
    fn test_d256_page8_selector_measured_crossovers() {
        for (tokens, context) in [(1, 65536), (2, 65536), (4, 128), (8, 128), (25, 32)] {
            assert!(matches!(
                select(&[context], &[tokens]),
                GQASplitKVVariant::SingleQ { .. }
            ));
            assert!(matches!(
                select_bucketed(&[context], &[tokens]),
                GQASplitKVVariant::SingleQ { .. }
            ));
        }
        for (tokens, context) in [(4, 512), (8, 512), (16, 128), (25, 128), (64, 0)] {
            assert!(matches!(
                select(&[context], &[tokens]),
                GQASplitKVVariant::TiledQ { .. }
            ));
            assert!(matches!(
                select_bucketed(&[context], &[tokens]),
                GQASplitKVVariant::TiledQ { .. }
            ));
        }
    }

    #[test]
    fn test_d256_page8_selector_prices_request_local_tail_tiles() {
        assert!(matches!(
            select(&[65536; 8], &[1; 8]),
            GQASplitKVVariant::SingleQ { .. }
        ));
        assert!(matches!(
            select(&[1024, 65536], &[64, 1]),
            GQASplitKVVariant::SingleQ { .. }
        ));
        assert!(matches!(
            select(&[65536, 1024], &[1, 8]),
            GQASplitKVVariant::SingleQ { .. }
        ));
        assert!(matches!(
            select(&[65536, 65536], &[8, 1]),
            GQASplitKVVariant::TiledQ { .. }
        ));
    }

    #[test]
    fn test_d256_page8_selector_models_current_tiled_q_builder_capacity() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 128);
        let tiled_q = split_kv().select(65, 9);
        let work = tiled_q_selection_work(tiled_q, 6, 128, None, &[1024, 65536], &[0, 64, 65]);
        let shape = metadata.update(&[0, 1], &[1024, 65536], &[0, 64, 65], tiled_q);
        assert_eq!(work.num_kv_splits, 72);
        assert_eq!(work.num_kv_splits, u64::from(shape.num_sdpa_map_task_templates));
        assert_eq!(work.active_partial_states, 128);
        assert_eq!(
            work.active_partial_states,
            materialized_active_partial_states(&metadata)
        );
        assert_eq!(work.padded_replay_work, 1024);
        assert_eq!(
            work.padded_replay_work,
            u64::from(shape.num_total_sdpa_map_task_templates) * 8
        );

        let work = tiled_q_selection_work(tiled_q, 6, 128, None, &[65536; 8], &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(work.num_kv_splits, 128);
        assert_eq!(work.active_partial_states, 128);
        assert_eq!(work.padded_replay_work, 1024);

        let policy = GQAReplayBucketPolicy::new(128, &[]);
        let tiled_q = split_kv().select(25, 4);
        let work = tiled_q_selection_work(tiled_q, 6, 128, Some(&policy), &[65536], &[0, 25]);
        let shape = metadata.update_bucketed(&[0], &[65536], &[0, 25], tiled_q, &policy);
        assert_eq!(work.num_kv_splits, 23);
        assert_eq!(work.num_kv_splits, u64::from(shape.num_sdpa_map_task_templates));
        assert_eq!(work.active_partial_states, 128);
        assert_eq!(
            work.active_partial_states,
            materialized_active_partial_states(&metadata)
        );
        assert_eq!(work.padded_replay_work, 24 * 8);
        assert_eq!(
            work.padded_replay_work,
            u64::from(shape.num_total_sdpa_map_task_templates) * 8
        );
    }
}
