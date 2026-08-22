//! GPU metadata for history and block attention partials.

use std::cell::Cell;

use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BlockSpecMetadata;
use inference_executor_core::attn::GQAReplayShape;

use crate::attn::block_spec::capacity::BlockSpecGQACapacity;

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;
const NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS: usize = 2;

struct BlockSpecMetadataUpload {
    visible_kv_token_ranges: Vec<u32>,
    q_token_ranges: Vec<u32>,
    sdpa_map_task_templates: Vec<u32>,
    cu_sdpa_partial_outputs: Vec<u32>,
    replay_shape: GQAReplayShape,
}

pub struct BlockSpecGQAMetadataBuffers {
    capacity: BlockSpecGQACapacity,
    req_slots: Buffer,
    flat_token_indices: Buffer,
    visible_kv_token_ranges: Buffer,
    q_token_ranges: Buffer,
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
    sdpa_execution: ExecutionVariant,
}

impl BlockSpecGQAMetadataBuffers {
    pub fn new(device: &Device, capacity: BlockSpecGQACapacity, sdpa_execution: ExecutionVariant) -> Self {
        assert_eq!(
            sdpa_execution.map.thread_block.max_q_tokens as usize, capacity.max_q_tokens,
            "block-spec history attention execution must match metadata capacity"
        );
        let task_template_values = capacity
            .max_sdpa_map_task_templates
            .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
            .expect("block-spec GQA TaskTemplate metadata capacity must fit usize");
        Self {
            capacity,
            req_slots: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            visible_kv_token_ranges: Buffer::new_zeroed_elements(
                device,
                capacity
                    .block
                    .max_tokens
                    .checked_mul(NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS)
                    .expect("block-spec GQA visible K/V-token-range capacity must fit usize"),
                Dtype::Uint32,
            ),
            q_token_ranges: Buffer::new_zeroed_elements(
                device,
                capacity
                    .max_q_token_ranges
                    .checked_mul(2)
                    .expect("block-spec GQA Q-token-range capacity must fit usize"),
                Dtype::Uint32,
            ),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(device, task_template_values, Dtype::Uint32),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                capacity
                    .max_q_token_ranges
                    .checked_add(1)
                    .expect("block-spec GQA cumulative partial-output capacity must fit usize"),
                Dtype::Uint32,
            ),
            replay_shape: Cell::new(None),
            sdpa_execution,
        }
    }

    pub fn update(&self, block: &BlockSpecMetadata) -> GQAReplayShape {
        assert!(
            block.num_requests() <= self.capacity.block.max_requests,
            "block-spec GQA request count exceeds capacity"
        );
        assert_eq!(
            block.block_size(),
            self.capacity.block.block_size,
            "block-spec GQA block size must match the static capacity"
        );
        assert!(
            block.num_tokens() <= self.capacity.block.max_tokens,
            "block-spec GQA token count exceeds capacity"
        );
        self.req_slots.write_typed(0, block.req_slots());
        self.flat_token_indices.write_typed(0, block.flat_token_indices());

        let metadata = build_metal_metadata(block, self.sdpa_execution, self.capacity.max_sdpa_map_task_templates);
        let num_q_token_ranges = metadata.replay_shape.num_q_token_tiles as usize;
        assert!(
            num_q_token_ranges <= self.capacity.max_q_token_ranges,
            "block-spec GQA Q-token-range count exceeds capacity"
        );
        self.visible_kv_token_ranges
            .write_typed(0, &metadata.visible_kv_token_ranges);
        self.q_token_ranges.write_typed(0, &metadata.q_token_ranges);
        self.sdpa_map_task_templates
            .write_typed(0, &metadata.sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs
            .write_typed(0, &metadata.cu_sdpa_partial_outputs);
        self.replay_shape.set(Some(metadata.replay_shape));
        metadata.replay_shape
    }

    pub fn req_slots(&self) -> &Buffer {
        &self.req_slots
    }

    pub fn flat_token_indices(&self) -> &Buffer {
        &self.flat_token_indices
    }

    pub fn visible_kv_token_ranges(&self) -> &Buffer {
        &self.visible_kv_token_ranges
    }

    pub fn q_token_ranges(&self) -> &Buffer {
        &self.q_token_ranges
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
            .expect("block-spec GQA metadata must be updated before recording")
    }

    pub fn sdpa_execution(&self) -> ExecutionVariant {
        self.sdpa_execution
    }
}

fn build_metal_metadata(
    block: &BlockSpecMetadata,
    variant: ExecutionVariant,
    max_sdpa_map_task_templates: usize,
) -> BlockSpecMetadataUpload {
    let map = variant.map.thread_block;
    let kv_tokens_per_iteration = map.kv_tokens_per_iteration;
    debug_assert!(
        kv_tokens_per_iteration > 0,
        "block-spec GQA KV-token iteration size must be positive"
    );
    debug_assert!(map.max_q_tokens > 0, "block-spec GQA Q-token range must contain tokens");
    let num_tokens = block.num_tokens();
    let visible_kv_token_ranges = block
        .history_token_begins()
        .iter()
        .zip(block.history_token_ends())
        .flat_map(|(&begin, &end)| [begin, end])
        .collect::<Vec<_>>();
    debug_assert_eq!(
        visible_kv_token_ranges.len(),
        num_tokens * NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS
    );
    let max_q_tokens = map.max_q_tokens as usize;
    let mut q_token_range_indices = Vec::new();
    for request_index in 0..block.num_requests() {
        let request_begin = request_index * block.block_size();
        let request_end = request_begin + block.block_size();
        let mut range_begin = request_begin;
        while range_begin < request_end {
            let range_end = (range_begin + max_q_tokens).min(request_end);
            q_token_range_indices.push(range_begin..range_end);
            range_begin = range_end;
        }
    }
    debug_assert!(
        max_sdpa_map_task_templates
            >= q_token_range_indices
                .len()
                .checked_mul(2)
                .expect("block-spec GQA minimum TaskTemplate count must fit usize")
    );
    debug_assert!(
        max_sdpa_map_task_templates.is_power_of_two(),
        "block-spec GQA TaskTemplate capacity must be a power of two"
    );

    let history_token_ranges = q_token_range_indices
        .iter()
        .map(|range| {
            let begin = block.history_token_begins()[range.clone()]
                .iter()
                .copied()
                .min()
                .expect("block-spec Q-token range must not be empty");
            let end = block.history_token_ends()[range.clone()]
                .iter()
                .copied()
                .max()
                .expect("block-spec Q-token range must not be empty");
            debug_assert!(begin < end, "block-spec GQA history range must not be empty");
            begin..end
        })
        .collect::<Vec<_>>();
    let num_kv_iterations = history_token_ranges
        .iter()
        .map(|range| (range.end - range.start).div_ceil(kv_tokens_per_iteration) as usize)
        .collect::<Vec<_>>();
    let max_history_task_templates = max_sdpa_map_task_templates - q_token_range_indices.len();
    let mut num_history_task_templates = vec![1usize; q_token_range_indices.len()];
    let mut total_history_task_templates = q_token_range_indices.len();
    while total_history_task_templates < max_history_task_templates {
        let candidate = num_kv_iterations
            .iter()
            .zip(&num_history_task_templates)
            .enumerate()
            .filter(|&(_, (&iterations, &task_templates))| task_templates < iterations)
            .max_by_key(|&(_, (&iterations, &task_templates))| iterations.div_ceil(task_templates))
            .map(|(token_index, _)| token_index);
        let Some(token_index) = candidate else {
            break;
        };
        num_history_task_templates[token_index] += 1;
        total_history_task_templates += 1;
    }

    let mut q_token_ranges = Vec::with_capacity(q_token_range_indices.len() * 2);
    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(q_token_range_indices.len() + 1);
    cu_sdpa_partial_outputs.push(0);
    for (q_token_range_index, q_token_range) in q_token_range_indices.iter().enumerate() {
        q_token_ranges.extend_from_slice(&[
            q_token_range
                .start
                .try_into()
                .expect("block-spec GQA Q-token-range begin must fit u32"),
            q_token_range
                .end
                .try_into()
                .expect("block-spec GQA Q-token-range end must fit u32"),
        ]);
        let history_range = &history_token_ranges[q_token_range_index];
        let num_iterations = num_kv_iterations[q_token_range_index];
        let num_tasks = num_history_task_templates[q_token_range_index];
        for task_index in 0..num_tasks {
            let iteration_begin = num_iterations * task_index / num_tasks;
            let iteration_end = num_iterations * (task_index + 1) / num_tasks;
            let kv_token_begin = history_range
                .start
                .checked_add(
                    (iteration_begin as u64 * kv_tokens_per_iteration as u64)
                        .try_into()
                        .expect("block-spec GQA history iteration begin must fit u32"),
                )
                .expect("block-spec GQA history token begin must fit u32");
            let kv_token_end = history_range.end.min(
                history_range
                    .start
                    .checked_add(
                        (iteration_end as u64 * kv_tokens_per_iteration as u64)
                            .try_into()
                            .expect("block-spec GQA history iteration end must fit u32"),
                    )
                    .expect("block-spec GQA history token end must fit u32"),
            );
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_range_index
                    .try_into()
                    .expect("block-spec GQA Q-token-range index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }

        sdpa_map_task_templates.extend_from_slice(&[u32::MAX; NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS]);
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("block-spec GQA cumulative partial-output count must fit u32"),
        );
    }

    let num_task_templates = sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS;
    let num_total_sdpa_map_task_templates = num_task_templates
        .checked_next_power_of_two()
        .expect("block-spec GQA replay TaskTemplate count must fit usize");
    assert!(
        num_total_sdpa_map_task_templates <= max_sdpa_map_task_templates,
        "block-spec GQA replay TaskTemplate count exceeds capacity"
    );
    sdpa_map_task_templates.resize(
        num_total_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
        u32::MAX,
    );
    let num_tokens = num_tokens.try_into().expect("block-spec GQA token count must fit u32");
    let replay_shape = GQAReplayShape {
        num_tokens,
        num_total_tokens: num_tokens,
        num_q_token_tiles: q_token_range_indices
            .len()
            .try_into()
            .expect("block-spec GQA Q-token-range count must fit u32"),
        num_total_q_token_tiles: q_token_range_indices
            .len()
            .try_into()
            .expect("block-spec GQA Q-token-range count must fit u32"),
        num_sdpa_map_task_templates: num_task_templates
            .try_into()
            .expect("block-spec GQA active TaskTemplate count must fit u32"),
        num_total_sdpa_map_task_templates: num_total_sdpa_map_task_templates
            .try_into()
            .expect("block-spec GQA TaskTemplate count must fit u32"),
        reduce_sdpa_partial_outputs: true,
    };
    replay_shape.validate();
    BlockSpecMetadataUpload {
        visible_kv_token_ranges,
        q_token_ranges,
        sdpa_map_task_templates,
        cu_sdpa_partial_outputs,
        replay_shape,
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::BlockSpecCapacity;

    use super::*;

    #[test]
    fn test_metadata_materializes_tiled_history_and_block_partials() {
        let device = Device::system_default();
        let capacity = BlockSpecGQACapacity::new(BlockSpecCapacity::new(2, 3), 8);
        let variant = backend_sdpa::ExecutionVariant::tiled_q(config(), 8, 16, 2);
        let buffers = BlockSpecGQAMetadataBuffers::new(&device, capacity, variant);
        let block = BlockSpecMetadata::new(
            &[3, 7],
            &[11, 12, 13, 20, 21, 22],
            &[0..11, 0..11, 0..11, 3..20, 3..20, 3..20],
            3,
        );

        let shape = buffers.update(&block);

        assert_eq!(shape.num_tokens, 6);
        assert_eq!(shape.num_q_token_tiles, 2);
        assert_eq!(shape.num_total_sdpa_map_task_templates, 8);
        assert_eq!(
            buffers.visible_kv_token_ranges().read_typed::<u32>(0, 12),
            [0, 11, 0, 11, 0, 11, 3, 20, 3, 20, 3, 20]
        );
        assert_eq!(buffers.q_token_ranges().read_typed::<u32>(0, 4), [0, 3, 3, 6]);
        assert_eq!(buffers.cu_sdpa_partial_outputs().read_typed::<u32>(0, 3), [0, 2, 5]);
        assert_eq!(variant, buffers.sdpa_execution());
    }

    #[test]
    fn test_long_history_is_partitioned_without_growing_static_scratch() {
        let anchor = 262_144;
        let block = BlockSpecMetadata::new(&[1], &(anchor..anchor + 6).collect::<Vec<_>>(), &vec![0..anchor; 6], 6);
        let metadata = build_metal_metadata(&block, backend_sdpa::ExecutionVariant::tiled_q(config(), 8, 16, 2), 16);

        let block_task = metadata.cu_sdpa_partial_outputs[1] as usize - 1;
        let history = metadata.sdpa_map_task_templates[..block_task * 3].as_chunks::<3>().0;
        assert_eq!(history.first().expect("history task")[1], 0);
        assert_eq!(history.last().expect("history task")[2], anchor);
        assert!(history.windows(2).all(|pair| pair[0][2] == pair[1][1]));
        assert_eq!(metadata.cu_sdpa_partial_outputs, [0, 16]);
        assert_eq!(metadata.replay_shape.num_total_sdpa_map_task_templates, 16);
    }

    #[test]
    fn test_q_tile_uses_history_union_and_preserves_per_query_masks() {
        let device = Device::system_default();
        let capacity = BlockSpecGQACapacity::new(BlockSpecCapacity::new(1, 3), 8);
        let variant = backend_sdpa::ExecutionVariant::tiled_q(config(), 8, 16, 2);
        let buffers = BlockSpecGQAMetadataBuffers::new(&device, capacity, variant);
        let block = BlockSpecMetadata::new(&[3], &[20, 21, 22], &[0..20, 1..20, 2..20], 3);

        let shape = buffers.update(&block);

        assert_eq!(shape.num_q_token_tiles, 1);
        assert_eq!(
            buffers.visible_kv_token_ranges().read_typed::<u32>(0, 6),
            [0, 20, 1, 20, 2, 20]
        );
        let first_task = buffers.sdpa_map_task_templates().read_typed::<u32>(0, 3);
        assert_eq!(first_task[1], 0);
        assert!(first_task[2] <= 20);
    }

    fn config() -> backend_sdpa::Config {
        backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 128,
            tokens_per_page: 8,
        }
    }
}
