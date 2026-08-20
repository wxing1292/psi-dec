use std::cell::Cell;

use inference_backend_metal::components::GQASDPAExecutionSpecialization;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::DSparkBlockMetadata;
use inference_executor_core::attn::GQAReplayShape;

use crate::attn::dspark::capacity::DSparkGQACapacity;

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;

struct DSparkGQAMetadata {
    sdpa_map_task_templates: Vec<u32>,
    cu_sdpa_partial_outputs: Vec<u32>,
    block_sdpa_map_task_template_indices: Vec<u32>,
    replay_shape: GQAReplayShape,
}

pub struct DSparkGQAMetadataBuffers {
    capacity: DSparkGQACapacity,
    req_slots: Buffer,
    flat_token_indices: Buffer,
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    block_sdpa_map_task_template_indices: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
    sdpa_execution: Cell<Option<GQASDPAExecutionSpecialization>>,
}

impl DSparkGQAMetadataBuffers {
    pub fn new(device: &Device, capacity: DSparkGQACapacity) -> Self {
        let task_template_values = capacity
            .max_sdpa_partial_outputs
            .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
            .expect("DSpark GQA TaskTemplate metadata capacity must fit usize");
        Self {
            capacity,
            req_slots: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(device, task_template_values, Dtype::Uint32),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                capacity
                    .block
                    .max_tokens
                    .checked_add(1)
                    .expect("DSpark GQA cumulative partial-output capacity must fit usize"),
                Dtype::Uint32,
            ),
            block_sdpa_map_task_template_indices: Buffer::new_zeroed_elements(
                device,
                capacity.block.max_tokens,
                Dtype::Uint32,
            ),
            replay_shape: Cell::new(None),
            sdpa_execution: Cell::new(None),
        }
    }

    pub fn update(&self, block: &DSparkBlockMetadata, execution: GQASDPAExecutionSpecialization) -> GQAReplayShape {
        let map = execution.map.thread_block;
        assert_eq!(
            map.max_q_tokens, 1,
            "DSpark history attention requires a single-Q SDPA specialization"
        );
        assert!(
            block.num_requests() <= self.capacity.block.max_requests,
            "DSpark GQA request count exceeds capacity"
        );
        assert_eq!(
            block.block_size(),
            self.capacity.block.block_size,
            "DSpark GQA block size must match the static capacity"
        );
        assert!(
            block.num_tokens() <= self.capacity.block.max_tokens,
            "DSpark GQA token count exceeds capacity"
        );
        self.req_slots.write_typed(0, block.req_slots());
        self.flat_token_indices.write_typed(0, block.flat_token_indices());

        let metadata = build_metal_metadata(
            block,
            map.kv_tokens_per_iteration,
            self.capacity.max_sdpa_partial_outputs,
        );
        self.sdpa_map_task_templates
            .write_typed(0, &metadata.sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs
            .write_typed(0, &metadata.cu_sdpa_partial_outputs);
        self.block_sdpa_map_task_template_indices
            .write_typed(0, &metadata.block_sdpa_map_task_template_indices);
        self.replay_shape.set(Some(metadata.replay_shape));
        self.sdpa_execution.set(Some(execution));
        metadata.replay_shape
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

    pub fn block_sdpa_map_task_template_indices(&self) -> &Buffer {
        &self.block_sdpa_map_task_template_indices
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
            .get()
            .expect("DSpark GQA metadata must be updated before recording")
    }

    pub fn sdpa_execution(&self) -> GQASDPAExecutionSpecialization {
        self.sdpa_execution
            .get()
            .expect("DSpark GQA metadata must be updated before recording")
    }
}

fn build_metal_metadata(
    block: &DSparkBlockMetadata,
    kv_tokens_per_iteration: u32,
    max_sdpa_map_task_templates: usize,
) -> DSparkGQAMetadata {
    assert!(
        kv_tokens_per_iteration > 0,
        "DSpark GQA KV-token iteration size must be positive"
    );
    let num_tokens = block.num_tokens();
    assert!(
        max_sdpa_map_task_templates
            >= num_tokens
                .checked_mul(2)
                .expect("DSpark GQA minimum TaskTemplate count must fit usize")
    );
    assert!(
        max_sdpa_map_task_templates.is_power_of_two(),
        "DSpark GQA TaskTemplate capacity must be a power of two"
    );

    let num_kv_iterations = block
        .history_token_begins()
        .iter()
        .zip(block.history_token_ends())
        .map(|(&begin, &end)| {
            assert!(begin < end, "DSpark GQA history range must not be empty");
            (end - begin).div_ceil(kv_tokens_per_iteration) as usize
        })
        .collect::<Vec<_>>();
    let max_history_task_templates = max_sdpa_map_task_templates - num_tokens;
    let mut num_history_task_templates = vec![1usize; num_tokens];
    let mut total_history_task_templates = num_tokens;
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

    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(num_tokens + 1);
    let mut block_sdpa_map_task_template_indices = Vec::with_capacity(num_tokens);
    cu_sdpa_partial_outputs.push(0);
    for q_token_index in 0..num_tokens {
        let history_begin = block.history_token_begins()[q_token_index];
        let history_end = block.history_token_ends()[q_token_index];
        let num_iterations = num_kv_iterations[q_token_index];
        let num_tasks = num_history_task_templates[q_token_index];
        for task_index in 0..num_tasks {
            let iteration_begin = num_iterations * task_index / num_tasks;
            let iteration_end = num_iterations * (task_index + 1) / num_tasks;
            let kv_token_begin = history_begin
                .checked_add(
                    (iteration_begin as u64 * kv_tokens_per_iteration as u64)
                        .try_into()
                        .expect("DSpark GQA history iteration begin must fit u32"),
                )
                .expect("DSpark GQA history token begin must fit u32");
            let kv_token_end = history_end.min(
                history_begin
                    .checked_add(
                        (iteration_end as u64 * kv_tokens_per_iteration as u64)
                            .try_into()
                            .expect("DSpark GQA history iteration end must fit u32"),
                    )
                    .expect("DSpark GQA history token end must fit u32"),
            );
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_index.try_into().expect("DSpark GQA Q-token index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }

        block_sdpa_map_task_template_indices.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("DSpark GQA block TaskTemplate index must fit u32"),
        );
        sdpa_map_task_templates.extend_from_slice(&[u32::MAX; NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS]);
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("DSpark GQA cumulative partial-output count must fit u32"),
        );
    }

    let num_task_templates = sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS;
    let num_total_sdpa_map_task_templates = num_task_templates
        .checked_next_power_of_two()
        .expect("DSpark GQA replay TaskTemplate count must fit usize");
    assert!(
        num_total_sdpa_map_task_templates <= max_sdpa_map_task_templates,
        "DSpark GQA replay TaskTemplate count exceeds capacity"
    );
    sdpa_map_task_templates.resize(
        num_total_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
        u32::MAX,
    );
    let num_tokens = num_tokens.try_into().expect("DSpark GQA token count must fit u32");
    let replay_shape = GQAReplayShape {
        num_tokens,
        num_total_tokens: num_tokens,
        num_q_token_tiles: num_tokens,
        num_total_q_token_tiles: num_tokens,
        num_sdpa_map_task_templates: num_task_templates
            .try_into()
            .expect("DSpark GQA active TaskTemplate count must fit u32"),
        num_total_sdpa_map_task_templates: num_total_sdpa_map_task_templates
            .try_into()
            .expect("DSpark GQA TaskTemplate count must fit u32"),
        reduce_sdpa_partial_outputs: true,
    };
    replay_shape.validate();
    DSparkGQAMetadata {
        sdpa_map_task_templates,
        cu_sdpa_partial_outputs,
        block_sdpa_map_task_template_indices,
        replay_shape,
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::GQASDPAConfig;
    use inference_backend_metal::components::GQASDPAExecutionSpecialization;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::DSparkBlockCapacity;

    use super::*;

    #[test]
    fn test_metadata_retains_backend_selected_sdpa_execution() {
        let device = Device::system_default();
        let capacity = DSparkGQACapacity::new(DSparkBlockCapacity::new(2, 3));
        let buffers = DSparkGQAMetadataBuffers::new(&device, capacity);
        let block = DSparkBlockMetadata::new(&[3, 7], &[11, 20], 3);
        let config = GQASDPAConfig {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 128,
            tokens_per_page: 8,
        };
        let execution = GQASDPAExecutionSpecialization::single_q(config, 4, 32, 2);

        let shape = buffers.update(&block, execution);

        assert_eq!(shape.num_tokens, 6);
        assert_eq!(execution, buffers.sdpa_execution());
    }

    #[test]
    #[should_panic(expected = "single-Q SDPA specialization")]
    fn test_metadata_rejects_tiled_q_partial_abi() {
        let device = Device::system_default();
        let capacity = DSparkGQACapacity::new(DSparkBlockCapacity::new(1, 3));
        let buffers = DSparkGQAMetadataBuffers::new(&device, capacity);
        let block = DSparkBlockMetadata::new(&[3], &[11], 3);

        let config = GQASDPAConfig {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 128,
            tokens_per_page: 8,
        };
        buffers.update(&block, GQASDPAExecutionSpecialization::tiled_q(config, 8, 16, 2));
    }

    #[test]
    fn test_metal_metadata_reserves_one_block_partial_per_query() {
        let block = DSparkBlockMetadata::new(&[3], &[11], 6);
        let metadata = build_metal_metadata(&block, 4, 32);

        assert_eq!(metadata.cu_sdpa_partial_outputs.len(), 7);
        assert_eq!(metadata.block_sdpa_map_task_template_indices.len(), 6);
        for q_token_index in 0..6 {
            let begin = metadata.cu_sdpa_partial_outputs[q_token_index] as usize;
            let end = metadata.cu_sdpa_partial_outputs[q_token_index + 1] as usize;
            let block_task = metadata.block_sdpa_map_task_template_indices[q_token_index] as usize;
            assert_eq!(block_task, end - 1);
            assert_eq!(
                metadata.sdpa_map_task_templates[block_task * 3..block_task * 3 + 3],
                [u32::MAX; 3]
            );
            for history_task in begin..block_task {
                let fields = &metadata.sdpa_map_task_templates[history_task * 3..history_task * 3 + 3];
                assert_eq!(fields[0], q_token_index as u32);
                assert!(fields[1] < fields[2]);
                assert!(fields[2] <= 11);
            }
        }
        assert_eq!(metadata.replay_shape.num_total_sdpa_map_task_templates, 32);
    }

    #[test]
    fn test_long_history_is_partitioned_without_growing_static_scratch() {
        let anchor = 262_144;
        let block = DSparkBlockMetadata::new(&[1], &[anchor], 6);
        let metadata = build_metal_metadata(&block, 256, 16);

        for q_token_index in 0..6 {
            let begin = metadata.cu_sdpa_partial_outputs[q_token_index] as usize;
            let block_task = metadata.block_sdpa_map_task_template_indices[q_token_index] as usize;
            let history = metadata.sdpa_map_task_templates[begin * 3..block_task * 3]
                .as_chunks::<3>()
                .0;
            assert_eq!(history.first().expect("history task")[1], 0);
            assert_eq!(history.last().expect("history task")[2], anchor);
            assert!(history.windows(2).all(|pair| pair[0][2] == pair[1][1]));
        }
        assert_eq!(metadata.replay_shape.num_total_sdpa_map_task_templates, 16);
    }
}
