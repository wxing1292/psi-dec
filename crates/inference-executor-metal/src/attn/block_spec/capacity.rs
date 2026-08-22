//! Static capacity for block-spec GQA.

use inference_executor_core::attn::BlockSpecCapacity;

/// Metal resource capacity derived from backend-neutral block-spec block facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockSpecGQACapacity {
    pub block: BlockSpecCapacity,
    pub max_q_tokens: usize,
    pub max_q_token_ranges: usize,
    pub max_sdpa_map_task_templates: usize,
    pub max_sdpa_partial_state_groups: usize,
}

impl BlockSpecGQACapacity {
    pub fn new(block: BlockSpecCapacity, max_q_tokens: u32) -> Self {
        block.validate();
        assert!(max_q_tokens > 0, "block-spec GQA Q-token range must contain tokens");
        let max_q_tokens = max_q_tokens as usize;
        let max_q_token_ranges_per_request = block.block_size.div_ceil(max_q_tokens);
        let max_q_token_ranges = block
            .max_requests
            .checked_mul(max_q_token_ranges_per_request)
            .expect("block-spec GQA Q-token-range capacity must fit usize");
        let min_composite_task_templates = max_q_token_ranges
            .checked_mul(2)
            .expect("block-spec GQA composite task-template capacity must fit usize");
        let min_sdpa_map_task_templates = block.max_tokens.max(min_composite_task_templates);
        let max_sdpa_map_task_templates = min_sdpa_map_task_templates
            .checked_next_power_of_two()
            .expect("block-spec GQA Map task-template capacity must fit usize");
        assert!(
            u32::try_from(max_sdpa_map_task_templates).is_ok(),
            "block-spec GQA Map task-template capacity must fit u32"
        );
        let max_sdpa_partial_state_groups = max_sdpa_map_task_templates
            .checked_mul(max_q_tokens)
            .expect("block-spec GQA partial-state-group capacity must fit usize");
        Self {
            block,
            max_q_tokens,
            max_q_token_ranges,
            max_sdpa_map_task_templates,
            max_sdpa_partial_state_groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_reserves_history_and_block_partials() {
        let capacity = BlockSpecGQACapacity::new(BlockSpecCapacity::new(3, 7), 8);

        assert_eq!(capacity.block.max_tokens, 21);
        assert_eq!(capacity.max_q_token_ranges, 3);
        assert_eq!(capacity.max_sdpa_map_task_templates, 32);
        assert_eq!(capacity.max_sdpa_partial_state_groups, 256);
    }
}
