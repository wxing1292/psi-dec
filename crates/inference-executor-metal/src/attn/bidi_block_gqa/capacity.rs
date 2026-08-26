//! Static capacity for BiDiBlockGQA.

use inference_executor_core::attn::BiDiBlockCapacity;

/// Metal resource capacity derived from backend-neutral BiDiBlockGQA block facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BiDiBlockGQACapacity {
    pub block: BiDiBlockCapacity,
    pub max_q_tokens: usize,
    pub max_q_token_ranges: usize,
    pub max_sdpa_map_task_templates: usize,
    pub max_sdpa_partial_state_groups: usize,
}

impl BiDiBlockGQACapacity {
    pub fn new(block: BiDiBlockCapacity, max_q_tokens: u32) -> Self {
        block.validate();
        assert!(max_q_tokens > 0, "BiDiBlockGQA Q-token range must contain tokens");
        let max_q_tokens = max_q_tokens as usize;
        let max_q_token_ranges_per_request = block.block_size.div_ceil(max_q_tokens);
        let max_q_token_ranges = block
            .max_requests
            .checked_mul(max_q_token_ranges_per_request)
            .expect("BiDiBlockGQA Q-token-range capacity must fit usize");
        let min_composite_task_templates = max_q_token_ranges
            .checked_mul(2)
            .expect("BiDiBlockGQA composite task-template capacity must fit usize");
        let min_sdpa_map_task_templates = block.max_tokens.max(min_composite_task_templates);
        let max_sdpa_map_task_templates = min_sdpa_map_task_templates
            .checked_next_power_of_two()
            .expect("BiDiBlockGQA Map task-template capacity must fit usize");
        assert!(
            u32::try_from(max_sdpa_map_task_templates).is_ok(),
            "BiDiBlockGQA Map task-template capacity must fit u32"
        );
        let max_sdpa_partial_state_groups = max_sdpa_map_task_templates
            .checked_mul(max_q_tokens)
            .expect("BiDiBlockGQA partial-state-group capacity must fit usize");
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
        let capacity = BiDiBlockGQACapacity::new(BiDiBlockCapacity::new(3, 7), 8);

        assert_eq!(capacity.block.max_tokens, 21);
        assert_eq!(capacity.max_q_token_ranges, 3);
        assert_eq!(capacity.max_sdpa_map_task_templates, 32);
        assert_eq!(capacity.max_sdpa_partial_state_groups, 256);
    }
}
