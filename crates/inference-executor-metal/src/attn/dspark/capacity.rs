use inference_executor_core::attn::DSparkBlockCapacity;

/// Metal resource capacity derived from backend-neutral DSpark block facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkGQACapacity {
    pub block: DSparkBlockCapacity,
    pub max_sdpa_partial_outputs: usize,
}

impl DSparkGQACapacity {
    pub fn new(block: DSparkBlockCapacity) -> Self {
        block.validate();
        let min_sdpa_partial_outputs = block
            .max_tokens
            .checked_mul(2)
            .expect("DSpark GQA partial-output capacity must fit usize");
        let max_sdpa_partial_outputs = min_sdpa_partial_outputs
            .checked_next_power_of_two()
            .expect("DSpark GQA partial-output capacity must fit usize");
        assert!(
            u32::try_from(max_sdpa_partial_outputs).is_ok(),
            "DSpark GQA partial-output capacity must fit u32"
        );
        Self {
            block,
            max_sdpa_partial_outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_reserves_history_and_block_partials() {
        let capacity = DSparkGQACapacity::new(DSparkBlockCapacity::new(3, 7));

        assert_eq!(capacity.block.max_tokens, 21);
        assert_eq!(capacity.max_sdpa_partial_outputs, 64);
    }
}
