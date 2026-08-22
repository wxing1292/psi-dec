use std::ops::Range;

use crate::attn::UngatedGQACore;

#[derive(Clone, Debug, PartialEq)]
pub struct BlockSpecGQACore {
    pub attention: UngatedGQACore,
    pub block_size: usize,
}

impl BlockSpecGQACore {
    pub fn new(attention: UngatedGQACore, block_size: usize) -> Self {
        let core = Self { attention, block_size };
        core.validate();
        core
    }

    pub fn validate(&self) {
        self.attention.validate();
        assert!(self.block_size > 0, "block-spec GQA block_size must be positive");
        assert!(
            u32::try_from(self.block_size).is_ok(),
            "block-spec GQA block_size must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockSpecCapacity {
    pub max_requests: usize,
    pub block_size: usize,
    pub max_tokens: usize,
}

impl BlockSpecCapacity {
    pub fn new(max_requests: usize, block_size: usize) -> Self {
        let max_tokens = max_requests
            .checked_mul(block_size)
            .expect("block-spec token capacity must fit usize");
        let capacity = Self {
            max_requests,
            block_size,
            max_tokens,
        };
        capacity.validate();
        capacity
    }

    pub fn validate(self) {
        assert!(self.max_requests > 0, "block-spec capacity requires requests");
        assert!(self.block_size > 0, "block-spec capacity requires block tokens");
        assert_eq!(
            self.max_tokens,
            self.max_requests
                .checked_mul(self.block_size)
                .expect("block-spec token capacity must fit usize"),
            "block-spec token capacity must match requests times block size"
        );
        assert!(
            u32::try_from(self.max_tokens).is_ok(),
            "block-spec token capacity must fit u32"
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSpecMetadata {
    req_slots: Vec<u32>,
    flat_token_indices: Vec<u32>,
    history_token_begins: Vec<u32>,
    history_token_ends: Vec<u32>,
    num_requests: usize,
    block_size: usize,
}

impl BlockSpecMetadata {
    pub fn new(
        req_slots: &[u32],
        flat_query_token_indices: &[u32],
        visible_history_token_ranges: &[Range<u32>],
        block_size: usize,
    ) -> Self {
        assert!(!req_slots.is_empty(), "block-spec metadata requires requests");
        assert!(block_size > 0, "block-spec metadata requires block tokens");
        let num_tokens = req_slots
            .len()
            .checked_mul(block_size)
            .expect("block-spec token count must fit usize");
        assert_eq!(
            flat_query_token_indices.len(),
            num_tokens,
            "block-spec metadata requires one query-token index per block row"
        );
        assert_eq!(
            visible_history_token_ranges.len(),
            num_tokens,
            "block-spec metadata requires one visible history token range per block row"
        );
        assert!(
            u32::try_from(block_size).is_ok(),
            "block-spec metadata block_size must fit u32"
        );
        assert!(
            req_slots
                .iter()
                .enumerate()
                .all(|(index, req_slot)| !req_slots[..index].contains(req_slot)),
            "block-spec metadata requires unique request slots"
        );
        assert!(
            visible_history_token_ranges.iter().all(|range| range.start < range.end),
            "block-spec visible history token ranges must be nonempty half-open ranges"
        );
        let mut req_slots_by_token = Vec::with_capacity(num_tokens);
        let mut history_token_begins = Vec::with_capacity(num_tokens);
        let mut history_token_ends = Vec::with_capacity(num_tokens);
        for (request_index, &req_slot) in req_slots.iter().enumerate() {
            let request_begin = request_index * block_size;
            let request_end = request_begin + block_size;
            for visible_history_token_range in &visible_history_token_ranges[request_begin..request_end] {
                req_slots_by_token.push(req_slot);
                history_token_begins.push(visible_history_token_range.start);
                history_token_ends.push(visible_history_token_range.end);
            }
        }

        Self {
            req_slots: req_slots_by_token,
            flat_token_indices: flat_query_token_indices.to_vec(),
            history_token_begins,
            history_token_ends,
            num_requests: req_slots.len(),
            block_size,
        }
    }

    pub fn req_slots(&self) -> &[u32] {
        &self.req_slots
    }

    pub fn flat_token_indices(&self) -> &[u32] {
        &self.flat_token_indices
    }

    pub fn history_token_begins(&self) -> &[u32] {
        &self.history_token_begins
    }

    pub fn history_token_ends(&self) -> &[u32] {
        &self.history_token_ends
    }

    pub fn num_requests(&self) -> usize {
        self.num_requests
    }

    pub fn num_tokens(&self) -> usize {
        self.req_slots.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_capacity_contains_only_batch_geometry() {
        let capacity = BlockSpecCapacity::new(3, 7);

        assert_eq!(capacity.max_requests, 3);
        assert_eq!(capacity.block_size, 7);
        assert_eq!(capacity.max_tokens, 21);
    }

    #[test]
    fn test_block_metadata_uses_history_end_as_first_query_row() {
        let metadata = BlockSpecMetadata::new(
            &[2, 9],
            &[11, 12, 13, 20, 21, 22],
            &[0..11, 0..11, 0..11, 4..20, 4..20, 4..20],
            3,
        );

        assert_eq!(metadata.req_slots(), [2, 2, 2, 9, 9, 9]);
        assert_eq!(metadata.flat_token_indices(), [11, 12, 13, 20, 21, 22]);
        assert_eq!(metadata.history_token_begins(), [0, 0, 0, 4, 4, 4]);
        assert_eq!(metadata.history_token_ends(), [11, 11, 11, 20, 20, 20]);
        assert_eq!(metadata.num_requests(), 2);
        assert_eq!(metadata.num_tokens(), 6);
        assert_eq!(metadata.block_size(), 3);
    }

    #[test]
    #[should_panic(expected = "unique request slots")]
    fn test_block_metadata_rejects_duplicate_request_slots() {
        let _ = BlockSpecMetadata::new(&[2, 2], &[11, 12, 13, 20, 21, 22], &vec![0..11; 6], 3);
    }

    #[test]
    #[should_panic(expected = "nonempty half-open ranges")]
    fn test_block_metadata_rejects_empty_history_range() {
        let _ = BlockSpecMetadata::new(&[2], &[11, 12, 13], &[11..11, 0..11, 0..11], 3);
    }

    #[test]
    fn test_block_metadata_preserves_per_query_history_ranges() {
        let metadata = BlockSpecMetadata::new(&[2], &[11, 12, 13], &[0..11, 1..11, 2..11], 3);

        assert_eq!(metadata.history_token_begins(), [0, 1, 2]);
        assert_eq!(metadata.history_token_ends(), [11, 11, 11]);
    }
}
