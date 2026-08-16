use crate::attn::UngatedGQACore;

#[derive(Clone, Debug, PartialEq)]
pub struct UngatedDSparkGQACore {
    pub attention: UngatedGQACore,
    pub block_size: usize,
}

impl UngatedDSparkGQACore {
    pub fn new(attention: UngatedGQACore, block_size: usize) -> Self {
        let core = Self { attention, block_size };
        core.validate();
        core
    }

    pub fn validate(&self) {
        self.attention.validate();
        assert!(self.block_size > 0, "DSpark GQA block_size must be positive");
        assert!(
            u32::try_from(self.block_size).is_ok(),
            "DSpark GQA block_size must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DSparkBlockCapacity {
    pub max_requests: usize,
    pub block_size: usize,
    pub max_tokens: usize,
}

impl DSparkBlockCapacity {
    pub fn new(max_requests: usize, block_size: usize) -> Self {
        let max_tokens = max_requests
            .checked_mul(block_size)
            .expect("DSpark block token capacity must fit usize");
        let capacity = Self {
            max_requests,
            block_size,
            max_tokens,
        };
        capacity.validate();
        capacity
    }

    pub fn validate(self) {
        assert!(self.max_requests > 0, "DSpark block capacity requires requests");
        assert!(self.block_size > 0, "DSpark block capacity requires block tokens");
        assert_eq!(
            self.max_tokens,
            self.max_requests
                .checked_mul(self.block_size)
                .expect("DSpark block token capacity must fit usize"),
            "DSpark block token capacity must match requests times block size"
        );
        assert!(
            u32::try_from(self.max_tokens).is_ok(),
            "DSpark block token capacity must fit u32"
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DSparkBlockMetadata {
    req_slots: Vec<u32>,
    flat_token_indices: Vec<u32>,
    history_token_begins: Vec<u32>,
    history_token_ends: Vec<u32>,
    num_requests: usize,
    block_size: usize,
}

impl DSparkBlockMetadata {
    pub fn new(req_slots: &[u32], anchor_positions: &[u32], block_size: usize) -> Self {
        assert!(!req_slots.is_empty(), "DSpark block metadata requires requests");
        assert_eq!(
            req_slots.len(),
            anchor_positions.len(),
            "DSpark block metadata requires one anchor position per request"
        );
        assert!(block_size > 0, "DSpark block metadata requires block tokens");
        assert!(
            u32::try_from(block_size).is_ok(),
            "DSpark block metadata block_size must fit u32"
        );
        assert!(
            req_slots
                .iter()
                .enumerate()
                .all(|(index, req_slot)| !req_slots[..index].contains(req_slot)),
            "DSpark block metadata requires unique request slots"
        );
        assert!(
            anchor_positions.iter().all(|&position| position > 0),
            "DSpark proposal anchors must follow at least one Main token"
        );

        let num_tokens = req_slots
            .len()
            .checked_mul(block_size)
            .expect("DSpark block token count must fit usize");
        let mut req_slots_by_token = Vec::with_capacity(num_tokens);
        let mut flat_token_indices = Vec::with_capacity(num_tokens);
        let mut history_token_begins = Vec::with_capacity(num_tokens);
        let mut history_token_ends = Vec::with_capacity(num_tokens);
        for (&req_slot, &anchor_position) in req_slots.iter().zip(anchor_positions) {
            for block_offset in 0..block_size {
                req_slots_by_token.push(req_slot);
                flat_token_indices.push(
                    anchor_position
                        .checked_add(block_offset as u32)
                        .expect("DSpark block token position must fit u32"),
                );
                history_token_begins.push(0);
                history_token_ends.push(anchor_position);
            }
        }

        Self {
            req_slots: req_slots_by_token,
            flat_token_indices,
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
        let capacity = DSparkBlockCapacity::new(3, 7);

        assert_eq!(capacity.max_requests, 3);
        assert_eq!(capacity.block_size, 7);
        assert_eq!(capacity.max_tokens, 21);
    }

    #[test]
    fn test_block_metadata_uses_anchor_as_first_query_row() {
        let metadata = DSparkBlockMetadata::new(&[2, 9], &[11, 20], 3);

        assert_eq!(metadata.req_slots(), [2, 2, 2, 9, 9, 9]);
        assert_eq!(metadata.flat_token_indices(), [11, 12, 13, 20, 21, 22]);
        assert_eq!(metadata.history_token_begins(), [0; 6]);
        assert_eq!(metadata.history_token_ends(), [11, 11, 11, 20, 20, 20]);
        assert_eq!(metadata.num_requests(), 2);
        assert_eq!(metadata.num_tokens(), 6);
        assert_eq!(metadata.block_size(), 3);
    }

    #[test]
    #[should_panic(expected = "unique request slots")]
    fn test_block_metadata_rejects_duplicate_request_slots() {
        let _ = DSparkBlockMetadata::new(&[2, 2], &[11, 20], 3);
    }
}
