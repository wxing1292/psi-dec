#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderSyncBlocks {
    block_index: usize,
    kv_page_ids: Vec<Vec<Vec<u32>>>,    // lane -> block index -> page IDs
    state_page_ids: Vec<Vec<Vec<u32>>>, // lane -> block index -> page IDs
}

impl DecoderSyncBlocks {
    pub fn new(block_index: usize, kv_page_ids: Vec<Vec<Vec<u32>>>, state_page_ids: Vec<Vec<Vec<u32>>>) -> Self {
        Self {
            block_index,
            kv_page_ids,
            state_page_ids,
        }
    }

    pub fn block_index(&self) -> usize {
        self.block_index
    }

    pub fn kv_page_ids(&self) -> &[Vec<Vec<u32>>] {
        &self.kv_page_ids
    }

    pub fn state_page_ids(&self) -> &[Vec<Vec<u32>>] {
        &self.state_page_ids
    }

    pub fn is_empty(&self) -> bool {
        self.kv_page_ids
            .iter()
            .all(|lane_kv_page_ids| lane_kv_page_ids.is_empty())
            && self
                .state_page_ids
                .iter()
                .all(|lane_state_page_ids| lane_state_page_ids.is_empty())
    }
}
