#[derive(Debug, Eq, PartialEq)]
pub struct DeviceResourcePlacement {
    arena_offset_bytes: u64,
    arena_len_bytes: u64,

    /// Each tuple is `(token_index, resource_index, num_resource_tokens)`.
    ///
    /// `token_index` is the absolute index in the initial request token sequence.
    /// `resource_index` is the logical index in the resource embedding sequence.
    /// `num_resource_tokens` is the number of consecutive tokens in both sequences.
    placements: Vec<(usize, usize, usize)>,
}

impl DeviceResourcePlacement {
    pub fn new(arena_offset_bytes: u64, arena_len_bytes: u64, placements: Vec<(usize, usize, usize)>) -> Self {
        Self {
            arena_offset_bytes,
            arena_len_bytes,
            placements,
        }
    }

    pub const fn arena_offset_bytes(&self) -> u64 {
        self.arena_offset_bytes
    }

    pub const fn arena_len_bytes(&self) -> u64 {
        self.arena_len_bytes
    }

    pub fn placements(&self) -> &[(usize, usize, usize)] {
        &self.placements
    }
}
