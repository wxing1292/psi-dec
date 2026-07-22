/// Per-request GDN state versions carried from batch preparation through commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateTxn {
    pub token_index: u32,
    pub num_total_tokens: u32,
    pub num_spec_tokens: u32,
}

impl GDNStateTxn {
    pub fn new(token_index: u32, num_total_tokens: u32, num_spec_tokens: u32) -> Self {
        assert!(num_total_tokens > 0, "GDN state txn requires at least one token");
        assert!(
            num_spec_tokens <= num_total_tokens,
            "GDN state txn spec suffix must fit input rows: spec={} total={}",
            num_spec_tokens,
            num_total_tokens
        );
        token_index
            .checked_add(num_total_tokens)
            .expect("GDN state version must fit u32");
        Self {
            token_index,
            num_total_tokens,
            num_spec_tokens,
        }
    }

    pub fn last_candidate_state_version(self) -> u32 {
        self.token_index
            .checked_add(self.num_total_tokens)
            .expect("GDN state version must fit u32")
    }

    pub fn first_candidate_state_version(self) -> u32 {
        self.last_candidate_state_version() - self.num_spec_tokens
    }

    pub fn contains_candidate_state_version(self, state_version: u32) -> bool {
        state_version >= self.first_candidate_state_version() && state_version <= self.last_candidate_state_version()
    }
}
