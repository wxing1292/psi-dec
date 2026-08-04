/// Per-request GDN state versions carried from prepare through commit.
///
/// Prepare uses the forward range to map output rows to state versions. It uses the candidate range
/// to keep every state that commit can select. Commit promotes one selected state to current.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateTxn {
    /// Exclusive lower boundary of the states that the forward generates.
    start_state_version: u32,
    /// Inclusive upper bound of the states that the forward generates.
    end_state_version: u32,
    /// Inclusive lower bound of the states that must remain selectable through commit.
    ///
    /// This version reuses the current slot when it equals `start_state_version`.
    candidate_start_state_version: u32,
    /// Inclusive upper bound of the states that require candidate slots.
    candidate_end_state_version: u32,
}

impl GDNStateTxn {
    /// Creates a transaction with independently defined forward and candidate state ranges.
    pub fn from_state_versions(
        start_state_version: u32,
        end_state_version: u32,
        candidate_start_state_version: u32,
        candidate_end_state_version: u32,
    ) -> Self {
        assert!(
            start_state_version < end_state_version,
            "GDN state txn end must advance its start"
        );
        assert!(
            candidate_start_state_version >= start_state_version,
            "GDN state txn candidate range must not precede its forward range"
        );
        assert!(
            candidate_start_state_version <= candidate_end_state_version,
            "GDN state txn candidate range must not be empty"
        );
        assert!(
            candidate_end_state_version <= end_state_version,
            "GDN state txn candidate range must not follow its forward range"
        );
        Self {
            start_state_version,
            end_state_version,
            candidate_start_state_version,
            candidate_end_state_version,
        }
    }

    /// Creates the common transaction whose candidate range is the speculative suffix and its
    /// fixed-token boundary.
    pub fn new(token_index: u32, num_total_tokens: u32, num_spec_tokens: u32) -> Self {
        assert!(num_total_tokens > 0, "GDN state txn requires at least one token");
        assert!(
            num_spec_tokens <= num_total_tokens,
            "GDN state txn spec suffix must fit input rows: spec={} total={}",
            num_spec_tokens,
            num_total_tokens
        );
        let end_state_version = token_index
            .checked_add(num_total_tokens)
            .expect("GDN state version must fit u32");
        Self::from_state_versions(
            token_index,
            end_state_version,
            end_state_version - num_spec_tokens,
            end_state_version,
        )
    }

    pub fn start_state_version(self) -> u32 {
        self.start_state_version
    }

    pub fn end_state_version(self) -> u32 {
        self.end_state_version
    }

    pub fn candidate_start_state_version(self) -> u32 {
        self.candidate_start_state_version
    }

    pub fn candidate_end_state_version(self) -> u32 {
        self.candidate_end_state_version
    }

    pub fn num_forward_tokens(self) -> u32 {
        self.end_state_version - self.start_state_version
    }

    pub fn num_candidate_states(self) -> u32 {
        self.candidate_end_state_version - self.candidate_start_state_version + 1
    }

    pub fn contains_candidate_state_version(self, state_version: u32) -> bool {
        state_version >= self.candidate_start_state_version && state_version <= self.candidate_end_state_version
    }

    pub fn contains_state_version(self, state_version: u32) -> bool {
        state_version >= self.start_state_version && state_version <= self.end_state_version
    }
}

#[cfg(test)]
mod tests {
    use super::GDNStateTxn;

    #[test]
    fn test_state_version_contract() {
        let txn = GDNStateTxn::from_state_versions(10, 14, 11, 13);

        assert_eq!(txn.start_state_version(), 10);
        assert_eq!(txn.end_state_version(), 14);
        assert_eq!(txn.num_forward_tokens(), 4);
        assert_eq!(txn.num_candidate_states(), 3);
        assert!(txn.contains_candidate_state_version(11));
        assert!(txn.contains_candidate_state_version(12));
        assert!(txn.contains_candidate_state_version(13));
        assert!(!txn.contains_candidate_state_version(14));
        assert!(txn.contains_state_version(10));
        assert!(txn.contains_state_version(14));
    }

    #[test]
    fn test_new_uses_exclusive_token_end_boundaries() {
        let txn = GDNStateTxn::new(10, 4, 2);

        assert_eq!(txn.start_state_version(), 10);
        assert_eq!(txn.end_state_version(), 14);
        assert_eq!(txn.candidate_start_state_version(), 12);
        assert_eq!(txn.candidate_end_state_version(), 14);
    }

    #[test]
    #[should_panic(expected = "GDN state txn candidate range must not follow its forward range")]
    fn test_candidate_range_must_not_follow_forward_range() {
        let _ = GDNStateTxn::from_state_versions(10, 12, 11, 13);
    }
}
