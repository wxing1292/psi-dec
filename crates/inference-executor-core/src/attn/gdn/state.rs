use std::ops::Range;

/// Converts a token index to the state version after that token.
///
/// The result is also the expected index of the next input token.
pub fn to_state_version(token_index: u32) -> u32 {
    token_index.checked_add(1).expect("GDN state version must fit u32")
}

/// Converts a state version to the index of the token that produced it.
pub fn from_state_version(state_version: u32) -> u32 {
    state_version
        .checked_sub(1)
        .expect("GDN state version must follow one token")
}

/// Per-request candidate state versions that commit can select, as a half-open range.
///
/// State version `v` follows Main tokens `[0, v)`. The next Main input starts at `v`.
/// Forward row positions and cache-boundary writes come from the batch, not this transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateTxn {
    /// Inclusive start of the states that commit can select.
    dst_start_state_version: u32,
    /// Exclusive end of the states that commit can select.
    dst_end_state_version: u32,
}

impl GDNStateTxn {
    /// Creates a transaction from a selectable state range.
    pub fn from_state_versions(dst_start_state_version: u32, dst_end_state_version: u32) -> Self {
        assert!(
            dst_start_state_version < dst_end_state_version,
            "GDN destination state range must not be empty"
        );
        Self {
            dst_start_state_version,
            dst_end_state_version,
        }
    }

    /// Creates a transaction for one token range and its speculative suffix.
    pub fn new(token_index: u32, num_total_tokens: u32, num_spec_tokens: u32) -> Self {
        assert!(num_total_tokens > 0, "GDN state txn requires at least one token");
        assert!(
            num_spec_tokens <= num_total_tokens,
            "GDN state txn spec suffix must fit input rows: spec={} total={}",
            num_spec_tokens,
            num_total_tokens
        );
        let final_state_version = token_index
            .checked_add(num_total_tokens)
            .expect("GDN final state version must fit u32");
        Self::from_state_versions(
            final_state_version - num_spec_tokens,
            to_state_version(final_state_version),
        )
    }

    pub fn dst_start_state_version(self) -> u32 {
        self.dst_start_state_version
    }

    pub fn dst_end_state_version(self) -> u32 {
        self.dst_end_state_version
    }

    pub fn dst_state_versions(self) -> Range<u32> {
        self.dst_start_state_version()..self.dst_end_state_version()
    }

    pub fn num_candidate_states(self) -> u32 {
        self.dst_end_state_version - self.dst_start_state_version
    }

    pub fn contains_dst_state_version(self, candidate_state_version: u32) -> bool {
        self.dst_state_versions().contains(&candidate_state_version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_version_conversions() {
        assert_eq!(to_state_version(10), 11);
        assert_eq!(from_state_version(11), 10);
    }

    #[test]
    fn test_state_version_ranges_are_half_open() {
        let txn = GDNStateTxn::from_state_versions(12, 15);
        assert_eq!(txn.dst_state_versions(), 12..15);
        assert_eq!(txn.num_candidate_states(), 3);
        assert!(txn.contains_dst_state_version(12));
        assert!(txn.contains_dst_state_version(14));
        assert!(!txn.contains_dst_state_version(15));
    }

    #[test]
    fn test_new_uses_a_candidate_suffix() {
        let txn = GDNStateTxn::new(10, 4, 2);

        assert_eq!(txn.dst_state_versions(), 12..15);
    }

    #[test]
    fn test_candidate_range_can_include_the_source_state() {
        let txn = GDNStateTxn::new(10, 2, 2);
        assert_eq!(txn.dst_state_versions(), 10..13);
    }

    #[test]
    #[should_panic(expected = "GDN destination state range must not be empty")]
    fn test_candidate_range_must_not_be_empty() {
        let _ = GDNStateTxn::from_state_versions(13, 13);
    }

    #[test]
    fn test_candidate_range_depends_on_actual_input_not_mtp_lanes() {
        for num_fixed_tokens in 1..=4 {
            for num_spec_tokens in 0..=3 {
                let txn = GDNStateTxn::new(3, num_fixed_tokens + num_spec_tokens, num_spec_tokens);
                assert_eq!(
                    txn.dst_state_versions(),
                    3 + num_fixed_tokens..4 + num_fixed_tokens + num_spec_tokens
                );
                for num_accepted_tokens in 0..=num_spec_tokens {
                    assert!(txn.contains_dst_state_version(3 + num_fixed_tokens + num_accepted_tokens));
                }
            }
        }
    }
}
