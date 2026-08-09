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

/// Converts an unshifted destination state version to its candidate state version.
pub fn to_candidate_state_version(dst_state_version: u32, candidate_state_version_shift: u32) -> u32 {
    dst_state_version
        .checked_sub(candidate_state_version_shift)
        .expect("GDN candidate state version shift must fit u32")
}

/// Converts a candidate state version to its unshifted destination state version.
pub fn from_candidate_state_version(candidate_state_version: u32, candidate_state_version_shift: u32) -> u32 {
    candidate_state_version
        .checked_add(candidate_state_version_shift)
        .expect("GDN destination state version shift must fit u32")
}

/// Per-request GDN destination and candidate state versions carried from prepare through commit.
///
/// The destination range contains the unshifted states that one forward produces. The candidate
/// range contains the shifted states that commit can select. Both ranges are half-open.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateTxn {
    /// Inclusive start of the unshifted states that the forward produces.
    dst_start_state_version: u32,
    /// Exclusive end of the unshifted states that the forward produces.
    dst_end_state_version: u32,
    /// Number of states that commit can select.
    num_candidate_states: u32,
    /// Offset from an unshifted destination state version to its candidate state version.
    candidate_state_version_shift: u32,
}

impl GDNStateTxn {
    /// Creates a transaction from one destination range and its candidate suffix mapping.
    pub fn from_state_versions(
        dst_start_state_version: u32,
        dst_end_state_version: u32,
        num_candidate_states: u32,
        candidate_state_version_shift: u32,
    ) -> Self {
        assert!(
            dst_start_state_version < dst_end_state_version,
            "GDN destination state range must not be empty"
        );
        assert!(num_candidate_states > 0, "GDN candidate state range must not be empty");
        let candidate_end_state_version =
            to_candidate_state_version(dst_end_state_version, candidate_state_version_shift);
        let candidate_start_state_version = candidate_end_state_version
            .checked_sub(num_candidate_states)
            .expect("GDN candidate state range must fit u32");
        assert!(
            candidate_start_state_version
                .checked_add(1)
                .is_some_and(|first_generated_candidate| first_generated_candidate >= dst_start_state_version),
            "GDN candidate range can precede the destination range only by its source state"
        );
        Self {
            dst_start_state_version,
            dst_end_state_version,
            num_candidate_states,
            candidate_state_version_shift,
        }
    }

    /// Creates an unshifted transaction for one token range and its speculative suffix.
    pub fn new(token_index: u32, num_total_tokens: u32, num_spec_tokens: u32) -> Self {
        assert!(num_total_tokens > 0, "GDN state txn requires at least one token");
        assert!(
            num_spec_tokens <= num_total_tokens,
            "GDN state txn spec suffix must fit input rows: spec={} total={}",
            num_spec_tokens,
            num_total_tokens
        );
        let dst_start_state_version = to_state_version(token_index);
        let dst_end_state_version = dst_start_state_version
            .checked_add(num_total_tokens)
            .expect("GDN destination state range must fit u32");
        let num_candidate_states = num_spec_tokens
            .checked_add(1)
            .expect("GDN candidate state count must fit u32");
        Self::from_state_versions(dst_start_state_version, dst_end_state_version, num_candidate_states, 0)
    }

    pub fn dst_start_state_version(self) -> u32 {
        self.dst_start_state_version
    }

    pub fn dst_end_state_version(self) -> u32 {
        self.dst_end_state_version
    }

    pub fn dst_state_versions(self) -> Range<u32> {
        self.dst_start_state_version..self.dst_end_state_version
    }

    pub fn candidate_start_state_version(self) -> u32 {
        self.candidate_end_state_version()
            .checked_sub(self.num_candidate_states)
            .expect("GDN candidate state range must fit u32")
    }

    pub fn candidate_end_state_version(self) -> u32 {
        to_candidate_state_version(self.dst_end_state_version, self.candidate_state_version_shift)
    }

    pub fn candidate_state_versions(self) -> Range<u32> {
        self.candidate_start_state_version()..self.candidate_end_state_version()
    }

    pub fn num_forward_tokens(self) -> u32 {
        self.dst_end_state_version - self.dst_start_state_version
    }

    pub fn num_candidate_states(self) -> u32 {
        self.num_candidate_states
    }

    pub fn candidate_state_version_shift(self) -> u32 {
        self.candidate_state_version_shift
    }

    pub fn contains_candidate_state_version(self, candidate_state_version: u32) -> bool {
        self.candidate_state_versions().contains(&candidate_state_version)
    }

    pub fn contains_dst_state_version(self, dst_state_version: u32) -> bool {
        self.dst_state_versions().contains(&dst_state_version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_version_conversions() {
        assert_eq!(to_state_version(10), 11);
        assert_eq!(from_state_version(11), 10);
        assert_eq!(to_candidate_state_version(14, 2), 12);
        assert_eq!(from_candidate_state_version(12, 2), 14);
    }

    #[test]
    fn test_state_version_ranges_are_half_open() {
        let txn = GDNStateTxn::from_state_versions(11, 15, 3, 1);

        assert_eq!(txn.dst_state_versions(), 11..15);
        assert_eq!(txn.candidate_state_versions(), 11..14);
        assert_eq!(txn.num_forward_tokens(), 4);
        assert_eq!(txn.num_candidate_states(), 3);
        assert!(txn.contains_candidate_state_version(11));
        assert!(txn.contains_candidate_state_version(13));
        assert!(!txn.contains_candidate_state_version(14));
        assert!(txn.contains_dst_state_version(11));
        assert!(txn.contains_dst_state_version(14));
        assert!(!txn.contains_dst_state_version(15));
    }

    #[test]
    fn test_new_uses_an_unshifted_candidate_suffix() {
        let txn = GDNStateTxn::new(10, 4, 2);

        assert_eq!(txn.dst_state_versions(), 11..15);
        assert_eq!(txn.candidate_state_versions(), 12..15);
    }

    #[test]
    #[should_panic(expected = "GDN candidate range can precede the destination range only by its source state")]
    fn test_candidate_range_must_not_precede_the_source_state() {
        let _ = GDNStateTxn::from_state_versions(10, 12, 4, 0);
    }
}
