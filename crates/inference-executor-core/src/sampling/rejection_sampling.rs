#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRejectionSamplingShape {
    pub num_active_reqs: u32,
    pub num_total_reqs: u32,
    pub num_active_draft_distributions: u32,
    pub num_total_draft_distributions: u32,
    pub num_active_target_distributions: u32,
    pub num_total_target_distributions: u32,
    pub top_k: u32,
    pub max_target_k: u32,
    pub max_draft_k: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRejectionSamplingReqParams {
    pub seed: u32,
    pub sample_position: u32,
    pub top_k: u32,
}

impl SparseRejectionSamplingShape {
    pub fn validate(self) {
        assert!(self.num_active_reqs > 0, "sparse rejection sampling requires reqs");
        assert!(self.num_active_reqs <= self.num_total_reqs);
        assert!(self.num_active_draft_distributions <= self.num_total_draft_distributions);
        assert!(self.num_active_target_distributions <= self.num_total_target_distributions);
        assert!(self.top_k > 0, "sparse rejection sampling requires top_k");
        assert!(
            self.max_target_k >= self.top_k,
            "sparse rejection target distribution must cover top_k"
        );
        assert!(
            self.max_draft_k >= self.top_k,
            "sparse rejection draft distribution must cover top_k"
        );
        let expected_num_target_distributions = self
            .num_active_draft_distributions
            .checked_add(self.num_active_reqs)
            .expect("sparse rejection target-distribution count must fit u32");
        assert_eq!(
            self.num_active_target_distributions, expected_num_target_distributions,
            "sparse rejection requires one target distribution per draft plus one final distribution per request"
        );
    }
}
