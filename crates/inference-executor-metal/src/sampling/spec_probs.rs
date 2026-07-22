use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_runtime_core::runtime::RawRequestSlot;

#[derive(Default)]
pub struct SpecProbsStore {
    max_num_spec_tokens: usize,
    max_requests: usize,
    max_k: usize,
    #[cfg(debug_assertions)]
    expected_draft_token_ids: Vec<Option<u32>>,
    draft_token_ids: Option<Buffer>,
    draft_probs: Option<Buffer>,
    target_token_ids: Option<Buffer>,
    target_probs: Option<Buffer>,
}

impl SpecProbsStore {
    pub fn new(device: &Device, max_num_spec_tokens: usize, max_requests: usize, max_k: usize) -> Self {
        if max_num_spec_tokens == 0 {
            return Self::default();
        }
        assert!(max_requests > 0, "draft store requires requests");
        assert!(max_k > 0, "draft store requires slots");
        let num_draft_distributions = max_requests
            .checked_mul(max_num_spec_tokens)
            .expect("draft-distribution count overflow");
        let num_target_distributions = max_requests
            .checked_mul(
                max_num_spec_tokens
                    .checked_add(1)
                    .expect("target distributions per request overflow"),
            )
            .expect("target-distribution count overflow");
        u32::try_from(num_draft_distributions).expect("draft-distribution count must fit u32");
        u32::try_from(num_target_distributions).expect("target-distribution count must fit u32");
        let draft_slots = num_draft_distributions
            .checked_mul(max_k)
            .expect("draft-distribution slot count overflow");
        let target_slots = num_target_distributions
            .checked_mul(max_k)
            .expect("target-distribution slot count overflow");
        Self {
            max_num_spec_tokens,
            max_requests,
            max_k,
            #[cfg(debug_assertions)]
            expected_draft_token_ids: vec![None; num_draft_distributions],
            draft_token_ids: Some(Buffer::new_zeroed_elements(device, draft_slots, Dtype::Int32)),
            draft_probs: Some(Buffer::new_zeroed_elements(device, draft_slots, Dtype::Float32)),
            target_token_ids: Some(Buffer::new_zeroed_elements(device, target_slots, Dtype::Int32)),
            target_probs: Some(Buffer::new_zeroed_elements(device, target_slots, Dtype::Float32)),
        }
    }

    pub fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        if !self.is_enabled() {
            return;
        }
        #[cfg(debug_assertions)]
        {
            for &req_slot in request_slots {
                let req_slot = req_slot as usize;
                assert!(
                    req_slot < self.max_requests,
                    "speculative-probability request slot exceeds capacity"
                );
                for spec_token_index in 0..self.max_num_spec_tokens {
                    let distribution_index = self.distribution_index(
                        req_slot
                            .try_into()
                            .expect("speculative-probability request slot must fit u32"),
                        spec_token_index,
                    );
                    self.expected_draft_token_ids[distribution_index] = None;
                }
            }
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = request_slots;
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.max_num_spec_tokens > 0
    }

    pub fn max_k(&self) -> usize {
        self.max_k
    }

    pub fn num_draft_distributions(&self) -> u32 {
        self.max_requests
            .checked_mul(self.max_num_spec_tokens)
            .and_then(|count| u32::try_from(count).ok())
            .expect("draft-distribution count must fit u32")
    }

    pub fn num_target_distributions(&self) -> u32 {
        self.max_requests
            .checked_mul(
                self.max_num_spec_tokens
                    .checked_add(1)
                    .expect("target distributions per request overflow"),
            )
            .and_then(|count| u32::try_from(count).ok())
            .expect("target-distribution count must fit u32")
    }

    pub fn draft_distribution_index(&self, req_slot: u32, spec_token_index: usize) -> u32 {
        self.distribution_index(req_slot, spec_token_index)
            .try_into()
            .expect("draft-distribution index must fit u32")
    }

    pub fn set_expected_draft_token(&mut self, req_slot: u32, spec_token_index: usize, token_id: u32) {
        #[cfg(debug_assertions)]
        {
            let distribution_index = self.distribution_index(req_slot, spec_token_index);
            self.expected_draft_token_ids[distribution_index] = Some(token_id);
        }
        #[cfg(not(debug_assertions))]
        let _ = (req_slot, spec_token_index, token_id);
    }

    pub fn assert_expected_draft_token(&self, req_slot: u32, spec_token_index: usize, token_id: u32) {
        #[cfg(debug_assertions)]
        {
            let distribution_index = self.distribution_index(req_slot, spec_token_index);
            assert_eq!(
                Some(token_id),
                self.expected_draft_token_ids[distribution_index],
                "draft token mismatch"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = (req_slot, spec_token_index, token_id);
    }

    pub fn draft_token_ids(&self) -> &Buffer {
        self.draft_token_ids
            .as_ref()
            .expect("draft-distribution token IDs missing")
    }

    pub fn draft_probs(&self) -> &Buffer {
        self.draft_probs
            .as_ref()
            .expect("draft-distribution probabilities missing")
    }

    pub fn target_token_ids(&self) -> &Buffer {
        self.target_token_ids
            .as_ref()
            .expect("target-distribution token IDs missing")
    }

    pub fn target_probs(&self) -> &Buffer {
        self.target_probs
            .as_ref()
            .expect("target-distribution probabilities missing")
    }

    fn distribution_index(&self, req_slot: u32, spec_token_index: usize) -> usize {
        assert!(self.is_enabled(), "draft store is disabled");
        request_major_distribution_index(req_slot, spec_token_index, self.max_requests, self.max_num_spec_tokens)
    }
}

fn request_major_distribution_index(
    req_slot: u32,
    spec_token_index: usize,
    max_requests: usize,
    max_num_spec_tokens: usize,
) -> usize {
    let req_slot = req_slot as usize;
    assert!(req_slot < max_requests, "draft request slot exceeds capacity");
    assert!(
        spec_token_index < max_num_spec_tokens,
        "draft token index exceeds capacity"
    );
    req_slot
        .checked_mul(max_num_spec_tokens)
        .and_then(|base| base.checked_add(spec_token_index))
        .expect("draft-distribution index overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_indices_are_request_major_for_any_speculator() {
        assert_eq!(request_major_distribution_index(0, 0, 3, 2), 0);
        assert_eq!(request_major_distribution_index(0, 1, 3, 2), 1);
        assert_eq!(request_major_distribution_index(2, 0, 3, 2), 4);
        assert_eq!(request_major_distribution_index(2, 1, 3, 2), 5);
    }
}
