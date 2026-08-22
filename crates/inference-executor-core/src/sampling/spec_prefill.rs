use super::SpecMicrobatch;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecPrefillSelection {
    pub main_row_indices: Vec<u32>,
    pub req_slots: Vec<u32>,
    pub flat_token_indices: Vec<u32>,
}

impl SpecPrefillSelection {
    pub fn main_rows_are_prefix(&self) -> bool {
        self.main_row_indices
            .iter()
            .enumerate()
            .all(|(row, &source_row)| source_row as usize == row)
    }
}

pub fn build_spec_prefill_selection(
    microbatch: &impl SpecMicrobatch,
    accepted_prefix_lengths: &[usize],
) -> SpecPrefillSelection {
    let num_reqs = microbatch.num_reqs();
    assert_eq!(microbatch.req_slots().len(), num_reqs);
    assert_eq!(microbatch.token_indices().len(), num_reqs);
    assert_eq!(microbatch.cu_tokens().len(), num_reqs + 1);
    assert_eq!(microbatch.cu_tokens()[0], 0);
    let num_decode_reqs = (0..num_reqs)
        .filter(|&req_index| microbatch.is_decode_req(req_index))
        .count();
    assert_eq!(
        accepted_prefix_lengths.len(),
        num_decode_reqs,
        "Spec Prefill accepted-prefix lengths must match decode requests"
    );

    let mut main_row_indices = Vec::new();
    let mut req_slots = Vec::new();
    let mut flat_token_indices = Vec::new();
    let mut decode_req_index = 0usize;

    for req_index in 0..num_reqs {
        let row_begin = microbatch.cu_tokens()[req_index] as usize;
        let row_end = microbatch.cu_tokens()[req_index + 1] as usize;
        assert!(
            row_begin <= row_end,
            "Spec Prefill cumulative row offsets must be ordered"
        );
        let committed_rows = if microbatch.is_decode_req(req_index) {
            let accepted = accepted_prefix_lengths[decode_req_index];
            decode_req_index += 1;
            let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                accepted <= num_spec_tokens,
                "Spec Prefill accepted prefix exceeds speculative suffix"
            );
            let request_rows = row_end - row_begin;
            assert!(
                num_spec_tokens <= request_rows,
                "Spec Prefill speculative suffix exceeds the request rows"
            );
            request_rows - num_spec_tokens + accepted
        } else {
            row_end - row_begin
        };
        let token_begin = microbatch.token_indices()[req_index];
        if committed_rows > 0 {
            let last_row_offset = (committed_rows - 1) as u32;
            assert!(
                token_begin <= u32::MAX - last_row_offset,
                "Spec Prefill token indices must fit u32"
            );
        }
        for request_row in 0..committed_rows {
            main_row_indices.push((row_begin + request_row) as u32);
            req_slots.push(microbatch.req_slots()[req_index]);
            flat_token_indices.push(token_begin + request_row as u32);
        }
    }
    debug_assert_eq!(decode_req_index, num_decode_reqs);
    assert!(
        !main_row_indices.is_empty(),
        "Spec Prefill requires committed Main rows"
    );
    SpecPrefillSelection {
        main_row_indices,
        req_slots,
        flat_token_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Batch {
        req_slots: Vec<u32>,
        token_indices: Vec<u32>,
        cu_tokens: Vec<u32>,
        decode: Vec<bool>,
        num_spec_tokens: Vec<u32>,
    }

    impl SpecMicrobatch for Batch {
        fn num_reqs(&self) -> usize {
            self.req_slots.len()
        }

        fn is_decode_req(&self, req_index: usize) -> bool {
            self.decode[req_index]
        }

        fn num_spec_tokens(&self, req_index: usize) -> u32 {
            self.num_spec_tokens[req_index]
        }

        fn req_slots(&self) -> &[u32] {
            &self.req_slots
        }

        fn token_indices(&self) -> &[u32] {
            &self.token_indices
        }

        fn cu_tokens(&self) -> &[u32] {
            &self.cu_tokens
        }

        fn flat_token_ids(&self) -> &[i32] {
            &[]
        }
    }

    #[test]
    fn excludes_rejected_draft_suffixes_and_preserves_request_coordinates() {
        let batch = Batch {
            req_slots: vec![3, 7, 9],
            token_indices: vec![20, 40, 80],
            cu_tokens: vec![0, 3, 11, 19],
            decode: vec![false, true, true],
            num_spec_tokens: vec![0, 7, 7],
        };

        let selection = build_spec_prefill_selection(&batch, &[2, 0]);

        assert_eq!(selection.main_row_indices, [0, 1, 2, 3, 4, 5, 11]);
        assert_eq!(selection.req_slots, [3, 3, 3, 7, 7, 7, 9]);
        assert_eq!(selection.flat_token_indices, [20, 21, 22, 40, 41, 42, 80]);
        assert!(!selection.main_rows_are_prefix());
    }

    #[test]
    fn one_request_committed_prefix_needs_no_compaction() {
        let batch = Batch {
            req_slots: vec![4],
            token_indices: vec![100],
            cu_tokens: vec![0, 8],
            decode: vec![true],
            num_spec_tokens: vec![7],
        };

        let selection = build_spec_prefill_selection(&batch, &[3]);

        assert_eq!(selection.main_row_indices, [0, 1, 2, 3]);
        assert_eq!(selection.flat_token_indices, [100, 101, 102, 103]);
        assert!(selection.main_rows_are_prefix());
    }
}
