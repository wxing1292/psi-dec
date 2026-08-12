use std::ops::Range;

use crate::compute::BatchDeviceRequest;
use crate::compute::BatchDeviceResponse;
use crate::runtime::RawPageID;
use crate::runtime::RawRequestSlot;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorHibernationPlan {
    All,
    Selected {
        request_slot_ranges: Vec<Range<RawRequestSlot>>,
        page_id_ranges: Vec<Range<RawPageID>>,
    },
}

impl ExecutorHibernationPlan {
    pub fn selected(request_slot_ranges: Vec<Range<RawRequestSlot>>, page_id_ranges: Vec<Range<RawPageID>>) -> Self {
        assert_canonical_ranges(&request_slot_ranges, "executor hibernation request slot");
        assert_canonical_ranges(&page_id_ranges, "executor hibernation page ID");
        Self::Selected {
            request_slot_ranges,
            page_id_ranges,
        }
    }

    pub fn assert_valid(&self) {
        if let Self::Selected {
            request_slot_ranges,
            page_id_ranges,
        } = self
        {
            assert_canonical_ranges(request_slot_ranges, "executor hibernation request slot");
            assert_canonical_ranges(page_id_ranges, "executor hibernation page ID");
        }
    }
}

fn assert_canonical_ranges(ranges: &[Range<u32>], name: &str) {
    assert!(
        ranges.iter().all(|range| range.start < range.end) && ranges.windows(2).all(|pair| pair[0].end < pair[1].start),
        "{name} ranges must be nonempty, sorted, disjoint, and nonadjacent"
    );
}

pub enum ReplayableModelExecutorRequest<BatchRequest = BatchDeviceRequest> {
    Batch(BatchRequest),
    Start(ExecutorHibernationPlan),
    Stop(ExecutorHibernationPlan),
}

pub enum ReplayableModelExecutorResponse<BatchResponse = BatchDeviceResponse> {
    Batch(BatchResponse),
    Started,
    Stopped,
}

#[cfg(test)]
mod tests {
    use super::ExecutorHibernationPlan;

    #[test]
    fn test_executor_hibernation_plan_accepts_canonical_ranges() {
        assert_eq!(
            ExecutorHibernationPlan::selected(vec![1..2, 4..5], vec![2..4, 8..9]),
            ExecutorHibernationPlan::Selected {
                request_slot_ranges: vec![1..2, 4..5],
                page_id_ranges: vec![2..4, 8..9],
            }
        );
    }

    #[test]
    #[should_panic(
        expected = "executor hibernation page ID ranges must be nonempty, sorted, disjoint, and nonadjacent"
    )]
    fn test_executor_hibernation_plan_rejects_adjacent_ranges() {
        let _ = ExecutorHibernationPlan::selected(Vec::new(), vec![2..4, 4..5]);
    }

    #[test]
    #[should_panic(
        expected = "executor hibernation request slot ranges must be nonempty, sorted, disjoint, and nonadjacent"
    )]
    fn test_executor_hibernation_plan_rejects_empty_range() {
        let request_slot_ranges = std::iter::once(2..2).collect();
        let _ = ExecutorHibernationPlan::selected(request_slot_ranges, Vec::new());
    }
}
