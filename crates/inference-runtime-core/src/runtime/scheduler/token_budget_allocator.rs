use std::cmp::min;
use std::collections::BinaryHeap;

use ahash::AHashMap;
use ordered_float::NotNan;

use crate::runtime::RawRequestID;

#[derive(Clone, Copy, Debug)]
pub struct ReqTokenInventory<'a> {
    req_id: RawRequestID,
    num_ready_tokens: usize,
    num_queued_tokens: usize,
    num_spec_tokens: usize,
    spec_confidences: &'a [NotNan<f32>],
    max_partial_token_consumption: usize,
}

impl<'a> ReqTokenInventory<'a> {
    pub fn new<const L: usize>(
        req_id: RawRequestID,
        num_ready_tokens: usize,
        num_queued_tokens: usize,
        num_spec_tokens: usize,
        spec_confidences: &'a [NotNan<f32>],
    ) -> Self {
        debug_assert!(L > 0, "request token inventory requires at least one cache lane");
        debug_assert!(
            L == 1 || num_spec_tokens < L,
            "speculative token count must fit the configured cache lanes"
        );
        // MTP binds each speculative position to an additional cache lane.
        // DSpark keeps one cache lane and stores its proposals separately.
        if 1 < L {
            if num_ready_tokens == 0 {
                debug_assert!(0 < num_queued_tokens);
            } else {
                debug_assert!(L - 1 <= num_queued_tokens);
            }
        }
        debug_assert_eq!(
            num_spec_tokens,
            spec_confidences.len(),
            "speculative token and confidence counts must match"
        );
        debug_assert!(
            spec_confidences
                .iter()
                .all(|confidence| (0.0..=1.0).contains(&confidence.into_inner())),
            "speculative confidence must be in [0, 1]"
        );

        let num_validated_tokens = num_ready_tokens + num_queued_tokens;
        Self {
            req_id,
            num_ready_tokens,
            num_queued_tokens,
            num_spec_tokens,
            spec_confidences,
            max_partial_token_consumption: num_validated_tokens.saturating_sub(L - 1),
        }
    }

    pub fn req_id(&self) -> RawRequestID {
        self.req_id
    }

    /// Evaluate an absolute request token budget without changing request state.
    pub fn token_consumption(&self, token_budget: usize) -> usize {
        let num_validated_tokens = self.max_validated_token_consumption();
        if token_budget == 0 || num_validated_tokens == 0 {
            0
        } else if token_budget >= num_validated_tokens {
            min(token_budget, num_validated_tokens + self.num_spec_tokens)
        } else {
            min(token_budget, self.max_partial_token_consumption)
        }
    }

    pub fn min_validated_token_consumption(&self) -> usize {
        let max_validated_token_consumption = self.max_validated_token_consumption();
        debug_assert!(
            max_validated_token_consumption != 0,
            "runnable request must have a progress budget"
        );
        if self.max_partial_token_consumption == 0 {
            max_validated_token_consumption
        } else {
            1
        }
    }

    pub fn max_validated_token_consumption(&self) -> usize {
        self.num_ready_tokens + self.num_queued_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchBudget {
    pub req_budget: usize,
    pub token_budget: usize,
    pub max_token_per_req: usize,
}

/// Allocate minimum validated, maximum validated, then speculative token budgets.
pub fn allocate_sticky_token_budgets(
    batch_budget: BatchBudget,
    sticky_token_inventories: &[ReqTokenInventory<'_>],
) -> AHashMap<RawRequestID, usize> {
    debug_assert!(0 < batch_budget.req_budget, "req_budget must be positive");
    debug_assert!(0 < batch_budget.token_budget, "token_budget must be positive");
    debug_assert!(0 < batch_budget.max_token_per_req, "max_token_per_req must be positive");
    debug_assert!(
        batch_budget.max_token_per_req <= batch_budget.token_budget,
        "max_token_per_req must not exceed token_budget"
    );
    debug_assert!(
        sticky_token_inventories.len() <= batch_budget.req_budget,
        "sticky request count exceeds req_budget"
    );
    debug_assert!(
        sticky_token_inventories
            .iter()
            .enumerate()
            .all(|(request_index, request)| {
                sticky_token_inventories[..request_index]
                    .iter()
                    .all(|previous| previous.req_id != request.req_id)
            }),
        "sticky request IDs must be unique within one batch"
    );
    debug_assert!(
        sticky_token_inventories
            .iter()
            .all(|request| request.min_validated_token_consumption() <= batch_budget.max_token_per_req),
        "request minimum token budget exceeds max_token_per_req"
    );

    let mut token_budgets = vec![0; sticky_token_inventories.len()];
    let mut remaining_tokens = batch_budget.token_budget;

    // Phase 1: allocate each sticky request's minimum validated token budget.
    for (request_index, request) in sticky_token_inventories.iter().enumerate() {
        let token_budget = request.min_validated_token_consumption();
        assert!(
            token_budget <= remaining_tokens,
            "sticky request minimum budgets exceed token_budget"
        );

        token_budgets[request_index] = token_budget;
        remaining_tokens -= token_budget;
    }

    // Phase 2: allocate each sticky request's maximum validated token budget.
    for (request_index, request) in sticky_token_inventories.iter().enumerate() {
        let current_token_budget = token_budgets[request_index];
        let token_budget_limit = min(
            request.max_validated_token_consumption(),
            min(batch_budget.max_token_per_req, current_token_budget + remaining_tokens),
        );
        let token_budget = request.token_consumption(token_budget_limit);
        debug_assert!(
            current_token_budget <= token_budget,
            "validated token consumption must not remove minimum progress"
        );

        remaining_tokens -= token_budget - current_token_budget;
        token_budgets[request_index] = token_budget;
    }

    // Phase 3: allocate speculative token budgets by cumulative confidence.
    let mut spec_token_candidates = BinaryHeap::new();
    for (request_index, request) in sticky_token_inventories.iter().enumerate() {
        push_next_spec_token_candidate(
            &mut spec_token_candidates,
            request,
            request_index,
            token_budgets[request_index],
            0,
            1.0,
            batch_budget.max_token_per_req,
        );
    }

    while remaining_tokens != 0 {
        let Some(candidate) = spec_token_candidates.pop() else {
            break;
        };

        token_budgets[candidate.request_index] += 1;
        remaining_tokens -= 1;
        push_next_spec_token_candidate(
            &mut spec_token_candidates,
            &sticky_token_inventories[candidate.request_index],
            candidate.request_index,
            token_budgets[candidate.request_index],
            candidate.spec_position + 1,
            candidate.cumulative_confidence.into_inner(),
            batch_budget.max_token_per_req,
        );
    }

    sticky_token_inventories
        .iter()
        .zip(token_budgets)
        .map(|(request, token_budget)| (request.req_id, token_budget))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpecTokenCandidate {
    cumulative_confidence: NotNan<f64>,
    request_index: usize,
    spec_position: usize,
}

impl Ord for SpecTokenCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cumulative_confidence
            .cmp(&other.cumulative_confidence)
            .then_with(|| other.request_index.cmp(&self.request_index))
            .then_with(|| other.spec_position.cmp(&self.spec_position))
    }
}

impl PartialOrd for SpecTokenCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn push_next_spec_token_candidate(
    spec_token_candidates: &mut BinaryHeap<SpecTokenCandidate>,
    request: &ReqTokenInventory<'_>,
    request_index: usize,
    current_token_budget: usize,
    spec_position: usize,
    previous_cumulative_confidence: f64,
    max_token_per_req: usize,
) {
    if spec_position == request.spec_confidences.len() || current_token_budget == max_token_per_req {
        return;
    }

    let candidate_token_budget = current_token_budget + 1;
    let candidate_consumption = request.token_consumption(candidate_token_budget);
    if candidate_consumption != candidate_token_budget {
        return;
    }

    // The initial policy applies the identity transform.
    // Calibration and cross-request comparability require telemetry.
    let confidence = f64::from(request.spec_confidences[spec_position].into_inner());
    let cumulative_confidence = previous_cumulative_confidence * confidence;
    spec_token_candidates.push(SpecTokenCandidate {
        cumulative_confidence: NotNan::new(cumulative_confidence).expect("validated confidence cannot be NaN"),
        request_index,
        spec_position,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_consumption_wo_tokens() {
        let inventory = ReqTokenInventory::new::<1>(1, 0, 0, 0, &[]);

        assert_eq!(
            (0..=2)
                .map(|token_budget| inventory.token_consumption(token_budget))
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
    }

    #[test]
    fn test_token_consumption_w_single_cache_lane() {
        let confidences = [NotNan::new(0.9).unwrap(), NotNan::new(0.8).unwrap()];
        let inventory = ReqTokenInventory::new::<1>(1, 0, 1, 2, &confidences);

        assert_eq!(inventory.min_validated_token_consumption(), 1);
        assert_eq!(inventory.max_validated_token_consumption(), 1);
        assert_eq!(
            (0..=4)
                .map(|token_budget| inventory.token_consumption(token_budget))
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 3]
        );
    }

    #[test]
    fn test_token_consumption_w_multiple_cache_lanes() {
        let confidences = [NotNan::new(1.0).unwrap(); 3];
        let inventory = ReqTokenInventory::new::<4>(1, 1, 3, 3, &confidences);

        assert_eq!(inventory.min_validated_token_consumption(), 1);
        assert_eq!(inventory.max_validated_token_consumption(), 4);
        assert_eq!(
            (0..=8)
                .map(|token_budget| inventory.token_consumption(token_budget))
                .collect::<Vec<_>>(),
            vec![0, 1, 1, 1, 4, 5, 6, 7, 7]
        );
    }

    #[test]
    fn test_token_consumption_w_atomic_validated_budget() {
        let confidences = [NotNan::new(1.0).unwrap(); 3];
        let inventory = ReqTokenInventory::new::<4>(1, 0, 2, 3, &confidences);

        assert_eq!(inventory.min_validated_token_consumption(), 2);
        assert_eq!(inventory.max_validated_token_consumption(), 2);
        assert_eq!(
            (0..=6)
                .map(|token_budget| inventory.token_consumption(token_budget))
                .collect::<Vec<_>>(),
            vec![0, 0, 2, 3, 4, 5, 5]
        );
    }

    #[test]
    fn test_allocate_sticky_token_budgets_wo_validated_budget_transition() {
        let sticky_token_inventories = [
            ReqTokenInventory::new::<4>(1, 5, 3, 0, &[]),
            ReqTokenInventory::new::<4>(2, 5, 3, 0, &[]),
        ];

        assert_eq!(
            allocate_sticky_token_budgets(
                BatchBudget {
                    req_budget: 2,
                    token_budget: 8,
                    max_token_per_req: 8,
                },
                &sticky_token_inventories,
            ),
            AHashMap::from([(1, 5), (2, 3)])
        );
    }

    #[test]
    fn test_allocate_sticky_token_budgets_w_validated_budget_transition() {
        let sticky_token_inventories = [
            ReqTokenInventory::new::<4>(1, 5, 3, 0, &[]),
            ReqTokenInventory::new::<4>(2, 5, 3, 0, &[]),
        ];

        assert_eq!(
            allocate_sticky_token_budgets(
                BatchBudget {
                    req_budget: 2,
                    token_budget: 9,
                    max_token_per_req: 8,
                },
                &sticky_token_inventories,
            ),
            AHashMap::from([(1, 8), (2, 1)])
        );
    }

    #[test]
    fn test_allocate_sticky_token_budgets_w_confidence_order() {
        let request_1_confidences = [NotNan::new(0.90).unwrap(), NotNan::new(0.10).unwrap()];
        let request_2_confidences = [NotNan::new(0.80).unwrap(), NotNan::new(0.80).unwrap()];
        let sticky_token_inventories = [
            ReqTokenInventory::new::<1>(1, 1, 0, 2, &request_1_confidences),
            ReqTokenInventory::new::<1>(2, 1, 0, 2, &request_2_confidences),
        ];

        assert_eq!(
            allocate_sticky_token_budgets(
                BatchBudget {
                    req_budget: 2,
                    token_budget: 5,
                    max_token_per_req: 3,
                },
                &sticky_token_inventories,
            ),
            AHashMap::from([(1, 2), (2, 3)])
        );
    }

    #[test]
    fn test_allocate_sticky_token_budgets_w_fifo_confidence_ties() {
        let request_1_confidences = [NotNan::new(1.0).unwrap(); 3];
        let request_2_confidences = [NotNan::new(1.0).unwrap(); 3];
        let sticky_token_inventories = [
            ReqTokenInventory::new::<4>(1, 1, 3, 3, &request_1_confidences),
            ReqTokenInventory::new::<4>(2, 1, 3, 3, &request_2_confidences),
        ];

        assert_eq!(
            allocate_sticky_token_budgets(
                BatchBudget {
                    req_budget: 2,
                    token_budget: 10,
                    max_token_per_req: 7,
                },
                &sticky_token_inventories,
            ),
            AHashMap::from([(1, 6), (2, 4)])
        );
    }
}
