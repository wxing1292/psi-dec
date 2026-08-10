# Token Budget Allocator

This document describes the first confidence-aware token-budget allocator policy.
The public allocator API is in `runtime::scheduler`.

## Scope

The runtime core owns request ordering and token-budget allocation.
The model executor continues to return fixed-width proposal tokens, probabilities, and confidence values.

The policy has these hard constraints:

- `max_requests`, `max_tokens`, and `max_tokens_per_request` are separate limits.
- `max_tokens_per_request` applies to every selected request.
- KV-page feasibility is separate from token-budget feasibility.
- `prepare()` is not a planning API because it changes request state.
- The allocator returns an `AHashMap<RawRequestID, usize>` with one scalar `token_budget` for each request.
- A Decode allocation can remove only a request-local speculative suffix.
- The sampled-token anchor and the other Decode input tokens are mandatory.
- Each runnable sticky request in the planning pass must receive its minimum progress budget.
- Each compute slot owns its sticky request ID order.
- Different compute slots can contain the same request ID.
- One device batch must not contain the same request ID more than once.

The first policy does not use an absolute confidence threshold.
Confidence changes allocation only when the hard batch budget cannot contain every proposal row.

## Request token inventory

The first version does not add a planning trait or a request budget-curve type.
Each user request produces an immutable `ReqTokenInventory` for the planning pass.
The inventory contains these fields:

```rust
struct ReqTokenInventory<'a> {
    req_id: RawRequestID,
    num_ready_tokens: usize,
    num_queued_tokens: usize,
    num_spec_tokens: usize,
    spec_confidences: &'a [NotNan<f32>],
    max_partial_token_consumption: usize,
}
```

`num_spec_tokens` must equal `spec_confidences.len()`.
The constructor uses the request's compile-time cache-lane count to derive `max_partial_token_consumption`.
The scheduler API does not expose the cache-lane count.

The inventory calculates `min_validated_token_consumption()` and `max_validated_token_consumption()` from these
fields.
Both methods return request-local bounds without speculative verification.
They do not replace the independent `max_tokens_per_request` hard limit.

`ReqTokenInventory` owns the piecewise token-consumption curve:

```rust
impl ReqTokenInventory<'_> {
    fn token_consumption(
        &self,
        token_budget: usize,
    ) -> usize {
        let num_validated_tokens =
            self.num_ready_tokens + self.num_queued_tokens;
        if token_budget == 0 || num_validated_tokens == 0 {
            0
        } else if token_budget >= num_validated_tokens {
            min(
                token_budget,
                num_validated_tokens + self.num_spec_tokens,
            )
        } else {
            min(
                token_budget,
                self.max_partial_token_consumption,
            )
        }
    }

    fn min_validated_token_consumption(&self) -> usize {
        let max_consumption = self.max_validated_token_consumption();
        debug_assert!(max_consumption != 0);
        if self.max_partial_token_consumption == 0 {
            max_consumption
        } else {
            1
        }
    }

    fn max_validated_token_consumption(&self) -> usize {
        self.num_ready_tokens + self.num_queued_tokens
    }
}
```

The production data flow is direct:

```rust
let sticky_token_inventories = sticky_requests
    .iter()
    .map(UserRequest::token_estimate)
    .collect::<Vec<_>>();

let sticky_token_budgets = allocate_sticky_token_budgets(
    BatchBudget {
        req_budget,
        token_budget,
        max_token_per_req,
    },
    &sticky_token_inventories,
);
```

This flow does not require a new request trait.

The request-local `L` remains an implementation parameter.
The allocator supplies one total absolute `token_budget` for each query.
The inventory query is stateless and always uses all currently available proposals.
The allocator queries and returns absolute request token budgets.
The output map owns request lookup. It does not preserve allocation order.
It does not expose budget deltas as planning actions or outputs.

The allocator subtracts `target_token_budget - current_token_budget` only to update the remaining global budget.

## Minimum validated consumption

The request inventory calculates the minimum budget that produces validated progress:

```rust
request.min_validated_token_consumption()
```

Failure is an invariant violation.
A runnable request must have a valid minimum consumption.
The allocator debug-checks that the result does not exceed `max_tokens_per_request`.

A sticky request must receive its full minimum progress budget.
The allocator panics if all sticky minimums do not fit `token_budget`.
The scheduler guarantees that the sticky request count does not exceed `req_budget`.

## Allocation phases

The sticky allocator executes three phases directly.
It does not admit new requests.

### Phase 1: Minimum validated token budgets

The allocator reserves the full minimum validated token budget for every sticky request.
This phase provides a liveness guarantee for the current working set.

### Phase 2: Maximum validated token budgets

The allocator visits sticky requests in their slot-local order.
It increases each request up to `max_tokens_per_request` or the remaining batch budget.
It also caps this phase at `request.max_validated_token_consumption()`.
This cap is the largest absolute budget that cannot include speculative verification.
It then supplies the capped absolute budget to `token_consumption()`.

The query preserves multi-lane discontinuities.
For example:

```text
budget 5 -> consumption 5
budget 6 -> consumption 5
budget 7 -> consumption 5
budget 8 -> consumption 8
```

If the complete jump fits, Phase 2 allocates it.
If the jump does not fit, the request remains at the last valid partial consumption.

### Phase 3: Speculative token budgets

The allocator adds only the next proposal position of each eligible request to a heap.
It ranks each candidate by cumulative confidence after an identity transform:

```text
score[r,j] = product(confidence[r,t], t=0..j)
```

After the allocator selects position `j`, it can add position `j+1` from that request.
The allocator never uses a future confidence value to select an earlier position.

The identity transform ranks raw executor outputs directly.
It does not prove that the values are calibrated or comparable across requests.
A telemetry gate must validate these properties before a later policy uses an absolute threshold or a measured cost
decision.

## Proposal modes

MTP confidence value `1.0` is a placeholder.
It is not a calibrated acceptance probability.
An MTP batch can use the same causal-candidate heap because all scores tie at `1.0`.
The FIFO tie-break then produces fixed FIFO prefix allocation.
This behavior is valid only while the policy has no absolute threshold and one runtime batch does not mix proposal
modes.

Runtime integration must keep proposal tokens, probabilities, and confidence values in one request-local state.
The three vectors must have the same length.
Every prefix trim must change all three vectors.

## Current policy status

This policy is lexicographic.
It is not a throughput optimum:

1. Preserve sticky request progress.
2. Increase sticky request validated-token consumption.
3. Allocate sticky proposal prefixes by causal confidence.
4. Let `FIFOBatcher` use the remaining request and token budgets for FIFO requests.

`max_tokens_per_request` bounds the effect of a request with many validated tokens.
The policy does not guarantee that each batch has capacity for speculative verification.

## Open decisions

The first policy does not decide these items:

- FIFO or round-robin order in Phase 2.
- Measured cost for request count, rows, padding, and replay buckets.
- The confidence calibration method and telemetry gate.
- Page-feasibility feedback to the planning pass.

The first version intentionally does not add a cost estimator.
The required latency and padding data does not exist yet.
