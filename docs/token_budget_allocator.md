# Token Budget Allocator

The allocator divides a batch's token budget among the sticky requests assigned to one compute slot.
It first preserves request progress, then consumes validated input, and then allocates speculative prefixes.
This document describes the current confidence-aware policy.
The public allocator API is in `runtime::scheduler`.
[`core.md`](core.md) owns scheduling and request lifecycle.
[`future_work.md`](future_work.md#confidence-aware-scheduling) owns the remaining calibration, cost, and integration work.

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

Each user request produces an immutable `ReqTokenInventory` for the planning pass.
The inventory contains the request ID, ready and queued token counts, speculative token count, and borrowed confidence values.
The constructor derives `prefill_decode_threshold` from the validated-token count and compile-time cache-lane count `L`.
The scheduler API does not expose the cache-lane count.

`num_spec_tokens` must equal `spec_confidences.len()`.

The inventory calculates `min_validated_token_consumption()` and `max_validated_token_consumption()` from these
fields.
Both methods return request-local bounds without speculative verification.
They do not replace the independent `max_tokens_per_request` hard limit.

`ReqTokenInventory` owns the piecewise token-consumption curve.
`token_consumption()` returns zero when the budget or validated-token count is zero.
A budget that covers all validated tokens can also include speculative verification, up to the available speculative count.
A smaller budget is capped at `prefill_decode_threshold`.

The [allocator source](../crates/inference-runtime-core/src/runtime/scheduler/token_budget_allocator.rs) owns the exact fields, formulas, and boundary tests.
The [scheduler source](../crates/inference-runtime-core/src/runtime/scheduler/simple_scheduler.rs) collects inventories in the compute slot's sticky request order.
It resolves each ID through the schedule queue and calls `UserRequest::token_estimate` for each present request.
It passes the inventories and `BatchBudget` to `allocate_sticky_token_budgets`.

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
For example, eight validated tokens and four cache lanes give a `prefill_decode_threshold` of five:

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
When cumulative confidence values tie, the heap selects the earlier proposal position first.
When positions also tie, it selects the earlier request in the slot's sticky order.

The identity transform ranks raw executor outputs directly.
It does not prove that the values are calibrated or comparable across requests.
A telemetry gate must validate these properties before a later policy uses an absolute threshold or a measured cost
decision.

### Worked allocation

This example uses sticky order `A, B`, a batch token budget of `5`, and a per-request limit of `3`.
Each request has one validated input token and two available proposals.
Both inventories use one cache lane.
Phase 1 assigns one token to each request.
Phase 2 consumes no additional budget because both validated inputs fit.
Phase 3 can allocate the remaining three tokens:

```text
+---------+----------------------+----------------------------+
| Request | Proposal confidence  | Cumulative candidate score |
+---------+----------------------+----------------------------+
| A       | 0.90, 0.10           | A0 = 0.90, A1 = 0.09       |
| B       | 0.80, 0.80           | B0 = 0.80, B1 = 0.64       |
+---------+----------------------+----------------------------+

+------+-----------------+----------+-----------------+
| Pick | Available       | Selected | Budgets (A, B)  |
+------+-----------------+----------+-----------------+
| 1    | A0, B0          | A0       | (2, 1)          |
| 2    | A1, B0          | B0       | (2, 2)          |
| 3    | A1, B1          | B1       | (2, 3)          |
+------+-----------------+----------+-----------------+
```

`B1` becomes eligible only after `B0` is selected.
The result contains one validated token and one proposal for `A`, and one validated token and two proposals for `B`.

With the same budgets and every confidence set to `1.0`, the selection is:

```text
candidate order: A0 -> B0 -> A1 -> B1
selected:        A0 -> B0 -> A1        (3 proposal tokens)
final budgets:  A = 3, B = 2
```

This tie case visits the earlier proposal position before it advances either request to the next position.

## Proposal modes

MTP confidence value `1.0` is a placeholder.
It is not a calibrated acceptance probability.
An MTP batch can use the same causal-candidate heap because all scores tie at `1.0`.
The tie-break allocates one proposal position across eligible requests before it advances to the next position.
Request order breaks ties within each position.
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
