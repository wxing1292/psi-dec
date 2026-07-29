# MoE Executor

This document describes the current MoE implementation.
It covers routing, backend path selection, expert execution, common-expert composition, tests, and production
benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency.
It owns backend-neutral MoE layer metadata.
`crates/inference-executor-metal` owns the current Metal replay backend.
MoE is the public semantic boundary.
The low-level sparse expert MLP remains an inner component of the MoE path.

```text
crates/inference-executor-core/src/mlp/moe/
  mod.rs      semantic MoE component boundary exports
  core.rs      GatedMoECore + GatedMoEReplayShape

crates/inference-executor-metal/src/mlp/moe/
  mod.rs      Metal MoE module root
  backend.rs   GatedMoEMetalConfig + GatedMoE
  scratch.rs   routing, top-k-expert, and optional common-expert scratch ownership and bindings

crates/inference-executor-metal/src/model/qwen/
  v3_x/layer/moe.rs  Qwen3xMoE, private checkpoint weights, load, and record
  v3_5/main/layer.rs Qwen3.5 Main dense-MLP/MoE layer variants
  v3_5/mtp/layer.rs  Qwen3.5 MTP dense-MLP/MoE layer variants
  v3_5/plan.rs       Qwen3.5 MoE geometry/config builder

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal MoE / sparse expert kernels live in:

```text
crates/inference-backend-metal/src/components/
  moe_routing.rs
  moe_expert_major.rs
  moe_combine.rs
  quantized_sparse_mlp.rs
```

## Shape model

`GatedMoECore` owns immutable layer metadata:

```text
model_layer_index
hidden_dim
intermediate_dim
common_expert_intermediate_dim (optional and independent from routed expert intermediate_dim)
num_experts
num_experts_per_token
norm_topk_prob
```

It derives MoE / sparse expert shapes:

```text
router_shape
gate_shape
up_shape
down_shape
```

`GatedMoE` owns the semantic MoE replay contract:

```text
router projection -> route top-k -> dispatch -> sparse expert MLP -> combine/scatter
```

`GatedMoE` owns the token-major and compact expert-major implementations.
It selects one implementation from the active replay shape.
The backend implements the executor `ReplayLayer` contract.
Qwen model and layer code use this contract to append MoE work to a larger e2e replay.
The contract has one semantic input, one output, and a caller-owned recorder.
The semantic replay input can include a common/shared expert branch:

```text
router projection -> route top-k -> dispatch -> sparse expert MLP
common expert + common gate
combine/scatter with common contribution
```

Current MoE replay records dispatch/layout as part of the selected token-major or expert-major path.
The scheduler does not own a separate dispatch phase.
Model and layer wiring treat the full MoE MLP as one component boundary in a larger layer/model ICB.
The MoE backend records internal barriers where commands have RAW dependencies.
These barriers separate router projection and routing, dispatch and layout, expert compute, common-expert work, and
combine/scatter.

`GatedMoEMetalConfig` keeps expert quantization bits separate from router and shared-gate
quantization bits:

```text
bits              top-k expert and common-expert MLP projections
router_bits       MoE router projection
common_gate_bits  common-expert gate projection
```

Qwen3.6-35B-A3B uses 4-bit expert MLPs and 8-bit router/common-gate tensors through config quantization overrides.
The Qwen component geometry helper resolves these overrides during semantic load.
Benches must not assume one global bit width for all projections in a MoE layer.

The router and common-gate projections each use one adaptive affine operator.
`GatedMoE` provides the fixed projection geometry and quantization layout when it creates each operator.
It provides the current active token count when it records each projection.
The affine operator selects QMV or a QMM tile.
`GatedMoE` does not store separate QMV/QMM kernels or a projection threshold.

`GatedMoECore::common_expert_intermediate_dim` is the only semantic source for common-expert presence and shape.
Code must not infer this information from `intermediate_dim`.
Routed and common experts can use different intermediate widths.

Weight loading derives from `common_expert_intermediate_dim`.
Common-expert MLP construction and optional scratch allocation also derive from it.
The Qwen weight owner groups the common gate and dense expert under one optional owner.
Thus, a partially populated common-expert branch is not representable.

`QuantizedSparseMLP` remains a lower-level expert compute component.
It exposes token-major and expert-major sparse expert MLP compute.
It does not own routing, dispatch, combine, common-expert work, or path selection.

Its token-major shape is `{ num_routes, num_tokens }`.
Its expert-major shape is `{ num_experts, num_routes }`.
Raw gather-matmul operators use semantic gather axes `{ num_routes, num_input_vectors }`.
Only their true matrix axes retain `n` and `k`.

## Replay contract

`GatedMoE` records one MoE MLP forward through `ReplayLayer::record(...)` and a caller-owned `Recorder`.
The full MoE path is the component boundary.
The sparse expert MLP kernel alone is not that boundary.

The semantic replay input is:

```text
GatedMoEReplayInput
  shape    GatedMoEReplayShape
  hidden_state      &Buffer
  next_hidden_state &Buffer
  scratch  MoEScratchBindings
  weights  GatedMoEWeights
  common_expert optional GatedMoECommonExpertReplayInput
```

Replay returns `next_hidden_state` directly.
It does not wrap the caller-owned buffer in a one-field output object.

Focused production-path tests use the same `ReplayLayer::record(...)` entrypoint as model replay.
The full-forward MoE benchmark composes the lower-level backend components directly.
This benchmark-only composition can force each implementation without adding a force option to the production API.

The no-common-expert replay order is:

```text
hidden_state
  -> router quantized projection
  -> router bf16 softmax
  -> MoE routing
  -> token-major or expert-major sparse expert dispatch/compute
  -> top-k weighted combine
  -> next_hidden_state
```

The shared-expert replay order adds the common branch before the final combine:

```text
hidden_state
  -> common dense expert
  -> common gate projection
  -> combine routed contribution + common contribution
```

MoE routing is a two-stage contract.
The router projection writes bf16 logits.
The softmax operator writes a bf16 `router_probs` buffer over all experts.
`MoERoutingKernel` selects top-k experts from `router_probs`.
The routing kernel does not read router logits.
Only the softmax operator reads the logits.

This contract aligns replay resource dependencies with the data flow.
The routing kernel renormalizes selected probabilities only when `norm_topk_prob=true`.
`expert_indices` and `expert_probs` are route-major with `num_tokens * num_experts_per_token` entries.

The shape contract is exact for the current microbatch:

```text
num_tokens              current microbatch token count
num_routes              num_tokens * num_experts_per_token
backend path            selected from num_tokens
```

The current routing kernel supports at most 256 experts and at most 16 selected experts per token.
`MoERoutingShape::validate()` treats other shapes as internal contract violations and panics.

Production callers may allocate scratch with the executor's maximum token capacity.
Each replay invocation selects the backend path from the current `num_tokens`.
It validates that capacity buffers cover the current route, input, and output shapes.
The token-major path consumes `token_indices` and `route_indices` directly.

The expert-major path builds compact expert-grouped routes and packs hidden rows.
It then runs route-major sparse expert compute.
Finally, it inverse-scatters and combines the results.
Route order inside an expert group is not a semantic contract.
The inverse route map is part of the contract.

Qwen model replay keeps MoE scratch in one model-owned `MoEScratch`.
It owns three explicit regions.
They are routing, `topk_experts`, and optional `common_expert`.
`bindings()` exposes routing and top-k-expert scratch.
`common_expert_bindings()` exposes the optional common-expert branch.

The model stream serializes Main and MTP execution.
Thus, MoE layers can reuse router logits/probs, route metadata, sparse activation, expert-major packing, and optional
common-expert scratch.
Qwen asserts that scratch layout determinants are uniform across all Main and MTP MoE layers.

The shared `Qwen3xMoE` leaf owns per-layer router, top-k, and common-expert weights.
The role-specific layer and scratch types own output buffers and composition.
These types are `Qwen35MainLayer`/`Qwen35MainLayerScratch` and `Qwen35MTPLayer`/`Qwen35MTPLayerScratch`.

Token-major `token_indices` and identity `route_indices` are capacity metadata.
They are not request metadata.
Qwen initializes them once in `MoEScratch`.
Each replay consumes the prefix for the current route count.

When a shared expert is present, the routed and common-expert branches form a fork/join data flow.
Both branches read the same normalized hidden input.
They write disjoint scratch buffers and join only at final combine/scatter.
Recommendation: Do not insert a barrier between these branches unless they share a buffer.

`MoEExpertMajorKernels` represents ragged expert-major MoE dispatch.
It groups routes by expert in a compact route buffer.
It packs token hidden states and runs route-major sparse expert compute.
It then inverse-scatters the results to token-major output.
The current implementation uses a parallel histogram/counting-sort layout:

```text
expert_indices
  -> expert_counts
  -> expert_offsets / expert_cursors
  -> routes_by_expert
  -> routes_by_token
  -> experts_by_route
```

`routes_by_expert[expert_route]` maps compact expert-major row to the original token-major route.
`routes_by_token[token_route]` maps the original token-major route back to its compact expert-major row.
`expert_offsets[e]..expert_offsets[e + 1]` is the ragged route segment for expert `e`.
MoE semantics require compact expert grouping and the inverse route map.
Original route order inside each expert is not a forward contract.

Expert-major sparse compute uses ragged rows with shape `{ num_experts, num_routes }`.
For each `expert_route` in `0..num_routes`, `experts_by_route[expert_route]` selects the expert.
It does not allocate `num_experts * routes_per_expert` rows.

Token-major and expert-major replay paths remain explicit benchmark probes.
The production Metal backend uses token-major for `num_tokens <= 4`.
It uses expert-major for larger microbatches.

## Data flow and backend stages

MoE starts as a hidden-state transform and introduces a route-major side stream:

```text
hidden_state[tokens, hidden_dim]
  -> router projection
  -> softmax over num_experts
  -> routing top-k
       expert_indices[routes]
       expert_probs[routes]
  -> selected sparse expert MLP path
  -> combine/scatter
  -> next_hidden_state[tokens, hidden_dim]
```

where:

```text
num_routes = num_tokens * num_experts_per_token
route_index = token_index * num_experts_per_token + expert_slot_index
```

Router projection writes bf16 logits.
Softmax writes bf16 probabilities over all experts.
Routing reads probabilities, not logits.
It writes route-major `expert_indices` and f32 `expert_probs`.
When `norm_topk_prob` is enabled, routing renormalizes selected top-k probabilities over the selected experts.
Otherwise, the values remain the selected softmax probabilities.

### Token-Major Sparse MLP

Token-major sparse MLP keeps routes in original token-major order:

```text
input hidden[tokens, hidden_dim]
token_indices[routes]       token row for each route
expert_indices[routes]      selected expert for each route
route_indices[routes]       activation row used by down projection

fused gate/up/silu
  reads input[token_indices[route]]
  reads expert_indices[route] expert weights
  writes activation[route, intermediate_dim]

down
  reads activation[route_indices[route]]
  reads expert_indices[route] expert weights
  writes routed_hidden[route, hidden_dim]
```

In the shared Qwen `MoEScratch`, `token_indices` and identity `route_indices` are capacity metadata.
Qwen initializes them once.
Current `tokens * topk` determines the active route prefix.

### Ragged Expert-Major Sparse MLP

Expert-major first converts token-major routes into compact expert-major rows:

```text
expert_indices[token_route]
  -> expert_counts[num_experts]
  -> expert_offsets[num_experts + 1]
  -> routes_by_expert[expert_route]
  -> routes_by_token[token_route]
  -> experts_by_route[expert_route]
```

The layout kernels are:

```text
layout_clear    zero expert_counts and expert_cursors
layout_count    count routes per expert from expert_indices
layout_prefix   prefix-sum expert_counts into expert_offsets and reset cursors
layout_scatter  assign each token_route to a compact expert_route
```

`expert_offsets[e]..expert_offsets[e + 1]` is the ragged segment for expert `e`.
`routes_by_expert` maps each compact expert-major row to its original token-major route.
`routes_by_token` is the inverse map for final scatter.
Ragged expert-major affine kernels use `experts_by_route`.
`experts_by_route` selects the expert for each compact route row.

After layout:

```text
pack_input
  reads hidden_state[token]
  reads routes_by_expert
  writes packed_input[expert_route, hidden_dim]

ragged sparse expert MLP
  reads packed_input + experts_by_route + expert weights
  writes route_output[expert_route, hidden_dim]

scatter/combine
  reads route_output via routes_by_token
  reads expert_probs[token_route]
  writes next_hidden_state[token, hidden_dim]
```

Route order inside one expert segment is not semantic.
The contract requires compact expert grouping and a correct inverse map for scatter.
Expert-major affine kernels can process the resulting contiguous ragged segments.
They do not require a rectangular `experts * routes_per_expert` layout.

### Combine and Common Expert

Without a shared/common expert, combine computes:

```text
next_hidden[token, dim] =
  sum_{slot in topk} expert_probs[token_route] * routed_hidden[token_route, dim]
```

With a shared expert, the routed branch and common branch are a fork/join:

```text
routed branch: hidden -> routing -> sparse expert MLP -> routed contribution
common branch: hidden -> common dense expert, hidden -> common gate projection

next_hidden[token, dim] =
  routed_sum[token, dim] + sigmoid(common_gate[token]) * common_hidden[token, dim]
```

The branches read the same normalized hidden input and write disjoint scratch.
Recommendation: Add replay barriers only at these actual dependencies:

- Router logits before softmax.
- Probabilities before routing.
- Routing/layout before sparse expert compute.
- Expert output and common branch before final combine or scatter.

### Backend selection boundary

`GatedMoE` owns backend path selection.
`QuantizedSparseMLP` owns only expert inner compute.
The current production selector is:

```text
tokens <= 4  -> token-major
tokens > 4   -> ragged expert-major
```

The runtime core and Qwen model config do not contain this threshold.
They also do not expose a forced implementation.
The full-forward benchmark directly composes the same lower-level backend components to force token-major or
expert-major execution.
Changing the selector is a Metal backend performance change.
Recommendation: Justify that change with full MoE wrapper numbers, not only isolated sparse MLP kernel timings.

Focused tests compare routing, token-major sparse MLP, and combine with CPU references.
They use fixed and random inputs.
The expert-major test records the production subgraph `layout -> pack -> sparse MLP -> scatter`.
It compares the final token-major output with the same CPU expert and bf16-combine references.
This comparison covers fixed and random fixtures.

## Tests and benchmarks

Current Metal component benches:

```text
cargo bench -p inference-backend-metal --bench moe
cargo bench -p inference-backend-metal --bench sparse_mlp
```

The benches include Metal replay/ICB cases for MoE routing/combine.
They include token-major sparse expert forward paths.
Synthetic forward cases record router projection, routing, sparse expert MLP, common expert MLP, common gate, and
scatter/combine.
They record these operations in one batch.
Token-major and expert-major implementations remain explicit replay cases.
They use the 35B-A3B MoE profile:

```text
hidden_dim=2048
moe_intermediate_dim=512
num_experts=256
topk_experts=8
tokens=1,2,4,8,16,32,64
```

Current Metal real-weight MoE full-forward bench:

```text
cargo bench -p inference-executor-metal --bench qwen35_moe -- \
  --model-dir <35b-a3b-model-dir> --layer 0 --tokens 1 \
  --impls token_major --iters 1 --warmup-iters 0 --runs 1
```

`qwen35_moe` loads a Qwen3.6-35B-A3B 4-bit checkpoint.
It composes one sparse layer from the production backend components.
It runs explicit token-major and compact expert-major replay without a production force flag.
CLI arguments select the model path, layer, token list, iteration counts, implementation, and parity checking:

```text
cargo bench -p inference-executor-metal --bench qwen35_moe -- \
  --model-dir <35b-a3b-model-dir> --layer 0 \
  --tokens 1,2,4,8,16,32,64 \
  --impls token_major,expert_major \
  --check-parity
```

`--check-parity` is an explicit non-timed diagnostic.
It is off by default.
Keep it separate from pure timing after correctness verification.

`sparse_mlp` includes replay-backed token-major sparse expert fused gate/up/silu probes.
It includes Metal replay/ICB sparse expert forward paths over the same token counts.
New sparse/MoE Metal code does not keep direct-submit component or forward paths.
The 27B dense checkpoint has no MoE routing, combine, or sparse-MLP path.
Thus, there is no meaningful 27B MoE bench.

Recommendation: Bisect MoE performance by contract boundary in this order:

1. Router projection, softmax, and routing.
2. Sparse expert MLP.
3. Combine.
4. Shared branch.
5. Full MoE wrapper.

Routing and combine belong to MoE.
`QuantizedSparseMLP` owns only expert inner compute.
Top-k ordering differences alone are not a performance target.
Route probability semantics must identify logits or already-softmaxed probabilities.
They must also identify whether routing renormalizes selected top-k probabilities.

The expected parity result is bitwise-equal token-major and expert-major output.
The current production backend uses token-major for `num_tokens <= 4`.
It uses expert-major for larger microbatches.

[`executor_benchmarks.md`](executor_benchmarks.md) defines shared GPU serialization, benchmark metrics, and
performance-evidence rules.
