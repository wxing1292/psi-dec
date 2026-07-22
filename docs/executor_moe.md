# MoE Executor

This document maps the current MoE implementation from routing and policy selection through expert execution,
common-expert composition, and production benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency. It keeps backend-neutral MoE layer
metadata and policy contracts; `crates/inference-executor-metal` owns the current Metal replay backend. The public
semantic boundary is MoE; the low-level sparse expert MLP remains an inner component of that MoE path.

```text
crates/inference-executor-core/src/mlp/moe/
  mod.rs      semantic MoE component boundary exports
  core.rs      GatedMoECore + GatedMoEReplayShape
  policy.rs    MoEExecutionPolicy + MoEExecutionPolicyConfig

crates/inference-executor-metal/src/mlp/moe/
  mod.rs      Metal MoE module root
  backend.rs   GatedMoEMetalConfig + GatedMoE
  scratch.rs   routing, top-k-expert, and optional common-expert scratch ownership and bindings

crates/inference-executor-metal/src/model/qwen/v3_5/layer/
  moe.rs       Qwen35MoE, private checkpoint weights, load, and record

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape
  Layer
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

It keeps token-major, compact expert-major, and auto execution policies explicit. The backend implements the executor
`Layer + ReplayLayer` contract so Qwen model/layer code can append MoE work into a larger e2e replay through one semantic
input/output and caller-owned recorder. The semantic replay input optionally carries a common/shared expert branch:

```text
router projection -> route top-k -> dispatch -> sparse expert MLP
common expert + common gate
combine/scatter with common contribution
```

Current MoE replay records dispatch/layout as part of the selected token-major or expert-major policy, not as a
separate scheduler-owned phase. Model/layer wiring treats the whole MoE MLP as one component boundary inside a larger
layer/model ICB. The MoE backend records internal barriers between router/routing, dispatch/layout, expert compute,
common expert work, and combine/scatter where those commands have RAW dependencies.

`GatedMoEMetalConfig` keeps expert quantization bits separate from router and shared-gate
quantization bits:

```text
bits              top-k expert and common-expert MLP projections
router_bits       MoE router projection
common_gate_bits  common-expert gate projection
```

Qwen3.6-35B-A3B uses 4-bit expert MLPs and 8-bit router/common-gate tensors through config quantization overrides.
The Qwen component geometry helper resolves those overrides into the Metal config during semantic load; benches must
not assume one global bit width for every projection in a MoE layer.

`GatedMoECore::common_expert_intermediate_dim` is the single semantic source for common-expert presence and shape. It
must not be inferred from `intermediate_dim`: routed and common experts may legally use different intermediate widths.
Weight loading, common-expert MLP construction, and optional common-expert scratch allocation all derive from this value.

`QuantizedSparseMLP` remains a lower-level expert compute component; it exposes token-major and expert-major sparse
expert MLP compute but does not own router, dispatch, combine, common expert, or policy selection. Its token-major shape
uses `{ num_routes, num_tokens }`; its expert-major shape uses `{ num_experts, num_routes }`. Raw gather-matmul operators
use semantic gather axes `{ num_routes, num_input_vectors }`; only their true matrix axes retain `n` and `k`.

## Replay contract

`GatedMoE` records one MoE MLP forward through `ReplayLayer::record(...)` and a
caller-owned `Recorder`. The component boundary is the full MoE path, not the sparse expert MLP kernel by itself.

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

Replay returns `next_hidden_state` directly; it does not wrap the caller-owned buffer in a one-field output object.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay.

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

MoE routing is a two-stage contract: router projection writes bf16 logits, the softmax operator writes a bf16
`router_probs` buffer over all experts, and `MoERoutingKernel` selects top-k experts from `router_probs`. The routing
kernel does not read router logits; logits are only the softmax input. This keeps replay resource dependencies aligned
with the actual dataflow. The routing kernel renormalizes selected probabilities only when `norm_topk_prob=true`.
`expert_indices` and `expert_probs` are route-major with `num_tokens * num_experts_per_token` entries.

The shape contract is exact for the current microbatch:

```text
num_tokens              current microbatch token count
num_routes              num_tokens * num_experts_per_token
policy                  token-major, expert-major, or auto resolved from num_tokens
```

Production callers may allocate scratch using the executor's maximum token capacity. Each replay invocation resolves the
execution policy from the current `tokens` and validates that the capacity buffers cover the current route/input/output
shape. The token-major path consumes `token_indices` and `route_indices` directly. The expert-major path builds compact
expert-grouped routes, packs hidden rows, runs route-major sparse expert compute, then inverse-scatters and combines.
The route order inside an expert group is not a semantic contract; the inverse route map is.

Qwen model replay keeps MoE scratch in one model-owned `MoEScratch`. It owns three explicit regions: routing,
`topk_experts`, and optional `common_expert`. `bindings()` exposes routing and top-k-expert scratch;
`common_expert_bindings()` exposes the optional common-expert branch. Main and MTP execution is serialized on the model stream,
so router logits/probs, route metadata, sparse activation, expert-major packing, and optional common-expert scratch are
reusable across MoE layers. Qwen asserts across every main and MTP MoE layer that the scratch layout determinants are
uniform; per-layer router/top-k/common expert weights and layer output buffers remain layer-owned.
Token-major `token_indices` and identity `route_indices` are capacity metadata, not request metadata; Qwen initializes
them once in `MoEScratch` and each replay consumes the prefix implied by the current route count.

When a shared expert is present, the routed sparse expert branch and the common expert branch form a fork/join dataflow.
Both branches read the same normalized hidden input and write disjoint scratch buffers; they only join at final
combine/scatter. Replay should not insert barriers between those branches unless a buffer is actually shared.

Ragged expert-major MoE dispatch is represented by `MoEExpertMajorKernels`: routes are grouped by expert into a compact
route buffer, packed from token hidden states, evaluated with route-major sparse expert compute, then inverse-scattered
back to token-major output. The current implementation produces that grouping with a parallel histogram/counting-sort
layout:

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
`expert_offsets[e]..expert_offsets[e + 1]` is the ragged route segment for expert `e`. MoE semantics only require
compact expert grouping plus the inverse route map; preserving original route order inside each expert is not a forward
contract. The sparse expert compute uses ragged expert-major affine kernels over compact expert-major rows and
`experts_by_route`; it uses a ragged route-to-expert
`experts * routes_per_expert` affine contract.
Token-major and expert-major replay paths remain explicit probes. The model-executor default auto policy uses
token-major for `tokens <= 4` and expert-major above that threshold.

## Data flow and backend stages

MoE starts as a hidden-state transform and introduces a route-major side stream:

```text
hidden_state[tokens, hidden_dim]
  -> router projection
  -> softmax over num_experts
  -> routing top-k
       expert_indices[routes]
       expert_probs[routes]
  -> sparse expert MLP policy
  -> combine/scatter
  -> next_hidden_state[tokens, hidden_dim]
```

where:

```text
num_routes = num_tokens * num_experts_per_token
route_index = token_index * num_experts_per_token + expert_slot_index
```

Router projection writes bf16 logits. Softmax writes bf16 probabilities over all experts. Routing reads probabilities,
not logits, and writes route-major `expert_indices` and f32 `expert_probs`. If `norm_topk_prob` is enabled, the selected
top-k probabilities are renormalized over the selected experts; otherwise they remain the selected softmax
probabilities.

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

In the shared Qwen `MoEScratch`, `token_indices` and identity `route_indices` are capacity metadata initialized once; the
active route prefix is determined by current `tokens * topk`.

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

`expert_offsets[e]..expert_offsets[e + 1]` is the ragged segment for expert `e`. `routes_by_expert` maps compact
expert-major row back to original token-major route. `routes_by_token` is the inverse map used by final scatter.
`experts_by_route` names the expert for each compact route row and is consumed by ragged expert-major affine kernels.

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

The route order inside one expert segment is not semantic. The required contract is compact grouping by expert plus a
correct inverse map for scatter. This lets the expert-major affine kernels process contiguous ragged expert segments
without a rectangular `experts * routes_per_expert` layout.

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

The branches read the same normalized hidden input and write disjoint scratch, so replay should only add barriers where
there is an actual dependency: router logits before softmax, probabilities before routing, routing/layout before sparse
expert compute, expert output/common branch before final combine or scatter.

### Policy boundary

`GatedMoE` owns policy selection; `QuantizedSparseMLP` only owns expert inner compute. Current auto policy is:

```text
tokens <= 4  -> token-major
tokens > 4   -> ragged expert-major
```

Token-major and expert-major are both correctness paths and explicit bench probes. Changing the auto threshold is a perf
policy change and should be justified with full MoE wrapper numbers, not only isolated sparse MLP kernel timings.

Focused tests compare routing, token-major sparse MLP, and combine against CPU references with fixed and random inputs.
The expert-major test records the production subgraph `layout -> pack -> sparse MLP -> scatter` and compares its final
token-major output against the same CPU expert and bf16-combine references, covering both fixed and random fixtures.

## Tests and benchmarks

Current Metal component benches:

```text
cargo bench -p inference-backend-metal --bench moe
cargo bench -p inference-backend-metal --bench sparse_mlp
```

The benches include Metal replay/ICB cases for MoE routing/combine, token-major sparse expert forward paths, and
synthetic MoE forward replay paths that record router projection, routing, sparse expert MLP, common expert MLP, common
gate, and scatter/combine into one batch. Token-major and expert-major policies remain explicit replay cases.
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

`qwen35_moe` loads a Qwen3.6-35B-A3B 4-bit checkpoint and uses the MoE replay backend to run explicit token-major and
compact expert-major MoE replay for one sparse layer. It uses CLI args for model path, layer, token list, iteration
counts, implementation selection, and parity checking:

```text
cargo bench -p inference-executor-metal --bench qwen35_moe -- \
  --model-dir <35b-a3b-model-dir> --layer 0 \
  --tokens 1,2,4,8,16,32,64 \
  --impls token_major,expert_major \
  --check-parity
```

`--check-parity` is an explicit, non-timed diagnostic and is off by default.
Keep it separate from pure timing after correctness has already been established.

`sparse_mlp` includes replay-backed token-major sparse expert fused gate/up/silu probes and Metal replay/ICB sparse expert
forward paths over the same token counts. New sparse/MoE Metal code does not keep direct-submit component or forward
paths. The 27B dense checkpoint has no MoE routing/combine/sparse-MLP path, so there is no meaningful 27B MoE
bench.

MoE perf should be bisected by contract boundary: router projection/softmax/routing, sparse expert MLP, combine, shared
branch, then full MoE wrapper. Routing and combine belong to MoE; `QuantizedSparseMLP` is only the expert inner compute.
Top-k ordering differences are not a perf target by themselves, but the route probability semantics must be explicit:
logits vs already-softmaxed probabilities, and whether selected top-k probabilities are renormalized.

Token-major and expert-major outputs should remain bitwise equal in parity runs.
The current auto policy uses token-major for `tokens <= 4` and expert-major for
larger microbatches.

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
