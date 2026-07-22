# Engineering Conventions

This document owns repository-wide naming, runtime-shape/storage-layout, replay/resource safety, optimization,
public-API, and test-style rules. Architecture and ownership boundaries remain in [`high_level.md`](high_level.md);
component-specific tensor vocabulary, source owners, and execution paths remain in the matching component document.

## Naming and coordinate domains

Names should encode domain semantics. Use established component abbreviations as type prefixes, such as `GQA*` and
`GDN*`, instead of spelling out the component name.

Use semantic names at model, layer, component, and non-matmul operator boundaries: `num_tokens`, `num_routes`,
`num_input_vectors`, and `num_experts_per_token`. Reserve `m`, `n`, and `k` for a true low-level matrix multiplication
shape. Routing, gather, page, and state coordinates retain their semantic names even when their implementation feeds a
matmul.

Use `num_*` for valid typed work counts and `total_*` for padded dispatch, replay, scratch, or capacity extents. Name
what is counted; do not substitute `element` or an unqualified `size` for the tensor/domain. Reserve `*_bytes` for raw
allocation lengths, binding offsets, address arithmetic, and byte copies.

Use unsigned integers for counts, slots, and IDs. Signed integers are for real negative values, sentinels, or imported
ABI contracts. Host byte offsets and address arithmetic use checked `u64`; flattened Metal addresses use `ulong`;
convert to `usize` only at Rust slice/pointer or Objective-C API boundaries. Keep bounded local counts and indices as
Rust `u32`/Metal `uint`; a large aggregate allocation does not require every local loop or tensor coordinate to be
64-bit.

Configuration names identify their owner:

```text
*ExecutorConfig   model-executor initialization
*RuntimeConfig    runtime core
*ServiceConfig    service transport/configuration
```

Checkpoint schemas identify their format, such as `HFGenerationConfig`, rather than using an unqualified subsystem
name.

Prefer precise coordinates:

```text
token_index
block_index
base_token_index + token_offset
base_block_index + block_offset
state_version
candidate_state_versions
verified_state_version
first_candidate_state_version
```

Avoid `absolute_*` unless a competing coordinate system exists. Avoid naked `offset`, `context`, `candidate_lens`, or
`commit_len` when the owner or unit can be named directly.

Use `flat` only for tensors or coordinates produced by flattening per-request token sequences. Per-request starting
`token_indices` may expand into `flat_token_indices`; request-slot IDs remain `req_slots` when repeated per token.
`cu_tokens` already names flattened segment boundaries and needs no `flat` prefix.

Use `cu_<items>` only for a monotonic cumulative-count table with `N + 1` entries. Adjacent values select the half-open
item segment owned by logical owner `i`: `cu_items[i]..cu_items[i + 1]`. The owning comment names both the owner and the
counted item unit. Do not use `cu_` for ordinary coordinates, byte offsets, capacities, or non-cumulative metadata.

Domain-standard abbreviations are appropriate when unambiguous at the owning boundary. Attention code may use `q`,
`k`, `v`, `qk`, `kv`, and projection names such as `qkv`, `qgkv`, and `qkvabz`. Do not reuse them for unrelated
query/key/value concepts; outside an established domain, prefer the complete semantic noun.

Use one symbolic convention for attention tensor and tile comments:

```text
Q: [Tq,  Hq,  D]    Tq  = Q tokens       Hq  = Q heads
K: [Tkv, Hkv, D]    Tkv = KV tokens      Hkv = KV heads
V: [Tkv, Hkv, D]    D   = head dimension
O: [Tq,  Hq,  D]

Q tile: [Tq_tile, Hq_tile, D]
K tile: [Tkv_tile, D]  // one fixed KV head
V tile: [Tkv_tile, D]
```

`T` names a token dimension and `H` a head dimension. Suffix the tiled axis with `_tile`; do not introduce ambiguous
`Bq`/`Bkv`. Outside the SDPA tensor/kernel boundary, use plain `token_tile`; reserve `q_*` for the Q tensor and its
dimensions.

## GPU work vocabulary

Keep mathematical decomposition separate from launch topology:

- A `*Tile` is the smallest named matmul-like logical unit at that component boundary. It is not a launch object.
- A `*Task` is the complete logical work executed by one threadblock. Task and threadblock are 1:1.
- A `*TaskTemplate` is an optional materialized subset of Task fields reused across regular grid coordinates.
- A threadblock is one cooperating group of GPU threads; Metal calls it a threadgroup.
- A grid is all threadblocks launched by one kernel dispatch.

`Task` and `TaskTemplate` are logical-work vocabulary, not CUDA/Metal launch primitives. A Task may execute one or more
Tile steps, including repeatedly advancing one tensor tile along an ordered axis. Put the path-specific Tile/Task/Grid
contract beside the Rust recorder or source owner, not on generic model metadata shared by multiple paths.

Do not add Rust/MSL structs or variables merely to represent a threadblock, grid, or fully derived Task. At the owning
Rust/MSL boundary, comments still list every logical Task coordinate and its source. Mark grid-derived coordinates
explicitly. If every field is derived, materialize no Task, TaskTemplate, or ABI buffer.

Name a Task only when a threadblock owns a coherent semantic work unit. For a flat elementwise/map dispatch where
threadblock grouping is incidental tuning, describe the tensor map and grid without inventing a `*Task` noun.

For irregular work, materialize only fields that cannot be derived regularly. One TaskTemplate combined with its
grid-derived coordinates must produce exactly one logical Task for exactly one threadblock.

For a map/reduce pipeline, name the map result `*PartialOutput` and the fully reduced result `*Output`. Reduce metadata
names the partial outputs it selects rather than the TaskTemplate that originally produced them. Component-specific
coordinates, ABI records, and cumulative-offset examples belong in the owning component doc and source boundary.

## Runtime shapes and persistent layouts

Keep runtime/replay shapes separate from init-time capacity and storage layouts.

`*ReplayShape` contains only values describing the current recorded execution, such as active token count, active
request count, and runtime partition count. It does not contain init-time capacities, persistent-buffer strides, or
storage coordinates.

An object whose main purpose is describing persistent tensor dimensions is named `*Layout`, not `*Shape`. For example,
a GQA page-ID table stored as `[num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]` exposes exactly:

```text
num_req_slots
num_gqa_layers
num_blocks
num_page_ids_per_block
```

Backend command shapes may carry a nested Layout when lowering requires persistent dimensions as command constants or
source specialization. Layer-local coordinates use the corresponding name: `model_layer_index` for the full model
stack and `gqa_layer_index` for the compact GQA table. Avoid names that encode one caller's interpretation of a tensor
axis, such as `num_page_table_layers`, when the stored dimension is simply layers.

Layouts and persistent state store independent model inputs, resource handles, and lifecycle data. Do not cache a count
uniquely derived from adjacent dimensions, dtype, or `page_bytes`; derive it at the typed-index or raw-byte boundary that
needs it. Forward paths may borrow initialized layouts but must not derive capacity from the current batch.

Logical structure comes first. Stable component identity, dtype and tuning choices, and meaningful resource views are
not redundant merely because one model uses one value. Remove a wrapper only when it forwards one-to-one and owns no
independent lifetime, slicing, resource, or semantic branch.

## Replay and asynchronous resource safety

Replay keys contain only facts that change recorded command structure, dispatch topology, static geometry, scratch
extent, or a required algorithm choice. Request slots and dynamic values such as valid counts, page IDs, offsets,
temperature, top-p, seed, sample position, and sampling domain belong in batch metadata or submission arguments. Do not
expand the cache key merely to avoid implementing a typed dynamic input.

Place synchronization at the exact consumer dependency. Missing barriers are correctness bugs; global
every-command barriers are not an acceptable substitute for identifying RAW, WAR, WAW, aliasing, or semantic phase
boundaries.

Asynchronous backend resources remain alive and resident until host-visible completion proves the submission finished.
Do not reset or reuse a command allocator, scratch owner, parameter buffer, or replay-local resource while an in-flight
submission can still reference it. Backend-specific completion and residency mechanisms belong in the backend
documentation and implementation.

Detailed replay composition belongs in [`executor.md`](executor.md); the Metal object model, completion mechanism, and
residency ownership belong in the Metal backend
[`README`](../crates/inference-backend-metal/README.md).

## Optimization correctness

An optimization must preserve the semantic boundary it replaces: tensor outputs, routing decisions, probability
distributions, state versions, and lifecycle effects. A speculative path must not publish unverified future state, and
an alternate sampler must operate on the same transformed distribution as the production path it accelerates. Exact
sampling, state, and component-path contracts remain in the matching current component document.

Kernel and composition policy is shape-sensitive. Preserve strong small- and large-shape paths instead of replacing
them universally with a locally faster specialization. A primitive microbenchmark is not sufficient evidence for a
model-layer or end-to-end change; validate the production composition, metadata updates, barriers, scratch lifetime,
and output ownership at representative shapes.

Establish output, route, probability, or state parity before making a performance claim. For speculative execution,
also keep the deterministic workload trajectory fixed or report proposals, sampled tokens, accepted tokens/chunks, and
acceptance efficiency; throughput from a different trajectory is not a like-for-like executor or kernel comparison.
The complete verification ladder and performance-evidence record live in
[`executor_benchmarks.md`](executor_benchmarks.md).

## Public API

Keep public surface minimal. Items remain private unless an external caller intentionally needs them; do not use
`pub(crate)` or `pub(super)`.

Do not export planning structs, kernel metadata, workaround modules, scratch internals, or backend-local tables merely
for convenience. A benchmark needing private internals is a reason to discuss a bench helper API, not to widen the
production API automatically. Do not reshape production `src` only to make a benchmark easier.

Keep backend details behind backend APIs. Metal components own kernels, dispatch parameters, tile scratch, and runtime
resource bindings. Model executors own model semantics, persistent model/request buffers, and component wiring; they do
not expose backend tiling or kernel-local tables in model-level APIs.

## Test style

Every Rust test function starts with `test_` and protects one explicit behavioral, correctness, ownership, or lifecycle
contract. Let the module path name the subject; use the shortest unambiguous case name inside that module, such as
`gqa_attention::tests::test_ragged_random` or `stream::tests::test_submission_drop`.

Exercise the production owner API and real contract boundary. A helper-only test does not protect the owner's
setter/record/submission wiring; retain it only when the helper owns reusable semantics that cannot be covered clearly
through production.

Use `fixture_*` for test data/builders and `reference_*` for slow or CPU oracles. Reserve `test_*` for test entry points.
Do not retain constructor-only tests already covered by stronger execution tests or tests of derived Rust behavior such
as `PartialEq`.

Optimized numerical paths should have concise fixed-input and random-input tests against a CPU/slow reference.
Reference implementations favor obvious correctness and clean clippy output: use a named input struct instead of a long
argument list, and prefer iterator/enumerate forms unless index arithmetic is the behavior under test.

Test-only constructors and fixtures may remain simple when production construction already enforces the contract. Do
not widen or reshape them solely to mirror production. Backend-specific execution constraints, such as serial Metal
testing, stay in the corresponding backend/executor document.

## General source clarity

Prefer concrete nouns such as `file_name_for`, `file_path`, and `mapped_files` over overloaded shorthand unless that
shorthand is a stable module contract. When a callback can keep, transform, or remove an item, use an explicit action
enum instead of encoding control flow in `Option<T>`; reserve `Option<T>` for genuinely optional values.

Comments explain non-obvious ownership, ordering, units, or backend constraints. They do not narrate mechanics already
visible in the code.
