# Engineering Conventions

This document gives repository-wide engineering conventions:

- Naming
- Runtime shapes and storage layouts
- Replay and resource safety
- Optimization
- Public APIs
- Test style

[`high_level.md`](high_level.md) defines architecture and ownership boundaries. Each component document defines its
tensor terms, source owners, and execution paths.

## Naming and coordinate domains

Recommendation: Names show their domain semantics. Use established component abbreviations as type prefixes. Examples
include `GQA*` and `GDN*`.

Use semantic names at model, layer, component, and non-matmul operator boundaries. Examples include:

- `num_tokens`
- `num_routes`
- `num_input_vectors`
- `num_experts_per_token`

Use `m`, `n`, and `k` only for a low-level matrix-multiplication shape. Keep semantic names for routing, gather, page,
and state coordinates. Keep these names when the implementation sends the values to a matmul.

Use `num_*` for a valid typed work count. Use `total_*` for a padded dispatch, replay, scratch, or capacity extent.
Name the item that the value counts.

Do not use `element` or an unqualified `size` for a tensor or domain. Use `*_bytes` for these byte values:

- Raw allocation lengths
- Binding offsets
- Address arithmetic
- Byte copies

Use unsigned integers for counts, slots, and IDs. Use signed integers for real negative values, sentinels, or imported
ABI contracts.

Use checked `u64` values for host byte offsets and address arithmetic. Use `ulong` for flattened Metal addresses.
Convert to `usize` only at a Rust slice, pointer, or Objective-C API boundary.

Keep bounded local counts and indices as Rust `u32` or Metal `uint`. A large allocation does not require 64-bit local
loops or tensor coordinates.

Configuration names identify their owner:

```text
*ExecutorConfig   model-executor initialization
*RuntimeConfig    runtime core
*ServerConfig     service/application initialization
```

Checkpoint schemas identify their format. For example, use `HFGenerationConfig` instead of an unqualified subsystem
name.

Recommendation: Use precise coordinates:

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

Recommendation: Do not use `absolute_*` unless another coordinate system exists. Do not use an unqualified `offset` or
`context`. Name the owner or unit.

This rule also applies to names such as `candidate_lens` and `commit_len`.

Use `flat` only for a tensor or coordinate from flattened per-request token sequences. Per-request `token_indices` can
expand into `flat_token_indices`.

Keep request-slot IDs as `req_slots` when they repeat for each token. `cu_tokens` already identifies flattened segment
boundaries. It does not need a `flat` prefix.

Use `cu_<items>` only for a monotonic cumulative-count table with `N + 1` entries. Adjacent values select this
half-open segment for logical owner `i`: `cu_items[i]..cu_items[i + 1]`.

The owning comment names the owner and the counted item. Do not use `cu_` for these values:

- Ordinary coordinates
- Byte offsets
- Capacities
- Non-cumulative metadata

Domain-standard abbreviations are appropriate when the owning boundary makes them unambiguous. Attention code may use
`q`, `k`, `v`, `qk`, and `kv`.

Attention code may also use projection names such as `qkv`, `qgkv`, and `qkvabz`. Do not reuse these names for unrelated
concepts.

Recommendation: Outside an established domain, use the complete semantic noun.

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

`T` identifies a token dimension. `H` identifies a head dimension. Add `_tile` to the tiled axis.

Do not introduce the ambiguous names `Bq` or `Bkv`. Outside the SDPA tensor or kernel boundary, use `token_tile`.
Use `q_*` only for the Q tensor and its dimensions.

## GPU work vocabulary

Keep mathematical decomposition separate from launch topology:

- A `*Tile` is the smallest named matmul-like unit at that component boundary. It is not a launch object.
- A `*Task` is the full logical work for one threadblock. A Task and a threadblock have a 1:1 relationship.
- A `*TaskTemplate` is an optional stored subset of Task fields. Regular grid coordinates reuse this subset.
- A threadblock is one cooperating group of GPU threads. Metal calls it a threadgroup.
- A grid is all threadblocks launched by one kernel dispatch.

`Task` and `TaskTemplate` identify logical work. They are not CUDA or Metal launch primitives.

A Task can run one or more Tile steps. It can move one tensor tile repeatedly along an ordered axis.

Put each path-specific Tile, Task, and Grid contract beside its Rust recorder or source owner. Do not put this contract
on generic model metadata that multiple paths share.

Do not add Rust or MSL items only to represent a threadblock, grid, or fully derived Task. This rule applies to structs
and variables.

At the owning Rust or MSL boundary, comments list each logical Task coordinate and its source. Identify grid-derived
coordinates explicitly. Do not store a Task, TaskTemplate, or ABI buffer when all fields are derived.

Name a Task only when a threadblock owns one semantic work unit. Some flat elementwise or map dispatches use
threadblock grouping only for tuning.

For these dispatches, describe the tensor map and grid. Do not add a `*Task` name.

For irregular work, store only fields that regular rules cannot derive. One TaskTemplate and its grid coordinates must
produce one Task for one threadblock.

For a map/reduce pipeline, use `*PartialOutput` for the map result. Use `*Output` for the fully reduced result.

Reduce metadata identifies its selected partial outputs. It does not identify the TaskTemplate that produced them.
Put component coordinates, ABI records, and cumulative-offset examples in the owning component document and source.

## Runtime shapes and persistent layouts

Keep runtime or replay shapes separate from initialization capacity and storage layouts.

`*ReplayShape` contains only values that describe the current recorded execution. Examples include active token,
request, and runtime-partition counts.

It does not contain initialization capacities, persistent-buffer strides, or storage coordinates.

Use `*Layout` for an object that primarily describes persistent tensor dimensions. Do not name this object `*Shape`.

For example, a GQA page-ID table stores
`[num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]`. It exposes exactly these fields:

```text
num_req_slots
num_gqa_layers
num_blocks
num_page_ids_per_block
```

Backend command shapes may contain a nested Layout. This permission applies when lowering needs persistent dimensions
as command constants or source specialization.

Layer-local coordinates use the matching name:

- Use `model_layer_index` for the full model stack.
- Use `gqa_layer_index` for the compact GQA table.

Recommendation: Do not encode one caller's tensor-axis interpretation in a name. For example, do not use
`num_page_table_layers` when the stored dimension contains layers.

Layouts and persistent state store independent model inputs, resource handles, and lifecycle data. Do not cache a count
that adjacent dimensions, a data type, or `page_bytes` uniquely determine.

Derive that count at the typed-index or raw-byte boundary that needs it. Forward paths may borrow initialized layouts.
They must not derive capacity from the current batch.

Logical structure has priority. Stable component identity and explicit data-type or tuning choices are not redundant.
Meaningful resource views are also not redundant because one model uses one value.

Remove a wrapper only when it forwards one-to-one. The wrapper must own no independent lifetime, slice, resource, or
semantic branch.

## Replay and asynchronous resource safety

Replay keys contain only facts that change these items:

- Recorded command structure
- Dispatch topology
- Static geometry
- Scratch extent
- A necessary algorithm choice

Put request slots and dynamic values in batch metadata or submission arguments. Dynamic values include these examples:

- Valid counts
- Page IDs and offsets
- Temperature, top-p, and seed
- Sample position and sampling domain

Do not expand the cache key to prevent the implementation of a typed dynamic input.

Put synchronization at the exact consumer dependency. A missing barrier is a correctness bug. A global barrier for
each command is not an acceptable replacement.

Identify each RAW, WAR, WAW, aliasing, or semantic-phase boundary.

Keep asynchronous backend resources alive and resident until host-visible completion proves that the submission
finished. An in-flight submission can still refer to these resources.

During this time, do not reset or reuse these resources:

- A command allocator
- A scratch owner
- A parameter buffer
- A replay-local resource

Put backend-specific completion and residency mechanisms in the backend documentation and implementation.

[`executor.md`](executor.md) defines detailed replay composition. The Metal backend
[`README`](../crates/inference-backend-metal/README.md) defines the Metal object model, completion mechanism, and
residency ownership.

## Optimization correctness

An optimization must preserve the semantic boundary that it replaces:

- Tensor outputs
- Routing decisions
- Probability distributions
- State versions
- Lifecycle effects

A speculative path must not publish unverified future state. An alternative sampler must use the same transformed
distribution as the production path.

Each current component document defines its exact sampling, state, and component-path contracts.

Kernel and composition policies depend on shape. Keep strong paths for small and large shapes. Do not replace all paths
with one locally faster specialization.

A primitive microbenchmark is not sufficient evidence for a model-layer or end-to-end change. At representative
shapes, validate these items:

- Production composition
- Metadata updates and barriers
- Scratch lifetime
- Output ownership

Establish output, route, probability, or state parity before you make a performance claim. For speculative execution,
keep the deterministic workload trajectory fixed.

If the trajectory changes, report these values:

- Proposals
- Sampled tokens
- Accepted tokens and chunks
- Acceptance efficiency

Throughput from a different trajectory is not a like-for-like executor or kernel comparison. See
[`executor_benchmarks.md`](executor_benchmarks.md) for the verification and performance-evidence rules.

## Public API

Keep the public surface small. Items remain private unless an external caller needs them. Do not use `pub(crate)` or
`pub(super)`.

Treat a model role as a structural ownership boundary. Main, MTP, and DSpark use different concrete layer types. This
rule applies when they use some of the same operators.

Share true leaf components and utilities across roles. Do not share a structural layer facade that mixes their
execution graphs.

Production source and benchmarks use crate-absolute paths for sibling and ancestor imports. Use `use super::...` only
in tests.

A trait can be the production interface for a concrete type. In this case, put the method implementation directly in
the trait implementation.

Keep constructors and other operations in the inherent implementation. Do not duplicate trait methods as inherent
methods. Do not forward between duplicate methods.

Do not export these items only for convenience:

- Planning structs
- Kernel metadata
- Workaround modules
- Scratch internals
- Backend-local tables

A benchmark can need private internals. Discuss a benchmark helper API before you widen the production API. Do not
change production `src` only to make a benchmark easier.

Keep backend details behind backend APIs. Metal components own kernels, dispatch parameters, tile scratch, and runtime
resource bindings.

Model executors own model semantics, persistent model or request buffers, and component wiring. They do not expose
backend tiling or kernel-local tables in model APIs.

## Commits

Recommendation: One commit owns one coherent architectural idea. While history is local, amend or rebase a fix into
its owning commit.

After you publish or share a commit, preserve the history. Use a focused follow-up commit.

Use a concise subject that starts with a lowercase verb. Focus the subject on the main purpose. Keep standard acronyms:

```text
add HTTP chat completions
clean tokenizer streaming
```

Do not list related changes in the subject. When a body is useful, give a few high-level points. Do not restate the
per-file diff.

Recommendation: Use reviewable and compilable boundaries when practical. An intentional local history split can
temporarily fail to compile.

Delivered history must be self-consistent.

## Test style

Each Rust test function starts with `test_`. It protects one behavior, correctness, ownership, or lifecycle contract.

Let the module path identify the subject. Inside the module, use the shortest unambiguous case name. Examples include:

- `gqa_attention::tests::test_ragged_random`
- `stream::tests::test_submission_drop`

Exercise the production owner API and its real contract boundary. A helper-only test does not protect owner wiring for
setters, recording, or submission.

Keep a helper-only test only when the helper owns reusable semantics that production tests cannot cover clearly.

Use `fixture_*` for structured test data or builders. Use `reference_*` for slow or CPU oracles.

Use a simple `new_<noun>` name for a small test-local mock constructor. For example, use `new_task`. Use `test_*` only
for test entry points.

List test entries before local helper functions. Keep the helpers at the end of the test module.

Do not keep constructor-only tests when stronger execution tests cover them. Do not test derived Rust behavior such as
`PartialEq`.

Keep a small and focused test module inline. Move longer test coverage to a sibling `*_test.rs` file.

The production owner declares this file under `cfg(test)`. Do not create a separate file for only a few concise cases.

Use `unwrap()` for test setup and coordination failures that are not the tested behavior. Do not use verbose
`expect(...)` messages for these failures.

Use these Mockall call counts:

- `.once()` for exactly one call
- `.times(n)` for repeated calls
- `.never()` for forbidden calls

Omit the call count only when it is intentionally unrestricted. Use `return_const` for a cloneable fixed result. Use
`return_once` for a one-use or move-capturing result.

Use `returning` for a repeatable computed result.

Recommendation: Optimized numerical paths have concise fixed-input and random-input tests against a CPU or slow
reference.

Recommendation: Reference implementations make correctness clear and keep Clippy output clean.

Use a named input struct instead of a long argument list.

Recommendation: Use iterators and `enumerate` unless index arithmetic is the tested behavior.

Test-only constructors and fixtures may remain simple when production construction enforces the contract. Do not change
them only to mirror production.

Put backend execution constraints in the matching backend or executor document. Serial Metal testing is one example.

## General source clarity

Recommendation: Use concrete nouns such as `file_name_for`, `file_path`, and `mapped_files`. Do not use overloaded
shorthand unless it is a stable module contract.

A callback can keep, transform, or remove an item. For this behavior, use an explicit action enum. Do not encode the
control flow in `Option<T>`.

Use `Option<T>` only for an optional value.

Comments explain ownership, ordering, units, or backend constraints that are not obvious. They do not describe
mechanics that the code shows.
