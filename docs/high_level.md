# High-Level Engineering Guidance

This document defines shared repo rules and the boundary between **runtime core** and the **model executor**.

## Doc style

Docs should be concise, readable, informative, and source-aligned.

Keep current-component docs about current `src`. Update the matching `docs/executor_*.md` when component source layout
or default paths change. Put active follow-ups in `docs/future_work.md`; promote durable repository-wide rules into
`docs/engineering_conventions.md` and component-specific findings into the owning component document. Do not keep broad
historical/perf archaeology directories.

Give each document one primary job: the top-level README establishes the project mental model, architecture docs define
ownership and data flow, component docs describe current source and component-specific validation, and workflow docs
own cross-cutting commands, verification rules, and operations. Link across those boundaries instead of copying the
same contract into several files.

## Core vs executor

| Layer        | Owns                                                                                                                            | Must not own                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Runtime core  | scheduling, request lifecycle, token/block metadata, KV/state page allocation/free, page ownership, cache/state notifications                    | model tensor layout, backend details, GQA/GDN/MLP internals      |
| Model executor | model layout parsing, backend tensor/state objects, GQA, Gated DeltaNet, dense MLP, MoE, component-local page interpretation              | scheduling policy, global lifecycle, page allocation/free policy |
| Metal backend  | Metal FFI, device/buffer/kernel/stream/runtime primitives, Apple Silicon component kernels consumed by the model executor                        | request scheduling, global lifecycle, page allocation/free policy |

Core provides stable metadata and page IDs. Executor and backend components consume that contract to run the model.

## Shared hard constraints

Use `panic!`, `assert!`, or `debug_assert!` for internal invariant violations and impossible contract states. Use release
`assert!` only at init-time, one-time structural or ownership boundaries, or contracts whose enforcement is absolutely
necessary in release. Repeated internal bug checks that would add release hot-path noise belong in `debug_assert!`; tests
and debug builds provide their coverage. Classify each check by lifecycle and cost instead of converting them mechanically.

Use recoverable `Err(Exception::custom(...))` only for user input, model loading, runtime failures, MLX failures, IO failures, or other genuinely recoverable errors.

Do not use `pub(crate)` or `pub(super)`. Use private items by default; use plain `pub` only for intentional API surface.

Run formatting as `cargo +nightly fmt`.

Before handing off broad Rust changes, run the compile-only repository gates:

```sh
cargo +nightly fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo +nightly clippy --workspace --all-targets --all-features -- -D warnings
git diff --check
```

Run focused tests at the narrowest production owner. Run Metal/GPU commands one at a time with explicit cross-process
coordination; do not use a parallel workspace test run as a GPU verification gate.

In Rust module roots, list normal `use` imports before module declarations. Prefer exposing child modules with
`pub mod` over re-exporting child items through the parent module. A child directory can define public items, but the
parent should not become a broad facade for that child API.

```rust
use some_crate::Thing;

pub mod direct;
pub mod replay;
```

Do not reshape production `src` only for benchmark convenience.

Production `src` defines the ownership, lifecycle, and API contract. Tests and
benches validate that contract and exercise its real paths; they must not add
optional production state, compatibility branches, or abstractions solely for
their own construction convenience. Work source-first: add a boundary test only
when it distinguishes a concrete structural risk or production invariant, not
as a mechanical response to every edit.

Keep production source and tests concise. Remove redundant derived state,
one-off forwarding helpers, one-field wrappers, and repeated setup abstractions
unless they express a real owner, invariant, lifecycle, reusable operation, or
the peer-component symmetry described below.

## Delegated work

When a child task owns implementation or writing, the parent must not duplicate that work. The parent owns coordination:
scope boundaries, shared-checkout and integration state, GPU/resource serialization, cross-task conflict prevention,
status collection, and final integration and verification. The child reports progress, questions, and results; the
parent remains available for those callbacks instead of block-polling or busy-waiting.

## Ownership style

Start by naming managed objects before adding APIs.

如无必要，勿增实体. Do not add wrappers, structs, fields, enums, helpers, buffers, scratch owners, compatibility paths,
validation layers, or tests unless they own a distinct semantic concept, invariant boundary, resource/lifecycle,
reusable operation, or materially improve clarity. Prefer direct data flow. This is not mechanical entity-count
minimization: the peer-component symmetry rule below is a first-class maintenance and cognitive-load benefit, and a
small symmetric entity is preferable when it makes peer ownership, data flow, or lifecycle easier to transfer.

Keep stable identity and explicit datatype/tuning choices. Derive facts and capacities from the model/config dimensions
that own them; do not store or repeatedly pass duplicates or convenient magic limits.

Keep shape validation structurally symmetric across peer backend components. Use the same readable sequence: validate
positive dimensions and relationships; compute named derived counts with checked arithmetic; assert the shader domain;
validate invocation buffer ranges; dispatch with the same derived count. A `u32` shader count must reject `2^32`, while
a `u32` element-index domain may contain exactly `2^32` elements because its maximum index is `u32::MAX`. Name the
assertion and boundary test after `count` or `index` so this distinction is explicit rather than module-specific lore.

Prefer small object-owned surfaces:

```text
get / get_ref / get_mut
set
push / pop
reset
```

`reset` means resetting the whole managed object at that scope. If only one field changes, use `set_<field>`.

Lifecycle belongs to the owner. Avoid hot-path cleanup calls that compensate for unclear ownership.

Use lifecycle verbs for lifecycle APIs:

```text
new         construct an owner without touching external resources unless that is the owner contract
load        load the named object/resource at that API scope
unload      release the same object/resource shape that load acquires
load_all    load every resource in the owner's current domain
unload_all  release every loaded resource in the owner's current domain
```

Keep lifecycle verbs symmetric within the same owner. Avoid APIs where
`load()` loads one kind of state but `unload()` releases another. Prefer
`load(file_name)` / `unload(file_name)` and `load_all()` / `unload_all()` when
the owner manages many resources.

Do not make callers pass derived contextual inputs repeatedly. For example, if
a model directory has a fixed file layout, expose `load(model_dir)` or
`from_model_dir(model_dir)` instead of requiring callers to build
`model.safetensors.index.json` paths at every call site.

Keep data and resource ownership separate. A mapping/index object should
represent the mapping only; path resolution, mapped-file caches, and resource
release belong to the store/owner that has the base directory and lifecycle.
Prefer explicit composition such as `new(model_dir, index)` plus
`from_model_dir(model_dir)` convenience constructors when both are useful.
Do not hide a second resource lifecycle behind the same `load`/`unload` pair.

If a condition is guaranteed by a core/executor contract, check it at the boundary where violations become visible instead
of adding defensive recovery code that hides lifecycle bugs. Use a release `assert!` there only when that boundary is
init-time, one-time structural/ownership, or otherwise absolutely necessary to enforce in release; use `debug_assert!`
for repeated internal paths.

Distinguish an invariant check from a data-flow bug. If the existing data flow already guarantees a condition, add the
smallest useful check at the owning boundary; choose `assert!` or `debug_assert!` by the lifecycle and release-cost rule
above. Do not add types, branches, grouping, or recovery state for an impossible case. Change structure only when valid
inputs can violate the current data flow or the owner cannot actually guarantee the condition.

## Design style

Prefer first-principles contracts over compatibility patches. Define the caller-visible inputs, outputs, ownership, and
state transitions first; then choose the backend implementation that satisfies that contract. Do not preserve a known
wrong implementation behind a feature flag, environment variable, or fallback once the correct default path is verified.

Keep interfaces symmetric across related components. For a pair such as sampling/rejection sampling, GQA/GDN state
tables, or dense MLP/MoE paths, use parallel names for shapes, inputs, outputs, scratch, kernels, and record methods
unless the semantics truly differ.

Structural and API symmetry is a maintenance tool, not cosmetic consistency. It lets a reader transfer ownership,
lifecycle, data-flow, test, and profiling knowledge between peer components instead of reconstructing a new mental
model; needless asymmetry increases maintenance and cognitive burden. This benefit takes precedence over mechanical
entity-count minimization when a small symmetric entity makes the peer contract clearer. Deviate only for a concrete
semantic or resource-lifecycle distinction, and state that distinction at the owning boundary.

Keep backend details behind backend APIs. Metal components may own kernels, dispatch parameters, tile scratch, and
runtime resource bindings. Model executors should own model semantics, persistent model/request buffers, and wiring
between components, but should not expose backend tiling, temporary scratch, or kernel-local tables in model-level APIs.

## Model / layer / operator boundary

Model executor code should preserve this ownership hierarchy:

```text
Model
  owns whole-model orchestration:
    embedding
    main layers
    final norm / unembedding
    sampling / rejection sampling
    MTP modules
    request state, replay caches, and stage ordering

Layer / Component
  owns semantic model computation:
    GQA, GDN, dense MLP, MoE
    embedding, unembedding, sampling, rejection sampling
    MTP layer/module semantics
    model-level shape/input/output contract
    weights, request metadata, state/page metadata at that semantic boundary
    record support for replay composition

Operator
  owns backend execution:
    kernels and dispatch shapes
    buffers, weights, scratch binding structs
    resource read/write/read_write declarations
    command barriers and backend dependency requirements
    replay recording

Backend Command
  owns one backend dispatch:
    exactly one backend pipeline or executable function
    resource and parameter bindings
    backend execution geometry
    consumer-side barrier attribute
```

The model answers "which whole-model stage runs next?". A layer/component
answers "what semantic model computation is this and what state does it own?".
An operator answers "how does this tensor operation lower on this backend?".
A backend command is one concrete dispatch produced by that lowering. Metal
represents it as one compute pipeline dispatch in an ICB slot; another backend
may use a kernel launch or graph node.
One component may emit multiple operators, and one operator may emit multiple
commands.

Do not let model code bind backend resources directly when a semantic
layer/component can own that boundary. Do not let backend operators encode
model-specific request semantics when a layer/component can translate those
semantics into backend buffers.

Traits should enforce this boundary where drift is likely:

```text
Layer                       model-executor semantic component contract
ReplayLayer                 replay contract over the same Layer input/output
Recorder                    backend replay recording contract
Operator                    backend recordable execution contract
```

Concrete types remain preferred for implementation clarity. Use traits as
boundary constraints, not as blanket dynamic-dispatch abstractions. Semantic
layers should record through `inference-executor-core::backend::recorder::Recorder`, not
directly depend on a concrete backend batch builder. The model replay-cache
boundary may lower those semantic records into the backend Metal recorder while
recording a cached replay; lower-level builder details should not leak into
semantic component inputs or public executor APIs.

Prefer direct calls when their related invariants are adjacent in the caller.
Do not introduce extent, state, or planner wrappers that merely repackage a few
arguments without adding ownership or contract meaning.

Reuse resources at the narrowest correct owner. Allocate immutable weights and stable kernels at init time. Allocate
component/model scratch once at the owner that can safely time-share it. Do not add GPU-to-GPU copies when the producer
can write directly into the consumer's persistent destination buffer.

Runtime hot paths should do only runtime work: write current token/page/state metadata, submit cached or recorded work,
and read the small CPU-visible outputs required by the runtime contract. Relayout, path resolution, kernel selection
tables, and reusable buffer allocation belong at init or cache-build time.

## Engineering conventions

[`engineering_conventions.md`](engineering_conventions.md) is the detailed source of truth for naming, coordinate and
numeric domains, GPU Tile/Task vocabulary, runtime shapes versus persistent layouts, public API, and test style. This
document retains only architecture, ownership, lifecycle, performance-evidence, and completion rules.

## Performance evidence

Performance claims must identify the exact commit and dirty state, machine and
environment, model, command, baseline, current result, and verdict. Compare the
same work rather than throughput alone. For speculative decoding, also report
sampled tokens, chunks or decisions, and acceptance efficiency; a throughput
change accompanied by a different deterministic acceptance trajectory is not
by itself a pure executor or kernel regression.

## Definition of done

A change is done only when:

```text
target behavior is implemented
default path uses the intended logic
obsolete fallback or experimental path is removed or isolated
slow/reference correctness coverage exists where applicable
relevant cargo check/test issues are fixed
relevant benches are runnable
profile keys are coherent
public API did not grow unnecessarily
runtime core vs executor ownership is still clean
current docs are updated if behavior/layout changes
```

## Common anti-patterns

Avoid:

```text
scheduler logic inside executor components
model-specific ownership of reusable component semantics
runtime core parsing model tensor layout
bench-driven production architecture changes
defensive state machines that hide contract violations
temporary workaround code behind polished APIs
feature/env switches that keep obsolete production paths alive
GPU copies between two buffers when the producer can write the final destination
random helper buckets
large inline test modules
public exports of kernel/planning internals
claiming completion before default path, tests, benches, and cleanup are verified
```
