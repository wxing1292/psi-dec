# High-Level Engineering Guidance

This document gives shared repository rules. It also defines the boundary between the **runtime core** and the
**model executor**.

## Doc style

Use the [technical English style](technical_english.md) for repository prose.

Recommendation: Keep the text concise, clear, useful, and aligned with the source.

Current-component documents describe the current `src`. Update the matching `docs/executor_*.md` when a source layout
or default path changes.

Put active follow-up work in `docs/future_work.md`. Put durable repository rules in
`docs/engineering_conventions.md`. Put component findings in the document that owns the component.

Do not create broad directories for historical or performance archaeology.

Give each document one primary purpose:

- The top-level README gives the project model.
- Architecture documents define ownership and data flow.
- Component documents describe the current source and validation.
- Workflow documents contain shared commands, verification rules, and operations.

Link to the document that owns a contract. Do not copy the same contract into multiple documents.

## Core vs executor

| Layer        | Owns                                                                                                                            | Must not own                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Runtime core  | scheduling, request lifecycle, token/block metadata, KV/state page allocation/free, page ownership, cache/state notifications                    | model tensor layout, backend details, GQA/GDN/MLP internals      |
| Model executor | model layout parsing, backend tensor/state objects, GQA, Gated DeltaNet, dense MLP, MoE, component-local page interpretation              | scheduling policy, global lifecycle, page allocation/free policy |
| Metal backend  | Metal FFI, device/buffer/kernel/stream/runtime primitives, Apple Silicon component kernels consumed by the model executor                        | request scheduling, global lifecycle, page allocation/free policy |

The runtime core provides stable metadata and page IDs. Executor and backend components use this contract to run the
model.

## Shared hard constraints

Use `panic!`, `assert!`, or `debug_assert!` for internal invariant violations and impossible contract states.

Use a release `assert!` only at these boundaries:

- Initialization
- A one-time structure or ownership check
- A contract that release code must enforce

Use `debug_assert!` for repeated internal checks that add noise to a release hot path. Tests and debug builds cover
these checks. Classify each check by its lifecycle and cost. Do not convert checks mechanically.

Use the shared typed `Error` for a recoverable failure. Select the variant from the result that the caller observes.
Examples include `InvalidArgument`, `Unavailable`, and `Internal`. Use assertions or panics for internal invariant
violations.

Do not use `pub(crate)` or `pub(super)`. Keep items private by default. Use plain `pub` only for an intentional API.

Run formatting as `cargo +nightly fmt`.

Match the verification work to the change stage:

```text
development       focused unit tests at the production owner
integration       relevant package tests plus repository compile gates
final acceptance  the real production path through an external caller
```

Recommendation: For final acceptance, start a release server when a change affects a server or RPC path. Exercise the
protocol from outside the process. Make sure that the server stops cleanly.

Do expensive GPU checks during final acceptance. Do them earlier only when they answer a correctness question.

Before handing off broad Rust changes, run the repository compile gates:

```sh
cargo +nightly fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo +nightly clippy --workspace --all-targets --all-features -- -D warnings
git diff --check
```

Run Metal and GPU commands one at a time. Coordinate these commands across processes. Do not use parallel workspace
tests as a GPU verification gate.

In a Rust module root, list normal `use` imports before module declarations.

Recommendation: Expose a child module with `pub mod`. Do not re-export child items through the parent without a clear
API need.

A child directory can define public items. Recommendation: Do not make the parent a broad facade for the child API.

```rust
use some_crate::Thing;

pub mod direct;
pub mod replay;
```

Do not change production `src` only to make a benchmark easier.

Production `src` defines the ownership, lifecycle, and API contract. Tests and benchmarks validate that contract
through production paths. They must not add optional production state, compatibility branches, or test-only
abstractions.

Start with the production source. Add a boundary test when it identifies a structural risk or a production invariant.
Do not add a test as a mechanical response to each edit.

Keep production source and tests concise. Remove redundant derived state and one-use forwarding helpers. Remove
one-field wrappers and repeated setup abstractions.

Keep an item when it expresses one of these concepts:

- A real owner
- An invariant
- A lifecycle
- A reusable operation
- The peer-component symmetry in this document

## Delegated work

When a child task owns work, the parent must not do the same work. The parent coordinates the tasks.

The parent owns these items:

- Scope boundaries
- Shared-checkout and integration state
- GPU and resource serialization
- Cross-task conflict prevention
- Status collection
- Final integration and verification

The child reports progress, questions, and results. The parent remains available for these reports. The parent does not
use block polling or busy waiting.

## Ownership style

Name the managed objects before you add APIs.

Add an entity only when it has a necessary purpose. This rule applies to these entities:

- Wrappers
- Structs
- Fields
- Enums
- Helpers
- Buffers
- Scratch owners
- Compatibility paths
- Validation layers
- Tests

An entity has a necessary purpose when it owns a distinct concept, invariant, resource, lifecycle, or reusable
operation. An entity can also be necessary when it materially improves clarity.

Recommendation: Use direct data flow. Do not minimize the number of entities mechanically.

Peer-component symmetry reduces maintenance work and cognitive load.

Recommendation: Use a small symmetric entity when it makes peer ownership, data flow, or lifecycle clear.

Keep stable identity and explicit data-type or tuning choices. Derive facts and capacities from the model or
configuration dimensions that own them. Do not store duplicates, pass duplicates, or use convenient magic limits.

Use the same shape-validation structure across peer backend components:

1. Validate positive dimensions and relationships.
2. Calculate named derived counts with checked arithmetic.
3. Assert the shader domain.
4. Validate the invocation-buffer ranges.
5. Dispatch with the same derived count.

A `u32` shader count must reject `2^32`. A `u32` element-index domain can contain exactly `2^32` elements. Its maximum
index is `u32::MAX`. Name each assertion and boundary test for its `count` or `index` domain.

Recommendation: Use small, object-owned surfaces:

```text
get / get_ref / get_mut
set
push / pop
reset
```

`reset` changes the full managed object at that scope. If only one field changes, use `set_<field>`.

The owner controls the lifecycle.

Recommendation: Do not use hot-path cleanup calls to compensate for unclear ownership.

Use lifecycle verbs for lifecycle APIs:

```text
new         construct an owner without touching external resources unless that is the owner contract
load        load the named object/resource at that API scope
unload      release the same object/resource shape that load acquires
load_all    load every resource in the owner's current domain
unload_all  release every loaded resource in the owner's current domain
```

Keep lifecycle verbs symmetric in one owner.

Recommendation: Do not let `load()` acquire one state type while `unload()` releases a different type.

Recommendation: When an owner manages many resources, use these symmetric pairs:

- `load(file_name)` and `unload(file_name)`
- `load_all()` and `unload_all()`

Do not make callers pass the same derived context many times. A model directory can have a fixed file layout. In this
case, expose `load(model_dir)` or `from_model_dir(model_dir)`. Do not make each caller build a
`model.safetensors.index.json` path.

Keep data ownership separate from resource ownership.

Recommendation: A mapping or index object represents only the mapping.

The store owns path resolution, mapped-file caches, and resource release. The store also owns the base directory and
the resource lifecycle.

Recommendation: Use explicit composition when both constructors are useful:

- `new(model_dir, index)`
- `from_model_dir(model_dir)`

Do not hide a second resource lifecycle behind the same `load` and `unload` pair.

A core or executor contract can guarantee a condition. Check that condition at the boundary where a violation becomes
visible. Do not add defensive recovery code that hides lifecycle bugs.

Use a release `assert!` only for an initialization, one-time structure, or ownership boundary. Also use it for a
contract that release code must enforce. Use `debug_assert!` for repeated internal paths.

Distinguish an invariant check from a data-flow bug. Existing data flow can already guarantee a condition. In this case,
add the smallest useful check at the owning boundary.

Select `assert!` or `debug_assert!` with the lifecycle and release-cost rule above. Do not add recovery state for an
impossible case. This restriction also applies to types, branches, and grouping.

Change the structure only in these cases:

- Valid inputs can violate the current data flow.
- The owner cannot guarantee the condition.

## Design style

Recommendation: Use first-principles contracts instead of compatibility patches. First, define caller-visible inputs,
outputs, ownership, and state transitions.

Then, select the backend implementation that satisfies the contract.

Remove a known incorrect implementation after you verify the correct default path. Do not keep it behind a feature
flag, environment variable, or fallback.

Recommendation: Use an established library when it reduces correctness or maintenance risk. Keep project-specific
semantics local.

For example, a generic tokenizer can own incremental decoding. A Qwen codec owns the Qwen response grammar.

Keep interfaces symmetric across related components. Examples include these component pairs:

- Sampling and rejection sampling
- GQA and GDN state tables
- Dense MLP and MoE paths

Use parallel names for shapes, inputs, outputs, scratch, kernels, and record methods. Use different names only when the
semantics are different.

Structural and API symmetry is a maintenance tool. It is not cosmetic consistency. A reader can transfer ownership,
lifecycle, data-flow, test, and profiling knowledge between peer components.

Unnecessary asymmetry increases maintenance work and cognitive load. Thus, symmetry is more important than mechanical
entity-count reduction. Use a small symmetric entity when it makes the peer contract clear.

Use an asymmetric design only for a concrete semantic or resource-lifecycle difference. State the difference at the
owning boundary.

Keep backend details behind backend APIs. Metal components may own these details:

- Kernels and dispatch parameters
- Tile scratch
- Runtime resource bindings

Recommendation: Model executors own model semantics, persistent model or request buffers, and component connections.
They do not expose backend tiling, temporary scratch, or kernel-local tables in model APIs.

## Model / layer / operator boundary

Recommendation: Model executor code preserves this ownership hierarchy:

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

The model answers, "Which whole-model stage runs next?" A layer or component answers, "Which model computation and state
does this layer own?"

An operator answers, "How does this tensor operation run on this backend?" A backend command is one concrete dispatch
from that operation.

Metal uses one compute-pipeline dispatch in an ICB slot. A different backend can use a kernel launch or graph node. One
component can emit multiple operators. One operator can emit multiple commands.

Do not let model code bind backend resources directly when a semantic layer can own that boundary. A component is also
a semantic layer.

Do not let backend operators contain model-specific request semantics when a layer or component can translate them into
backend buffers.

Recommendation: Use traits to enforce this boundary where drift is likely:

```text
ReplayLayer                 model-executor semantic replay contract with typed input/output
Recorder                    backend replay recording contract
Operator                    backend recordable execution contract
```

Recommendation: Use concrete types for clear implementations.

Use traits as boundary constraints. Do not use traits as general dynamic-dispatch abstractions.

Recommendation: Semantic layers record through `inference-executor-core::backend::recorder::Recorder`. They do not
depend directly on a concrete backend batch builder.

The model replay-cache boundary can lower semantic records into the Metal recorder. It does this operation when it
records a cached replay.

Recommendation: Do not expose lower-level builder details in component inputs or public executor APIs.

Recommendation: Use direct calls when the caller contains the related invariants.

Do not add a wrapper that only repackages arguments.

An extent, state, or planner wrapper must add ownership or contract meaning.

Reuse resources at the narrowest correct owner. Allocate immutable weights and stable kernels during initialization.
Allocate component or model scratch one time.

The scratch owner must be able to share the scratch safely across time. Do not add a GPU-to-GPU copy when the producer
can write directly into the consumer's persistent destination buffer.

Recommendation: Runtime hot paths do only runtime work:

- Write the current token, page, and state metadata.
- Submit cached or recorded work.
- Read the small CPU-visible outputs that the runtime contract requires.

Do relayout and path resolution during initialization. Build kernel-selection tables and allocate reusable buffers
during initialization or cache construction.

## Engineering conventions

[`engineering_conventions.md`](engineering_conventions.md) gives the detailed rules for these subjects:

- Naming
- Coordinate and numeric domains
- GPU Tile and Task terms
- Runtime shapes and persistent layouts
- Public APIs
- Test style

This document contains only architecture, ownership, lifecycle, performance-evidence, and completion rules.

## Performance evidence

Each performance claim must identify these facts:

- The exact commit and dirty state
- The machine and environment
- The model and command
- The baseline and current result
- The verdict

Compare the same work. Do not compare throughput alone. For speculative decoding, also report sampled tokens, chunks or
decisions, and acceptance efficiency.

A throughput change can have a different deterministic acceptance trajectory. This change alone does not identify an
executor or kernel regression.

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

Recommendation: Do not use these patterns:

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
