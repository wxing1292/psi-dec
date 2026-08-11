# Model Executor Architecture

The model executor turns runtime-owned batch metadata and page IDs into model execution. It owns model structure,
weights, component state, replay composition, and sampling. It does not schedule requests or allocate globally owned
cache pages.

Read this document after the top-level [README](../README.md) and [`core.md`](core.md). It explains the stable executor
mental model. Component documents describe current source and algorithms. Workflow documents contain shared commands
and cross-component measurement rules.

## Boundary and ownership

```text
runtime core
  schedule requests
  own request/cache lifecycle and physical page allocation
  produce batch metadata and page IDs
          |
          v
model executor
  bind model weights and component state
  interpret page IDs for GQA/GDN
  compose model stages and replay programs
  return sampled decisions and component results
          |
          v
Metal backend
  own device, buffers, kernels, command recording, and replay submission
```

The executor consumes runtime decisions. It must not recreate scheduler policy.

The runtime transports model inputs and outputs. It must not parse model tensor layouts or component-local page
contents.

## Shared component model

GQA, GDN, dense MLP, MoE, and sampling use the same ownership pattern:

```text
backend-neutral core/config and shape contract
  -> backend component with immutable kernels/tuning
  -> model-executor adapter with weights, state, metadata, and scratch
  -> typed record input/output at the model boundary
  -> CPU/reference oracle and focused production-owner tests
```

Component-specific complexity stays behind that boundary. Backend APIs remain model-independent. Qwen adapters supply
model dimensions, weights, and measured defaults.

Validation follows the same ownership:

- Config loading returns errors when it cannot parse or normalize checkpoint data.
- Component `Core` and backend constructors assert their static geometry once.
- Replay recording checks only current-batch shape, capacity, and binding contracts.

Do not restate component geometry during model normalization. Do not revalidate immutable core and config state on each
record.

Do not add wrappers only to align names. A type is useful when it owns a semantic boundary, invariant, resource, or
lifecycle.

`GQAMetadataBuffers` and `GDNMetadataBuffers` own reusable GPU metadata buffers for one batch. `GQAInput` and `GDNInput`
borrow the record-time tensors and component metadata.

`GQAOutput` and `GDNOutput` name the corresponding component outputs. They do not introduce another allocation owner.

## Current source areas

```text
crates/inference-executor-core/src/
  attn/gqa       backend-neutral GQA metadata and shapes
  attn/gdn       backend-neutral GDN metadata and shapes
  mlp/dense      backend-neutral dense gated-MLP metadata
  mlp/moe        backend-neutral MoE metadata and execution policy
  model/qwen     Qwen config, microbatch, and pending-transaction contracts
  sampling       sampling config, RNG domains, shapes, and CPU references

crates/inference-executor-metal/src/
  attn           GQA/GDN adapters, batch metadata, page/state tables, scratch
  mlp            dense-MLP and MoE adapters
  model/qwen     semantic model/layer components, weights, replay stages, MTP, DSpark
  sampling       top-k/top-p, DSpark Markov, and sparse rejection replay owners

crates/inference-backend-metal/src/
  metal          reusable Metal device/buffer/kernel/stream/replay runtime
  operators      recordable backend operations without model semantics
  components     reusable GQA, GDN, MLP, sampling, norm, embedding, and page-I/O kernels
```

For exact files and current paths, use the component documents:

- [`executor_qwen.md`](executor_qwen.md): Qwen semantic model loading, request state, replay stages, MTP, and DSpark.
- [`executor_gqa.md`](executor_gqa.md): GQA projection, KV pages, attention map and reduce, and outputs.
- [`executor_gdn.md`](executor_gdn.md): GDN projection, short convolution, recurrence, and state pages.
- [`executor_dense_mlp.md`](executor_dense_mlp.md): dense gated MLP.
- [`executor_moe.md`](executor_moe.md): routing and sparse expert execution.
- [`executor_sampling.md`](executor_sampling.md): ordinary sampling and sparse rejection.

## Model composition

Model-specific code connects reusable components. It does not absorb their implementations. A Qwen main forward has
this conceptual flow:

```text
token IDs
  -> embedding
  -> repeated transformer layers
       input norm
       GQA or GDN
       residual / post-attention norm
       dense MLP or MoE
       residual
  -> final norm
  -> unembedding
  -> ordinary sampling or Main verification distributions
  -> optional MTP or DSpark proposal and rejection flow
```

Normalized model configuration selects each layer variant. Exact typed binding subtrees identify the weights.

Semantic layer and component `load` functions consume those inputs directly.
Each model role owns only the plan needed to convert its configuration to reusable components.

The layer owns stage ordering and scratch handoff. The component owns reusable math, backend dispatch, and
component-local state interpretation.

Backend-neutral core types name component geometry and runtime replay shapes. Metal `ReplayLayer` names the typed
record input and output and the record operation.

`ReplayLayer` is intentionally lightweight. Page tables, routing keys, state transactions, and other component metadata
remain explicit typed input. An artificial tensor-to-tensor API does not hide this metadata.

## Weight contract

Model weights are immutable while loaded.
Each later load must materialize the same checkpoint representation.

- Load only the tensor set for the current semantic owner or layer into a `TensorMap`.
- Preserve each tensor key, storage dtype, shape, and bytes in the map value.
- Let the semantic owner remove its exact tensors from the map.
- Treat a nonempty map after owner construction as an incomplete ownership transfer.
- Parse model layout and validate shapes while loading.
- Complete required relayout, slicing, head reordering, and byte-level fusion during initialization.
- Materialize backend-owned immutable buffers and views, then release checkpoint mmap and file ownership when possible.
- Do not rewrite, relayout, or fuse model weights per request or token.
- Do not silently dequantize a full unsupported quantized weight. Fail explicitly when no runtime kernel supports it.

Persistent model parameters must preserve the checkpoint storage dtype and quantized representation. Kernels perform
dequantization, precision promotion, and model-defined numeric transforms during execution.

Runtime numerical state and scratch may use the compute dtype required for stability. These buffers are not model
weights.

If measured execution requires a precomputed parameter transform, give the derived execution resource an explicit name
and owner. Do not expose it as the loaded checkpoint weight. Record the supporting performance evidence.

Recommendation: Do not use hot-path `contiguous` calls. If execution requires a layout, prepare it during
initialization.

`SafeTensorStore` reads safetensors into a bounded `TensorMap` and releases the mapped shards after the copy.
It does not materialize the full checkpoint at one time.
Exact binding trees contain keys only.
They do not own tensor bytes or duplicate checkpoint values.
Generic embedding and unembedding owners apply the same contract to their exact quantized tensor binding.

## Metal lowering boundary

Executor code lowers semantic components into backend recordable operations:

```text
ReplayLayer::record(typed input)
  -> Recorder<ReplayOp>
  -> backend Operator::record
  -> pipeline + buffer/constant bindings + dispatch
  -> ReplayProgram
  -> submission-time ReplayArguments
```

The executor owns semantic stage order and the buffers exchanged between stages. Backend operators own kernel binding,
resource usage, dispatch, and internal phase barriers.

The backend [`README`](../crates/inference-backend-metal/README.md) contains these details:

- The complete Metal object model
- Residency rules
- The stream lifecycle
- The minimal Add One example

This document does not duplicate those details.

### Buffer and scratch ownership

Keep these domains distinct:

| Object                  | Owner                                      | Meaning                                                             |
| ----------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| Immutable weight buffer | model or component                         | initialized once and shared across replays                          |
| Runtime page buffer     | runtime allocates and component interprets | persistent KV or GDN state addressed by runtime page IDs            |
| Batch metadata buffer   | component batch-metadata owner             | current batch's offsets, page IDs, or state slots                   |
| Scratch buffer          | component, layer, or model scratch owner   | temporary partials and intermediates with explicit reuse boundaries |
| Replay parameter buffer | backend replay program                     | submission-time scalar values for one recorded program              |

A `Buffer` is raw storage. A tensor or weight view adds dtype, shape, layout, and byte offset.

Different views may intentionally alias one buffer. Scratch reuse is correct only when the next writer cannot destroy
data that a later stage still consumes.

### Barriers

A barrier belongs to the consumer command that must wait. Layer entry barriers protect cross-component dependencies.
Backend components retain their internal phase barriers.

Do not infer barriers from method order. Do not duplicate them at both layers. Do not add barriers to independent
reads.

## Replay composition

Recording is expensive relative to replay. The executor therefore caches stable command topology.

A replay key contains only values that change recorded commands, dispatch topology, static geometry, or scratch layout.
Metadata buffers or `ReplayArguments` contain dynamic values that fit an existing recording.

```text
static / replay-defining
  component geometry and tuning
  capacity bucket or exact shape when inactive lanes are unsupported
  command topology and scratch extent

dynamic / submission-scoped
  valid request, token, row, task-template, or partial-output counts
  page IDs, offsets, state slots, sampling parameters
  other values consumed through current batch metadata
```

Capacity-bucket replay is safe only when each participating kernel causes inactive lanes to return before these
actions:

- Reading input
- Mutating state
- Advancing RNG
- Writing output

Otherwise, the replay key keeps the exact count. Padding is a dispatch property. It does not permit changes to valid
work or semantic descriptor counts.

The shared default capacity policy starts with these buckets:

```text
1 2 4 6 8 12 16 20 24 32 40 48 56 64
```

Above `64`, the policy divides each power-of-two interval into quarters. For example, the next buckets are `80`,
`96`, `112`, and `128`. The configured capacity is always a terminal bucket. A bucket must not exceed allocated
scratch or buffer capacity.

A component must insert a terminal bucket before each recorded-topology boundary. The component must keep categorical
topology choices in its replay key. Capacity bucketing must not combine different kernel selections, command counts,
or dispatch structures. A topology boundary identifies the first active count for the new topology. The policy ignores
a boundary above the configured capacity. The backend component that selects the topology must own these boundaries.

Some work domains permit zero active work. In this case, a policy result of zero means that the domain does not record
or dispatch work. Zero is not a replay capacity.

`ReplayArguments` contain keyed submission values that recording declares. Submission validates that the caller
provides each declared value exactly once and within its recorded bounds.

These arguments prevent a program rebuild for scalar activity changes. They do not replace component batch metadata.

### Qwen replay stages

Qwen keeps separate replay caches for these semantically separate stages:

- Main forward
- Main output
- GDN state restore
- Ordinary sampling
- MTP proposal
- DSpark context append
- DSpark proposal
- Spec sampling
- Rejection sampling

A cache boundary exists when command topology, lifecycle, or CPU dependency differs. A component name alone does not
create a cache boundary.

Normal forward, output, and sampling commands can share one ordered command buffer when they share one dependency chain.

MTP and DSpark proposal work remains separate from Main where the sampled result crosses the CPU boundary.
GDN state candidate preparation and cache-boundary publication retain their transaction lifecycle when their GPU work
is replayed.

The generic executor lifecycle uses role-qualified Main and Spec hooks:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> wait -> read_main

embed_spec -> forward_spec -> unembed_spec -> sample_spec
submit_spec -> wait -> read_spec
```

Each record hook owns one semantic stage.
`forward_spec` must not also record Spec embedding, unembedding, or sampling.
Main and Spec may have different data contracts, but they must preserve this lifecycle shape.

Detailed keys, stage order, and request lifecycle are in [`executor_qwen.md`](executor_qwen.md). Sampling and rejection RNG
and write-distribution contracts are in [`executor_sampling.md`](executor_sampling.md).

## Concurrency and lifecycle

The current service path uses one executor synchronously. The executor prepares a batch, executes it, obtains the
result, and commits before the next batch.

Replay caches, scratch owners, request-slot state, and pending GDN transactions therefore remain executor-owned. They
remain confined to one thread unless an API explicitly states otherwise.

Runtime core still owns the durable request and cache lifecycle. The executor reports sampled decisions and component
results. It does not free globally owned pages or commit scheduler state independently.

`ReplayableModel` also defines synchronous model residency operations.
Current Qwen model executors support this order:

```text
stop: clear_replay_cache -> unload_state -> unload_weights
start: load_weights -> load_state
```

The model shell remains allocated while its resources are unloaded.
Weight-bearing component shells remain allocated and fail fast if execution accesses missing weights.
The unload traversal must remove all shared `Rc` owners.
The final owner releases the shared Metal resource.

`unload_state` writes full `PageArena`, GQA, and GDN payloads to SSD before it releases their buffers.
The snapshot also stores durable GDN request state and future publish page IDs.
The executor finishes or clears transient restore, publish, and batch transactions before it writes the snapshot.

`load_state` validates the complete snapshot before it allocates state resources.
It attaches consumers only after all state reads succeed.
It releases all new resources after a read failure.

`ReplayableModelEventLoop` invokes these operations for idempotent `Start` and `Stop` commands.
It also starts a stopped model before it executes a batch.
Runtime core currently sends only batch requests.
Executor residency tracking and idle policy remain design work.
See [`model_idle_unload.md`](model_idle_unload.md) for the current boundary and remaining wiring.

## Verification boundary

Recommendation: Use the narrowest production owner that can express the invariant:

- CPU references prove math.
- Backend tests prove shader build, dispatch, ABI, and parity.
- Executor component tests prove real metadata, state, and scratch ownership.
- Layer and end-to-end tests prove composition and lifecycle.

Do not reshape production source for test construction. Do not add naming-only tests. Run Metal tests serially.

[`executor_benchmarks.md`](executor_benchmarks.md) defines these shared topics:

- The full verification ladder
- Benchmark targets
- Profiling vocabulary
- Performance-evidence rules

## Operational workflows

- Model download, server and client commands, logging, cold-start separation, and end-to-end helpers:
  [`service.md`](service.md).
- Tests, benchmarks, profiling, and performance claims: [`executor_benchmarks.md`](executor_benchmarks.md).
- Shared naming, API, ownership, and definition-of-done rules: [`engineering_conventions.md`](engineering_conventions.md)
  and [`high_level.md`](high_level.md).
- Active investigations: [`future_work.md`](future_work.md).

The Metal backend embeds MLX-derived headers at build time through `build.rs`. The repository
[`NOTICE`](../NOTICE) retains their MIT attribution.
