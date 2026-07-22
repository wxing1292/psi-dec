# Model Executor Architecture

The model executor turns runtime-owned batch metadata and page IDs into model execution. It owns model structure,
weights, component state, replay composition, and sampling. It does not schedule requests or allocate globally owned
cache pages.

Read this document after the top-level [README](../README.md) and [`core.md`](core.md). It explains the stable executor
mental model; component documents describe current source and algorithms, while workflow documents carry shared
commands and cross-component measurement rules.

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
  return sampled decisions and lifecycle notifications
          |
          v
Metal backend
  own device, buffers, kernels, command recording, and replay submission
```

The executor consumes runtime decisions; it must not recreate scheduler policy. The runtime transports model inputs and
outputs; it must not parse model tensor layouts or component-local page contents.

## Shared component model

GQA, GDN, dense MLP, MoE, and sampling use the same ownership pattern:

```text
backend-neutral core/config and shape contract
  -> backend component with immutable kernels/tuning
  -> model-executor adapter with weights, state, metadata, and scratch
  -> typed record input/output at the model boundary
  -> CPU/reference oracle and focused production-owner tests
```

Component-specific complexity stays behind that boundary. Backend APIs remain model-independent; Qwen adapters supply
model dimensions, weights, and measured defaults.

Validation follows the same ownership. Config loading returns errors when checkpoint data cannot be parsed or normalized;
component `Core`/backend constructors assert their static geometry once; replay recording checks only current-batch shape,
capacity, and binding contracts. Do not restate component geometry in model normalization or revalidate immutable
core/config state on every record.

Do not add wrappers merely to make names line up. A type is useful when it owns a semantic boundary, invariant,
resource, or lifecycle. For example, `GQAMetadataBuffers` and `GDNMetadataBuffers` own reusable GPU metadata buffers for one
batch, while `GQAInput`/`GDNInput` borrow the actual record-time tensors and component metadata. `GQAOutput` and
`GDNOutput` name the corresponding component outputs without introducing another allocation owner.

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
  model/qwen     semantic model/layer components, weights, replay stages, MTP
  sampling       top-k/top-p and sparse rejection replay owners

crates/inference-backend-metal/src/
  metal          reusable Metal device/buffer/kernel/stream/replay runtime
  operators      recordable backend operations without model semantics
  components     reusable GQA, GDN, MLP, sampling, norm, embedding, and page-I/O kernels
```

For exact files and current paths, use the component documents:

- [`executor_qwen.md`](executor_qwen.md): Qwen semantic model loading, request state, replay stages, and MTP.
- [`executor_gqa.md`](executor_gqa.md): GQA projection, KV pages, attention map/reduce, and outputs.
- [`executor_gdn.md`](executor_gdn.md): GDN projection, short convolution, recurrence, and state pages.
- [`executor_dense_mlp.md`](executor_dense_mlp.md): dense gated MLP.
- [`executor_moe.md`](executor_moe.md): routing and sparse expert execution.
- [`executor_sampling.md`](executor_sampling.md): ordinary sampling and sparse rejection.

## Model composition

Model-specific code wires reusable components together; it does not absorb their implementations. A Qwen main forward
is conceptually:

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
  -> ordinary sampling or target distributions
  -> optional MTP proposal and rejection flow
```

Normalized model configuration selects each layer variant, and exact typed binding subtrees identify its weights.
Semantic layer/component `load` functions consume those inputs directly; there is no parallel Main/MTP plan tree. The
layer owns stage ordering and scratch handoff. The component owns reusable math, backend dispatch, and component-local
state interpretation.

The backend-neutral `Layer` trait names typed input/output and input/output shapes. Metal `ReplayLayer` extends it with a
record operation. It is intentionally lightweight: page tables, routing keys, state transactions, and other real
component metadata remain explicit typed input rather than being hidden behind an artificial tensor-to-tensor API.

## Weight contract

Model weights are immutable after initialization.

- Parse model layout and validate shapes while loading.
- Perform required relayout, slicing, head reordering, fusion, and format normalization once at init.
- Materialize backend-owned immutable buffers/views, then release checkpoint mmap/file ownership when possible.
- Do not rewrite, relayout, or fuse model weights per request or token.
- Do not silently dequantize a full unsupported quantized weight. Fail explicitly when no runtime kernel supports it.

Dense norms or biases may be materialized as dense buffers; that is different from expanding quantized matrix weights.
Avoid hot-path `contiguous` calls. If a layout is required for execution, prepare it at init.

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
resource usage, dispatch, and internal phase barriers. The complete Metal object model, residency rules, stream
lifecycle, and minimal Add One example live in the backend
[`README`](../crates/inference-backend-metal/README.md); they are not duplicated here.

### Buffer and scratch ownership

Keep these domains distinct:

| Object | Owner | Meaning |
| --- | --- | --- |
| Immutable weight buffer | model/component | initialized once and shared across replays |
| Runtime page buffer | runtime allocation; component interpretation | persistent KV or GDN state addressed by runtime page IDs |
| Batch metadata buffer | component batch-metadata owner | current batch's offsets, page IDs, or state slots |
| Scratch buffer | component/layer/model scratch owner | temporary partials and intermediates with explicit reuse boundaries |
| Replay parameter buffer | backend replay program | submission-time scalar values for one recorded program |

A `Buffer` is raw storage. A tensor/weight view adds dtype, shape, layout, and byte offset. Different views may alias one
buffer intentionally; scratch reuse is correct only when the next writer cannot destroy data still consumed by a later
stage.

### Barriers

A barrier belongs to the consumer command that must wait. Layer entry barriers protect cross-component dependencies;
backend components retain their internal phase barriers. Do not infer barriers from method order, duplicate them at
both layers, or add them to independent reads.

## Replay composition

Recording is expensive relative to replay, so stable command topology is cached. A replay key contains only values that
change recorded commands, dispatch topology, static geometry, or scratch layout. Dynamic values that fit an existing
recording are written into metadata buffers or `ReplayArguments`.

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

Power-of-two capacity replay is safe only when every participating kernel returns inactive lanes before reading input,
mutating state, advancing RNG, or writing output. Otherwise the replay key keeps the exact count. Padding is a dispatch
property, not permission to change valid work or semantic descriptor counts.

`ReplayArguments` are keyed submission values declared while recording. Submission validates that every declared value
is provided exactly once and within its recorded bounds. They avoid rebuilding a program for scalar activity changes;
they do not replace component batch metadata.

### Qwen replay stages

Qwen keeps separate replay caches for semantically separate stages such as main forward, target output, GDN state
restore, ordinary sampling, MTP proposal, draft sampling, and target rejection. A cache boundary exists when command
topology, lifecycle, or CPU dependency differs—not merely because a component has its own name.

Normal forward/output/sampling commands that share one dependency chain can be submitted in one ordered command buffer.
MTP proposal and target rejection remain separated where the accepted-token decision crosses the CPU boundary. GDN
state candidate preparation and cache-boundary publication preserve their own transaction lifecycle even when their
GPU work is replayed.

Detailed keys, stage order, and request lifecycle are in [`executor_qwen.md`](executor_qwen.md). Sampling/rejection RNG
and sparse-distribution contracts are in [`executor_sampling.md`](executor_sampling.md).

## Concurrency and lifecycle

The current service path uses one executor synchronously: prepare a batch, execute it, obtain the result, and commit
before the next batch. Replay caches, scratch owners, request-slot state, and pending GDN transactions therefore remain
executor-owned and single-thread confined unless an API explicitly states otherwise.

Runtime core still owns the durable request/cache lifecycle. The executor reports decisions and component lifecycle
events; it does not free globally owned pages or commit scheduler state on its own.

## Verification boundary

Tests should prove the narrowest production owner that can express the invariant:

- CPU references prove math.
- Backend tests prove shader build, dispatch, ABI, and parity.
- Executor component tests prove real metadata/state/scratch ownership.
- Layer and end-to-end tests prove composition and lifecycle.

Do not reshape production source for test construction or add naming-only tests. Run Metal tests serially. The full
verification ladder, benchmark targets, profiling vocabulary, and performance-evidence rules live in
[`executor_benchmarks.md`](executor_benchmarks.md).

## Operational workflows

- Model download, server/client commands, logging, cold-start separation, and end-to-end helpers:
  [`service.md`](service.md).
- Tests, benchmarks, profiling, and performance claims: [`executor_benchmarks.md`](executor_benchmarks.md).
- Shared naming, API, ownership, and definition-of-done rules: [`engineering_conventions.md`](engineering_conventions.md)
  and [`high_level.md`](high_level.md).
- Active investigations: [`future_work.md`](future_work.md).

The Metal backend embeds MLX-derived headers at build time through `build.rs`; their MIT attribution is retained in the
repository [`NOTICE`](../NOTICE).
