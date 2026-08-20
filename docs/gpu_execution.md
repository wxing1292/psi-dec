# GPU Execution Vocabulary

This document defines the shared GPU execution vocabulary for the repository. Component documents define the
component-specific operation, task, constants, variant, selection, and data layout.

The vocabulary follows the execution hierarchy and scope concepts in
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/). Use that material as a conceptual
reference. Do not copy Blackwell-specific CTA, warpgroup, TMA, TMEM, or tile-size names into a backend-independent API.

The project uses `threadblock` in backend-independent identifiers. Metal source uses `threadgroup` and `simdgroup` at
the backend boundary.

## Execution hierarchy

Use these project relations:

```text
KernelLaunch
    = CompiledKernel
    + KernelArguments
    + KernelExecutionConfiguration

CompiledKernel
    = compile(KernelSource, KernelConstants)
```

`KernelExecutionConfiguration` contains `grid_dimensions` and `thread_block_dimensions`. It does not contain an
executed Grid object. A launch creates this execution hierarchy:

```text
KernelLaunch
    -> one Grid
    -> many ThreadBlocks
    -> many Threads per ThreadBlock
```

All threads execute the same `CompiledKernel`. All threads can read the same `KernelArguments`. Each thread has a
different `thread_block_index` and `thread_index`.

The hardware hierarchy and the semantic work hierarchy are related but distinct:

```text
KernelLaunch
├── CompiledKernel
│   ├── KernelSource
│   └── KernelConstants
├── KernelArguments
└── KernelExecutionConfiguration
    ├── grid_dimensions
    └── thread_block_dimensions

one ThreadBlock
└── component-defined ThreadBlockTask
    └── component-defined ThreadTask per Thread
```

A non-persistent kernel can define a `1:1` relation between `ThreadBlock` and `ThreadBlockTask`. This relation is a
kernel contract. It is not a global GPU programming rule. Each component must state its relation.

## Compile-time constants and execution variants

`KernelConstants` contains values that are fixed when the project compiles or looks up one `CompiledKernel`. Group
fields by the scope that owns them:

```text
ComponentKernelConstants
├── model or storage geometry
└── kernels
    └── one semantic kernel phase
        ├── thread_block
        │   └── collective thread-block properties
        └── optional lower scope
            └── simdgroup properties
```

Use nesting to show scope. Do not repeat the scope in each nested field name.

- `max_*` is a compile-time upper bound.
- `num_*` is an exact count in its stated scope.
- `required_threads` is a compile-time launch requirement.
- `KernelExecutionConfiguration.thread_block_dimensions` contains the actual launch dimensions.

The required and actual thread counts must match.

`KernelArguments` contains launch data. It can contain buffers, metadata, and scalar values. Do not copy a compile-time
constant into `KernelArguments` only because the shader uses that value.

Dynamic workload shapes can derive tasks and an execution configuration. They do not select a kernel unless the
component has multiple legal implementations with a workload-dependent choice.

Use `specialization` for the act or result of compiling an implementation for fixed constants. Do not use
`KernelSpecialization` as the name of the constant value. The MLC reference uses compile-time constants such as
`BLK_M` and describes kernels as specialized for fixed tile shapes. This project uses the same distinction.

An `ExecutionVariant` is the complete unit that a runtime selector can choose:

```text
ExecutionVariant
├── algorithm or execution-path identity
├── one or more kernel-family identities
└── matching KernelConstants
```

An algorithm and a shape-specific kernel implementation are not separate selector levels. One registry entry contains
the complete combination. For example, a future GDN registry can contain recurrent and chunkwise variants. Each entry
can also identify the constants and kernels for one shape family.

An `ExecutionVariant` does not have to own pipeline objects. A registry can exist before device-specific pipeline
construction. In that case, the variant is a stable description that a later owner uses to compile or look up the
matching `CompiledKernel` values.

## Tasks, tiles, and layouts

A task describes semantic work. For a component-specific non-persistent kernel, use this model:

```text
ThreadBlockTask
    = derive(
        KernelArguments,
        thread_block_index,
        KernelConstants,
      )

ThreadTask
    = derive(
        ThreadBlockTask,
        thread_index,
        KernelConstants,
      )
```

These are semantic definitions. The CPU does not have to construct or upload task objects. A kernel can derive task
coordinates from grid indices and compact metadata.

A tile is a bounded subregion of tensor coordinates used by an implementation. A tile is not a launch, an execution
configuration, or a generic task object.

For GEMM, `BM`, `BN`, and `BK` describe operand and output subregions. `BK` is an implementation loop extent. For SDPA
or recurrence, use component coordinates such as Q-token ranges, KV-token ranges, head ranges, or V-row ranges. Do not
force these coordinates into GEMM names when the mathematical domains differ.

A layout maps logical coordinates to storage locations or thread ownership. State the coordinate domains and owning
scope. Do not use `Tile`, `Task`, or `Layout` as an unqualified substitute for this information.

## Registration and selection

Use this component-local model when a component has runtime-selectable execution variants:

```text
StaticConfig
    -> Registry.supports(...)
    -> legal ExecutionVariants

DynamicWorkload + legal ExecutionVariants
    -> Selector::select(...)
    -> Selection
    -> metadata, KernelArguments, and KernelExecutionConfiguration
```

`Registry` and `Selector` are short internal names because their module path supplies the component scope. Do not add a
cross-component registry or selector trait.

`Registry::supports(...)` must contain correctness and static capability checks. It must not contain dynamic workload
performance thresholds. `Selector` owns the dynamic choice. Recording must not select a second variant.

The selector must compare complete candidate cost. If one candidate needs dynamic task partitioning, scratch extents,
or replay extents, the selector must materialize those values before it compares candidates. Do not add a separate
`Planner` layer for this work.

Use a small selection value when the result is small. A simple selector can return `(VariantKey, &ExecutionVariant)`.
The key must have a real use, such as replay identity, logging, or cache lookup. A complex component can return a
component-local `Selection` struct when metadata upload, replay, and recording must consume multiple coupled values.
Do not add a generic `Plan` type.

A component with one fixed implementation can omit `Registry`, `Selector`, and `Selection`. It must still use clear
`KernelConstants`, task, argument, and execution-configuration names. Add the selection structure when a second real
variant appears. Do not add an empty framework only for visual symmetry.

## Current component mappings

GQA SDPA has multiple legal Map/Reduce execution variants and a workload-dependent crossover. It uses a component-local
`Registry`, `Selector`, and `Selection`. See [GQA SDPA selection](gqa_sdpa_selection.md).

GDN currently registers one recurrent algorithm. Its final-state and candidate-state kernels implement different state
materialization contracts. They are phases of one current execution variant, not independent candidates. Its private
selector returns `(VariantKey, &Variant)`. A future chunkwise algorithm must be a separate complete variant. See
[GDN Executor](executor_gdn.md).

The following table defines the cross-component execution model. It also records intentional differences. A component
must not add a selector only to match another component.

| Component | Semantic execution | Non-persistent thread-block task | Dynamic selection owner |
| --- | --- | --- | --- |
| Quantized embedding | Dequantize selected vocabulary rows into hidden rows. | A bounded flat range of `(token, hidden)` output values. | One current fixed variant. No registry or selector. |
| Unembedding | Apply one affine quantized projection from hidden rows to vocabulary logits. | The selected affine QMV or QMM kernel defines the task. | `AffineQuantizedMatmul` owns the row-dependent QMV/QMM selection. `Unembed` does not select it again. |
| Row gather | Copy indexed input rows to a dense output. | A bounded flat range of `(output row, column)` values. | Dtype fixes one variant during initialization. No dynamic selector. |
| RMSNorm | Normalize one hidden row and apply its weight row. | One token row. | Dtype fixes one variant during initialization. No dynamic selector. |
| Residual add | Add two flat tensors or two row-major active prefixes. | A bounded flat range of output values. | Dtypes fix one variant during initialization. No dynamic selector. |
| Residual-add RMSNorm | Add two hidden rows, preserve the residual row, and normalize it. | One token row. | The backend selects the scalar or BF16-vectorized kernel at initialization. The runtime shape does not change this choice. |
| RMSNorm/RoPE | Normalize and rotate one attention-head row. | One `(flat Q token, Q head)` row. | Model geometry and RoPE constants fix one variant. No dynamic selector. |
| Dense MLP | Record `gate_up affine -> SwiGLU -> down affine`. | Each leaf kernel defines its own task. SwiGLU uses a bounded flat range of `(token, intermediate)` output values. | Each affine owner selects QMV or QMM independently. Dense MLP does not select the same decision again. |
| Sparse expert MLP | Apply expert-indexed MLP weights to routed rows. | The selected expert kernel defines its task over routed rows and expert weights. | The MoE owner selects the complete token-major or expert-major command graph. The sparse leaf does not select the outer graph again. |
| MoE | Record routing and either token-major or expert-major expert execution. | Each routing, layout, pack, expert, combine, or scatter phase defines its own task. | A component-local `Registry` and `Selector` select the complete command-graph variant. The deterministic selector can be called again when the same identity is needed. |
| Top-K sampling | Map vocabulary partitions to partial candidates, then reduce the partial candidates for each sampling row. | Map: one `(sampling row, vocabulary partition)`. Reduce: one sampling row. | A component-local registry and selector choose the Map variant from the output contract and Top-K width. Replay capacity remains a separate owner concern. |
| Sparse rejection sampling | Walk one request's ordered draft sequence and sample its fallback or continuation token. | One request. | One current fixed variant. Replay bucket policies select capacities, not kernels. |
| DSpark Markov sampling | Produce partial Top-K candidates with a Markov correction, then use the common Top-K reduce phase. | Map: one `(sampling row, vocabulary partition)`. Reduce: one sampling row. | The Markov Map kernel has one current variant. Its partial-candidate layout is an explicit producer/consumer contract. |

### Registry and selector audit

GQA SDPA is the current component in this table with a rich materialized `Selection`. Each legal Map/Reduce variant
produces different Q-token ranges, KV ranges, partial-state offsets, replay extents, and candidate metrics. The
component-local `Selection` keeps these coupled results with the selected execution variant.
`GQAMetadataBuffers::update(...)` consumes this result as one unit. It does not rebuild work partitioning or select a
second variant.

The other current components do not need a rich selection value:

- GDN uses a small `(VariantKey, &Variant)` selection. The current registry contains only `Recurrent`.
- Embedding, row gather, RMSNorm, residual operations, rejection sampling, and DSpark Markov Map currently derive one
  fixed variant.
- Dense MLP and unembedding delegate row-dependent QMV/QMM selection to `AffineQuantizedMatmul`.
- MoE can repeat its pure selector where it needs the command-graph identity.
- Top-K sampling uses a small selection value for its Map kernel family.
- Sparse MLP implements expert inner compute. MoE owns the outer command-graph choice.

`ExecutorHibernationPlan` and `StateSnapshotPlan` describe requested model-state persistence. They are not GPU kernel or
execution selections, and this execution rule does not apply to them.

### Selection and replay identity

Kernel-variant selection and replay capacity selection are different operations:

```text
dynamic semantic shape
    -> kernel or execution-variant selection, when required
    -> recorded command topology

active item count
    -> replay bucket policy
    -> recorded capacity that must preserve the selected topology
```

A replay key must identify every dynamic choice that changes the recorded command graph or compiled kernel. An active
count must not enter the key when a replay parameter can supply that count without changing topology.

Dense MLP and unembedding use the topology boundaries from their adaptive affine owners. MoE adds the boundary where
its complete execution variant changes. Top-K sampling keeps the active Top-K width in its replay shape because that width
changes candidate geometry and can change the Map kernel family. Rejection sampling keeps request and distribution
capacities separate because they have different semantic domains.

### Intentional asymmetries

Embedding and unembedding are not symmetric GPU algorithms. Embedding is a quantized row lookup. Unembedding is an
adaptive quantized matrix multiplication. They can share weight-lifecycle and replay-capacity conventions, but they
must not share a kernel selector.

Dense MLP and sparse expert MLP are not two layouts of one kernel. Dense MLP applies the same weights to every token.
Sparse expert MLP applies expert-indexed weights to routed rows. The MoE owner selects token-major or expert-major
execution before it invokes the sparse expert leaf.

Top-K sampling and rejection sampling share runtime-parameter and replay-preparation rules. They do not share one
thread-block task. Top-K sampling partitions a vocabulary row. Rejection sampling processes one ordered request.
