# GPU Execution Vocabulary

This document defines the shared GPU execution vocabulary for the repository. Component documents define the
component-specific operation, task, specialization, and data layout.

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
    = compile(KernelSource, KernelSpecialization)
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
│   └── KernelSpecialization
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

## Static specialization and dynamic execution

`KernelSpecialization` contains compile-time choices. Group fields by the scope that owns them:

```text
ComponentSpecialization
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

`KernelArguments` contains launch data. It can contain buffers, metadata, and scalar values. Do not copy a static
specialization value into `KernelArguments` only because the shader uses that value.

Dynamic workload shapes can derive tasks and an execution configuration. They do not select a kernel unless the
component has multiple legal implementations with a workload-dependent choice.

## Tasks, tiles, and layouts

A task describes semantic work. For a component-specific non-persistent kernel, use this model:

```text
ThreadBlockTask
    = derive(
        KernelArguments,
        thread_block_index,
        KernelSpecialization,
      )

ThreadTask
    = derive(
        ThreadBlockTask,
        thread_index,
        KernelSpecialization,
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

## Registration and planning

Add a runtime planner only when all of these conditions apply:

1. The component has more than one legal implementation or specialization.
2. The best choice depends on the dynamic workload.
3. A complete candidate plan can express dispatch, metadata, scratch, and reduction cost.

Use this ownership model when a planner is necessary:

```text
StaticProblem
    -> SpecializationRegistry.supports(...)
    -> legal KernelSpecializations

DynamicWorkload + each legal specialization
    -> complete candidate plan
    -> compare complete-plan cost
    -> selected plan
    -> metadata, KernelArguments, and KernelExecutionConfiguration
```

`supports(...)` must contain correctness and static capability checks. It must not contain workload performance
thresholds. The planner owns dynamic selection. Recording must not select a second implementation.

Do not add a registry or planner when one static specialization implements the current algorithm. Derive that
specialization during initialization. Derive each execution configuration from the dynamic shape at recording time.

## Current component mappings

GQA SDPA has multiple legal Map/Reduce specializations and a workload-dependent crossover. It uses a specialization
registry and a dynamic planner. See [GQA SDPA planning](gqa_sdpa_planner.md).

GDN has one mathematical recurrent algorithm. Its final-state and candidate-state kernels implement different state
materialization contracts. They are not cost candidates. GDN derives one `GDNComputeSpecialization` during
initialization and does not use a runtime planner. See [GDN Executor](executor_gdn.md).

The following table defines the cross-component execution model. It also records intentional differences. A component
must not add a planner only to match another component.

| Component | Semantic execution | Non-persistent thread-block task | Dynamic selection owner |
| --- | --- | --- | --- |
| Quantized embedding | Dequantize selected vocabulary rows into hidden rows. | A bounded flat range of `(token, hidden)` output values. | No planner. The backend derives one fixed kernel specialization at initialization. |
| Unembedding | Apply one affine quantized projection from hidden rows to vocabulary logits. | The selected affine QMV or QMM kernel defines the task. | `AffineQuantizedMatmul` owns the row-dependent QMV/QMM choice. `Unembed` does not select it again. |
| RMSNorm | Normalize one hidden row and apply its weight row. | One token row. | No planner. Dtype selects one static kernel at initialization. |
| Residual add | Add two flat tensors or two row-major active prefixes. | A bounded flat range of output values. | No planner. Dtypes select one static kernel at initialization. |
| Residual-add RMSNorm | Add two hidden rows, preserve the residual row, and normalize it. | One token row. | The backend selects the scalar or BF16-vectorized kernel at initialization. The runtime shape does not change this choice. |
| RMSNorm/RoPE | Normalize and rotate one attention-head row. | One `(flat Q token, Q head)` row. | No planner. Model geometry and RoPE constants define one compiled specialization. |
| Dense MLP | Record `gate_up affine -> SwiGLU -> down affine`. | Each leaf kernel defines its own task. SwiGLU uses a bounded flat range of `(token, intermediate)` output values. | Each affine owner selects QMV or QMM independently. Dense MLP does not add a second planner. |
| MoE | Record routing and either token-major or expert-major expert execution. | Each routing, layout, pack, expert, combine, or scatter phase defines its own task. | The pure `GatedMoEComputePath::select(...)` helper selects the complete token-major or expert-major command graph. Sparse expert kernels do not select this graph. |
| Top-K sampling | Map vocabulary partitions to partial candidates, then reduce the partial candidates for each sampling row. | Map: one `(sampling row, vocabulary partition)`. Reduce: one sampling row. | The pure Map-specialization helper selects the kernel family from the output contract and Top-K width. The component does not materialize a planner or plan object. Replay capacity selection remains a separate owner concern. |
| Sparse rejection sampling | Walk one request's ordered draft sequence and sample its fallback or continuation token. | One request. | No kernel planner. Replay bucket policies select only capacities. |
| DSpark Markov sampling | Produce partial Top-K candidates with a Markov correction, then use the common Top-K reduce phase. | Map: one `(sampling row, vocabulary partition)`. Reduce: one sampling row. | The Markov Map kernel has one current specialization. Its partial-candidate layout is an explicit producer/consumer contract. |

### Planning and replay identity

Kernel planning and replay capacity selection are different operations:

```text
dynamic semantic shape
    -> kernel or execution planning, when required
    -> recorded command topology

active item count
    -> replay bucket policy
    -> recorded capacity that must preserve the planned topology
```

A replay key must identify every dynamic choice that changes the recorded command graph or compiled kernel. An active
count must not enter the key when a replay parameter can supply that count without changing topology.

Dense MLP and unembedding use the topology boundaries from their adaptive affine owners. MoE adds the boundary where
its complete execution plan changes. Top-K sampling keeps the active Top-K width in its replay shape because that width
changes candidate geometry and can change the Map kernel family. Rejection sampling keeps request and distribution
capacities separate because they have different semantic domains.

### Intentional asymmetries

Embedding and unembedding are not symmetric GPU algorithms. Embedding is a quantized row lookup. Unembedding is an
adaptive quantized matrix multiplication. They can share weight-lifecycle and replay-capacity conventions, but they
must not share a kernel planner.

Dense MLP and sparse expert MLP are not two layouts of one kernel. Dense MLP applies the same weights to every token.
Sparse expert MLP applies expert-indexed weights to routed rows. The MoE owner selects token-major or expert-major
execution before it invokes the sparse expert leaf.

Top-K sampling and rejection sampling share runtime-parameter and replay-preparation rules. They do not share one
thread-block task. Top-K sampling partitions a vocabulary row. Rejection sampling processes one ordered request.
