# GQA SDPA Selection

This document defines the GQA SDPA selection model. It separates reference vocabulary from current implementation
facts. It also records the scope of the current design change.

## Reference vocabulary

Use [GPU execution vocabulary](gpu_execution.md) for the shared launch, constant, variant, task, tile, layout, and
selection terms. This section defines the SDPA-specific mapping.

### SDPA kernel constants

The Map constants use this hierarchy:

```text
MapKernelConstants
├── thread_block
│   ├── max_q_tokens
│   ├── max_q_heads
│   ├── kv_tokens_per_iteration
│   └── required_threads
└── kv_cache
    └── tokens_per_page
```

The Reduce constants do not describe the KV cache because Reduce does not read it:

```text
ReduceKernelConstants
└── thread_block
    ├── max_q_tokens
    ├── max_q_heads
    └── required_threads
```

All fields under `thread_block` describe collective properties of one threadblock. A `max_*` field is an upper bound.
`kv_tokens_per_iteration` is the exact number of KV tokens that the complete threadblock processes in one internal
iteration. `required_threads` is a compile-time requirement. It must match the actual threadblock dimensions in the
execution configuration.

`tokens_per_page` belongs to `kv_cache` because the kernel uses it for paged KV address translation. It is not a work
tile extent.

The project does not define a generic `Tile` Rust type for this hierarchy. A concrete kernel can use internal tensor
tiles and data layouts. These implementation details do not change the semantic task boundary.

### SDPA tasks

For the current non-persistent SDPA kernels, one threadblock executes one task. This relation is a kernel contract. It
is not a global GPU programming rule.

```text
one Map ThreadBlock -> one MapThreadBlockTask
one Reduce ThreadBlock -> one ReduceThreadBlockTask
```

Use these semantic derivations:

```text
MapThreadBlockTask
    = derive(
        MapKernelArguments,
        thread_block_index,
        MapKernelConstants,
      )

MapThreadTask
    = derive(
        MapThreadBlockTask,
        thread_index,
        MapKernelConstants,
      )
```

A conceptual Map task has these fields:

```rust
struct MapThreadBlockTask {
    request_index: u32,

    flat_q_token_indices: Range<u32>,
    q_head_indices: Range<u32>,

    kv_head_index: u32,
    request_local_kv_token_indices: Range<u32>,

    partial_state_group_index: u32,
}
```

One Map task defines this Cartesian product:

```text
flat Q-token range
x Q-head range
x one KV head
x request-local KV-token range
```

The Map task produces one `PartialAttentionState` for each `(flat Q token, Q head)` coordinate. The scalar
`partial_state_group_index` identifies the base of this group. It does not identify one scalar partial state.

`MapThreadTask` is shader-derived. The CPU does not have to construct or upload it. The kernel defines the exact thread
and data layout.

The Reduce task follows the same rule:

```rust
struct ReduceThreadBlockTask {
    request_index: u32,

    flat_q_token_indices: Range<u32>,
    q_head_indices: Range<u32>,

    partial_state_group_indices: Range<u32>,
}
```

```text
ReduceThreadTask
    = derive(
        ReduceThreadBlockTask,
        thread_index,
        ReduceKernelConstants,
      )
```

One Reduce task merges one or more `PartialAttentionState` values for each output coordinate. It writes one final
Q-token and Q-head output range. `ReduceThreadTask` is shader-derived.

### SDPA execution

One current SDPA execution contains two ordered launches:

```text
SDPAExecution
├── Map KernelLaunch
│   └── produces PartialAttentionState values
└── Reduce KernelLaunch
    └── merges PartialAttentionState values and writes final output
```

The Reduce launch depends on Map completion. The current implementation uses the Map/Reduce path even when one final
output has only one partial state.

The generic `PartialAttentionState` boundary is representation-independent. It does not require separate numerator or
denominator buffers.

## Current implementation facts

### Ownership

The GQA component supplies semantic request facts. The Metal backend supplies legal execution variants. The Metal
executor selector owns dynamic work partitioning and complete-candidate comparison.

```text
backend_sdpa::Config
    -> backend_sdpa::Registry
    -> gqa::sdpa::Selector

gqa::sdpa::RequestShape[]
    -> one candidate gqa::sdpa::Selection per legal variant
    -> selection policy
    -> selected gqa::sdpa::Selection
    -> GQAMetadataBuffers::update(...)
    -> Map kernel
    -> Reduce kernel
```

The runtime core does not select an execution variant. It continues to own request scheduling, page allocation,
and cache lifecycle.

### Static variants

`backend_sdpa::Config` contains `io_dtype`, `num_q_heads`, `num_kv_heads`, `head_dim`, and `tokens_per_page`.

`backend_sdpa::Registry` performs static capability filtering. It does not use batch lengths or performance
thresholds. Each legal `backend_sdpa::ExecutionVariant` contains one compatible Map and Reduce pair.

The current Rust constant fields follow the reference hierarchy:

```text
backend_sdpa::ExecutionVariant
├── map
│   ├── thread_block
│   │   ├── max_q_tokens
│   │   ├── max_q_heads
│   │   ├── kv_tokens_per_iteration
│   │   └── required_threads
│   └── kv_cache
│       └── tokens_per_page
└── reduce
    └── thread_block
        ├── max_q_tokens
        ├── max_q_heads
        └── required_threads
```

`map.thread_block.max_q_tokens == 1` identifies the current SingleQ kernel geometry. A larger value identifies the
current TiledQ kernel geometry. The selection does not use a SingleQ/TiledQ selector enum. The concrete low-level
kernel types keep these names because they use different Metal sources and launch geometry.

### Dynamic selection

`gqa::sdpa::RequestShape` contains `num_history_tokens` and `num_q_tokens` for one request. For a causal Q-token offset:

```text
num_visible_kv_tokens = num_history_tokens + q_token_offset + 1
```

The selector creates request-local `gqa::sdpa::QTokenRange` values. A range contains `request_index`,
`flat_q_token_indices`, and `max_visible_kv_tokens`. A `QTokenRange` is dynamic task metadata. It is not a kernel
tile or a complete Map task.

The selector calculates the number of KV iterations:

```text
num_kv_iterations
    = ceil(max_visible_kv_tokens / map.thread_block.kv_tokens_per_iteration)
```

It then distributes consecutive KV iterations across Map tasks. A tail task can contain fewer KV iterations. Adjacent
task ranges do not overlap and do not leave gaps.

`gqa::sdpa::MapTaskTemplate` stores only the fields that the regular grid cannot derive:

```text
q_token_range_index
request_local_kv_token_indices
```

The current GPU ABI flattens one template to three `u32` values:

```text
[q_token_range_index, request_local_kv_token_begin, request_local_kv_token_end]
```

The grid derives the Q-head range and KV-head index. The template and grid coordinates together derive one semantic
`MapThreadBlockTask`. The CPU does not upload a fully materialized `MapThreadBlockTask[]` array.

`cu_sdpa_partial_outputs` selects the partial-state groups that Reduce merges for each Q-token range. Reduce metadata
names the partial states. It does not name the Map template that produced them.

### Selection

`gqa::sdpa::Selection` contains one complete dynamic decision:

```text
variant
q_token_ranges
map_task_templates
cu_partial_outputs_by_q_token_range
replay_shape
metrics
```

The selection does not store a duplicate registry index. The complete execution variant is the replay-topology identity
and the concrete-kernel recording input.

`GQAMetadataBuffers::update(...)` uploads the selection. It does not select another variant or recompute KV
partitioning.

The materialized selection is necessary because the selected variant, Q-token ranges, Map task templates,
partial-state offsets, replay shape, and metrics are one coupled result. A kernel-kind enum or threshold helper cannot
represent this result. Moving candidate construction into `GQAMetadataBuffers` would mix selection with GPU ABI upload.
Returning unrelated parallel values would weaken the boundary and permit mismatched variant and task metadata.

`gqa::sdpa::Selector` is not a stateless wrapper. It owns the legal variant registry and the allocation limits.
For each dynamic workload, it materializes every complete candidate before it compares them. This work allocates and
fills request-local vectors. Callers must not repeat it only to recover one field.

The selector includes candidate materialization. GQA does not have a separate `Planner` type or a `Plan` type. The
`Selection` name describes the returned value and does not introduce a second decision layer.

### Metrics and selection

`gqa::sdpa::SelectionMetrics` names each counted unit:

```text
num_scheduled_qk_token_pairs
num_active_qk_token_pairs
num_map_threadblocks_per_kv_head
num_map_simdgroup_waves_per_kv_head
num_active_partial_state_groups
num_active_partial_states
num_reserved_partial_state_groups
num_replay_reserved_partial_state_groups
max_kv_iterations_per_map_task
num_logical_qk_token_pairs
```

The D=256, eight-token-page policy keeps the measured crossover threshold. It rejects a TiledQ candidate when active QK
token pairs use less than half of its scheduled QK token pairs. It then applies the existing selection score. The score
is a tuning heuristic. It is not a FLOP, byte, token, or elapsed-time unit.

The selector evaluates request-local Q-token ranges. It does not use only aggregate token and context counts. This rule
keeps one-token ragged tails and batches of independent one-token requests on SingleQ.

### Current partial-state ABI

The current Map ABI stores `partial_max_logits`, `partial_exp_sums`, and normalized `partial_output`.
`partial_exp_sums` is the semantic denominator. The current ABI does not store the numerator.

For partial states `a` and `b`:

```text
max = max(max_a, max_b)
denom = exp(max_a - max) * denom_a
      + exp(max_b - max) * denom_b
numerator = exp(max_a - max) * denom_a * output_a
          + exp(max_b - max) * denom_b * output_b
output = numerator / denom
```

A possible future ABI can store `max`, `denom`, and `numerator`. That representation is not part of this refactor.

### Replay and resource contracts

This refactor does not change these contracts:

- `GQAReplayShape` count domains and exact or bucketed meanings.
- Replay parameter keys.
- Partial-state scratch allocation and layout.
- Page-table layout and KV-cache ownership.
- Map-before-Reduce dependency.
- SingleQ and TiledQ numerical kernels.

Replay padding can change the recorded Map grid. Therefore, the selector includes the selected replay capacity when it
materializes and compares candidates.

## Proposed design scope

This change moves dynamic selection and complete-candidate construction into `gqa::sdpa::Selector`. It replaces the
old path enum and duplicate identity with one `backend_sdpa::ExecutionVariant`. It also separates Q-token ranges, Map
task templates, and kernel constant fields.

This change does not propose a new partial-state ABI, replay key, resource contract, KV-segment allocation policy, or
Metal numerical kernel.

## Validation

The implementation must keep registry capability tests, selector crossover tests, ragged-tail tests, metadata ABI
tests, SingleQ and TiledQ Metal parity tests, the Qwen3.8 D=256 page-eight parity test, and representative one-layer
production performance checks.
