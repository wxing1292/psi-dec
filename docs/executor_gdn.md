# GDN Executor

This document describes the current GDN implementation. It covers tensor geometry, state transactions, Metal
projection, short convolution, ragged recurrence, and state-page I/O.

## Source layout

```text
crates/inference-executor-core/src/attn/
  mod.rs                    MLX-free attention module exports
  gdn/
    mod.rs                  GDN module root
    core.rs                 GDNCore metadata
    reference.rs            CPU short-convolution and recurrent correctness oracles
    state.rs                backend-neutral per-request GDNStateTxn lifecycle metadata

crates/inference-executor-metal/src/attn/
  mod.rs                    Metal attention module exports
  gdn/
    mod.rs                  GDN Metal module root
    batch_metadata.rs       state-domain-owned, capacity-sized GPU metadata updated per microbatch
    backend.rs              GDN Metal replay wiring and candidate-state materialization
    scratch.rs              reusable GDN scratch allocation owner and borrowed replay bindings
    request_slots.rs        private CPU request-slot/version/candidate mapping
    request_slots/
      file_io.rs            snapshot readiness validation
    state_table.rs          public GDNRequestStateTable, live arenas, GDNStatePageIO, and lifecycle
    state_table_test.rs     request-state API-set and transaction lifecycle tests
    state_table/
      file_io.rs            symmetric full and selected metadata and arena file I/O

crates/inference-executor-metal/src/model/qwen/
  v3_x/
    layer/gdn.rs            Qwen3xGDN, private checkpoint weights, load, and record
    state/gdn.rs            Qwen3xGDNState prepare/restore/commit/publish/reset lifecycle
    state/gdn/file_io.rs    Qwen3xGDNState full and selected state file I/O
  v3_5/
    main/layer.rs           Qwen3.5 QGKV-GQA/GDN layer variants
    plan.rs                 Qwen3.5 GDN geometry/config builder

crates/inference-backend-metal/src/components/
  gdn_compute.rs        reusable Metal GDN compute graph and kernel specializations
  gdn_compute_test.rs   GDN compute reference tests
  gdn_qkvabz_split.rs  reusable QKVABZ split component
  gdn_state_pages.rs    reusable GDN state-page read/write helpers
  metal/
    gdn_compute.metal        short-convolution, ragged recurrent, and output-norm/gate source
    gdn_qkvabz_split.metal  QKVABZ split source
    gdn_state_page_read.metal
    gdn_state_page_write.metal
```

`crates/inference-executor-core` owns the backend-neutral GDN semantic metadata. `crates/inference-executor-metal` owns
the Metal replay wiring and request state table.

## Tensor and axis vocabulary

GDN names tensors at the boundary that owns their current value. Axis letters have one meaning throughout Rust,
Metal, references, tests, and benches:

```text
R      number of requests, num_reqs
T      number of valid flat tokens across those requests
Hqk    number of Q/K heads
Dqk    Q/K head width
Hv     number of V/state heads
Dv     V/state head width
Cqkv   concatenated Q/K/V channel width
Kc     short-convolution kernel size
Ks     short-convolution history length, Kc - 1
S      state-slot axis
```

`Cqkv = 2 * Hqk * Dqk + Hv * Dv`. `C` identifies only this concatenated channel axis at the projection and
short-convolution boundaries.

`C` does not identify a head axis, head width, or convolution-kernel extent. Short convolution and convolution state
operate independently along `Cqkv`. Their temporal geometry is `Kc`/`Ks`.

The forward tensors are:

```text
hidden_state                 [T, hidden_dim]
qkvabz                       [T, Cqkv + 2 * Hv + Hv * Dv]
qkv                           [T, Cqkv]
a, b                         [T, Hv]
z                            [T, Hv, Dv]
conv_weight                  [Cqkv, Kc]
conv_state                   [S, Cqkv, Ks]
conv_qkv                     [T, Cqkv]
recurrent_state              [S, Hv, Dv, Dqk]
recurrent_output             [T, Hv, Dv]
norm_gated_output            [T, Hv, Dv]
next_hidden_state            [T, hidden_dim]
```

Persistent checkpoint parameters keep their checkpoint storage dtype:

```text
qkvabz/output packed weights       packed U32
qkvabz/output scales and biases    bf16
conv_weight                        bf16
norm_weight                        bf16
a_log                              bf16
dt_bias                            bf16
```

Metal kernels promote BF16 parameters to F32 at the operation that consumes them. The loader does not create persistent
F32 copies or derive `-exp(a_log)`. Convolution and recurrent state remain F32 runtime state.

`qkv` is the projection-split output and short-convolution input. `conv_qkv` is the short-convolution output
and recurrent-core input.

`recurrent_output` is the recurrent result before RMS normalization and the output gate. It is not attention output.
`norm_gated_output` is the normalized/gated tensor that output projection consumes.

Request segments use `flat_token_begin`, `flat_token_end`, `num_req_tokens`, and `token_index_in_req`. Reserve `q` and
`k` for actual Q/K tensor values and coordinates.

`num_*` identifies valid logical work. Reserve `total_*` for a padded dispatch, replay, or scratch extent.

`cu_tokens` has `R + 1` cumulative flat-token counts. Adjacent entries select the half-open flat-token segment that one
request owns.

## Kernel specialization and thread-block tasks

`GDNCompute::new` derives one `GDNComputeSpecialization` from `GDNComputeConfig`. This is an initialization-time static
operation. GDN does not use a runtime registry or planner because the current compute graph has no
workload-dependent implementation choice.

The specialization has this hierarchy:

```text
GDNComputeSpecialization
├── model
│   ├── num_qk_heads
│   ├── qk_head_dim
│   ├── num_v_heads
│   ├── v_head_dim
│   └── conv_kernel_size
└── kernels
    ├── short_conv
    │   └── thread_block.required_threads
    ├── candidate_conv_state
    │   └── thread_block.required_threads
    ├── final_state_recurrent
    │   └── thread_block
    │       ├── num_qk_dim_threads
    │       ├── num_v_rows
    │       └── required_threads
    ├── candidate_state_recurrent
    │   └── thread_block
    │       ├── num_qk_dim_threads
    │       ├── num_simdgroups
    │       ├── simdgroup.num_v_rows
    │       └── required_threads
    └── output_norm_gate
        └── thread_block.required_threads
```

`GDNComputeConfig` also contains `q_scale` and `norm_eps`. The host passes these values as kernel arguments. They are
not compile-time specialization values.

At recording time, `GDNComputeShape` and `GDNComputeSpecialization` derive each kernel execution configuration.
`GDNComputeShape` supplies recorded request and token extents. The specialization supplies required thread-block
geometry. The `dispatch_1d` and `dispatch_threadblocks` calls pass the actual grid and thread-block dimensions. For each
kernel, the actual thread count must equal `thread_block.required_threads`.

The final-state recurrent and candidate-state recurrent kernels implement one recurrent algorithm. They have different
state-materialization contracts. The final-state recurrent kernel can write the state after the last request token. The
candidate-state recurrent kernel can write a distinct state after each request token. The two kernels are not cost
candidates.

The current final-state recurrent kernel uses this task relation:

```text
one ThreadBlock
    → one GDNFinalStateRecurrentThreadBlockTask

GDNFinalStateRecurrentThreadBlockTask
    request_index
    v_head_index
    v_dim_indices: Range<u32>
    flat_token_indices: Range<u32>
```

`v_dim_indices` selects a half-open V-row range for one V head. `flat_token_indices` selects one request's half-open
token range. The thread block owns
`recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk]`. It advances this state over `flat_token_indices` in
order.

The kernel derives `GDNFinalStateRecurrentThreadBlockTask` from kernel arguments, the thread-block index, and
`GDNComputeSpecialization.kernels.final_state_recurrent`. The CPU does not construct or pass a task object.

The shader derives the per-thread work:

```text
GDNFinalStateRecurrentThreadTask
    = derive(
        GDNFinalStateRecurrentThreadBlockTask,
        thread_index,
        GDNComputeSpecialization.kernels.final_state_recurrent,
      )
```

One final-state recurrent thread owns one V row and a strided Dqk fragment. It keeps that fragment thread-local while it
advances the request's token range. The CPU does not construct or pass a thread-task object.

For Qwen's current geometry:

```text
recurrent_state[slot, Hv=32, Dv=128, Dqk=128]

final_state_recurrent.thread_block
  num_qk_dim_threads = 32
  num_v_rows = 8
  required_threads = 256

grid_dimensions = (Dv / num_v_rows, num_total_reqs * Hv, 1)
                = (16,              num_total_reqs * 32, 1)

thread_block_dimensions = (num_qk_dim_threads, num_v_rows, 1)
                        = (32,                     8, 1)
```

One final-state recurrent thread block owns an `[8, 128]` state slice. This slice contains
`8 * 128 * sizeof(f32) = 4 KiB` of logical state.

For `Dqk=128`, one thread owns four strided state values:

```text
                              Dqk = 128
                  round 0     round 1     round 2     round 3
                 qk 0..31    qk 32..63   qk 64..95  qk 96..127
               +-----------+-----------+-----------+-----------+
V row y        | lane x    | lane x    | lane x    | lane x    |
               +-----------+-----------+-----------+-----------+

thread(x, y).state_fragments = [
  state[y, x],
  state[y, x + 32],
  state[y, x + 64],
  state[y, x + 96],
]
```

The 32 x-lanes collectively hold one 128D state row. Eight y-rows collectively hold the complete `[8, 128]` state
slice. MSL declares `thread float state_fragments[4]`. Physical register placement remains a compiler decision.

In each round, the 32 x-lanes access one contiguous 128-byte span (`32 * sizeof(f32)`). Four rounds cover one 512-byte
row. This pattern describes memory access. It does not guarantee one 512-byte hardware transaction.

The candidate-state recurrent kernel uses this task relation:

```text
one ThreadBlock
    → one GDNCandidateStateRecurrentThreadBlockTask

GDNCandidateStateRecurrentThreadBlockTask
    request_index
    v_head_index
    v_dim_indices: Range<u32>
    flat_token_indices: Range<u32>
```

The candidate-state recurrent specialization has `num_qk_dim_threads=32`, `num_simdgroups=2`, and
`simdgroup.num_v_rows=2`. One 64-thread block owns a `[4, 128]` state slice. Each SIMDgroup owns two V rows. The two
SIMDgroups share normalized Q/K values and gate scalars.

The shader derives `GDNCandidateStateRecurrentThreadTask` from the thread-block task, thread index, and the
candidate-state recurrent specialization. One candidate-state recurrent thread owns strided Dqk fragments for two V
rows in its SIMDgroup. It can write those fragments to a row-specific candidate state slot after each token.

The recurrent kernels write `recurrent_output [T, Hv, Dv]`. They preserve token order within one request. Requests,
V heads, V-row ranges, and Dqk lanes remain parallel.

Output norm + gate uses this task relation:

```text
one ThreadBlock
    → one GDNOutputNormGateThreadBlockTask

GDNOutputNormGateThreadBlockTask
    flat_token_index
    v_head_index
```

The grid derives both task fields. One 128-thread block RMS-normalizes and gates one `[Dv]` recurrent-output vector.
The CPU does not construct or pass a task object.

The shader derives `GDNOutputNormGateThreadTask` from the thread-block task, thread index, and output-norm specialization.
One thread owns strided Dv elements and one square-sum partial. The thread block reduces these partials before each
thread writes its normalized and gated output elements.

Short convolution, candidate convolution materialization, and QKVABZ split use flat element dispatches. Their
thread-block grouping is launch geometry. The implementation does not add semantic thread-block task types for these
kernels.

State-page I/O is a separate copy component. Each state-I/O request selects one logical state version, one recurrent
physical slot, one convolution physical slot, and its page IDs across every GDN layer. The read and write kernels use
this task relation:

```text
one ThreadBlock
    → one GDNStatePageReadThreadBlockTask
      or one GDNStatePageWriteThreadBlockTask

GDNStatePageReadThreadBlockTask / GDNStatePageWriteThreadBlockTask
    state_io_request_index
    gdn_layer_index
    state_kind
    page_index_in_state
```

The grid derives all task coordinates. `page_id`, `recurrent_state_slot`, and `conv_state_slot` are data inputs.
`state_kind` selects the applicable physical slot. One thread block copies one page with `float4` lanes.

## Canonical metadata and host/Metal ABI

Canonical host structures use these dynamic work domains:

```text
GDNReplayShape
  num_reqs, num_total_reqs,
  num_tokens, num_total_tokens

GDNComputeShape
  num_total_reqs, num_total_tokens

GDNComputeConfig
  num_qk_heads, qk_head_dim,
  num_v_heads, v_head_dim,
  conv_kernel_size, q_scale, norm_eps

GDNComputeSpecialization
  compile-time model geometry,
  per-kernel thread-block geometry

GDNCore
  model_layer_index, hidden_dim,
  num_qk_heads, qk_head_dim,
  num_v_heads, v_head_dim,
  conv_kernel_size, q_scale

GDNQKVABZSplitConfig
  qkv_dim, num_v_heads, v_dim

GDNQKVABZSplitShape
  num_total_tokens
```

Generic `GDNComputeConfig` is the public construction input. `GDNCompute::new` validates the config and derives one
private `GDNComputeSpecialization`. The specialization contains all generated Metal constants and each kernel's required
thread-block geometry. `GDNCompute` stores `q_scale` and `norm_eps` separately because the host passes them as kernel
arguments.

For a bucketed invocation, `GDNComputeShape` contains recorded capacities. Submission arguments contain active counts.
For an exact invocation, the shape counts are both active counts and dispatch extents. Neither shape causes runtime
kernel selection.

The Qwen adapter supplies dimensions and weights. Generic Rust and Metal contain no Qwen name or config type.

The canonical binding order and dispatch topology are:

```text
QKVABZ split
  buffers 0..4: qkvabz, qkv, a, b, z
  scalars 5..8: num_active_tokens, qkv_dim, num_v_heads, v_dim
  dispatch: num_total_tokens * (Cqkv + 2 * Hv + Hv * Dv), 256 threads/threadblock

short convolution
  buffers 0..7: conv_qkv, next_conv_state, qkv, conv_state,
                conv_weight, src_conv_state_slots, flat_materialized_conv_state_slots, cu_tokens
  parameter dtype: conv_weight bf16
  scalars 8..12: num_active_reqs, num_active_tokens, conv_state_offset_bytes,
                 next_conv_state_offset_bytes, write_final_state
  dispatch: max(num_total_tokens * Cqkv, num_total_reqs * Cqkv * Ks), 256 threads/threadblock

final-state recurrent
  buffers 0..9: recurrent_output, recurrent_state_arena, conv_qkv, a, b,
                a_log, dt_bias, src_recurrent_state_slots,
                flat_materialized_recurrent_state_slots, cu_tokens
  parameter dtype: a_log and dt_bias bf16
  scalars 10..12: q_scale, num_active_reqs, recurrent_state_offset_bytes
  grid: (Dv / kernels.final_state_recurrent.thread_block.num_v_rows,
         num_total_reqs * Hv, 1)
  threads: (kernels.final_state_recurrent.thread_block.num_qk_dim_threads,
            kernels.final_state_recurrent.thread_block.num_v_rows, 1)

candidate-state recurrent
  buffers and scalars: same binding domains as final-state recurrent
  grid: (Dv / kernels.candidate_state_recurrent.thread_block.num_v_rows(),
         num_total_reqs * Hv, 1)
  threads: (kernels.candidate_state_recurrent.thread_block.num_qk_dim_threads,
            kernels.candidate_state_recurrent.thread_block.num_simdgroups, 1)

output_norm_gate
  buffers 0..3: norm_gated_output, recurrent_output, z, norm_weight
  parameter dtype: norm_weight bf16
  scalars 4..5: eps, num_active_tokens
  dispatch: num_total_tokens * Hv * 128, 128 threads/threadblock

batched state-page read/write
  buffers 0..5: pages, recurrent_states, conv_states, page_ids,
                recurrent_state_slots, conv_state_slots
  scalars 6..13: num_gdn_layers, num_state_slots, num_state_io_requests,
                 num_recurrent_pages_per_state_slot, recurrent_state_bytes,
                 num_conv_pages_per_state_slot, conv_state_bytes, page_bytes
  grid: (total_pages, 1, 1), threads: (256, 1, 1)
```

Candidate convolution materialization uses buffers 0..5 for `next_conv_state`, `qkv`, `conv_state`,
`src_conv_state_slots`, `flat_materialized_conv_state_slots`, and `cu_tokens`. Its scalars 6..9 are `num_active_reqs`,
`num_active_tokens`, `conv_state_offset_bytes`, and `next_conv_state_offset_bytes`.

Candidate recurrent materialization uses `src_recurrent_state_slots` at buffer 7,
`flat_materialized_recurrent_state_slots` at buffer 8, and `cu_tokens` at buffer 9. It uses scalars 10..12. One SIMDgroup
owns one `[2, Dqk]` register-resident state slice. Two SIMDgroups share normalized Q/K and gate scalars through
threadgroup memory. The grid is `(Dv / 4, num_total_reqs * Hv, 1)`. The thread-block dimensions are `(32, 2, 1)`.
The candidate-state path requires `Dqk % 32 == 0` and `Dv % 4 == 0`. These requirements are initialization-time
geometry contracts.

The invalid state-slot sentinel is `u32::MAX`. All compute variants use the same row-level contract. A row always
produces its normal output. A kernel writes the row's convolution or recurrent state only when the corresponding domain
entry contains a valid slot.

## Ownership

The Qwen GDN weight owner loads one bounded `TensorMap` from its exact GDN binding subtree.
It removes all QKV/A/B/Z, convolution, norm, state-parameter, and output tensors from that map.
It materializes the fused QKVABZ buffers required by the backend ABI during initialization.
The map must be empty after construction.

`GDNCore` owns immutable layer metadata:

```text
model_layer_index
hidden_dim
num_qk_heads / qk_head_dim
num_v_heads / v_head_dim
conv_kernel_size
q_scale
```

The independent dimensions derive `qk_dim`, `v_dim`, `qkv_dim = Cqkv`, and the convolution history length.
`GDNCore` and the backend invocation shape do not store duplicate fields.

The current GDN Metal execution contract uses BF16 model input and output boundaries. It uses F32 for projection outputs,
GDN compute, and persistent recurrent state. `GDNCore` remains backend-neutral and does not define these data types.

`GDNMetalConfig` owns numeric and storage configuration. It includes norm epsilon, `input_dtype`, `output_dtype`,
`qkvabz_scale_bias_dtype`, and `output_scale_bias_dtype`. The current implementation accepts only BF16 model boundaries.
F32 model boundaries remain explicit future work. The config does not expose GDN kernel specialization controls.

`GDN` owns one adaptive `AffineQuantizedMatmul` for the qkvabz projection and one for the output projection.
Each operator owns its QMV and QMM candidates.
The backend selects the kernel family and tile from the fixed affine config and the recorded row capacity.
The GDN token bucket policy unions both affine topology boundaries. Thus, one bucket cannot cross a kernel-selection
boundary.
`GDN` does not select or name an affine kernel.

`GDN` translates immutable `GDNCore` geometry into `GDNComputeConfig`. `GDNCompute::new` derives one
`GDNComputeSpecialization`. The final-state recurrent specialization uses eight V rows per thread block when
`v_head_dim` permits it. It falls back to four V rows for other valid dimensions. The candidate-state recurrent
specialization uses two V rows per SIMDgroup and two SIMDgroups per thread block. The specialization also records the
required thread count for short convolution, candidate convolution materialization, and output norm + gate.

The selected model and thread-block geometry specializes the generated Metal source. The model adapter does not select
this geometry. `GDNComputeShape` contains only recorded request and token extents. It does not cause runtime kernel
selection.

Kernel source-hash caching shares compiled pipelines for identical component configs across layers and models. The backend
API does not contain model names or model config types. Batch metadata objects and scratch bindings do not copy static
geometry or tuning.

`Qwen3xGDNState` owns one shared `GDN` backend and one shared `GDNScratch` for compatible Main GDN layers. It also owns
the shared `Rc<GDNRequestStateTable>`, reusable `GDNMetadataBuffers`, cached restore replay, and optional pending publish.
Each `Qwen3xGDN` layer owns immutable weights, a compact `gdn_layer_index`, and cloned backend, scratch, and state-table
handles.

Construction validates every shared `GDNCore` against the representative core. Only `model_layer_index` may differ.
The shared backend must have one compute and affine layout. The `GDN` backend retains the device and creates its scratch
from the validated core geometry.

The current Qwen3.5 executor imports and owns `Qwen3xGDNState` directly. Its model layers own `Qwen3xGDN` directly.
Sharing the implementation does not move the GDN lifecycle into another executor. The backend records qkvabz projection,
projection split, recurrent core/state update, optional candidate state materialization, and output projection into the
caller’s `Recorder`.

State preparation also keeps the leaf boundary model-neutral. `Qwen3xGDNState::prepare_states` receives the request-slot,
block-index, token-index, cumulative-token, state-transaction, and state-page slices that `GDNRequestStateTable`
consumes. `prepare_metadata_bucketed` receives cumulative tokens and the prepared state. It selects independent request
and token capacities with the component-local policy. `prepare_metadata_bucketed_with_token_capacity` accepts a token
capacity that a composite replay stage already selected. This path buckets only the private request capacity. It does not
bucket the token capacity again. The Qwen3.5 executor extracts these slices from its own microbatch before it calls the
shared leaf.

`GDNRequestStateTable` owns all GDN layers at model level. It owns two contiguous aggregate arenas: one recurrent arena
and one convolution arena.

Its `num_pages_per_state_slot()` reports the physical page count from the instantiated layout. Service cache capacity uses
this owner-derived value instead of a second GDN shape formula.

```text
recurrent_states[layer, state_slot, v_head, v_dim, qk_dim]
conv_states[layer, state_slot, qkv_channel, conv_history]
```

The logical model-level GDN storage shape is:

```text
recurrent_states[num_gdn_layers][num_state_slots][num_v_heads][v_head_dim][qk_head_dim]
conv_states[num_gdn_layers][num_state_slots][Cqkv][Ks]

one logical state_version -> (recurrent_state_slot, conv_state_slot)

one recurrent slot:
  recurrent_states[gdn_layer_index][recurrent_state_slot]

one convolution slot:
  conv_states[gdn_layer_index][conv_state_slot]

page_ids_staging[state_io_request]
  [num_gdn_layers]
    [num_recurrent_pages_per_state_slot]
    [num_conv_pages_per_state_slot]
recurrent_state_slots_staging[state_io_request]
conv_state_slots_staging[state_io_request]
```

Each arena currently has the same `num_state_slots` physical capacity. The state-table boundary names recurrent and
convolution physical slots independently. Fresh C0-only allocation can produce equal numeric IDs because the two pools
start in the same order. ID equality is not a contract. C0 prepare, compute, commit, publish, restore, and snapshot paths
consume the two IDs independently. The trailing dimensions come directly from the shared GDN core. They are not separate
request-slot axes.

`page_bytes` is the raw allocation unit. Page I/O divides by `sizeof(f32)` only when it indexes f32 state. A layout or
state object never stores that derived capacity.

Runtime page IDs remain CPU transaction data in `GDNStatePages` vectors. `GDNStatePageIO` owns the reusable
`page_ids`, `recurrent_state_slots`, and `conv_state_slots` GPU staging buffers and the batched read/write kernels. It
fills the staging buffers immediately
before restore or publish recording. The buffers do not represent persistent request-page ownership.

At initialization, Qwen wiring derives direct per-request state-slot, candidate-state, and publish-staging capacities.
`GDNRequestStateTable` consumes these direct resource bounds. It does not inspect scheduler, MTP, DSpark, or sampling
configuration.

Speculative candidate suffixes and cache-block boundary versions can be disjoint. Therefore, the resource bound adds
the maximum candidate count and maximum crossed-boundary count. For `P = num_spec_tokens`, both DSpark and MTP have
`P + 1` candidates. The candidates represent accepted counts `0..=P`. MTP shifts the complete candidate range by
`P - 1`. It does not add candidates.
Publish staging permits every block boundary that one maximum-length request can cross across all active request slots.

For `M = max_tokens_per_request` and cache-block width `B`, Qwen3.5 uses this safe per-request bound:

```text
max_materialized_states = (P + 1) + ceil(M / B)
num_state_slots         = 1 + max_materialized_states
                          ^ current state
```

Candidate states and boundary states can overlap. The bound does not depend on that overlap. Prepare removes overlap by
merging the two ordered inputs.

The public table directly owns a private `GDNRequestSlots` mapping, pending restore/publish state transactions, and one
`GDNStatePageIO`. It has no second public state table or mutable aggregate wrapper.

`GDNRequestSlots` owns `current_recurrent_state_slots` and `current_conv_state_slots`. It also owns separate free pools
and separate transaction maps from state version to physical slot for the two domains. `begin_txn(...)` receives
`recurrent_materialized_state_versions` and `conv_materialized_state_versions` independently. Current C0 preparation
passes the same ordered union to both inputs. The owner pairs domain slots only when a committed current state or a
page publish requires both.

`GDNStateTxn` is backend-neutral per-request metadata for one microbatch. It lives from
`GDNRequestStateTable::prepare(...)` through `commit(...)`. The prepare boundary receives request slots, block and token
indices, cumulative token counts, transactions, and runtime state-page IDs. It does not depend on a Qwen microbatch
type.

The transaction stores two canonical half-open ranges:

```text
destination states  [dst_start_state_version, dst_end_state_version)
candidate states    [candidate_start_state_version, candidate_end_state_version)
```

If the first input has `token_index = V`, its output state has version `V + 1`. A state version is also the expected
index of the next input token. The destination range contains the unshifted states that the forward produces. The
candidate range contains the states that commit can select. A candidate can equal the committed source version. In that
case, it reuses the current slot and does not require a forward-row write.

Prepare restores the source state before it materializes the transaction. The restored or current state version must
equal the request `token_index`.

Qwen passes only an unshifted destination version to GDN commit. GDN applies the transaction's candidate shift and
validates the shifted version against the candidate range. Qwen and sampling code do not expose candidate versions.
`GDNStateTxn` is common Qwen3.x GDN metadata.

`GDNMetadataBuffers` is the state-domain-owned, capacity-sized GPU metadata object that all GDN layers share. Prepare
writes `cu_tokens`, `src_recurrent_state_slots`, `src_conv_state_slots`,
`flat_materialized_recurrent_state_slots`, and `flat_materialized_conv_state_slots`. Prepare then returns and stores the
authoritative `GDNReplayShape`.

`GDNMetadataBuffers` is the sole owner of the current replay shape. `GDNInput` borrows the metadata object instead of a
duplicate shape. Backend recording and replay-key construction both read the stored shape.

`GDNStateArenaBindings` borrows both aggregate arenas and the selected layer's `u64` byte bases. Initialization validates
the aggregate arena lengths and layer strides. Record-time layer selection uses direct arithmetic and debug bounds.
Production binds each arena at Metal offset zero. It passes the bases as Metal `ulong` kernel arguments.

`GDNQKVABZSplitBuffers` carries `qkv`, `a`, `b`, and `z`. In qkvabz naming, `a` is the raw gate/dt
projection. `b` is the raw beta projection. `z` is the output gate projection.

`g` is not projected. Gate preparation derives it as part of `beta = sigmoid(b)`,
`g = -exp(a_log) * softplus(a + dt_bias)`, and `decay = exp(g)`. API and docs use q/k/v/a/b/z at projection boundaries.
They reserve `g`/`beta` for prepared low-level core values.

`GDNStateLayout` is the model-owned logical layout of the contiguous allocations. It contains the leading
`[gdn_layer_index, state_slot]` dimensions and `page_bytes`. During construction, the shared GDN core supplies the trailing
recurrent/conv tensor dimensions.

Arena lengths and the leading dimensions derive the per-slot and per-layer byte strides. The layout does not duplicate
these strides. It directly derives aggregate allocation lengths and the all-layer page-ID count. It does not store derived
f32-per-page counts, recurrent/conv page counts, or a selected layer coordinate.

Backend code then runs the recurrent state update and output projection. GDN math keeps `qkv`, gates,
`conv_qkv`, recurrent state, `recurrent_output`, and `norm_gated_output` in f32.

Qwen checkpoint weights and affine parameters remain packed U32 or BF16 in persistent Metal buffers. Quantized matmul
kernels dequantize packed weights and promote BF16 affine parameters during execution. GDN core kernels promote
`conv_weight`, `norm_weight`, `a_log`, and `dt_bias` when they read each value. The recurrent kernel computes
`-exp(a_log)` in F32.

`GDNMetalConfig` requires BF16 at both Qwen3.6 model boundaries. GDN state and internal math remain F32 because BF16 can
cause downstream NaN/Inf.

## Replay contract

`GDN` records one GDN layer forward through `ReplayLayer::record(...)` and a caller-owned `Recorder`. It does not submit
commands. It does not own request scheduling or the request-state lifecycle.

The semantic replay input is:

```text
GDNInput
  hidden_state  &Buffer
  next_hidden_state &Buffer
  batch_metadata  &GDNMetadataBuffers
  state          GDNLayerStateBindings
  scratch        GDNScratchBindings
  materialize_candidate_states
  weights        GDNWeights
  replay_mode    GDNReplayMode
```

`GDNOutput<'a>` is the named alias for the returned `&'a Buffer`. It is the caller-owned `next_hidden_state` buffer.
It does not allocate or add a wrapper.

`GDNReplayMode::Exact` preserves the fixed-scalar leaf APIs. An exact GDN program has no replay parameters.
`GDNReplayMode::Bucketed` records request and token capacities and uses these submission parameters:

```text
gdn.num_active_requests  u32 [1, num_total_reqs]
gdn.num_active_tokens    u32 [1, num_total_tokens]
```

`GDNReplayMode::BucketedWithTokenKey` replaces `gdn.num_active_tokens` with one caller-owned
`ReplayParameterKey`. A composite stage uses this mode so all token consumers share one active-token parameter. The stage
sets that parameter once. GDN adds only its private `gdn.num_active_requests` argument. The default bucketed API and
`add_gdn_replay_arguments` retain both GDN-owned keys for standalone users.

Each command binds only the domains that it consumes:

```text
qkvabz affine                         active tokens
QKVABZ split                          active tokens
short convolution                     active requests and active tokens
candidate convolution materialization active requests and active tokens
final-state or candidate-state recurrent active requests
output norm + gate                    active tokens
output affine                         active tokens
```

One bucketed GDN program therefore has two deduplicated `u32` parameters. Every padded command returns before an
inactive lane reads input or metadata, mutates state, reaches a threadblock barrier, or writes output. Each recurrent
request guard is uniform for the complete thread block.

`GDNReplayBucketPolicy` owns independent request and token policies. The token policy includes the topology boundaries
from both affine operators. `GDNReplayTopology` contains `materialize_candidate_states`, `qkvabz_affine`, and
`output_affine`. The GDN replay subkey contains `num_total_reqs`, `num_total_tokens`, and this topology. Active counts do
not enter the GDN subkey. `replay_token_topology_boundaries` exposes the affine boundaries to a composite-stage policy.
A caller-owned token capacity must contain all active tokens and must not exceed the initialized token capacity. It must
also select the same QKVABZ and output affine topologies as the active token count. GDN validates these conditions before
it updates metadata. Qwen3.5 Main selects one composite token capacity before it updates GQA or GDN metadata. It forces
GDN metadata to use this capacity. The outer Main key records the composite token capacity and the GDN capacity and
topology subkey. Main supplies the stage-owned active-token key. The GDN request count remains a private replay
dimension.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay.

State page restore/publish belongs to `GDNRequestStateTable`, not to individual layer backends. These lifecycle stages
retain exact keys and fixed arguments. They are not part of the forward request/token bucket policy. Runtime supplies one
state-page vector per cache block. The vector contains every GDN layer in model order.

The manager splits that vector into recurrent and convolution page-ID staging. It then records one flattened all-layer
page command:

```text
threadblock -> (state_io_request_index, gdn_layer_index, state_kind, page_index_in_state)
```

Qwen3.5 service replay defines that cache block as 2048 tokens. A GDN snapshot page vector is therefore a state at exactly
2048, 4096, ... tokens. The trie runtime and GQA page tables use the identical logical boundary.

Physical GQA pages remain smaller. The logical block groups these pages. This alignment remains a requirement. A trie
prefix hit must never begin where the corresponding GDN state snapshot cannot exist.

Qwen3.5 derives every GDN layer from one text configuration. Runtime state-page sizing uses the same single-layer
dimensions multiplied by the GDN layer count. `GDNRequestStateTable::new` validates that all GDN cores share these
dimensions. This init-time invariant lets the page kernel use arithmetic indexing. It does not need a per-layer layout
table or executor layout object.

GDN prepare separates at its real dependency boundary. It executes synchronously on the executor thread.
`GDNRequestStateTable::prepare(...)` validates and applies request/version/page transactions. It returns prepared state
slots. `GDN::prepare(...)` then writes the dependent `GDNMetadataBuffers`.

The executor makes explicit sequential calls for Main GQA page preparation and metadata, GDN state preparation and
metadata, and optional MTP GQA page preparation. It has no prepare worker, channel, receiver, or inferred reset.

All prepare branches and any restore complete before main-model replay begins. Publish is a separate replay. The executor
submits it after host commit selects verified state versions.

The model retains the publish submission while it returns the device response to the scheduler. It does not wait in
`commit_batch(...)`.

The next `prepare(...)` or request-slot reset waits for the pending publish. The wait occurs before page/state staging
mutation or more model work. This wait is the device-state happens-before boundary.

The scheduler can commit trie metadata, release a terminal request, or reassign a page ID while publish is in flight.
These operations change host-only ownership. Any later GPU use of that page ID returns through the same executor. It first
crosses the pending-publish wait.

Dropping the model also waits through `ReplaySubmission` ownership. Runtime core does not own or poll a Metal submission.

The replay order is:

```text
hidden_state (BF16)
  -> qkvabz: AffineQuantizedMatmul (BF16 -> F32)
  -> qkvabz (F32)
  -> qkvabz_to_qkv_a_b_z
     |- qkv (F32)
     |- a (F32)
     |- b (F32)
     `- z (F32)
          |
          v
       GDNCompute (F32)
     short_conv -> final_state_recurrent -> output_norm_gate
          |
          v
       norm_gated_output (F32)
          |
          v
       output: AffineQuantizedMatmul (F32 -> BF16)
          |
          v
       next_hidden_state (BF16)
```

Stage nouns identify the operation. They do not overload one generic “attention” pipeline:

```text
qkvabz_to_qkv_a_b_z   elementwise map from qkvabz to qkv/a/b/z
short_conv              temporal convolution from qkv to conv_qkv plus next_conv_state
final_state_recurrent   ordered recurrent state transition and recurrent_output production
output_norm_gate        per-(token,V-head) RMS reduction, norm, and z-gate operation
```

In recurrent execution, each Q/K lane produces `q_square_sum_partial`, `k_square_sum_partial`, `state_k_partial`, and
`state_q_partial`. SIMD reductions produce `q_square_sum`, `k_square_sum`, `state_k_dot`, and one
`recurrent_output_value`. These values are local reduction values. They are not extra global tensors or Task fields.

Output norm + gate uses `square_sum_partial` and threadgroup `square_sum_partials` before it computes the inverse RMS.
No partial changes the existing dispatch, scratch, or ABI.

`GDNCompute` owns one mathematical recurrent algorithm. It has a final-state kernel and a candidate-state kernel for
the two materialization contracts. It handles one or more flat tokens per request with
`cu_tokens`. Each recurrent kernel computes Q/K inverse norms, decay, and beta. It advances each request's tokens in
order. It parallelizes across requests, V heads, V-row ranges, and Q/K-dimension lanes.

### Execution strategy

The final-state recurrent and candidate-state recurrent kernels use the same operation order and request segmentation.
They use independently tuned V-row partitions for their
different write contracts. Do not add `GDNComputePath`, a registry, or a runtime planner until a second mathematical
production implementation exists with a workload-dependent crossover. The current path is:

```text
shape: num_tokens >= num_reqs, segmented by cu_tokens
parallelism: request x v_head x V-row range, with Q/K-dimension lanes inside the threadblock
input: one or more contiguous rows per request
state: load source-state fragments once, advance them in MSL thread-local storage, then optionally write the final state
```

The final-state kernel uses 32 Q/K-dimension threads and eight V rows for Qwen's current geometry. This geometry produces
a 256-thread block. The candidate-state kernel uses two 32-thread SIMDgroups and two V rows per SIMDgroup.
`v_head_dim` and the selected V-row count derive the number of thread blocks. The backend does not store this dynamic
count. The final-state dataflow is:

```text
recurrent_state_arena[src slot, v_head, 8 V rows, 128 Dqk values]
                              |
                              | four 32-lane load rounds per row
                              v
distributed thread-local state_fragments[4] per thread
                              |
                              | for flat_token_begin .. flat_token_end, in order
                              |
             +----------------+----------------+
             |                                 |
             | Q/K stream loads                | state recurrence
             | from conv_qkv                    | on thread-local fragments
             |                                 |
             | lane-local square/dot partials  | decay and rank-one update
             | -> simd_sum                     | -> state_k/state_q simd_sum
             +----------------+----------------+
                              |
                              | optional segment-end final store
                              v
recurrent_state_arena[dst slot, v_head, 8 V rows, 128 Dqk values]
                              or discard when dst slot is u32::MAX
```

The normal final-state kernel reads only the materialization entry for `flat_token_end - 1`. It can store only the
segment-end state. The candidate-state register-V kernel reads each row's materialization entry while it scans the
segment. It can store registered candidate and cache-boundary states after any current row.

For each token, Q and K stream from global memory/cache. Lane-local scalar partials accumulate them before SIMD-group
reductions.

The recurrent kernel's threadgroup storage contains only four scalars: `q_inv_norm_shared`, `k_inv_norm_shared`,
`decay_shared`, and `beta_shared`. It has no threadgroup-resident state or Q/K buffer. This design uses register-oriented
state residency, not double buffering.

The threadblock walks a request segment in token order. Token `t + 1` depends on token `t`'s updated recurrent state.
Separate requests, V heads, and V-row ranges remain parallel. The current backend has no alternative recurrent
execution mode.

GDN bench fixtures distinguish fresh state from state-present execution. `ctx=0` leaves source conv/recurrent state
zeroed. `ctx>0` initializes only the source slot with deterministic non-zero data. It leaves the candidate destination
slot zeroed.

This setup matches the production lifecycle. The lifecycle reads a verified current state and produces a candidate state.

The replay shape keeps active work separate from recorded capacities:

```text
num_reqs          active request count
num_total_reqs    recorded request capacity
num_tokens        active flattened-token count
num_total_tokens  recorded flattened-token capacity
cu_tokens         active prefix length num_reqs + 1
```

Each active request must contribute at least one token. Thus, `num_reqs <= num_tokens`. Request and token capacities are
independent and do not require `num_total_reqs <= num_total_tokens`. Inactive metadata tails do not represent requests,
tokens, or state. The committed recurrent and convolution source slots represent existing context.

The state contract is slot based:

```text
GDNRequestStateTable
  current recurrent and convolution state slots per request slot
  current state_version per request slot
  txn materialized state_version -> recurrent and convolution state-slot mappings
  txn cache-boundary publish state_version -> page_ids mappings

src_recurrent_state_slots          current recurrent source slot per request
src_conv_state_slots               current convolution source slot per request
flat_materialized_recurrent_state_slots
                                    persistent recurrent slot per forward row, or u32::MAX
flat_materialized_conv_state_slots
                                    persistent convolution slot per forward row, or u32::MAX
conv_state               f32 slot arena for convolution state
next_conv_state          destination conv-state arena; may be the same backing as conv_state
recurrent_state_arena    f32 slot arena for recurrent state
```

When `conv_state` and `next_conv_state` share backing storage, source and destination slot IDs must name distinct slots
for committed updates. Qwen replay allocates current and candidate state slots from the request-state table.

Each forward starts a transaction and registers two state-version sets:

```text
candidate_state_versions
  half-open range that commit can select as the new current state

publish_state_versions
  ordered cache-boundary versions whose selected snapshots can be written to runtime-owned state pages
```

Prepare materializes this ordered union:

```text
publish_state_versions where version < candidate_end_state_version
union
[candidate_start_state_version, candidate_end_state_version)
```

Both inputs are ordered and unique. Prepare merges them in one pass. It does not sort the result. A publish version can
precede `candidate_start_state_version`. A publish version at or after `candidate_end_state_version` remains pending for
a later transaction.

The two `flat_materialized_*_state_slots` arrays map each materialized version to the forward row that produces it. All
other rows contain `u32::MAX`. Fresh aligned C0-only pools can assign the same numeric ID to both domains. The metadata,
compute, selected snapshot, restore, publish, and page-I/O boundaries never depend on that equality. Commit promotes the
selected recurrent and convolution candidate slots independently. It releases the other transaction slots in each
domain.

A commit to the current version leaves the current recurrent and convolution slots unchanged. It clears uncommitted
transaction slots in both domains.

Speculative Main verification must not promote a candidate written after rejected rows. Qwen wiring sets the candidate
version range to the versions that commit can select. GDN materializes those versions without interpreting their model
meaning.

For one Main request, Qwen calculates `num_fixed_tokens = q_len - num_spec_tokens`.
It selects `input_state_version + num_fixed_tokens + num_accepted_tokens`.
For input state 93, two fixed tokens, and two speculative tokens, accept counts 0, 1, and 2 select states 95, 96, and 97.
`num_spec_tokens` does not directly change this calculation.

The cache-lane topology owns the candidate shift:

```text
candidate_shift = num_cache_lanes.saturating_sub(2)

Vanilla  L = 1      shift = 0
DSpark   L = 1      shift = 0
MTP      L = P + 1  shift = P - 1
```

GDN converts a destination version to a candidate version with
`candidate_state_version = dst_state_version - candidate_shift`. It uses the inverse addition only when it must map a
candidate version back to a destination decision. `QueryTokens::Prefill` commits its full window and uses shift zero.

The complete MTP candidate range shifts. Its length stays equal to the DSpark range length:

```text
P = 3, S = 3, shift = P - 1 = 2

DSpark Main rows        fixed   spec0   spec1   spec2
state after row           V+1     V+2     V+3     V+4
accepted count              0       1       2       3
candidate state           V+1     V+2     V+3     V+4

MTP Main rows           fixed0  fixed1  fixed2  spec0   spec1   spec2
state after row           V+1     V+2     V+3     V+4     V+5     V+6
accepted count              0       1       2       3
candidate state           V+1     V+2     V+3     V+4
                          |------- S + 1 states -------|

MTP destination range    [V+1, V+7)
MTP candidate range      [V+1, V+5)
```

The MTP destination end can follow the candidate end. This condition does not require another candidate state.

A cache-block boundary before `candidate_end_state_version` can be published by this transaction. It can precede the
candidate range. The state table allocates one materialized slot for each boundary that is not already a candidate.

The normal GDN forward materializes requested states while it scans rows. If the final row is not materialized, its state
remains local to the kernel and is discarded. Commit retains the selected candidate state and discards the other
transaction states before the next forward.

Cache-boundary publish is a separate requirement. When commit selects a registered publish version, publish must write
the matching candidate/current slot to its page IDs.

GDN page read/write helpers remain separate recordable backend-metal components. They restore or publish verified state
pages. Runtime core owns page IDs and cache notifications. The model executor owns GDN state layout, request-slot
interpretation, and candidate slot promotion. It owns only CPU transaction copies of runtime-provided page-ID vectors.

`begin_txn(...)` registers candidate state-slot mappings and future immutable-page mappings.
It stores page mappings as typed `GDNStatePages` values for the current request txn.
After registration, `candidate_recurrent_state_slot(...)` and `candidate_conv_state_slot(...)` are read-only lookups.
Each lookup asserts only if the requested version is absent from that state domain.

`restore(...)` returns a `GDNStateRestore` job. It updates the table's current recurrent and convolution slots and its
current state version.

`commit_txn(...)` returns `GDNStatePublish` jobs for registered publish versions that the committed path satisfies. Qwen
includes publish versions inside the current forward in the candidate-state materialization set. A commit can therefore
publish intermediate cache boundaries through the verified committed version from already-materialized slots.

Publish versions beyond the current forward remain queued until a later transaction materializes and commits them.

If an earlier txn registered a future publish version that later falls inside a forward, Qwen adds that version to the
candidate materialization set. This rule applies even if the current batch does not repeat its page IDs.

Qwen compacts publish jobs into model-owned recurrent page-ID, convolution page-ID, and state-slot staging buffers. Page
IDs remain state-I/O-request-major across all GDN layers. Restore records one all-layer batch read before model forward.
Publish records one all-layer batch write after commit.

Publish is a separate replay from main forward and sampling. It consumes the already-selected committed state. It does not
affect response tokens. It can execute while the scheduler processes that response.

Qwen model replay keeps selected-path GDN transient scratch in one model-owned `GDNScratch`. The shared `GDN` backend
creates this scratch from its retained device and validated geometry. The scratch includes the F32 qkvabz
projection/split buffers and the F32 convolution/core/output-gate buffers. GDN layers execute serially in the replay
slice. Therefore, this scratch is reusable across layers.

State-page I/O writes directly between global state pages and the model-owned contiguous state arenas. It does not use
page-value scratch. Every production state kernel binds the aggregate arena at Metal offset zero.

Forward kernels receive an initialization-validated host `u64` layer byte base. They add it with Metal `ulong`. Page
I/O derives the all-layer state address directly with `ulong`.

Layer-local element indices remain `uint`. The executor validates them independently from the aggregate arena allocation.
This design preserves contiguous storage without an ICB nonzero buffer-binding offset above 4 GiB. It matters for MTP
rejection because a committed prefix can select an intermediate candidate state.

Per-layer owners retain weights and immutable component configuration. `GDNRequestStateTable` shares current/candidate
state, request-slot lifecycle, page-ID staging, and restore/publish jobs. Their versions and slots are common across all
GDN layers.
Weight reload uses the retained core and Metal configuration that created each backend.

During `unload_state`, each layer first drops its shared GDN backend, scratch, and request-state resources.
The model state owner then releases the final references, batch metadata, state arenas, and page-I/O resources.
State load rebuilds transient resources before it restores the full arenas and durable request metadata.

Durable metadata includes current slots, state versions, free-slot order, and future publish page IDs.
Submitted restore jobs, submitted publish jobs, and current batch transactions are transient.
The executor must finish or clear them before it writes a snapshot.

## State data flow

The replay-order section defines the hidden-state pipeline. Mutable request state flows beside it:

```text
src_recurrent_state_slots[num_reqs]  committed recurrent source slot for each request
src_conv_state_slots[num_reqs]       committed convolution source slot for each request
flat_materialized_recurrent_state_slots[num_tokens]
                                      persistent recurrent slot per flat token, or u32::MAX
flat_materialized_conv_state_slots[num_tokens]
                                      persistent convolution slot per flat token, or u32::MAX
conv_states[layer, slot, Cqkv, Ks]
recurrent_states[layer, slot, v_head, v_dim, qk_dim]
```

Short convolution reads the source conv-state slot and `qkv`. It writes `conv_qkv` for every current row. It writes the
next conv-state only when the row's convolution state slot is valid.

The recurrent core reads `conv_qkv`, raw F32 `a`/`b`, and raw BF16 `a_log`/`dt_bias`. It promotes the BF16 parameters
and derives normalized q/k, beta, decay, and output values in F32. It then advances the recurrent state in token order
for each request segment.

For candidate materialization, one SIMDgroup loads a `[2, Dqk]` source-state slice into registers. It keeps the slice
local across the segment. Two SIMDgroups share one normalized Q/K vector and gate pair. Each SIMDgroup writes its slice
to persistent state only when the row's recurrent state slot is valid. The operation order remains
`decayed_state = state * decay`, `delta = (v - decayed_state * k) * beta`, and
`state = decayed_state + k * delta`.

Candidate state materialization is part of the normal forward. If the first input token has index `V`, row zero produces
state version `V + 1`. Each later row increments the version by one.

If that version appears in the materialized union, the core writes the current conv/recurrent state into that row's
corresponding domain slots. The union contains commit candidates and cache-boundary candidates. Commit receives a
destination version, converts it to a candidate version, and selects the corresponding recurrent and convolution
slots.

Cache-boundary publish separately consumes the same materialized candidate/current slots. It emits a publish job only when
the committed verified path satisfies that publish version.

The important invariant is:

```text
all selectable versions must be materialized during the forward that computes them
commit shifts and validates one destination state_version in the GDN owner
publish writes only committed/verified versions
rejected speculative rows leave their candidate slots uncommitted
```

The recurrent algorithm handles every current row shape. This includes decode and MTP verification batches with one or more
rows per request, segmented by `cu_tokens`. One thread block selects a request, V head, and V-row range. Its
Q/K-dimension lanes load distributed source-state fragments. They then scan the request segment in order.

For `num_tokens=1,num_reqs=1`, this operation is still a one-step state update. For
`num_tokens=spec+1,num_reqs=1`, it verifies the full Main segment. It materializes any requested prefix candidate
versions while it scans.

Restore and publish page I/O are outside the core math:

```text
restore before forward
  runtime page IDs -> current recurrent and convolution slots
  updates GDNRequestStateTable current state_version

forward
  current recurrent and convolution slots -> candidate slots in each domain
  may materialize prefix/cache-boundary candidate versions

commit after rejection/sampling
  state_version -> current recurrent and convolution slots
  satisfied publish versions -> page write jobs

publish
  committed recurrent and convolution slots -> runtime page IDs
```

Runtime core owns state page IDs and cache lifecycle notifications. The executor owns GDN state tensor layout,
request-slot current/candidate slot mapping, and all-layer page-I/O command records.

`state_version` is the canonical absolute coordinate of verified mutable state. Immutable fp32 state pages are boundary
checkpoints. Restore loads one into mutable state after a prefix hit. Publish writes only a verified commit.

Backend page-I/O components receive compact page IDs, recurrent and convolution state slots, and `page_bytes`. Request
slots, versions, cache policy, and Qwen transaction semantics remain in the model-level state owner.

## Profile keys

The GDN benchmark uses these subcomponent names:

```text
qkvabz
qkvabz-to-qkv-a-b-z
compute
output
```

Do not add dynamic values to profile paths.

## GDN kernel family

The current forward replay path uses `GDNCompute` in `crates/inference-backend-metal/src/components/`. It records QKVABZ
split, short convolution, ragged recurrent, and `output_norm_gate` through explicit replay invocations. State page read and
write helpers belong to the separate exact restore/publish lifecycle stages.

Focused backend tests, component benches with parity checks, and Qwen real-weight wrapper/layer tests provide correctness
coverage. Slow/reference implementations are test oracles. They are not runtime fallbacks.

`gdn_compute_test.rs` compares Metal execution with the CPU short-convolution and recurrent references. It covers fixed
one-request ragged decode, random ragged input, and a random multi-request ragged batch. Candidate-state tests compare each
speculative prefix state with an independently evaluated CPU prefix reference. The bucketed candidate test reuses one
program for `1 -> 2 -> 1` active requests and tokens. It poisons inactive token and metadata tails. It verifies active
output/state parity and inactive scratch/state canaries. Its one-request rounds also use `u32::MAX` as the final
row slot and verify that normal output remains correct while persistent state stays unmodified.

The state-page component test writes and reads two requests across two GDN layers. It compares the written page layout
with an independent expected layout. It also verifies the selected recurrent and convolution slots and preserves all
unselected state-slot canaries.

## Tests and benches

Recommendation: Current tests and benches cover these areas:

- Backend component correctness
- Real-weight GDN wrapper correctness
- State slot promotion
- Page read/write helpers
- Qwen layer integration

Current benches:

```text
cargo bench -p inference-backend-metal --bench gdn_attn
cargo bench -p inference-backend-metal --bench gdn_state_io
cargo bench -p inference-executor-metal --bench qwen35_gdn -- \
  --model-dir <35b-a3b-model-dir> --tokens 1 --contexts 0 --num-reqs 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_gdn -- \
  --model-dir <35b-a3b-model-dir> --tokens 2 --contexts 128 --num-reqs 1 \
  --candidate-states --subcomponents --iters 1 --warmup-iters 0 --runs 1
```

Append `-- --profile-time 1 --noplot` to either backend Criterion target for a representative full-target smoke run.

`gdn_attn` records `GDNCompute` with state and candidate-state-update building blocks into Metal replay/ICB paths.
`gdn_state_io` covers the reusable GDN state-page read/write component. Neither bench exposes direct-submit component or
forward wiring.

The full-forward `qwen35_gdn` bench uses CLI arguments, not environment variables. Across GQA and GDN, `--tokens` is the
total current microbatch row count. `--num-reqs` is the number of request segments in that microbatch. `--contexts` means
context/state that exists before the measured forward.

The bench distributes rows as evenly as possible across requests. It builds `cu_tokens`, source state slots, and
candidate destination slots from these options.

`--candidate-states` materializes every current row into a distinct candidate slot. It selects the production
convolution and recurrent candidate-state kernels. `--subcomponents` reports the full set of projection, split,
compute, and output subcomponents. Candidate compute uses the `gdn.compute_candidate_state` key. Normal recurrent
compute uses the `gdn.compute` key.

For current GDN paths, the source state slot represents prior history. The bench reports `ctx` for comparison hygiene.
The value does not change recurrent kernel metadata yet. Invalid batch-shape combinations print a structured `skip` line.

The current backend records explicit data-dependency barriers. The replay layer also infers RAW/WAR/WAW hazards from
declared buffer usage. It does not add a conservative every-command fallback.

This bench loads real Qwen3.6 GDN weights. It adapts separate checkpoint qkv/a/b/z projections into the executor qkvabz
replay layout without changing their checkpoint dtype. It measures the full replay path: qkvabz projection, projection
split, `GDNCompute`, and output projection. Do not compare component-only `GDNCompute` or candidate-state-update timings
with full-forward numbers.

Recommendation: GDN replay debugging separates transient scratch from persistent state. Layers execute serially.
Thus, model-level code can reuse projection/core scratch.

Current/candidate conv/recurrent slot arenas are model-owned persistent resources.
GPU page-ID staging buffers are transient model-owned resources.

`GDNRequestStateTable` owns CPU-side current recurrent and convolution state slots, current `state_version`s,
transaction candidate mappings, future publish page IDs, and submitted restore/publish jobs.

`GDNRequestStateTable` and `Qwen3xGDNState` implement `FullStateIO` with `GDNStateSnapshotFiles`. The metadata path uses
native-endian `wincode` directly on `GDNRequestSlots`. It does not clone state into a snapshot DTO. Stop validates the
live transaction and future-publish invariants before serialization. Start decodes the table from the same executor
instance and attaches it only after all GDN resources load successfully.

Recommendation: Barrier audits follow this data flow:

1. Batched state page read
2. Core update
3. Candidate write
4. Verified commit/publish

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
