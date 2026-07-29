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
    backend.rs              GDN Metal replay wiring and core candidate state update sub-backend
    scratch.rs              reusable GDN scratch allocation owner and borrowed replay bindings
    request_state_table.rs  private CPU request-slot/version/candidate mapping
    state_table.rs          public GDNRequestStateTable, live arenas, GDNStatePageIO, and lifecycle

crates/inference-executor-metal/src/model/qwen/
  v3_x/
    layer/gdn.rs            Qwen3xGDN, private checkpoint weights, load, and record
    state/gdn.rs            Qwen3xGDNState prepare/restore/commit/publish/reset lifecycle
  v3_5/
    main/layer.rs           Qwen3.5 QGKV-GQA/GDN layer variants
    plan.rs                 Qwen3.5 GDN geometry/config builder

crates/inference-backend-metal/src/components/
  gdn_attention.rs      reusable Metal GDN core component kernels
  gdn_projection.rs     reusable Metal GDN projection-split component kernels
  gdn_state_pages.rs    reusable Metal GDN single-state and batched state-page read/write helpers
  metal/
    gdn_core.metal                  short-convolution, ragged recurrent, and output-norm/gate source
    gdn_projection_split.metal      projection-split source
    gdn_state_page_read.metal       batched state-page restore source
    gdn_state_page_write.metal      batched state-page publish source
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
projected_qkv                [T, Cqkv]
a, b                         [T, Hv]
z                            [T, Hv, Dv]
conv_weight                  [Cqkv, Kc]
conv_state                   [S, Cqkv, Ks]
conv_qkv                     [T, Cqkv]
recurrent_state              [S, Hv, Dv, Dqk]
recurrent_output             [T, Hv, Dv]
pre_output_hidden_states     [T, Hv, Dv]
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

`projected_qkv` is the projection-split output and short-convolution input. `conv_qkv` is the short-convolution output
and recurrent-core input.

`recurrent_output` is the recurrent result before RMS normalization and the output gate. It is not attention output.
`pre_output_hidden_states` is the normalized/gated tensor that output projection consumes.

Request segments use `flat_token_begin`, `flat_token_end`, `num_req_tokens`, and `token_index_in_req`. Reserve `q` and
`k` for actual Q/K tensor values and coordinates.

`num_*` identifies valid logical work. Reserve `total_*` for a padded dispatch, replay, or scratch extent.

`cu_tokens` has `R + 1` cumulative flat-token counts. Adjacent entries select the half-open flat-token segment that one
request owns.

## Tiles, Tasks, threadblocks, and grids

`GDNRecurrentStateTile` is the smallest matmul-like logical GDN compute tile. It is one `[Dv_tile, Dqk]` slice of
`recurrent_state` for a fixed state slot and V head.

The configuration names `Dv_tile` as `v_dim_tile_size`. The full recurrent state head is `[Dv, Dqk]`.

For Qwen's current geometry:

```text
recurrent_state[slot, Hv=32, Dv=128, Dqk=128]

grid:        (Dv / Dv_tile, num_reqs * Hv, 1)
           = (16,           num_reqs * 32, 1)

threadblock: (32, Dv_tile, 1)
           = (32, 8,       1)
           = 256 threads

threadblock(req, v_head, v_dim_tile)
  owns recurrent_state[slot, v_head, 8 * v_dim_tile .. 8 * (v_dim_tile + 1), 0..128]
  owns one [Dv_tile=8, Dqk=128] tile
  owns 8 * 128 * sizeof(f32) = 4 KiB of logical state
```

One logical `GDNRaggedRecurrentTask` maps 1:1 to one ragged recurrent threadblock. It owns one state tile and advances
that tile once per request token:

```text
GDNRaggedRecurrentTask {  // logical; one per threadblock
  req_index          grid-derived from threadblock_position.y / Hv
  v_head_index       grid-derived from threadblock_position.y % Hv
  v_dim_tile_index   grid-derived from threadblock_position.x
  flat_token_begin   derived from cu_tokens[req_index]
  flat_token_end     derived from cu_tokens[req_index + 1]
}
```

The grid is `(Dv / Dv_tile, R * Hv, 1)`. The threadblock shape is `(32, Dv_tile, 1)`.

The grid and `cu_tokens` derive every Task coordinate. Therefore, the implementation does not materialize a Task value,
TaskTemplate, or ABI buffer.

Thread position `x = 0..31` selects a Dqk lane. Position `y = 0..7` selects a V row in the tile.

For Qwen `Dqk=128`, one thread owns four strided state values:

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

32 x-lanes collectively hold one 128D state row.
Each y-row maps to one SIMD group; 8 SIMD groups collectively hold the complete [8, 128] tile.
```

MSL declares the fragment as `thread float state_fragments[4]`. The compiler normally keeps these thread-local values
in registers when possible.

Physical register placement remains a compiler decision. Source loads and final destination stores use four fragment
rounds.

In each round, the 32 x-lanes access one contiguous 128-byte span (`32 * sizeof(f32)`). Four rounds cover one 512-byte
row.

This pattern describes memory access. It does not guarantee one 512-byte hardware transaction.

The threadblock loads its source tile into distributed thread-local fragments. It advances them across the request's
flat-token segment.

The threadblock writes `recurrent_output [T, Hv, Dv]`. It materializes registered prefix candidates when requested.
It then writes the final fragments to the destination recurrent-state slot.

```text
parallel: requests, V heads, V-dimension tiles, Dqk lanes
ordered:  tokens within one request
```

State-page I/O is a copy operation, not matmul-like math. Therefore, it has no required `*Tile`.

Each state-I/O request selects one state slot and its page IDs across every GDN layer and state kind. The owning Metal
kernels map one logical `GDNStatePageReadTask` or `GDNStatePageWriteTask` 1:1 to one threadblock:

```text
GDNStatePageReadTask / GDNStatePageWriteTask {  // logical; one per threadblock
  state_io_request_index  grid-derived
  gdn_layer_index         grid-derived
  state_kind              grid-derived: recurrent or convolution
  page_index_in_state     grid-derived
}
```

The implementation does not materialize a Task value, TaskTemplate, or ABI buffer. `page_id` and `state_slot` are data
inputs, not Task coordinates.

One threadblock copies one page with `float4` lanes. The grid launches all requested state-page copies.

Output norm + gate is a cooperative reduction/map, not a matmul-like Tile. One comment-only
`GDNOutputNormGateTask` maps 1:1 to one 128-thread threadblock.

The task owns `{ flat_token_index, v_head_index }`. The grid derives both coordinates. The task RMS-normalizes and
gates one `[Dv]` recurrent-output vector.

Short convolution and projection split use flat map dispatches. Their threadblock grouping is incidental launch
tuning. Thus, their documentation describes tensors and grids without new `*Task` or `*Tile` nouns.

## Canonical metadata and host/Metal ABI

Canonical host structure order is unchanged:

```text
GDNCoreShape / GDNReplayShape
  num_reqs, num_tokens

GDNCoreConfig
  num_qk_heads, qk_head_dim,
  num_v_heads, v_head_dim,
  conv_kernel_size, v_dim_tile_size

GDNCore
  model_layer_index, hidden_dim,
  num_qk_heads, qk_head_dim,
  num_v_heads, v_head_dim,
  conv_kernel_size, q_scale

GDNProjectionSplitShape
  num_tokens, qkv_dim, num_v_heads, v_dim, input_dtype
```

Generic `GDNCoreConfig` owns static geometry and tuning. The replay shape/key owns dynamic batch work.

The Qwen adapter supplies dimensions, weights, and measured defaults. Generic Rust and Metal contain no Qwen name or
config type.

The canonical binding order and dispatch topology are:

```text
projection split
  buffers 0..4: qkvabz, projected_qkv, a, b, z
  scalars 5..8: num_tokens, qkv_dim, num_v_heads, v_dim
  dispatch: T * (Cqkv + 2 * Hv + Hv * Dv), 256 threads/threadblock

short convolution
  buffers 0..7: conv_qkv, next_conv_state, projected_qkv, conv_state,
                conv_weight, src_state_slots, dst_state_slots, cu_tokens
  parameter dtype: conv_weight bf16
  scalars 8..11: num_reqs, num_tokens, conv_state_offset_bytes,
                 next_conv_state_offset_bytes
  dispatch: max(T * Cqkv, R * Cqkv * Ks), 256 threads/threadblock

ragged recurrent
  buffers 0..9: recurrent_output, recurrent_state_arena, conv_qkv, a, b,
                a_log, dt_bias, src_state_slots, dst_state_slots, cu_tokens
  parameter dtype: a_log and dt_bias bf16
  scalars 10..13: q_scale, num_reqs, num_tokens, recurrent_state_offset_bytes
  grid: (Dv / Dv_tile, R * Hv, 1)
  threads: (32, Dv_tile, 1)

output_norm_gate
  buffers 0..3: pre_output_hidden_states, recurrent_output, z, norm_weight
  parameter dtype: norm_weight bf16
  scalars 4..6: eps, num_reqs, num_tokens
  dispatch: T * Hv * 128, 128 threads/threadblock

batched state-page read/write
  buffers 0..4: pages, recurrent_states, conv_states, page_ids, state_slots
  scalars 5..12: num_gdn_layers, num_state_slots, num_state_io_requests,
                 num_recurrent_pages_per_state_slot, recurrent_state_bytes,
                 num_conv_pages_per_state_slot, conv_state_bytes, page_bytes
  grid: (total_pages, 1, 1), threads: (256, 1, 1)
```

Candidate recurrent materialization adds `flat_candidate_state_slots` at buffer 9. It shifts `cu_tokens` to 10 and
uses scalars 11..14.

The invalid candidate-slot sentinel remains `u32::MAX`. The kernel does not write for that token.

## Ownership

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

`GDNCore` fixes the internal dtype to f32 and the output boundary to bf16. Backend contract methods define both values.
Configurable fields do not define them.

`GDNMetalConfig` owns shared execution tuning and numeric configuration. It includes the recurrent `Dv_tile` size, norm
epsilon, input dtype, `qkvabz_scale_bias_dtype`, and `output_scale_bias_dtype`.

The current mixed-dtype affine path owns one QMV BN8/BK32 kernel and one QMM BM32/BN32 kernel per projection.
`GDN` selects between them from the active row count.
The same-dtype adaptive `AffineQuantizedMatmul` does not own this mixed-dtype selection yet.

The Qwen adapter supplies its measured default `Dv_tile` value of 8. The reusable backend remains model-agnostic.

During backend construction, the executor translates immutable `GDNCore` geometry and selected `Dv_tile` tuning into
`GDNCoreConfig`. This backend-owned config specializes the generated Metal source for
`num_qk_heads/qk_head_dim`, `num_v_heads/v_head_dim`, `conv_kernel_size`, derived `qkv_dim`, and
`v_dim_tile_size`. `GDNCoreShape` contains only replay-varying `num_reqs/num_tokens`.

Kernel source-hash caching shares compiled pipelines for identical component configs across layers and models. The backend
API does not contain model names or model config types. Batch metadata objects and scratch bindings do not copy static
geometry or tuning.

`Qwen3xGDNState` owns one shared `GDN` backend and one shared `GDNScratch` for compatible Main GDN layers. It also owns
the shared `Rc<GDNRequestStateTable>`, reusable `GDNMetadataBuffers`, cached restore replay, and optional pending publish.
Each `Qwen3xGDN` layer owns immutable weights, a compact `gdn_layer_index`, and cloned backend, scratch, and state-table
handles.

The current Qwen3.5 executor imports and owns `Qwen3xGDNState` directly. Its model layers own `Qwen3xGDN` directly.
Sharing the implementation does not move the GDN lifecycle into another executor. The backend records qkvabz projection,
projection split, recurrent core/state update, optional candidate state materialization, and output projection into the
caller’s `Recorder`.

State preparation also keeps the leaf boundary model-neutral. `Qwen3xGDNState::prepare_states` receives the request-slot,
block-index, token-index, cumulative-token, state-transaction, and state-page slices that `GDNRequestStateTable`
consumes. `prepare_metadata` receives cumulative tokens and the prepared state. The Qwen3.5 executor extracts these slices
from its own microbatch before it calls the shared leaf.

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

one state_slot:
  recurrent_states[gdn_layer_index][state_slot]
  conv_states[gdn_layer_index][state_slot]

page_ids_staging[state_io_request]
  [num_gdn_layers]
    [num_recurrent_pages_per_state_slot]
    [num_conv_pages_per_state_slot]
```

`num_state_slots` is the only state-slot dimension. One slot names one complete GDN state with its recurrent and
convolution substates. Their trailing dimensions come directly from the shared GDN core. They are not separate request-slot
axes.

`page_bytes` is the raw allocation unit. Page I/O divides by `sizeof(f32)` only when it indexes f32 state. A layout or
state object never stores that derived capacity.

Runtime page IDs remain CPU transaction data in `GDNStatePages` vectors. `GDNStatePageIO` owns the reusable
`page_ids`/`state_slots` GPU staging buffers and the batched read/write kernels. It fills the staging buffers immediately
before restore or publish recording. The buffers do not represent persistent request-page ownership.

At initialization, GDN derives per-request state-slot and publish-staging capacity from the scheduler's
`max_tokens_per_request` and the logical cache-block size. The candidate-state bound is the larger of the
speculative-prefix count and the unaligned normal-forward boundary count.

Speculative prefixes already include the boundary versions that they cross. Therefore, these two bounds do not add.
Publish staging permits every block boundary that one maximum-length request can cross across all active request slots.

The public table directly owns a private `GDNRequestSlots` mapping, pending restore/publish state transactions, and one
`GDNStatePageIO`. It has no second public state table or mutable aggregate wrapper.

`GDNStateTxn` is backend-neutral per-request metadata for the state versions that one microbatch produces. It lives from
`GDNRequestStateTable::prepare(...)` through `commit(...)`. The prepare boundary receives explicit request slots,
block/token indices, cumulative token counts, transactions, and runtime state-page IDs. It does not depend on a Qwen
microbatch type.

`GDNMetadataBuffers` is the state-domain-owned, capacity-sized GPU metadata object that all GDN layers share. Prepare
writes its `cu_tokens` and src/dst/candidate state slots. Prepare then returns and stores the authoritative
`GDNReplayShape`.

`GDNMetadataBuffers` is the sole owner of the current replay shape. `GDNInput` borrows the metadata object instead of a
duplicate shape. Backend recording and replay-key construction both read the stored shape.

`GDNStateArenaBindings` borrows both aggregate arenas and the selected layer's checked `u64` byte bases. Production binds
each arena at Metal offset zero. It passes the bases as Metal `ulong` kernel arguments.

`GDNProjectionSplitBuffers` carries `projected_qkv`, `a`, `b`, and `z`. In qkvabz naming, `a` is the raw gate/dt
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

Backend code then runs the recurrent state update and output projection. GDN math keeps `projected_qkv`, gates,
`conv_qkv`, recurrent state, `recurrent_output`, and `pre_output_hidden_states` in f32.

Qwen checkpoint weights and affine parameters remain packed U32 or BF16 in persistent Metal buffers. Quantized matmul
kernels dequantize packed weights and promote BF16 affine parameters during execution. GDN core kernels promote
`conv_weight`, `norm_weight`, `a_log`, and `dt_bias` when they read each value. The recurrent kernel computes
`-exp(a_log)` in F32.

`GDNMetalConfig::boundary_dtype()` returns bf16 at the Qwen3.6 model boundary. GDN state and pre-output math remain f32
because bf16 can cause downstream NaN/Inf.

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
```

`GDNOutput<'a>` is the named alias for the returned `&'a Buffer`. It is the caller-owned `next_hidden_state` buffer.
It does not allocate or add a wrapper.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay.

State page restore/publish belongs to `GDNRequestStateTable`, not to individual layer backends. Runtime supplies one
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
bf16 hidden_state
  -> f32 hidden_state_f32 cast when qkvabz input dtype is f32
  -> qkvabz packed-weight dequantization and BF16 affine-parameter promotion into f32
  -> qkvabz projection split
  -> f32 GDN core with BF16 parameter promotion: short convolution, ragged recurrent, output_norm_gate
  -> f32-to-bf16 pre-output cast when the boundary dtype is bf16
  -> output projection
  -> bf16 next_hidden_state
```

Stage nouns identify the operation. They do not overload one generic “attention” pipeline:

```text
projection_split   elementwise map from qkvabz to projected_qkv/a/b/z
short_conv         temporal convolution map from projected_qkv to conv_qkv plus next_conv_state
ragged_recurrent   ordered recurrent state transition and recurrent_output production
output_norm_gate  per-(token,V-head) RMS reduction, norm, and z-gate map
```

In ragged recurrent, each Q/K lane produces `q_square_sum_partial`, `k_square_sum_partial`, `state_k_partial`, and
`state_q_partial`. SIMD reductions produce `q_square_sum`, `k_square_sum`, `state_k_dot`, and one
`recurrent_output_value`. These values are local reduction values. They are not extra global tensors or Task fields.

Output norm + gate uses `square_sum_partial` and threadgroup `square_sum_partials` before it computes the inverse RMS.
No partial changes the existing dispatch, scratch, or ABI.

The production GDN core uses only ragged recurrent execution. It handles one or more flat tokens per request with
`cu_tokens`. The recurrent kernel computes Q/K inverse norms, decay, and beta. It advances each request's tokens in order.
It parallelizes across requests, V heads, V-dimension tiles, and Q/K-dimension lanes.

### Execution strategy

`ragged_recurrent` is the current GDN recurrent execution path. It does not define GDN itself. Another execution path can
share the tensor and state-tile vocabulary. That path owns a different Task and Grid contract. The current path is:

```text
shape: num_tokens >= num_reqs, segmented by cu_tokens
parallelism: request x v_head x v_dim_tile, with Q/K-dimension lanes inside the threadblock
input: one or more contiguous rows per request
state: load source-state fragments once, advance them in MSL thread-local storage, then write the final destination slot
```

This path uses 32 Q/K-dimension threads and the configured `Dv_tile`. Qwen's measured default tile of 8 produces a
256-thread threadblock. `v_head_dim` and the configured tile derive the number of V-dimension tiles. The backend does not
store this number. The current state dataflow is:

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
                              +--> optional registered prefix candidate writes
                              |
                              | segment-end final store
                              v
recurrent_state_arena[dst slot, v_head, 8 V rows, 128 Dqk values]
```

For each token, Q and K stream from global memory/cache. Lane-local scalar partials accumulate them before SIMD-group
reductions.

The recurrent kernel's threadgroup storage contains only four scalars: `q_inv_norm_shared`, `k_inv_norm_shared`,
`decay_shared`, and `beta_shared`. It has no threadgroup state tile or threadgroup Q/K tile. This design uses
register-oriented state residency, not double buffering.

The threadblock walks a request segment in token order. Token `t + 1` depends on token `t`'s updated recurrent state.
Separate requests, V heads, and V-dimension tiles remain parallel. The current backend has no alternative recurrent
execution mode.

GDN bench fixtures distinguish fresh state from state-present execution. `ctx=0` leaves source conv/recurrent state
zeroed. `ctx>0` initializes only the source slot with deterministic non-zero data. It leaves the candidate destination
slot zeroed.

This setup matches the production lifecycle. The lifecycle reads a verified current state and produces a candidate state.

The replay shape contract is exact for the current microbatch. `GDNCoreConfig` owns static geometry:

```text
num_reqs       number of request rows in the ragged batch
num_tokens   total flattened tokens across those requests
cu_tokens      length num_reqs + 1, cumulative flat-token counts for each request
```

Each active request in a recorded GDN replay must contribute at least one row. The committed source state slot represents
existing context. Padding rows do not represent it.

The state contract is slot based:

```text
GDNRequestStateTable
  current state slot per request slot
  current state_version per request slot
  txn candidate state_version -> state_slot mappings
  txn cache-boundary publish state_version -> page_ids mappings

src_state_slots          current source state slot per request
dst_state_slots          candidate destination slot per request
conv_state               f32 slot arena for convolution state
next_conv_state          destination conv-state arena; may be the same backing as conv_state
recurrent_state_arena    f32 slot arena for recurrent state
```

When `conv_state` and `next_conv_state` share backing storage, source and destination slot IDs must name distinct slots
for committed updates. Qwen replay allocates current and candidate state slots from the request-state table.

Each forward starts a txn and registers two absolute-version sets:

```text
candidate_state_versions
  versions that rejection/commit may select as the new current state

publish_state_versions
  cache-boundary versions whose selected snapshot should be written to runtime-owned state pages
```

The candidate set contains every version that commit/publish can select. Replay records these candidate state slots in the
GDN metadata. Commit selects the candidate whose `state_version` matches the verified state version.

A commit to the current version leaves the current slot unchanged. It clears uncommitted txn state slots.

Speculative Main verification must not promote a candidate written after rejected rows. If a forward contains
`base + draft` rows and rejection accepts only a shorter verified prefix, Qwen replay records prefix candidate states.
It uses additional per-request slots.

The normal GDN forward materializes candidate states while it scans rows. It writes each requested row to its candidate
slot. Commit selects by verified state version. Therefore, it discards rejected candidate slots before the next forward.

Cache-boundary publish is a separate requirement. When commit selects a registered publish version, publish must write
the matching candidate/current slot to its page IDs.

GDN page read/write helpers remain separate recordable backend-metal components. They restore or publish verified state
pages. Runtime core owns page IDs and cache notifications. The model executor owns GDN state layout, request-slot
interpretation, and candidate slot promotion. It owns only CPU transaction copies of runtime-provided page-ID vectors.

`begin_txn(...)` registers candidate state-slot mappings and future immutable-page mappings. It stores them as typed
`GDNStatePages` values for the current request txn. After registration, `candidate_state_slot(...)` is a read-only lookup.
It asserts if the mapping was not registered.

`restore(...)` returns a `GDNStateRestore` job. It updates the table's current state slot/version.

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

Qwen model replay keeps selected-path GDN transient scratch in one model-owned `GDNScratch`. This scratch includes hidden
f32 cast scratch, qkvabz projection/split buffers, and convolution/core/pre-output buffers. GDN layers execute serially in
the replay slice. Therefore, this scratch is reusable across layers.

State-page I/O writes directly between global state pages and the model-owned contiguous state arenas. It does not use
page-value scratch. Every production state kernel binds the aggregate arena at Metal offset zero.

Forward kernels receive a checked host `u64` layer byte base. They add it with Metal `ulong`. Page I/O derives the
all-layer state address directly with `ulong`.

Layer-local element indices remain `uint`. The executor validates them independently from the aggregate arena allocation.
This design preserves contiguous storage without an ICB nonzero buffer-binding offset above 4 GiB. It matters for MTP
rejection because a committed prefix can select an intermediate candidate state.

Per-layer owners retain weights and immutable component configuration. `GDNRequestStateTable` shares current/candidate
state, request-slot lifecycle, page-ID staging, and restore/publish jobs. Their versions and slots are common across all
GDN layers.

## State data flow

The replay-order section defines the hidden-state pipeline. Mutable request state flows beside it:

```text
src_state_slots[num_reqs]          committed current state slot for each request
dst_state_slots[num_reqs]          final candidate slot for the full forward
flat_candidate_state_slots[num_tokens]  optional prefix candidate slot per flat token, or u32::MAX
conv_states[layer, slot, Cqkv, Ks]
recurrent_states[layer, slot, v_head, v_dim, qk_dim]
```

Short convolution reads the source conv-state slot and `projected_qkv`. It writes `conv_qkv` for every current row. It
writes the next conv-state into the destination slot.

The recurrent core reads `conv_qkv`, raw F32 `a`/`b`, and raw BF16 `a_log`/`dt_bias`. It promotes the BF16 parameters
and derives normalized q/k, beta, decay, and output values in F32. It then advances the recurrent state in token order
for each request segment.

Each Q/K-dimension lane loads its strided source-state fragment once. It keeps that MSL `thread` fragment local across the
segment. It writes the final fragment to the destination slot after the last token.

Candidate state materialization is part of the normal forward. For a request with base state version `V` and `n` rows,
the row after `i` tokens corresponds to state version `V + i`.

If that version appears in the txn's candidate set, the core writes the current conv/recurrent state into that row's
candidate slot. Commit later selects the slot whose state version equals the verified state version.

Cache-boundary publish separately consumes the same materialized candidate/current slots. It emits a publish job only when
the committed verified path satisfies that publish version.

The important invariant is:

```text
all selectable versions must be materialized during the forward that computes them
commit selects by absolute state_version
publish writes only committed/verified versions
rejected speculative rows leave their candidate slots uncommitted
```

`ragged_recurrent` handles every current row shape. This includes decode and MTP verification batches where each request
has one row. One threadblock selects a request, V head, and V-dimension tile. Its Q/K-dimension lanes load distributed
source-state fragments. They then scan the request segment in order.

For `num_tokens=1,num_reqs=1`, this operation is still a one-step state update. For
`num_tokens=spec+1,num_reqs=1`, it verifies the full Main segment. It materializes any requested prefix candidate
versions while it scans.

Restore and publish page I/O are outside the core math:

```text
restore before forward
  runtime page IDs -> current state slot
  updates GDNRequestStateTable current state_version

forward
  current slot -> candidate slots
  may materialize prefix/cache-boundary candidate versions

commit after rejection/sampling
  verified_state_version -> current slot
  satisfied publish versions -> page write jobs

publish
  committed slot -> runtime page IDs
```

Runtime core owns state page IDs and cache lifecycle notifications. The executor owns GDN state tensor layout,
request-slot current/candidate slot mapping, and all-layer page-I/O command records.

`state_version` is the canonical absolute coordinate of verified mutable state. Immutable fp32 state pages are boundary
checkpoints. Restore loads one into mutable state after a prefix hit. Publish writes only a verified commit.

Backend page-I/O components receive compact page IDs, state slots, and `page_bytes`. Request slots, versions, cache policy,
and Qwen transaction semantics remain in the model-level state owner.

## Profile keys

The GDN benchmark uses these subcomponent names:

```text
qkvabz-proj
split
core
output-proj
```

Do not add dynamic values to profile paths.

## GDN kernel family

The current replay path uses the Metal GDN core component in
`crates/inference-backend-metal/src/components/`. It records projection split, short convolution, ragged recurrent,
output_norm_gate, and state page read/write helpers through explicit replay invocations.

Focused backend tests, component benches with parity checks, and Qwen real-weight wrapper/layer tests provide correctness
coverage. Slow/reference implementations are test oracles. They are not runtime fallbacks.

`gdn_attention` compares Metal execution with the CPU short-convolution and recurrent references. It covers fixed
one-request ragged decode, random ragged input, and a random multi-request ragged batch. Candidate-state tests compare each
speculative prefix state with an independently evaluated CPU prefix reference.

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
```

Append `-- --profile-time 1 --noplot` to either backend Criterion target for a representative full-target smoke run.

`gdn_attn` records GDN core-with-state and candidate-state-update building blocks into Metal replay/ICB paths.
`gdn_state_io` covers the reusable GDN state-page read/write component. Neither bench exposes direct-submit component or
forward wiring.

The full-forward `qwen35_gdn` bench uses CLI arguments, not environment variables. Across GQA and GDN, `--tokens` is the
total current microbatch row count. `--num-reqs` is the number of request segments in that microbatch. `--contexts` means
context/state that exists before the measured forward.

The bench distributes rows as evenly as possible across requests. It builds `cu_tokens`, source state slots, and
candidate destination slots from these options.

For current GDN paths, the source state slot represents prior history. The bench reports `ctx` for comparison hygiene.
The value does not change recurrent kernel metadata yet. Invalid batch-shape combinations print a structured `skip` line.

The current backend records explicit data-dependency barriers. The replay layer also infers RAW/WAR/WAW hazards from
declared buffer usage. It does not add a conservative every-command fallback.

This bench loads real Qwen3.6 GDN weights. It adapts separate checkpoint qkv/a/b/z projections into the executor qkvabz
replay layout without changing their checkpoint dtype. It measures the full replay path: qkvabz projection, projection
split, the GDN core, and output projection. Do not compare component-only GDN core or candidate state update timings
with full-forward numbers.

Recommendation: GDN replay debugging separates transient scratch from persistent state. Layers execute serially.
Thus, model-level code can reuse projection/core scratch.

Current/candidate conv/recurrent slot arenas and GPU page-ID staging buffers are model-owned persistent resources.

`GDNRequestStateTable` owns CPU-side current state slots, current `state_version`s, txn candidate slot mappings, and
restore/publish job metadata.

Recommendation: Barrier audits follow this data flow:

1. Batched state page read
2. Core update
3. Candidate write
4. Verified commit/publish

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
