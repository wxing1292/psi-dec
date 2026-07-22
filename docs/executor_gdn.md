# GDN Executor

This document maps the current GDN implementation from tensor geometry and state transactions through Metal projection,
short convolution, ragged recurrence, and state-page I/O.

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

crates/inference-executor-metal/src/model/qwen/v3_5/
  layer/gdn.rs              Qwen35GDN, private checkpoint weights, load, and record
  state/gdn.rs              Qwen35GDNState prepare/restore/commit/publish/reset lifecycle

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

`crates/inference-executor-core` is the backend-neutral home for GDN semantic metadata. `crates/inference-executor-metal`
owns the current Metal replay wiring and request state table.

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

`Cqkv = 2 * Hqk * Dqk + Hv * Dv`. `C` is used only for this concatenated channel axis at the projection and
short-convolution boundaries; it is not a head axis, head width, or convolution-kernel extent. Short convolution and
convolution state operate independently along `Cqkv`, with temporal geometry `Kc`/`Ks`.

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

`projected_qkv` is projection-split output and short-convolution input. `conv_qkv` is short-convolution output and
recurrent-core input. `recurrent_output` is the recurrent result before RMS normalization and output gate; it is not
attention output. `pre_output_hidden_states` is the normalized/gated tensor consumed by output projection.

Request segments use `flat_token_begin`, `flat_token_end`, `num_req_tokens`, and `token_index_in_req`. `q` and `k` are
reserved for actual Q/K tensor values and coordinates. `num_*` is valid logical work; `total_*` is reserved for padded
dispatch, replay, or scratch extent. `cu_tokens` has `R + 1` cumulative flat-token counts; adjacent entries select the
half-open flat-token segment owned by one request.

## Tiles, Tasks, threadblocks, and grids

`GDNRecurrentStateTile` is the smallest matmul-like logical GDN compute tile: one `[Dv_tile, Dqk]` slice of
`recurrent_state` for a fixed state slot and V head. `Dv_tile` is configured as `v_dim_tile_size`; the full recurrent
state head is `[Dv, Dqk]`.

One logical `GDNRaggedRecurrentTask` maps 1:1 to one ragged recurrent threadblock, owns one state tile, and advances it
once per request token:

```text
GDNRaggedRecurrentTask {  // logical; one per threadblock
  req_index          grid-derived from threadblock_position.y / Hv
  v_head_index       grid-derived from threadblock_position.y % Hv
  v_dim_tile_index   grid-derived from threadblock_position.x
  flat_token_begin   derived from cu_tokens[req_index]
  flat_token_end     derived from cu_tokens[req_index + 1]
}
```

The grid is `(Dv / Dv_tile, R * Hv, 1)` and the threadblock shape is `(32, Dv_tile, 1)`. The grid and `cu_tokens` derive
every Task coordinate, so no Task value, TaskTemplate, or ABI buffer is materialized. The threadblock copies the source
`[Dv_tile, Dqk]` state tile, advances it over the request's flat-token segment, writes `recurrent_output [T, Hv, Dv]`,
and leaves the advanced tile in the destination recurrent-state slot.

```text
parallel: requests, V heads, V-dimension tiles, Dqk lanes
ordered:  tokens within one request
```

State-page I/O is a copy operation, not matmul-like math, so it has no forced `*Tile`. Each state-I/O request selects one
state slot and its page IDs across every GDN layer and state kind. The owning Metal kernels map one logical
`GDNStatePageReadTask` or `GDNStatePageWriteTask` 1:1 to one threadblock:

```text
GDNStatePageReadTask / GDNStatePageWriteTask {  // logical; one per threadblock
  state_io_request_index  grid-derived
  gdn_layer_index         grid-derived
  state_kind              grid-derived: recurrent or convolution
  page_index_in_state     grid-derived
}
```

No Task value, TaskTemplate, or ABI buffer is materialized. `page_id` and `state_slot` are data inputs, not Task
coordinates. One threadblock copies one page with `float4` lanes; the grid launches all requested state-page copies.

Output norm + gate is a cooperative reduction/map, not a matmul-like Tile. One comment-only `GDNOutputNormGateTask`
maps 1:1 to one 128-thread threadblock and owns `{ flat_token_index, v_head_index }`, both derived from the grid. It
RMS-normalizes and gates one `[Dv]` recurrent-output vector. Short convolution and projection split use flat map
dispatches whose threadblock grouping is incidental launch tuning, so they document tensors and grids without
inventing `*Task` or `*Tile` nouns.

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

Static geometry and tuning stay in generic `GDNCoreConfig`; dynamic batch work stays in replay shape/key. The Qwen
adapter supplies dimensions, weights, and measured defaults. No Qwen name or config type enters generic Rust or Metal.

The canonical binding order and dispatch topology are:

```text
projection split
  buffers 0..4: qkvabz, projected_qkv, a, b, z
  scalars 5..8: num_tokens, qkv_dim, num_v_heads, v_dim
  dispatch: T * (Cqkv + 2 * Hv + Hv * Dv), 256 threads/threadblock

short convolution
  buffers 0..7: conv_qkv, next_conv_state, projected_qkv, conv_state,
                conv_weight, src_state_slots, dst_state_slots, cu_tokens
  scalars 8..11: num_reqs, num_tokens, conv_state_offset_bytes,
                 next_conv_state_offset_bytes
  dispatch: max(T * Cqkv, R * Cqkv * Ks), 256 threads/threadblock

ragged recurrent
  buffers 0..9: recurrent_output, recurrent_state_arena, conv_qkv, a, b,
                a_log_decay, dt_bias, src_state_slots, dst_state_slots, cu_tokens
  scalars 10..13: q_scale, num_reqs, num_tokens, recurrent_state_offset_bytes
  grid: (Dv / Dv_tile, R * Hv, 1)
  threads: (32, Dv_tile, 1)

output_norm_gate
  buffers 0..3: pre_output_hidden_states, recurrent_output, z, norm_weight
  scalars 4..6: eps, num_reqs, num_tokens
  dispatch: T * Hv * 128, 128 threads/threadblock

batched state-page read/write
  buffers 0..4: pages, recurrent_states, conv_states, page_ids, state_slots
  scalars 5..12: num_gdn_layers, num_state_slots, num_state_io_requests,
                 num_recurrent_pages_per_state_slot, recurrent_state_bytes,
                 num_conv_pages_per_state_slot, conv_state_bytes, page_bytes
  grid: (total_pages, 1, 1), threads: (256, 1, 1)
```

Candidate recurrent materialization adds `flat_candidate_state_slots` at buffer 9, shifts `cu_tokens` to 10, and uses
scalars 11..14. The invalid candidate-slot sentinel remains `u32::MAX` with no write for that token.

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

`qk_dim`, `v_dim`, `qkv_dim = Cqkv`, and convolution history length are derived from those independent dimensions;
the core and backend invocation shape do not store duplicate fields for them. The GDN internal dtype is fixed f32 and
the output boundary is fixed bf16, so they are backend contract methods rather than configurable fields.

`GDNMetalConfig` owns shared execution tuning and numeric configuration, including recurrent `Dv_tile` size, norm
epsilon, input dtype, and affine dtypes. The Qwen adapter supplies its measured default `Dv_tile` of 8;
the reusable backend remains model-agnostic.

At backend construction, the executor translates immutable `GDNCore` geometry plus the selected `Dv_tile` tuning into
`GDNCoreConfig`. This backend-owned config specializes the generated Metal source for
`num_qk_heads/qk_head_dim`, `num_v_heads/v_head_dim`, `conv_kernel_size`, derived `qkv_dim`, and
`v_dim_tile_size`. `GDNCoreShape` contains only replay-varying `num_reqs/num_tokens`. Kernel source-hash caching therefore
shares compiled pipelines for identical component configs across layers and models without putting model names or model
config types in the backend API. Batch metadata objects and scratch bindings do not carry copies of static geometry or tuning.

`Qwen35GDNState` owns one shared `GDN` backend and one shared `GDNScratch` for compatible Main GDN layers, plus the
shared `Rc<GDNRequestStateTable>`, reusable `GDNMetadataBuffers`, cached restore replay, and optional pending publish.
Each `Qwen35GDN` layer retains immutable weights, a compact `gdn_layer_index`, and clones of the backend, scratch, and
state-table handles. The backend records qkvabz projection,
projection split, recurrent core/state update, optional candidate state materialization, and output projection into the
caller’s `Recorder`.

`GDNRequestStateTable` is the model-level owner for all GDN layers. It owns two contiguous aggregate arenas, one recurrent and
one convolution:

Its `num_pages_per_state_slot()` reports the physical page count from that instantiated layout; service cache capacity
uses this owner-derived value instead of maintaining a second GDN shape formula.

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

`num_state_slots` is the only state-slot dimension: one slot names one complete GDN state containing both its recurrent
and convolution substate. Their trailing dimensions come directly from the shared GDN core; they are not separate
request-slot axes. `page_bytes` is the raw allocation unit. Page I/O divides by `sizeof(f32)` only when it indexes f32
state; that derived capacity is never stored in a layout or state object.

Runtime page IDs remain CPU transaction data in `GDNStatePages` vectors. `GDNStatePageIO` owns the reusable
`page_ids`/`state_slots` GPU staging buffers together with the batched read/write kernels. The staging buffers are filled
immediately before restore or publish recording; they are not persistent request-page ownership.

At initialization, GDN derives its per-request state-slot and publish-staging capacity from the scheduler's
`max_tokens_per_request` and the logical cache-block size. The candidate-state bound is the larger of the
speculative-prefix count and the unaligned normal-forward boundary count; speculative prefixes already include
the boundary versions they cross, so those two bounds do not add. Publish staging permits every block boundary
that one maximum-length request can cross across all active request slots.

The public table directly owns a private `GDNRequestSlots` mapping, pending restore/publish state transactions, and one
`GDNStatePageIO`. There is no second public state table or mutable aggregate wrapper. `GDNStateTxn` is backend-neutral
per-request metadata for the state versions produced by one microbatch;
it lives from `GDNRequestStateTable::prepare(...)` through `commit(...)`. The prepare boundary receives explicit request
slots, block/token indices, cumulative token counts, transactions, and runtime state-page IDs. It does not depend on a
Qwen microbatch type.

`GDNMetadataBuffers` is the state-domain-owned, capacity-sized GPU metadata object shared by all GDN layers. Prepare writes its
`cu_tokens` and src/dst/candidate state slots, then returns and stores the authoritative `GDNReplayShape`. It is the sole
owner of the current replay shape: `GDNInput` borrows the metadata object instead of carrying a duplicate shape. Backend
recording and replay-key construction both read that stored shape. `GDNStateArenaBindings` borrows both aggregate arenas plus the
selected layer's checked `u64` byte bases. Production binds each arena at Metal offset zero and passes those bases as
Metal `ulong` kernel arguments.

`GDNProjectionSplitBuffers` carries `projected_qkv`, `a`, `b`, and `z`. In qkvabz naming, `a` is the raw gate/dt projection, `b` is the raw beta projection, and `z` is the output gate projection. `g` is not projected; it is derived inside gate preparation as part of `beta = sigmoid(b)`, `g = -exp(a_log) * softplus(a + dt_bias)`, and `decay = exp(g)`. API and docs use q/k/v/a/b/z at projection boundaries and reserve `g`/`beta` for prepared low-level core values.

`GDNStateLayout` is the model-owned logical layout of the contiguous allocations: the leading
`[gdn_layer_index, state_slot]` dimensions plus `page_bytes`. The shared GDN core supplies the trailing recurrent/conv
tensor dimensions during construction. Per-slot and per-layer byte strides are derived from arena lengths and those
leading dimensions; the layout does not duplicate them. It derives aggregate allocation lengths and the all-layer
page-ID count directly; it does not store derived f32-per-page counts, recurrent/conv page counts, or a selected layer
coordinate.

Backend code then runs recurrent state update and output projection. GDN math keeps `projected_qkv`, gates,
`conv_qkv`, recurrent state, `recurrent_output`, and `pre_output_hidden_states` in f32. `hidden_storage_dtype`
is bf16 at Qwen3.6 model boundaries, but reducing GDN pre-output or state math to bf16 can introduce precision loss and
downstream NaN/Inf.

## Replay contract

`GDN` records one GDN layer forward through `ReplayLayer::record(...)` and a caller-owned
`Recorder`. It does not submit commands and it does not own request scheduling or request-state lifecycle.

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

`GDNOutput<'a>` is the named alias for the returned `&'a Buffer`; it is the caller-owned `next_hidden_state` buffer and
does not allocate or add a wrapper.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay.

State page restore/publish belongs to `GDNRequestStateTable`, not to individual layer backends. Runtime supplies one state-page vector
per cache block containing every GDN layer in model order. The manager splits that vector into recurrent and convolution
page-ID staging, then records one flattened all-layer page command:

```text
threadblock -> (state_io_request_index, gdn_layer_index, state_kind, page_index_in_state)
```

Qwen3.5 service replay defines that cache block as 2048 tokens. A GDN
snapshot page vector is therefore a state at exactly 2048, 4096, ... tokens;
the trie runtime and GQA page tables use the identical logical boundary.
Physical GQA pages remain smaller and are grouped under that one block. This
alignment is required: a trie prefix hit must never begin at a position for
which the corresponding GDN state snapshot cannot exist.

Qwen3.5 derives every GDN layer from one text configuration, and runtime state-page sizing uses the same single-layer
dimensions multiplied by the GDN layer count. `GDNRequestStateTable::new` validates that all GDN cores share those dimensions.
This init-time invariant lets the page kernel use arithmetic indexing without a per-layer layout table or executor layout object.

GDN prepare is split at its real dependency boundary but executes synchronously on the executor thread.
`GDNRequestStateTable::prepare(...)` validates and applies request/version/page transactions and returns prepared state
slots. `GDN::prepare(...)` then writes the dependent `GDNMetadataBuffers`. Main GQA page preparation and metadata,
GDN state preparation and metadata, and optional MTP GQA page preparation are explicit sequential calls. There is no
prepare worker, channel, receiver, or inferred reset.

All prepare branches and any restore are completed before main-model replay begins. Publish is a separate replay
submitted after host commit selects verified state versions. The model retains the publish submission while returning
the device response to the scheduler; it does not wait in `commit_batch(...)`.

The next `prepare(...)` or request-slot reset waits for the pending publish before mutating page/state staging or
submitting more model work. This is the device-state happens-before boundary. The scheduler may commit trie metadata,
release a terminal request, or reassign a page ID while publish is in flight because those are host-only ownership
changes; any later GPU use of that page ID returns through the same executor and crosses the pending-publish wait first.
Dropping the model also waits through `ReplaySubmission` ownership. Runtime core therefore does not own or poll a Metal
submission.

The replay order is:

```text
bf16 hidden_state
  -> f32 hidden_state_f32 cast when qkvabz input dtype is f32
  -> qkvabz quantized projection into f32
  -> qkvabz projection split
  -> f32 GDN core: short convolution, ragged recurrent, output_norm_gate
  -> f32-to-bf16 pre-output cast when the boundary dtype is bf16
  -> output projection
  -> bf16 next_hidden_state
```

Stage nouns follow the operation rather than overloading one generic “attention” pipeline:

```text
projection_split   elementwise map from qkvabz to projected_qkv/a/b/z
short_conv         temporal convolution map from projected_qkv to conv_qkv plus next_conv_state
ragged_recurrent   ordered recurrent state transition and recurrent_output production
output_norm_gate  per-(token,V-head) RMS reduction, norm, and z-gate map
```

In ragged recurrent, each Q/K lane produces `q_square_sum_partial`, `k_square_sum_partial`, `state_k_partial`, and
`state_q_partial`. SIMD reductions produce `q_square_sum`, `k_square_sum`, `state_k_dot`, and one
`recurrent_output_value`. These are local reduction values, not extra global tensors or Task fields. Output norm + gate uses
`square_sum_partial` and threadgroup `square_sum_partials` before computing the inverse RMS. No partial changes the
existing dispatch, scratch, or ABI.

Current production GDN core uses only ragged recurrent execution. It handles one or more flat tokens per request using
`cu_tokens`, computes Q/K inverse norms plus decay and beta inside the recurrent kernel, and advances each request's
tokens in order while parallelizing across requests, V heads, V-dimension tiles, and Q/K-dimension lanes.

### Execution strategy

`ragged_recurrent` is the current GDN recurrent execution path, not the definition of GDN itself. Another execution path
may share the tensor and state-tile vocabulary while owning a different Task and Grid contract. The current path is:

```text
shape: num_tokens >= num_reqs, segmented by cu_tokens
parallelism: request x v_head x v_dim_tile, with Q/K-dimension lanes inside the threadblock
input: one or more contiguous rows per request
state: copy source state slot to candidate slot, then advance rows in order
```

This path uses 32 Q/K-dimension threads and the configured `Dv_tile`; Qwen's measured default tile of 8 produces a
256-thread threadblock with only small threadblock scratch. The number of V-dimension tiles is derived from `v_head_dim`
and the configured tile instead of being stored. Q/K-dimension reductions use SIMD-group reductions over the 32 lanes;
per-V-dimension-lane intermediate values are broadcast inside the SIMD group instead of materialized in global scratch.
The threadblock walks a request segment in token order because token `t + 1` depends on token `t`'s updated recurrent
state, while separate requests, V heads, and V-dimension tiles remain parallel. No alternative recurrent execution mode
remains in the current backend.

Bench fixtures for GDN distinguish fresh state from state-present execution. `ctx=0` leaves source conv/recurrent state
zeroed. `ctx>0` initializes only the source slot with deterministic non-zero data and leaves the candidate destination
slot zeroed, matching the production lifecycle where a verified current state is read and a candidate state is produced.

The replay shape contract is exact for the current microbatch; static geometry is owned by `GDNCoreConfig`:

```text
num_reqs       number of request rows in the ragged batch
num_tokens   total flattened tokens across those requests
cu_tokens      length num_reqs + 1, cumulative flat-token counts for each request
```

Each active request in a recorded GDN replay must contribute at least one row. Existing context is represented by the
committed source state slot, not by padding rows.

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

When `conv_state` and `next_conv_state` share backing storage, the source and destination slot IDs must name distinct
slots for committed updates. Qwen replay allocates current and candidate state slots from the request-state table. Each
forward begins a txn by registering two absolute-version sets:

```text
candidate_state_versions
  versions that rejection/commit may select as the new current state

publish_state_versions
  cache-boundary versions whose selected snapshot should be written to runtime-owned state pages
```

The candidate set contains every version that commit/publish may select, and replay records those candidate state slots
into the GDN metadata. Commit selects the candidate whose `state_version` matches the verified state version. A commit
to the current version leaves the current slot unchanged while clearing uncommitted txn state slots.

Speculative target verification must not promote a candidate written after rejected rows. If a forward contains
`base + draft` rows and rejection accepts only a shorter verified prefix, Qwen replay records prefix candidate states in
additional per-request slots. The normal GDN forward materializes candidate states while scanning rows by writing each
requested row to its candidate slot. Commit selects by verified state version, so rejected candidate slots are discarded
before the next forward. Cache-boundary publish is a separate requirement: when a registered publish version is
selected, the matching candidate/current slot must be written to its page IDs.

GDN page read/write helpers remain separate recordable backend-metal components for restoring or publishing verified
state pages. Runtime core owns page IDs and cache notifications; the model executor owns GDN state layout, request-slot
interpretation, candidate slot promotion, and only CPU transaction copies of runtime-provided page-ID vectors.
`begin_txn(...)` registers candidate state-slot mappings and future immutable-page mappings as typed
`GDNStatePages` values for the current request txn. After that, `candidate_state_slot(...)` is a read-only
lookup and asserts if the mapping was not registered. `restore(...)` returns a `GDNStateRestore` job and
updates the table's current state slot/version.
`commit_txn(...)` returns `GDNStatePublish` jobs for registered publish versions that are satisfied by the
committed path. Qwen includes publish versions that fall inside the current forward in the candidate-state materialization
set, so a commit can publish intermediate cache boundaries up to the verified committed version from already-materialized
slots. Publish versions beyond the current forward remain queued until a later transaction materializes and commits them.
When an earlier txn registered a future publish version that later falls inside a forward, Qwen adds that version to the
candidate materialization set even if the current batch does not repeat its page IDs. Qwen compacts publish jobs into
model-owned recurrent page-ID, convolution page-ID, and state-slot staging buffers. Page IDs remain state-I/O-request-major across
all GDN layers. Restore records one all-layer batch read before model forward; publish records one all-layer batch write
after commit.
Publish is a separate replay from main forward and sampling: it consumes the
already-selected committed state, has no bearing on the response tokens, and may execute while the scheduler processes
that response.

Qwen model replay keeps selected-path GDN transient scratch in one model-owned `GDNScratch`: hidden f32 cast
scratch, qkvabz projection/split buffers, and convolution/core/pre-output buffers. GDN layers execute serially in the
replay slice, so this scratch is reusable across layers. State-page I/O writes directly between global state pages and
the model-owned contiguous state arenas and does not use page-value scratch. Every production state kernel binds the
aggregate arena at Metal offset zero. Forward kernels receive a checked host `u64` layer byte base and add it with Metal
`ulong`; page I/O derives the all-layer state address directly with `ulong`. Layer-local element indices remain `uint` and
are validated independently from the aggregate arena allocation. This preserves contiguous storage without using an ICB
nonzero buffer-binding offset above 4 GiB. It matters for MTP rejection because a committed prefix can select an
intermediate candidate state.

Per-layer owners retain weights and immutable component configuration. Current/candidate state, request-slot lifecycle,
page-ID staging, and restore/publish jobs are shared by `GDNRequestStateTable` because their versions and slots are common
across all GDN layers.

## State data flow

The replay-order section defines the hidden-state pipeline. Mutable request state flows beside it:

```text
src_state_slots[num_reqs]          committed current state slot for each request
dst_state_slots[num_reqs]          final candidate slot for the full forward
flat_candidate_state_slots[num_tokens]  optional prefix candidate slot per flat token, or u32::MAX
conv_states[layer, slot, Cqkv, Ks]
recurrent_states[layer, slot, v_head, v_dim, qk_dim]
```

Short convolution reads the source conv-state slot and `projected_qkv`. It writes `conv_qkv` for every current row and
writes the next conv-state into the destination slot. The recurrent core reads `conv_qkv`,
raw `a`/`b`, `a_log_decay`, and `dt_bias`; derives normalized q/k, beta, decay, and output values; then advances the
recurrent state in token order for each request segment.

Candidate state materialization is part of the normal forward. For a request with base state version `V` and `n` rows,
the row after `i` tokens corresponds to state version `V + i`. If that version appears in the txn's candidate set, the
core writes the current conv/recurrent state into the candidate slot for that row. Commit later selects the slot whose
state version equals the verified state version. Cache-boundary publish is a separate consumer of the same materialized
candidate/current slots: a publish job is emitted only when the committed verified path satisfies that publish version.

The important invariant is:

```text
all selectable versions must be materialized during the forward that computes them
commit selects by absolute state_version
publish writes only committed/verified versions
rejected speculative rows leave their candidate slots uncommitted
```

`ragged_recurrent` handles every current row shape, including decode and MTP verification batches where each request has
one row. It
uses one threadblock over request, V-head, V-dimension-tile, and Q/K-dimension lanes, copies the source recurrent slot to the
candidate slot, and then scans each request segment in order. For `num_tokens=1,num_reqs=1`, this is still a one-step
state update. For `num_tokens=spec+1,num_reqs=1`, it verifies the full target segment and materializes any requested
prefix candidate versions while scanning.

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
request-slot current/candidate slot mapping, and the all-layer page-I/O command records.

`state_version` is the canonical absolute coordinate of verified mutable state. Immutable fp32 state pages are boundary
checkpoints: restore loads one into mutable state after a prefix hit, and publish writes only a verified commit. Backend
page-I/O components receive compact page IDs, state slots, and `page_bytes`; request slots, versions, cache policy, and
Qwen transaction semantics remain in the model-level state owner.

## Profile keys

Stable segments include:

```text
gated-delta-net
input-project
short-conv
advance-recurrent-state
advance-decode-candidate-state
commit
output-project
```

Do not add dynamic values to profile paths.

## GDN kernel family

The current replay path uses the Metal GDN core component in `crates/inference-backend-metal/src/components/`. It records
projection split, short convolution, ragged recurrent, output_norm_gate, and state page read/write helpers through explicit replay
invocations.

Correctness coverage belongs in focused backend tests, component benches with parity checks, and Qwen real-weight
wrapper/layer tests. Slow/reference implementations are oracles for tests, not
runtime fallbacks.

`gdn_attention` compares Metal execution against the CPU short-convolution and recurrent references for fixed
one-request ragged decode, random ragged input, and a random multi-request ragged batch. Candidate-state tests compare
every speculative prefix state against an independently evaluated CPU prefix reference.

## Tests and benches

Current tests and benches should cover backend component correctness, real-weight GDN wrapper correctness, state slot
promotion, page read/write helpers, and Qwen layer integration.

Current benches:

```text
cargo bench -p inference-backend-metal --bench gdn_attn
cargo bench -p inference-backend-metal --bench gdn_state_io
cargo bench -p inference-executor-metal --bench qwen35_gdn -- \
  --model-dir <35b-a3b-model-dir> --tokens 1 --contexts 0 --num-reqs 1 \
  --iters 1 --warmup-iters 0 --runs 1
```

Append `-- --profile-time 1 --noplot` to either backend Criterion target for
a representative full-target smoke run.

`gdn_attn` records GDN core-with-state and candidate-state-update building
blocks into Metal replay/ICB paths. `gdn_state_io` covers the reusable GDN
state-page read/write component. Neither bench exposes direct-submit component
or forward wiring.

The full-forward `qwen35_gdn` bench uses CLI arguments, not environment variables. Across GQA and GDN, `--tokens` is
the total current microbatch row count, `--num-reqs` is the number of request segments in that microbatch, and
`--contexts` means context/state that exists before the measured forward. The bench distributes rows as evenly as
possible across requests and builds
`cu_tokens`, source state slots, and candidate destination slots from those options. For current GDN paths, prior history
is represented by the source state slot, so the `ctx` value is reported for comparison hygiene but does not change
recurrent kernel metadata yet. Invalid batch-shape combinations print a structured `skip` line. The
current backend records explicit data-dependency barriers, and the replay layer also infers RAW/WAR/WAW hazards from
declared buffer usage; it does not add a conservative every-command fallback. This bench loads real Qwen3.6 GDN weights, adapts separate
checkpoint qkv/a/b/z projections into the executor qkvabz replay layout, and measures the full replay path:
qkvabz projection, projection split, the GDN core, and output projection. Do not compare component-only GDN
core or candidate state update timings against full-forward numbers.

GDN replay debugging should separate transient scratch from persistent state. Projection/core scratch can be
model-level reusable because layers execute serially. Current/candidate conv/recurrent slot arenas and GPU page-id
staging buffers are model-owned persistent resources. `GDNRequestStateTable` owns CPU-side current state slots,
current `state_version`s, txn candidate slot mappings, and restore/publish job metadata. Barrier audits should follow
data flow: batched state page read, core update, candidate write, then verified commit/publish.

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
