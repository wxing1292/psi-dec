# GQA Executor

This document maps the current GQA implementation from tensor geometry and batch metadata through Metal replay,
KV-page interpretation, and correctness coverage.

## Source layout

```text
crates/inference-executor-core/src/attn/
  mod.rs                    MLX-free attention module exports
  gqa/
    mod.rs                  GQA module root
    core.rs                 gated QGKV GQACore metadata and projection shapes
    ungated_core.rs         ungated QKV UngatedGQACore metadata and projection shapes
    reference.rs            CPU projected-GQA correctness oracle

crates/inference-executor-metal/src/attn/
  mod.rs                    Metal attention module exports
  gqa/
    mod.rs                  GQA Metal module root
    batch_metadata.rs       state-domain-owned, capacity-sized GPU metadata updated per microbatch
    backend.rs              gated QGKV Metal replay wiring
    scratch.rs              gated QGKV scratch allocation and borrowed replay bindings
    ungated_backend.rs      ungated QKV Metal replay wiring
    ungated_scratch.rs      ungated QKV scratch allocation and borrowed replay bindings
    request_page_table.rs   per-request, per-layer KV page table for runtime-supplied page IDs

crates/inference-executor-metal/src/model/qwen/
  v3_x/
    layer/gqa.rs            Qwen3xGQA, private checkpoint weights, load, and record
    state/gqa.rs            Qwen3xGQAState page/metadata/reset lifecycle grouping
  v3/
    main/gqa.rs             Qwen3 Main ungated GQA weights, state, load, and record
    main/layer.rs           fixed QKV Qwen3MainLayer composition
    main/plan.rs            Qwen3 Main QKV GQA geometry/config builder
  v3_5/
    main/layer.rs           Qwen3.5 Main QGKV-GQA/GDN layer variants
    mtp/layer.rs            Qwen3.5 MTP GQA layer composition
    plan.rs                 Qwen3.5 QGKV GQA geometry/config builder and dSpark plan

crates/inference-backend-metal/src/components/
  gqa_attention.rs          reusable Metal paged SDPA component kernels
  gqa_local_attention.rs    reusable dense bidirectional local-SDPA partial kernel
  gqa_projection.rs         gated QGKV projection split component
  ungated_gqa_projection.rs ungated QKV projection split component
  gqa_norm_rope.rs          reusable Metal q/k fused and single-input norm/RoPE component kernels
  gqa_kv_pages.rs           reusable Metal KV page update component kernels
  gqa_tiled_attention.rs    reusable token/Q-head tiled paged SDPA component
  metal/
    gqa_projection_split.metal  gated QGKV projection split source
    ungated_gqa_projection_split.metal  ungated QKV projection split source
    gqa_norm_rope.metal         Metal q/k norm and RoPE source
    gqa_kv_pages.metal          Metal KV page-update source
    gqa_paged_sdpa_map.metal     Metal paged SDPA map source
    gqa_paged_sdpa_reduce.metal  Metal paged SDPA partial-output reduce source
    gqa_local_sdpa.metal         Metal dense local-SDPA partial-output source
    gqa_tiled_attention.metal    Metal tiled paged SDPA map/reduce source
    gqa_activation_gate.metal    Metal attention-output gate source
```

`crates/inference-executor-core` is the backend-neutral home for GQA semantic metadata and replay shape. `crates/inference-executor-metal`
owns the current Metal replay wiring and request page table.

The Metal GQA executor backend implements the executor `ReplayLayer` contract so Qwen model/layer code can append GQA
work into a larger e2e replay through a semantic layer input/output and caller-owned `Recorder`.
`request_page_table.rs` owns the executor-side request-slot KV page table used to accumulate runtime-supplied page IDs
between reset notifications; runtime core still owns physical page allocation/free.

## Ownership

Each derived `GQACore` carries its source model coordinate plus the common GQA dimensions:

```text
model_layer_index
hidden_dim
head_dim
num_q_heads / num_kv_heads
attention scale
```

`GQACore` is the gated contract: its fused projection is always QGKV and its replay always applies the trained
attention-output gate. `UngatedGQACore` is the separate ungated contract: its fused projection is always QKV and it
has no gate resources or gate recording step. Both derive query, key, and value widths from head counts and
`head_dim`; neither uses a layout enum or gate flag.

GQA tensor and tile comments use one symbolic convention:

```text
Q: [Tq,  Hq,  D]       Q tile: [Tq_tile, Hq_tile, D]
K: [Tkv, Hkv, D]       K tile: [Tkv_tile, D]  // one fixed KV head
V: [Tkv, Hkv, D]       V tile: [Tkv_tile, D]
O: [Tq,  Hq,  D]

Q tile [Tq_tile, Hq_tile, D] x K tile^T [D, Tkv_tile]
  -> scores [Tq_tile, Hq_tile, Tkv_tile]
scores [Tq_tile, Hq_tile, Tkv_tile] x V tile [Tkv_tile, D]
  -> output partial [Tq_tile, Hq_tile, D]
```

`T` names token dimensions, `H` names head dimensions, and `D` is `head_dim`. `SDPAMapTile` is the smallest
matmul-like logical description `(q_token_tile_index, kv_head_index, q_head_tile_index, kv_token_tile_index)`.

Only `model_layer_index` is per layer. Qwen validates the remaining fields once and uses them through the shared backend.

Qwen3 Main owns `Qwen3MainGQAState`, `UngatedGQA`, and `UngatedGQAScratch`. Qwen3.5 owns
`Qwen3xGQAState`, which holds one shared gated `GQA` backend and `GQAScratch` for compatible invocations plus one
`Rc<GQARequestPageTable>` and reusable `GQAMetadataBuffers`. Each `Qwen3xGQA` layer component retains clones of the
backend, scratch, and page-table handles together with its own weights and compact layer coordinate. Qwen3 Main instead
uses its model-owned ungated leaf. `Qwen35MainLayer` and `Qwen35MTPLayer` continue to use the gated shared leaf inside
their separate role-specific owners.

The gated and ungated executors share lower-level norm/RoPE, KV update, paged SDPA, and output-projection components,
but their projection, scratch, weights, and replay graphs are concrete and separate. This keeps the existing gated
QGKV command sequence free of mode checks while making absence of a gate structural in the ungated QKV path.

Qwen3 Main constructs `UngatedGQACore` and `UngatedGQA`; Qwen3.5 Main/MTP construct `GQACore` and gated `GQA`.
Qwen3.5 dSpark also uses `UngatedGQACore` for its separate QKV attention graph. Head dimensions, head counts, RoPE
constants, and page geometry flow through init-time component specialization rather than a model-specific runtime
branch.

`GQARequestPageTable` stores executor-side request-slot KV page IDs between runtime reset/update
notifications in a fixed-stride GPU buffer:

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

Runtime still owns physical page allocation/free.

`GQAMetadataBuffers` stores the GPU arrays shared by every GQA layer in one model replay:

```text
req_slots[num_tokens]
flat_token_indices[num_tokens]
q_token_tiles[num_q_token_tiles][flat_token_start/flat_token_end]  // TiledQTokens
sdpa_map_task_templates[total_sdpa_map_task_templates][q_token_tile_index/kv_token_begin/kv_token_end]
cu_sdpa_partial_outputs[num_tokens + 1]                              // SingleQToken
cu_sdpa_partial_outputs[num_q_token_tiles + 1]                       // TiledQTokens
```

Each three-`u32` entry materializes one compact `SDPAMapTaskTemplate`: Q-token-tile index followed by the half-open
KV-token segment. Combining it with grid-derived `kv_head_index` and `q_head_tile_index` produces one logical
`SDPAMapTask` owned 1:1 by one threadblock; those regular coordinates are not duplicated in the buffer. `SingleQToken`
uses one-token Q tiles; `TiledQTokens` first builds request-local Q-token tiles. Additional TaskTemplates are
assigned to the Q-token tile with the most remaining KV-tile work. TaskTemplates for one Q-token tile are contiguous.
For a fixed Q-token/head output coordinate, adjacent `cu_sdpa_partial_outputs` values select the `SDPAPartialOutput`s
merged by reduce. `total_sdpa_map_task_templates` is the power-of-two replay extent; unused tail TaskTemplates contain
an invalid Q-token-tile index and perform no map write.

The `SingleQToken` paged map also permits an invalid-Q-token-tile `SDPAMapTaskTemplate` inside one token's
TaskTemplate range. It performs no paged partial-output write for that slot. A caller may populate the reserved
max-logit, exp-sum, and normalized `SDPAPartialOutput` through `GQALocalSDPAKernel` before invoking the unchanged
partial-output reducer. This generic composition
is used when an attention connection combines paged history with a dense bidirectional local block; the backend
component does not own model-specific proposal or cache semantics.

Replay recording borrows `&GQAMetadataBuffers` directly; there is no duplicate bindings wrapper.

`cu_tokens`, per-request `req_slots`, and per-request starting `token_indices` are CPU inputs used to build these
token-major arrays. GQA kernels do not consume `cu_tokens`, so `GQAMetadataBuffers` does not retain a GPU copy.

The model-level GQA storage shape is:

```text
pages[num_cache_pages][page_bytes]

main_page_ids[num_req_slots][num_gqa_layers][num_blocks][num_page_ids_per_block]
optional_mtp_page_ids[num_req_slots][1][num_blocks][num_page_ids_per_block]

one KV page, viewed with the model KV dtype:
  [K/V][num_kv_heads][num_tokens_per_page][head_dim]
```

The Metal config stores `page_bytes` and the shared activation/KV dtype. `num_tokens_per_page` is derived from those
values plus `num_kv_heads` and `head_dim`; neither it nor a duplicate KV dtype is stored.

For a flat token, the page lookup is exactly:

```text
req_slot -> gqa_layer_index -> block_index -> page_id_index -> page_id -> KV page
```

The main page-ID table uses compact main `gqa_layer_index`; the optional supported MTP has an independent table and its
single full-attention body uses `gqa_layer_index = 0`. The layout type uses generic `num_gqa_layers` for both: each
table instance owns its own capacity and may have a different GQA configuration.

The Qwen executor updates Main state once for each Main batch and optional MTP metadata once for the MTP stage. Main GQA
layers borrow the Main state domain; the optional MTP owns its distinct backend, scratch, and compact page-ID table.
Layer owners retain immutable weights and a `gqa_layer_index`; a main layer does not retain a copy of model-level GQA
configuration or batch metadata. Each current MTP full-attention body uses coordinate 0 in its own table.

## Backend specialization

Stable profile segments include:

```text
gqa-attn
input-project
apply-rope
update-kv-cache
paged-sdpa
attention-core
output-project
```

GQA owns KV page-table/cache interpretation inside the executor. Runtime core owns physical page allocation/free and provides page IDs.

The current replay paged SDPA path reads the shared KV page arena through token metadata plus the executor-side GPU page
table. It does not materialize a forward-local dense context window or upload per-forward block tables before
launching the selected Metal attention kernels.

`SingleQToken` and `TiledQTokens` map/reduce generate Metal source from the exact recorded component geometry. Immutable head,
dtype, page, scale, and tile choices are source constants; replay work such as `num_tokens`, Q-token tiles, total map
TaskTemplate extent, and selected Q-head tile width determines the cached recorded variant. Paged partial-output reduce likewise
generates source for stable Q-head/head-dimension geometry while keeping `num_tokens` as its replay argument. The common
kernel source-hash cache reuses identical generated pipelines, so this specialization does not introduce model-specific
backend types or names.

Paged `SingleQToken` SDPA exposes static geometry/tuning separately from dynamic replay work:

```text
GQAPagedSDPAConfig              GQAPagedSDPAShape
  num_q_heads                     num_tokens
  num_kv_heads                    total_sdpa_map_task_templates
  head_dim
  scale
  page_bytes
  page_table_layout
  gqa_layer_index
  kv_token_tile_size
  num_threads_per_threadblock
  q_head_tile_size
  dtype
```

`total_sdpa_map_task_templates` is the padded extent of compact TaskTemplates, not the raw number of KV-token tiles. One
TaskTemplate may cover several consecutive KV-token tiles; `num_sdpa_map_task_templates` is rounded up to produce the
total replay dispatch/scratch extent.

The backend configuration is model-independent. `model/qwen/v3/main/plan.rs` builds Qwen3's ungated core;
`model/qwen/v3_5/plan.rs` builds gated Main/MTP cores and ungated dSpark cores. Each concrete backend converts its
core and `GQAMetalConfig` into a projection split plus the shared norm/RoPE, KV-update, paged-SDPA, and
output-projection components. Only gated `GQA` constructs the activation-gate component. Backend source and APIs
contain no Qwen model names or Qwen configuration types.

## Replay contract

`GQA` records one GQA layer forward through `ReplayLayer::record(...)` and a caller-owned
`Recorder`. It does not submit commands and it does not own request scheduling or page allocation.

The semantic replay input is:

```text
GQAInput
  page_table_layout GQAPageTableLayout
  gqa_layer_index   compact coordinate into the bound page table
  batch_metadata     &GQAMetadataBuffers
  hidden_state      &Buffer
  next_hidden_state &Buffer
  kv_cache          GQAKVCacheBindings
  weights           GQAWeights
  scratch           GQAScratchBindings
```

`GQAOutput<'a>` is the named alias for the returned `&'a Buffer`; it is the caller-owned `next_hidden_state` buffer and
does not allocate or add a wrapper.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay. The data-flow section owns
the stage order and buffer dependencies.

KV page update uses the same model KV dtype as projection scratch and paged SDPA. Its page stride and
`num_tokens_per_page` are derived with that dtype; the Metal component selects the matching bf16/f32 update kernel.
The stable `num_kv_heads`, `head_dim`, `page_bytes`, dtype, and derived tokens-per-page belong to
`GQAKVPageUpdateConfig`; `num_token_writes`, `gqa_layer_index`, and page-table coordinates remain invocation data.

The replay shape separates fixed page-table layout from execution work:

```text
num_tokens                 number of flat tokens in the microbatch
num_q_token_tiles          request-local Q-token tiles; equals num_tokens for SingleQToken replay
total_sdpa_map_task_templates       padded SDPA map TaskTemplate extent used by dispatch and scratch
reduce_sdpa_partial_outputs   whether the selected batch plan semantically requires partial reduction
```

`SingleQToken` replay always records the reduce command, including when every
token has only one map TaskTemplate. This lets both TaskTemplate layouts share one
recorded program for the same Q-token-tile and map-TaskTemplate geometry; the flag remains
batch-plan metadata and does not enter the replay key.

`GQAMetadataBuffers::update_*` derives and stores this shape from the compact request metadata. It is the sole owner of the
current replay shape: `GQAInput` borrows the metadata object instead of carrying a duplicate shape. Backend recording and
replay-key construction both read that stored shape, so one batch plan cannot be paired with a different dispatch shape.

The fixed page-table layout is separate init-time state:

```text
num_req_slots               request-slot dimension of the bound page table
num_gqa_layers              GQA-layer dimension of the bound page-ID table
num_blocks                  block dimension of the bound page table
num_page_ids_per_block      physical page IDs assigned to one cache block
```

Qwen3.5 service replay uses a 2048-token logical cache block. Physical KV
pages remain 32 KiB, so the page-ID count is derived from the model's
tokens-per-physical-page and its GQA-layer count: 27B uses 4,096 GQA pages per
logical block (16 layers × 2048/8) and 35B-A3B uses 1,280 (10 × 2048/16).
The runtime trie and GDN state table use this same logical boundary.

Qwen3 has no GDN snapshot boundary. Its service uses a 16-token logical cache
block; Qwen3-14B stores eight tokens per 32 KiB physical page, so one logical
block owns two pages per GQA layer, or 80 pages across all 40 layers.

For Qwen3.5 model replay, `GQARequestPageTable::prepare(...)` validates and writes the current runtime page updates into
the bound table. `GQA::prepare(...)` selects the SDPA path and builds the batch plan once; every GQA layer reuses it.
`SingleQToken` replay always records partial-output reduction, including batches whose tokens each have one TaskTemplate.
`Qwen35MainReplayKey` and `Qwen35MTPReplayKey` therefore include only `num_q_token_tiles` and
`total_sdpa_map_task_templates`.
The main key also contains the non-optional GDN request-count subkey; MTP keeps its separate pure-GQA key. An ICB
recorded for one selector/dispatch geometry therefore cannot be reused for another, while both map-TaskTemplate layouts
share the same recorded program when that geometry matches.

### Execution strategy

`GQASDPAPath` selects one of two production map/reduce kernel paths. Both paths partition a long visible KV context into
independent `SDPAMapTaskTemplate`s; they differ in how many Q tokens and Q heads one map threadblock computes:

```text
shape: num_tokens grouped into request-local Q-token tiles
parallelism: TaskTemplate x KV head x Q-head tile
input: normalized Q plus paged K/V selected through the request page table
output: context-segment partials, followed by one numerically stable reduce

SingleQToken    one Q token per Q-token tile; scalar dot/reduction work
TiledQTokens    several Q tokens/Q heads per tile; SIMD-group matrix work
```

GQA head sharing is common to both paths. If `G = Hq / Hkv`, KV head `k` supplies K/V to Q heads
`[k * G, (k + 1) * G)`. A buffered TaskTemplate contributes the Q-token tile and half-open KV range; regular head
coordinates come from the grid:

```text
TaskTemplate                         grid coordinates
[q_tile, kv_begin, kv_end)      +   [kv_head, q_head_tile]
              \                         /
               +------ one map threadblock

visible context [0, N)
  -> one or more TaskTemplates
  -> independent partial output + max + exponential sum
  -> reduce to final [Q token, Q head, D]
```

`cu_sdpa_partial_outputs` selects the consecutive partials for each Q-token tile. Padded replay TaskTemplates use
`q_token_tile_index = u32::MAX`; their threadblocks return without writing.

The selector uses `SingleQToken` unless `TiledQTokens` supports the current shape. The explicit bf16 production profiles
are `(D=128, 8 KV tokens/page)` and `(D=256, 16 KV tokens/page)`, both with at most 8 Q heads per KV head. The gated
backend currently reaches only the `D=256` profile. For supported shapes the selector uses the average useful tokens per
request-local Q tile without floating-point division:

```text
num_tokens < 2 * num_q_token_tiles       -> SingleQToken
D=128 profile                            -> TiledQTokens, full Q/KV group
D=256 and num_tokens < 4 * tiles         -> TiledQTokens, roughly half the Q/KV group
otherwise                                -> TiledQTokens, full Q/KV group
```

The Q-head tile is capped at 256 threads. Current reachable model paths are:

| Model | `Hq / Hkv / D` | KV tokens/page | Production path |
| --- | --- | ---: | --- |
| Qwen3-14B | `40 / 8 / 128` | 8 | selector above; tiled `Hq_tile=5` |
| Qwen3.6-27B | `24 / 4 / 256` | 8 | `SingleQToken` |
| Qwen3.6-35B-A3B | `16 / 2 / 256` | 16 | selector above |
| dSpark | `head_dim=128` | model-derived | custom `SingleQToken` composition |

For 35B, `TiledQTokens` uses `Hq_tile=4` below four useful tokens/tile and `Hq_tile=8` otherwise.

#### `SingleQToken`

`SingleQToken` always uses `Tq_tile=1`. Qwen3-14B specializes `Tkv_tile=128`, 128 threads, and `Hq_tile=5`; the Qwen3.5
profiles retain `Tkv_tile=256`, 256 threads, and their model-derived Q-head tile:

```text
one block owns
  one Q token
  one KV head
  up to Hq_tile Q heads sharing that KV head
  one TaskTemplate's [kv_begin, kv_end) segment

Q[q_token, Hq_tile, D]                 paged K/V
          |                                |
          +-- each thread scores K token --+
                           |
             threadgroup logits[Hq_tile, Tkv_tile]
             + block max/sum reduction
                           |
             each thread owns output dim d
             and streams V[token, d]
                           |
                online-softmax merge
                           |
        partial output[D] + max + sum per Q head
```

With Qwen3-14B's `D=128` and 128 threads, one thread owns one output dimension and the full five-head group stays in one
map block. Its `logits[5, 128]` plus `reduce_scratch[128]` use 3 KiB of threadgroup memory. Reducing the head tile to four
would require separate `4+1` head blocks; reducing the thread count to 64 would double the output accumulators held by
each thread. The Qwen3.5 `D=256` profiles use one active thread per output dimension with 7 KiB for 27B
(`Hq_tile=6`) and 9 KiB for 35B (`Hq_tile=8`). K/V are streamed from global memory rather than staged as threadgroup
tiles. Running statistics and owned output dimensions are MSL thread-local values.

#### `TiledQTokens`

The common tiled geometry is:

```text
Q tile:          [up to 8 Q tokens, Hq_tile Q heads, D]
K/V tile:        [16 KV tokens, D] for one KV head
grid:            (Hkv * ceil((Hq/Hkv) / Hq_tile), TaskTemplates, 1)
threads/block:   (Tq_tile / 8) * Hq_tile * 32

Qwen3-14B Hq_tile=5  -> 160 threads = 5 SIMD-groups
Hq_tile=4  -> 128 threads = 4 SIMD-groups
Hq_tile=8  -> 256 threads = 8 SIMD-groups
```

One 32-lane SIMD-group owns one Q head and one eight-token fragment. Its lanes collectively hold the Q rows in MSL
thread-local `q_fragments`; Qwen3-14B has 16 dimension fragments per thread and the `D=256` profiles have 32. An
incomplete request tail loads only active rows.

```text
thread-local Q fragments stay resident for the TaskTemplate
                              |
paged K/V -- 32 lanes x 16 B per row
                              |
                              v
threadgroup K[16, D+8] + V[16, D+8] bf16
                              |
                 Q x K^T, causal mask
                 online-softmax update
                 probability x V
                              |
                thread-local max/sum/output
                              |
                  reuse K/V storage for
                     the next KV tile
                              |
                              v
partial output[up to 8 tokens, Hq_tile, D] + statistics
```

For each K or V row, lanes perform contiguous 16-byte loads. The two threadgroup tiles occupy 8.5 KiB for Qwen3-14B
(`2 * 16 * 136 * sizeof(bf16)`) and 16.5 KiB for `D=256` (`2 * 16 * 264 * sizeof(bf16)`). Q, scores, running
statistics, and output fragments are MSL thread-local. The current kernel has one K workspace and one V workspace; it
does not double-buffer consecutive K/V tiles.

Both paths reduce context-segment partials by rescaling them to one global maximum. `SingleQToken` reduces flat
`[token, Q head, D]` elements; `TiledQTokens` launches one block per `(Q head, Q-token tile)` and strides over active
`token x D` elements. `SingleQToken` records reduce even for one TaskTemplate per token so replay topology stays stable;
`TiledQTokens` always records its tiled reduce. Focused fixed, request-tail, multi-tile, and ragged cases compare both
paths against the CPU reference.

Qwen3.5 keeps reusable gated-GQA scratch in the directly owned `Qwen3xGQAState`, not in individual GQA layers. The
executor owns one Main `GQAScratch`; the optional MTP owns one matching scratch because its GQA configuration may differ.
This scratch contains the QGKV projection, projected gate, gated output, norm/RoPE, and SDPA buffers required by the
fixed gated graph.

Qwen3 Main instead owns `UngatedGQAScratch`. It contains the QKV projection, norm/RoPE, and SDPA buffers required by the
fixed ungated graph and has no gate buffers. Both scratch types expose matching borrowed replay bindings. Main and MTP
execution are serialized on the model stream, so their buffers are reused across submissions without per-layer
allocation. SDPA partial scratch is bounded by
`max_tokens * tiled_q_token_tile_size * num_q_heads`, independent
of `max_position_embeddings`.

`GQAMetadataBuffers` is the matching submission metadata owner. It is capacity-sized once and updated per submission; unlike
scratch, its buffers are read-only during a recorded GQA layer forward.

The buffer contract is:

```text
hidden_state / next_hidden_state     bf16 model boundary buffers shaped [num_tokens, hidden_dim]
req_slots                            request slot repeated per flat token
flat_token_indices                   request-absolute token index per flat token; used for RoPE, KV write address, and causal context length
q_token_tiles                       request-local flat-token ranges consumed by TiledQTokens SDPA
sdpa_map_task_templates              materialized Q-token-tile index and KV-token segment for SDPA map Tasks
cu_sdpa_partial_outputs              cumulative partial-output counts selected per Q-token tile by SDPA reduce
page_ids                             fixed-stride [req_slot, gqa_layer_index, block_index, page_id_index]
kv_pages                             shared runtime-provided KV arena backing
scratch                              caller-owned capacity buffers, used only up to current replay shape
weights                              immutable fused projection, q/k norm, and output projection buffers
```

The KV arena is intentionally recorded as both write and read resource: KV update writes the current tokens, and paged SDPA
reads the request-visible pages. KV update computes the cache block and physical page position from the absolute token
index, then looks up the page through the same page-ID table as SDPA. Runtime core owns page IDs and reset/free lifecycle;
the executor owns model-layer page-ID interpretation, validates shape-local buffer capacities, and rejects a runtime page
ID that is outside the bound global KV page arena.

KV update validates the current-token flat K/V input buffers, token metadata buffers, fixed-stride page table capacity,
and each supplied page ID against the model's global `kv_pages` capacity. Page IDs remain runtime-owned global cache
identifiers; this executor check enforces the runtime/executor ownership contract at the ingestion boundary.

## Data flow and bindings

Both GQA data flows are a single hidden-state stream plus side effects into the runtime-owned KV arena. The gated graph
is:

```text
hidden_state[num_tokens, hidden_dim]
  -> QGKV projection
  -> split q / gate / k / v
  -> q norm + RoPE, k norm + RoPE
  -> write current k/v tokens to KV pages
  -> paged SDPA map reads visible KV pages through the selected block lowering
  -> activation gate
  -> output projection
  -> next_hidden_state[num_tokens, hidden_dim]
```

The separate ungated graph is:

```text
hidden_state[num_tokens, hidden_dim]
  -> QKV projection
  -> split q / k / v
  -> q norm + RoPE, k norm + RoPE
  -> write current k/v tokens to KV pages
  -> paged SDPA map reads visible KV pages through the selected block lowering
  -> output projection
  -> next_hidden_state[num_tokens, hidden_dim]
```

The CPU inputs are `cu_tokens`, per-request `req_slots`, and per-request starting `token_indices`.
`GQAMetadataBuffers` expands them once into the token-major arrays defined in the owner-model section; all GQA layers in
the replay borrow that same plan. The kernels do not consume `cu_tokens` directly.

The page table is storage layout, not a replay shape:

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

KV page writes happen before SDPA and use the same page table interpretation as reads. For each write token, the KV update
kernel computes:

```text
block_index      = flat_token_index / (num_tokens_per_page * num_page_ids_per_block)
page_id_index    = (flat_token_index / num_tokens_per_page) % num_page_ids_per_block
page_token_index = flat_token_index % num_tokens_per_page
page_id          = page_ids[req_slot, gqa_layer_index, block_index, page_id_index]
```

Then it writes the projected K and V token into the shared page arena at that page and token offset. Within one page, the
logical view is `[K/V, kv_head, page_token_index, head_dim]`; its exact byte footprint must equal `page_bytes`. The
physical page id and page lifetime remain runtime-owned.

Paged SDPA is split into map and reduce stages:

```text
map(sdpa_map_task_template_index, kv_head, q_head_tile)
  reads one Q-token tile
  expands one TaskTemplate into one SDPAMapTask using the grid head coordinates
  walks the Task's KV-token segment in fixed-size KV-token tiles
  merges SDPAMapTile partials with online softmax
  resolves each KV token through page IDs + KV page arena
  writes partial_max_logits, partial_exp_sums, and SDPAPartialOutput

reduce(flat_token, q_head)
  uses cu_sdpa_partial_outputs to read that token's partial outputs
  combines stable online-softmax partials
  writes the final attention output token
```

Reduce uses per-partial-output max logits to combine exp sums and weighted outputs without materializing a dense context
window. `SingleQToken` replay records this reduce even when each token has one TaskTemplate so replay topology is stable.
Qwen3-14B's `SingleQToken` configuration uses `kv_token_tile_size=128`,
`num_threads_per_threadblock=128`, and `q_head_tile_size=5`. Qwen3.5 retains `kv_token_tile_size=256`,
`num_threads_per_threadblock=256`, and `q_head_tile_size <= 8`. Q heads are grouped by KV head; each map work item handles
one KV head and a tile of Q heads that share that KV head.

The common resource dependencies are explicit:

```text
q/k norm+RoPE reads q/k scratch and flat_token_indices, writes normalized q/k scratch
KV update reads k/v scratch + flat token metadata + page IDs, writes KV pages
SDPA map reads q scratch + KV pages + sdpa_map_task_templates + page IDs, writes SDPA scratch
SDPA reduce reads SDPA scratch + cu_sdpa_partial_outputs, writes attention output
```

Gated `GQA` prefixes these stages with QGKV projection and split into q/g/k/v scratch, then suffixes them with the
activation gate and an output projection from gated attention. `UngatedGQA` prefixes them with QKV projection and split
into q/k/v scratch, then sends attention output directly to the output projection.

Barriers are recorded only between stages with real dependencies. The model/layer boundary should not add an implicit
every-command barrier around GQA; explicit component barriers and backend-inferred buffer hazards provide the internal
ordering it needs.

## Tests and benches

Current correctness coverage is split between focused backend/component tests and Qwen wiring/model tests.

`gqa_attention` compares paged SDPA against the CPU projected-GQA reference with fixed input, random input, a random
ragged batch, and one TaskTemplate spanning multiple KV-token tiles. The cases validate compact TaskTemplate indexing,
online-softmax tile merging, request slots, page-table lookup, and causal visibility.

Metal backend component replay sanity lives in:

```text
cargo bench -p inference-backend-metal --bench gqa_attn -- --profile-time 1 --noplot
```

The GQA backend bench records paged-SDPA building blocks into Metal replay/ICB paths only. GQA Metal code does
not benchmark or expose direct-submit component or forward wiring.

Metal backend real full-forward replay bench lives in:

```text
cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 0,128,1024 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-paths single_q_token \
  --iters 1 --warmup-iters 0 --runs 1
```

Both benches use CLI arguments, not environment variables. `qwen3_gqa` loads the Qwen3 model config and first-layer
ungated weights, accepts explicit per-request token counts and context lengths, and reports both full replay and SDPA-only
measurements. Its single-Q and tiled tile/thread arguments are configurable; `--validate` compares full single-Q and tiled
outputs for a workload where production selects the tiled path and prints the derived threadgroup/register shape.

For `qwen35_gqa`, `--gqa-model 27b|35b` selects the real-weight layer profile;
the matching model directory is passed with `--model-dir`. For GQA, `--tokens` is the total current flat-token count,
`--num-reqs` is the number of request segments in that microbatch, and `--contexts` is the existing context length for
each request before its measured tokens. The bench distributes tokens as evenly as possible across requests and builds
`req_slots`, `flat_token_indices`, and a fixed-stride request page table
from those options. A single-request decode-style context sweep should use `--tokens 1 --num-reqs 1` and vary
`--contexts`; a multi-request decode batch can use `--tokens 8 --num-reqs 8`; a prefill/suffix sweep can use
`--tokens 64 --num-reqs 1 --contexts 0,2048,4096`. Without an explicit context list, the bench uses existing context
length zero. `--gqa-tokens-per-req` supplies explicit ragged per-request token counts. The comparison replay reports
`path=single_q_token` or `path=tiled_q_tokens`; model execution uses the automatic selector described above.
`--gqa-single-q-token-kv-token-tile-size`, `--gqa-single-q-token-num-threads-per-threadblock`, and
`--gqa-single-q-token-max-q-head-tile-size` override the `SingleQToken` defaults;
`--gqa-tiled-q-token-tile-size`, `--gqa-tiled-kv-token-tile-size`, and
`--gqa-tiled-q-head-tile-size` configure the `TiledQTokens` comparison path. When the Q-head override is omitted, the bench
uses the same half/full Q/KV-group rule as production. Bench output uses the corresponding
`q_token_tile_size`, `kv_token_tile_size`, and `q_head_tile_size` names.
`--print-limits` prints the device
threadblock-memory limit and the derived `SingleQToken` threadblock-memory footprint. The current backend records
explicit data-dependency barriers, and the replay layer also infers hazards from declared buffer usage; it does not add
a conservative every-command fallback. This bench
loads real Qwen3.6 layer weights and measures the full replay path: qgkv projection, projection split, q/k norm+RoPE, KV
page update, paged SDPA, activation gate, and output projection. Do not compare component-only paged-SDPA timings
against full-forward numbers.

Subcomponent probes use the same request-slot/page-table capacity contract as full-forward replay. In particular,
multi-request `kv-update` probes must pass the true `num_req_slots` through the page-table layout in
`GQAKVPageUpdateShape`; hard-coding a single
request slot under-validates the page-table contract even if the kernel happens to read the larger backing buffer.

Production Qwen KV cache dtype follows model config; Qwen3.6 bf16/default config creates bf16 KV pages. The paged KV writer stores projected K/V into those page-table pages.

The Metal executor keeps forward wiring out of `components/`: gated `GQA` composes QGKV projection, gate, and the
shared attention building blocks, while `UngatedGQA` composes the fixed QKV graph without gate resources.
`ReplayLayer::record(...)` appends into a caller-owned whole-layer/model replay recorder; focused tests and benches
build replay programs from the same recorder path. Each model state binds one shared scratch allocation for compatible
layers so decode replay reuse does not multiply scratch by layer count.
The core `scale` is part of both attention contracts and is passed through to paged SDPA kernels; kernels must not
silently substitute `1 / sqrt(head_dim)`.

Q and K norm/RoPE use separately specialized commands because their stable head counts differ.

GQA replay bugs usually reduce to these contracts: typed buffer offsets use typed indices; raw Metal buffer bindings use byte offsets,
page/stride arithmetic needs 64-bit address math in Metal when multiplying large strides, resource usage marks must
cover KV page write/read exactly, dynamic threadblock memory must be recorded into both direct and ICB paths, and
replay-cache shape keys must distinguish fixed page-table stride from execution partition/tile count. Debug in this
order: component primitive, real-weight GQA wrapper, attention slice inside a layer, then the layer ladder.

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
