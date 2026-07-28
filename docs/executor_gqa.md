# GQA Executor

This document describes the current GQA implementation. It covers tensor geometry, batch metadata, Metal replay,
KV-page interpretation, and correctness tests.

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

`crates/inference-executor-core` owns the backend-neutral GQA semantic metadata and replay shape.
`crates/inference-executor-metal` owns the Metal replay wiring and request page table.

The Metal GQA executor backend implements the executor `ReplayLayer` contract. Qwen model and layer code can use this
contract to append GQA work to a larger replay. A semantic layer input and output connect the work to a caller-owned
`Recorder`.

`request_page_table.rs` owns the executor request-slot KV page table. This table accumulates runtime-supplied page IDs
between reset notifications. The runtime core owns physical page allocation and release.

## Ownership

Each derived `GQACore` carries its source model coordinate plus the common GQA dimensions:

```text
model_layer_index
hidden_dim
head_dim
num_q_heads / num_kv_heads
attention scale
```

`GQACore` defines the gated contract. Its fused projection is always QGKV. Its replay always applies the trained
attention-output gate.

`UngatedGQACore` defines the separate ungated contract. Its fused projection is always QKV. It has no gate resources
or gate recording step.

Both contracts derive query, key, and value widths from the head counts and `head_dim`. Neither contract uses a layout
enum or gate flag.

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

`T` identifies token dimensions. `H` identifies head dimensions. `D` is `head_dim`.

`SDPAMapTile` is the smallest matmul-like logical description
`(q_token_tile_index, kv_head_index, q_head_tile_index, kv_token_tile_index)`.

Only `model_layer_index` is per layer. Qwen validates the remaining fields once and uses them through the shared backend.

Qwen3 Main owns `Qwen3MainGQAState`, `UngatedGQA`, and `UngatedGQAScratch`.

Qwen3.5 owns `Qwen3xGQAState`. This state holds one shared gated `GQA` backend and one `GQAScratch` for compatible
invocations. It also holds one `Rc<GQARequestPageTable>` and reusable `GQAMetadataBuffers`.

Each `Qwen3xGQA` layer component retains clones of the backend, scratch, and page-table handles. It also owns its
weights and compact layer coordinate. Qwen3 Main uses its model-owned ungated leaf.

`Qwen35MainLayer` and `Qwen35MTPLayer` use the gated shared leaf in separate role-specific owners.

The gated and ungated executors share the lower-level norm/RoPE, KV update, paged SDPA, and output-projection
components. Their projection, scratch, weights, and replay graphs remain concrete and separate.

This structure removes mode checks from the gated QGKV command sequence. It also makes the missing gate structural in
the ungated QKV path.

Qwen3 Main constructs `UngatedGQACore` and `UngatedGQA`. Qwen3.5 Main and MTP construct `GQACore` and gated `GQA`.
Qwen3.5 DSpark also uses `UngatedGQACore` for its separate QKV attention graph.

Init-time component specialization supplies the head dimensions, head counts, RoPE constants, and page geometry. A
model-specific runtime branch does not supply these values.

`GQARequestPageTable` stores executor request-slot KV page IDs in a fixed-stride GPU buffer. It retains the IDs between
runtime reset and update notifications.

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

The runtime still owns physical page allocation and release.

`GQAMetadataBuffers` stores the GPU arrays shared by every GQA layer in one model replay:

```text
req_slots[num_tokens]
flat_token_indices[num_tokens]
q_token_tiles[num_q_token_tiles][flat_token_start/flat_token_end]  // TiledQTokens
sdpa_map_task_templates[total_sdpa_map_task_templates][q_token_tile_index/kv_token_begin/kv_token_end]
cu_sdpa_partial_outputs[num_tokens + 1]                              // SingleQToken
cu_sdpa_partial_outputs[num_q_token_tiles + 1]                       // TiledQTokens
```

Each three-`u32` entry materializes one compact `SDPAMapTaskTemplate`. It contains a Q-token-tile index followed by the
half-open KV-token segment.

The grid supplies `kv_head_index` and `q_head_tile_index`. These coordinates combine with the template to produce one
logical `SDPAMapTask`. One threadblock owns each task in a `1:1` relation. The buffer does not duplicate the grid-derived
coordinates.

`SingleQToken` uses one-token Q tiles. `TiledQTokens` first builds request-local Q-token tiles.

The planner assigns additional TaskTemplates to the Q-token tile with the most remaining KV-tile work. TaskTemplates
for one Q-token tile are contiguous.

For a fixed Q-token/head output coordinate, adjacent `cu_sdpa_partial_outputs` values select the
`SDPAPartialOutput`s for the reducer.

`total_sdpa_map_task_templates` is the power-of-two replay extent. Unused tail TaskTemplates contain an invalid
Q-token-tile index and do not write a map result.

The `SingleQToken` paged map also permits an invalid-Q-token-tile `SDPAMapTaskTemplate` in one token's TaskTemplate
range. This template does not write a paged partial output for that slot.

A caller may populate the reserved max-logit, exp-sum, and normalized `SDPAPartialOutput` through
`GQALocalSDPAKernel`. It does this before it invokes the unchanged partial-output reducer.

This generic composition supports an attention connection that combines paged history with a dense bidirectional
local block. The backend component does not own model-specific proposal or cache semantics.

Replay recording borrows `&GQAMetadataBuffers` directly. It does not use a duplicate bindings wrapper.

The CPU uses `cu_tokens`, per-request `req_slots`, and per-request starting `token_indices` to build these token-major
arrays. GQA kernels do not consume `cu_tokens`. Therefore, `GQAMetadataBuffers` does not retain a GPU copy.

The model-level GQA storage shape is:

```text
pages[num_cache_pages][page_bytes]

main_page_ids[num_req_slots][num_gqa_layers][num_blocks][num_page_ids_per_block]
optional_mtp_page_ids[num_req_slots][1][num_blocks][num_page_ids_per_block]

one KV page, viewed with the model KV dtype:
  [K/V][num_kv_heads][num_tokens_per_page][head_dim]
```

The Metal config stores `page_bytes` and the shared activation/KV dtype. It derives `num_tokens_per_page` from these
values, `num_kv_heads`, and `head_dim`. It does not store that derived value or a duplicate KV dtype.

For a flat token, the page lookup is exactly:

```text
req_slot -> gqa_layer_index -> block_index -> page_id_index -> page_id -> KV page
```

The main page-ID table uses compact main `gqa_layer_index`. The supported optional MTP has an independent table. Its
single full-attention body uses `gqa_layer_index = 0`.

The layout type uses generic `num_gqa_layers` for both tables. Each table instance owns its capacity and can use a
different GQA configuration.

The Qwen executor updates Main state once for each Main batch. It updates optional MTP metadata once for the MTP stage.

Main GQA layers borrow the Main state domain. The optional MTP owns its backend, scratch, and compact page-ID table.

Layer owners retain immutable weights and a `gqa_layer_index`. A main layer does not retain model-level GQA
configuration or batch metadata. Each current MTP full-attention body uses coordinate 0 in its own table.

## Backend specialization

The GQA benchmark uses these subcomponent names:

```text
qgkv-proj
split
q-norm-rope
k-norm-rope
kv-update
sdpa-single-q-token
sdpa-tiled-q-tokens
gate
output-proj
```

GQA owns KV page-table and cache interpretation inside the executor. The runtime core owns physical page allocation and
release. It also provides page IDs.

The replay paged SDPA path reads the shared KV page arena through token metadata and the executor GPU page table. It does
not materialize a forward-local dense context window.

The path does not upload per-forward block tables before it launches the selected Metal attention kernels.

`SingleQToken` and `TiledQTokens` map/reduce generate Metal source from the exact recorded component geometry.
Immutable head, dtype, page, scale, and tile choices become source constants.

Replay work determines the cached recorded variant. This work includes `num_tokens`, Q-token tiles, the total map
TaskTemplate extent, and the selected Q-head tile width.

Paged partial-output reduce also generates source for stable Q-head and head-dimension geometry. It keeps `num_tokens`
as a replay argument.

The common kernel source-hash cache reuses identical generated pipelines. This specialization does not introduce
model-specific backend types or names.

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

`total_sdpa_map_task_templates` is the padded extent of compact TaskTemplates. It is not the raw number of KV-token
tiles.

One TaskTemplate can cover several consecutive KV-token tiles. The planner rounds up
`num_sdpa_map_task_templates` to produce the total replay dispatch and scratch extent.

The backend configuration is model-independent. `model/qwen/v3/main/plan.rs` builds the Qwen3 ungated core.
`model/qwen/v3_5/plan.rs` builds gated Main/MTP cores and ungated DSpark cores.

Each concrete backend converts its core and `GQAMetalConfig` into a projection split. It also constructs the shared
norm/RoPE, KV-update, paged-SDPA, and output-projection components.

Only gated `GQA` constructs the activation-gate component. Backend source and APIs contain no Qwen model names or Qwen
configuration types.

## Replay contract

`GQA` records one GQA layer forward through `ReplayLayer::record(...)` and a caller-owned `Recorder`. It does not
submit commands. It does not own request scheduling or page allocation.

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

`GQAOutput<'a>` is the named alias for the returned `&'a Buffer`. It is the caller-owned `next_hidden_state` buffer.
The alias does not allocate or add a wrapper.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay. The data-flow section
defines the stage order and buffer dependencies.

KV page update uses the same model KV dtype as projection scratch and paged SDPA. The component derives its page stride
and `num_tokens_per_page` with that dtype.

The Metal component selects the matching bf16/f32 update kernel. `GQAKVPageUpdateConfig` owns the stable
`num_kv_heads`, `head_dim`, `page_bytes`, dtype, and derived tokens-per-page.

`num_token_writes`, `gqa_layer_index`, and page-table coordinates remain invocation data.

The replay shape separates fixed page-table layout from execution work:

```text
num_tokens                 number of flat tokens in the microbatch
num_q_token_tiles          request-local Q-token tiles; equals num_tokens for SingleQToken replay
total_sdpa_map_task_templates       padded SDPA map TaskTemplate extent used by dispatch and scratch
reduce_sdpa_partial_outputs   whether the selected batch plan semantically requires partial reduction
```

`SingleQToken` replay always records the reduce command. It records this command even when each token has only one map
TaskTemplate.

This rule lets both TaskTemplate layouts share one recorded program for the same Q-token-tile and map-TaskTemplate
geometry. The flag remains batch-plan metadata and does not enter the replay key.

`GQAMetadataBuffers::update_*` derives and stores this shape from the compact request metadata. It is the sole owner of
the current replay shape.

`GQAInput` borrows the metadata object instead of carrying a duplicate shape. Backend recording and replay-key
construction both read the stored shape. Therefore, a batch plan cannot use a different dispatch shape.

The fixed page-table layout is separate init-time state:

```text
num_req_slots               request-slot dimension of the bound page table
num_gqa_layers              GQA-layer dimension of the bound page-ID table
num_blocks                  block dimension of the bound page table
num_page_ids_per_block      physical page IDs assigned to one cache block
```

Qwen3.5 service replay uses a 2048-token logical cache block. Physical KV pages remain 32 KiB.

The model's tokens-per-physical-page and GQA-layer count determine the page-ID count. The 27B model uses 4,096 GQA
pages per logical block (16 layers × 2048/8). The 35B-A3B model uses 1,280 pages (10 × 2048/16).

The runtime trie and GDN state table use this same logical boundary.

Qwen3 has no GDN snapshot boundary. Its service uses a 16-token logical cache block. Qwen3-14B stores eight tokens in
each 32 KiB physical page.

One logical block therefore owns two pages per GQA layer. It owns 80 pages across all 40 layers.

For Qwen3.5 model replay, `GQARequestPageTable::prepare(...)` validates and writes the current runtime page updates to
the bound table. `GQA::prepare(...)` selects the SDPA path and builds the batch plan once. Every GQA layer reuses this
plan.

`SingleQToken` replay always records partial-output reduction. This rule also applies when each batch token has one
TaskTemplate.

`Qwen35MainReplayKey` and `Qwen35MTPReplayKey` therefore include only `num_q_token_tiles` and
`total_sdpa_map_task_templates`. The main key also contains the non-optional GDN request-count subkey.

MTP keeps its separate pure-GQA key. An ICB recorded for one selector or dispatch geometry cannot serve another
geometry. Both map-TaskTemplate layouts share the recorded program when their geometry matches.

### Execution strategy

`GQASDPAPath` selects one of two production map/reduce kernel paths. Both paths partition a long visible KV context
into independent `SDPAMapTaskTemplate`s.

The paths differ in the number of Q tokens and Q heads that one map threadblock computes:

```text
shape: num_tokens grouped into request-local Q-token tiles
parallelism: TaskTemplate x KV head x Q-head tile
input: normalized Q plus paged K/V selected through the request page table
output: context-segment partials, followed by one numerically stable reduce

SingleQToken    one Q token per Q-token tile; scalar dot/reduction work
TiledQTokens    several Q tokens/Q heads per tile; SIMD-group matrix work
```

Both paths use GQA head sharing. If `G = Hq / Hkv`, KV head `k` supplies K/V to Q heads
`[k * G, (k + 1) * G)`.

A buffered TaskTemplate supplies the Q-token tile and half-open KV range. The grid supplies the regular head
coordinates:

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
`q_token_tile_index = u32::MAX`. Their threadblocks return without writing.

The selector uses `SingleQToken` unless `TiledQTokens` supports the current shape. The explicit bf16 production
profiles are `(D=128, 8 KV tokens/page)` and `(D=256, 16 KV tokens/page)`.

Both profiles support at most 8 Q heads per KV head. The gated backend currently reaches only the `D=256` profile.

For supported shapes, the selector uses the average useful tokens per request-local Q tile. It does not use
floating-point division:

```text
num_tokens < 2 * num_q_token_tiles       -> SingleQToken
D=128 profile                            -> TiledQTokens, full Q/KV group
D=256 and num_tokens < 4 * tiles         -> TiledQTokens, roughly half the Q/KV group
otherwise                                -> TiledQTokens, full Q/KV group
```

The Q-head tile is capped at 256 threads. Current reachable model paths are:

| Model | `Hq / Hkv / D` | KV tokens/page | Production path |
| --- | --- | ---: | --- |
| Qwen3-14B | `40 / 8 / 128` | 8 | selector above, tiled `Hq_tile=5` |
| Qwen3.6-27B | `24 / 4 / 256` | 8 | `SingleQToken` |
| Qwen3.6-35B-A3B | `16 / 2 / 256` | 16 | selector above |
| DSpark | `head_dim=128` | model-derived | custom `SingleQToken` composition |

For 35B, `TiledQTokens` uses `Hq_tile=4` below four useful tokens/tile and `Hq_tile=8` otherwise.

#### `SingleQToken`

`SingleQToken` always uses `Tq_tile=1`. Qwen3-14B specializes `Tkv_tile=128`, 128 threads, and `Hq_tile=5`.

The Qwen3.5 profiles retain `Tkv_tile=256`, 256 threads, and their model-derived Q-head tile:

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

For Qwen3-14B, `D=128` and 128 threads let one thread own one output dimension. The full five-head group stays in one
map block.

Its `logits[5, 128]` and `reduce_scratch[128]` use 3 KiB of threadgroup memory. A head tile of four requires separate
`4+1` head blocks.

A thread count of 64 doubles the output accumulators that each thread holds.

The Qwen3.5 `D=256` profiles use one active thread per output dimension. The 27B profile uses 7 KiB
(`Hq_tile=6`). The 35B profile uses 9 KiB (`Hq_tile=8`).

The kernels stream K/V from global memory instead of staging them as threadgroup tiles. Running statistics and owned
output dimensions are MSL thread-local values.

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
thread-local `q_fragments`.

Qwen3-14B has 16 dimension fragments per thread. The `D=256` profiles have 32. An incomplete request tail loads only
active rows.

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

For each K or V row, lanes load contiguous 16-byte segments. The two threadgroup tiles occupy 8.5 KiB for Qwen3-14B
(`2 * 16 * 136 * sizeof(bf16)`).

The tiles occupy 16.5 KiB for `D=256` (`2 * 16 * 264 * sizeof(bf16)`). Q, scores, running statistics, and output
fragments are MSL thread-local.

The current kernel has one K workspace and one V workspace. It does not double-buffer consecutive K/V tiles.

Both paths reduce context-segment partials by rescaling them to one global maximum. `SingleQToken` reduces flat
`[token, Q head, D]` elements.

`TiledQTokens` launches one block per `(Q head, Q-token tile)`. The block strides over active `token x D` elements.

`SingleQToken` records reduce even for one TaskTemplate per token. This rule keeps the replay topology stable.
`TiledQTokens` always records its tiled reduce.

Focused fixed, request-tail, multi-tile, and ragged cases compare both paths with the CPU reference.

Qwen3.5 keeps reusable gated-GQA scratch in the directly owned `Qwen3xGQAState`. Individual GQA layers do not own this
scratch.

The executor owns one Main `GQAScratch`. The optional MTP owns one matching scratch because its GQA configuration can
differ.

This scratch contains the buffers for QGKV projection, the projected gate, gated output, norm/RoPE, and SDPA. The fixed
gated graph requires these buffers.

Qwen3 Main owns `UngatedGQAScratch`. It contains the QKV projection, norm/RoPE, and SDPA buffers for the fixed ungated
graph. It has no gate buffers.

Both scratch types expose matching borrowed replay bindings. The model stream serializes Main and MTP execution.
Therefore, submissions reuse their buffers without per-layer allocation.

The bound for SDPA partial scratch is `max_tokens * tiled_q_token_tile_size * num_q_heads`. It is independent of
`max_position_embeddings`.

`GQAMetadataBuffers` owns the matching submission metadata. The owner receives its capacity once and updates its data
for each submission. Its buffers are read-only during a recorded GQA layer forward.

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

Q and K norm weights keep the checkpoint BF16 storage type. The norm/RoPE kernel reads them directly. It preserves the
configured activation-type arithmetic and rounding order. RMS reduction and RoPE trigonometry use F32.

The recording marks the KV arena as both a write and read resource. KV update writes the current tokens. Paged SDPA
reads the request-visible pages.

KV update calculates the cache block and physical page position from the absolute token index. It then finds the page
through the same page-ID table that SDPA uses.

The runtime core owns page IDs and their reset and release lifecycle. The executor owns model-layer page-ID
interpretation. It validates shape-local buffer capacities.

The executor rejects a runtime page ID outside the bound global KV page arena.

KV update validates the current-token flat K/V input buffers and token metadata buffers. It also validates the
fixed-stride page table capacity.

The update checks each supplied page ID against the model's global `kv_pages` capacity. Page IDs remain runtime-owned
global cache identifiers. This check enforces the runtime/executor ownership contract at the ingestion boundary.

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
`GQAMetadataBuffers` expands them once into the token-major arrays in the ownership section.

All GQA layers in the replay borrow that plan. The kernels do not consume `cu_tokens` directly.

The page table is storage layout, not a replay shape:

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

KV page writes occur before SDPA. Writes and reads use the same page-table interpretation. For each write token, the KV
update kernel calculates:

```text
block_index      = flat_token_index / (num_tokens_per_page * num_page_ids_per_block)
page_id_index    = (flat_token_index / num_tokens_per_page) % num_page_ids_per_block
page_token_index = flat_token_index % num_tokens_per_page
page_id          = page_ids[req_slot, gqa_layer_index, block_index, page_id_index]
```

The kernel then writes the projected K and V token to the shared page arena. It uses the selected page and token
offset.

Within one page, the logical view is `[K/V, kv_head, page_token_index, head_dim]`. Its exact byte footprint must equal
`page_bytes`. The runtime owns the physical page ID and page lifetime.

Paged SDPA has map and reduce stages:

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

Reduce uses per-partial-output max logits to combine exp sums and weighted outputs. It does not materialize a dense
context window.

`SingleQToken` replay records this reduce even when each token has one TaskTemplate. This rule keeps the replay topology
stable.

Qwen3-14B `SingleQToken` uses `kv_token_tile_size=128`, `num_threads_per_threadblock=128`, and
`q_head_tile_size=5`.

Qwen3.5 retains `kv_token_tile_size=256`, `num_threads_per_threadblock=256`, and `q_head_tile_size <= 8`.

The layout groups Q heads by KV head. Each map work item handles one KV head and a tile of its Q heads.

The common resource dependencies are explicit:

```text
q/k norm+RoPE reads q/k scratch and flat_token_indices, writes normalized q/k scratch
KV update reads k/v scratch + flat token metadata + page IDs, writes KV pages
SDPA map reads q scratch + KV pages + sdpa_map_task_templates + page IDs, writes SDPA scratch
SDPA reduce reads SDPA scratch + cu_sdpa_partial_outputs, writes attention output
```

Gated `GQA` prefixes these stages with QGKV projection and a split into q/g/k/v scratch. It adds the activation gate
and an output projection after gated attention.

`UngatedGQA` prefixes the stages with QKV projection and a split into q/k/v scratch. It sends attention output directly
to the output projection.

Record barriers only between stages with real dependencies.

Recommendation: The model/layer boundary does not add an implicit every-command barrier around GQA.

Explicit component barriers and backend-inferred buffer hazards provide the required internal order.

## Tests and benches

Focused backend and component tests provide part of the correctness coverage. Qwen wiring and model tests provide the
remaining coverage.

`gqa_attention` compares paged SDPA with the CPU projected-GQA reference. It uses fixed input, random input, and a
random ragged batch.

Another case uses one TaskTemplate that spans multiple KV-token tiles. The cases validate compact TaskTemplate
indexing, online-softmax tile merging, request slots, page-table lookup, and causal visibility.

Metal backend component replay sanity lives in:

```text
cargo bench -p inference-backend-metal --bench gqa_attn -- --profile-time 1 --noplot
```

The GQA backend bench records paged-SDPA building blocks only in Metal replay/ICB paths. GQA Metal code does not
benchmark or expose direct-submit component or forward wiring.

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

Both benches use CLI arguments instead of environment variables.

`qwen3_gqa` loads the Qwen3 model config and first-layer ungated weights. It accepts explicit per-request token counts
and context lengths. It reports full-replay and SDPA-only measurements.

Its single-Q and tiled tile/thread arguments are configurable. `--validate` compares full single-Q and tiled outputs
for a workload where production selects the tiled path.

The validation also prints the derived threadgroup/register shape.

For `qwen35_gqa`, `--gqa-model 27b|35b` selects the real-weight layer profile. Pass the matching model directory with
`--model-dir`.

For GQA, `--tokens` is the total current flat-token count. `--num-reqs` is the number of request segments in that
microbatch. `--contexts` is the existing context length for each request before its measured tokens.

The bench distributes tokens as evenly as possible across requests. It builds `req_slots`, `flat_token_indices`, and a
fixed-stride request page table from these options.

Recommendation: For a single-request decode-style context sweep, use `--tokens 1 --num-reqs 1` and vary `--contexts`.

For a multi-request decode batch, you may use `--tokens 8 --num-reqs 8`.

For a prefill/suffix sweep, you may use `--tokens 64 --num-reqs 1 --contexts 0,2048,4096`.

Without an explicit context list, the bench uses existing context length zero. `--gqa-tokens-per-req` supplies explicit
ragged per-request token counts.

The comparison replay reports `path=single_q_token` or `path=tiled_q_tokens`. Model execution uses the automatic
selector described above.

`--gqa-single-q-token-kv-token-tile-size`, `--gqa-single-q-token-num-threads-per-threadblock`, and
`--gqa-single-q-token-max-q-head-tile-size` override the `SingleQToken` defaults.

`--gqa-tiled-q-token-tile-size`, `--gqa-tiled-kv-token-tile-size`, and `--gqa-tiled-q-head-tile-size` configure the
`TiledQTokens` comparison path.

When the Q-head override is absent, the bench uses the production half/full Q/KV-group rule. Bench output uses the
corresponding `q_token_tile_size`, `kv_token_tile_size`, and `q_head_tile_size` names.

`--print-limits` prints the device threadblock-memory limit. It also prints the derived `SingleQToken`
threadblock-memory footprint.

The current backend records explicit data-dependency barriers. The replay layer also infers hazards from declared
buffer usage. It does not add a conservative every-command fallback.

This bench loads real Qwen3.6 layer weights. It measures the full replay path: qgkv projection, projection split, q/k
norm+RoPE, KV page update, paged SDPA, activation gate, and output projection.

Do not compare component-only paged-SDPA timings with full-forward numbers.

Subcomponent probes use the same request-slot/page-table capacity contract as full-forward replay. Multi-request
`kv-update` probes must pass the true `num_req_slots` through the page-table layout in `GQAKVPageUpdateShape`.

Do not hard-code one request slot. That value under-validates the page-table contract, even if the kernel reads the
larger backing buffer.

Production Qwen KV cache dtype follows the model config. The Qwen3.6 bf16/default config creates bf16 KV pages. The
paged KV writer stores projected K/V in those page-table pages.

The Metal executor keeps forward wiring out of `components/`. Gated `GQA` composes QGKV projection, the gate, and the
shared attention building blocks.

`UngatedGQA` composes the fixed QKV graph without gate resources.

`ReplayLayer::record(...)` appends to a caller-owned whole-layer/model replay recorder. Focused tests and benches build
replay programs through the same recorder path.

Each model state binds one shared scratch allocation for compatible layers. Therefore, decode replay reuse does not
multiply scratch by the layer count.

The core `scale` is part of both attention contracts. The executor passes it to paged SDPA kernels. Kernels must not
silently substitute `1 / sqrt(head_dim)`.

Q and K norm/RoPE use separately specialized commands because their stable head counts differ.

Use typed indices for typed buffer offsets. Use byte offsets for raw Metal buffer bindings.

Use 64-bit Metal address math when page or stride arithmetic multiplies large strides. Resource usage marks must cover
KV page writes and reads exactly.

Record dynamic threadblock memory in both direct and ICB paths. Replay-cache shape keys must distinguish fixed
page-table stride from execution partition and tile count.

Debug in this order:

1. Component primitive
2. Real-weight GQA wrapper
3. Attention slice in a layer
4. Layer ladder

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
