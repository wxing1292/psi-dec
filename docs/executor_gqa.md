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
    dspark_core.rs          DSpark block geometry and metadata
    reference.rs            CPU projected-GQA correctness oracle

crates/inference-executor-metal/src/attn/
  mod.rs                    Metal attention module exports
  gqa/
    mod.rs                  GQA Metal module root
    sdpa.rs                 executor-owned complete-plan construction and execution selection
    batch_metadata.rs       capacity-sized GPU upload target for one selected SDPA plan
    backend.rs              gated QGKV Metal replay wiring
    scratch.rs              gated QGKV scratch allocation and borrowed replay bindings
    ungated_backend.rs      ungated QKV Metal replay wiring
    ungated_scratch.rs      ungated QKV scratch allocation and borrowed replay bindings
    request_page_table.rs   per-request, per-layer KV page table for runtime-supplied page IDs
    request_page_table/
      file_io.rs            symmetric full and selected state file I/O
  dspark/
    mod.rs                  DSpark attention module root
    backend.rs              ungated paged-history plus block-bidirectional replay graph
    capacity.rs             Metal DSpark GQA partial-output capacity
    context.rs              persistent DSpark context K/V append
    metadata.rs             proposal history and block metadata
    scratch.rs              fixed-capacity local Q/K/V and attention partials
    state.rs                DSpark page-table and proposal metadata lifecycle
    state/
      file_io.rs            DSpark full and selected state file I/O

crates/inference-executor-metal/src/model/qwen/
  v3_x/
    dspark/                 Qwen3x DSpark attention, layer, and model
    layer/gqa.rs            Qwen3xGQA, private checkpoint weights, load, and record
    state/gqa.rs            Qwen3xGQAState page/metadata/reset lifecycle grouping
    state/gqa/file_io.rs    Qwen3xGQAState full and selected state file I/O
  v3/
    main/gqa.rs             Qwen3 Main ungated GQA weights, state, load, and record
    main/gqa/file_io.rs     Qwen3 Main GQA full and selected state file I/O
    main/layer.rs           fixed QKV Qwen3MainLayer composition
    main/plan.rs            Qwen3 Main QKV GQA geometry/config builder
  v3_5/
    main/layer.rs           Qwen3.5 Main QGKV-GQA/GDN layer variants
    mtp/layer.rs            Qwen3.5 MTP GQA layer composition
    plan.rs                 Qwen3.5 QGKV GQA geometry/config builder

crates/inference-backend-metal/src/components/
  gqa_split_kv.rs           backend-owned SDPA specialization registry and capability checks
  gqa_split_kv_single_q.rs  reusable Metal SplitKV SingleQ and activation-gate kernels
  gqa_split_kv_single_q_test.rs
                            SplitKV SingleQ Metal parity and replay contracts
  gqa_split_kv_tiled_q.rs   reusable Metal SplitKV TiledQ component kernels
  gqa_split_kv_tiled_q_test.rs
                            SplitKV TiledQ Metal parity and padded replay contracts
  gqa_block_attention.rs    reusable dense bidirectional block-SDPA partial kernel
  gqa_qgkv_split.rs         gated QGKV split component
  gqa_qkv_split.rs          ungated QKV split component
  rms_norm_rope.rs          reusable Metal head-row RMSNorm/RoPE component
  gqa_kv_page_write.rs      reusable Metal KV page-write component
  metal/
    gqa_qgkv_split.metal       gated QGKV split source
    gqa_qkv_split.metal        ungated QKV split source
    rms_norm_rope.metal         Metal head-row RMSNorm/RoPE source
    gqa_kv_page_write.metal     Metal KV page-write source
    gqa_split_kv_single_q_map.metal     Metal SplitKV SingleQ map source
    gqa_split_kv_single_q_reduce.metal  Metal SplitKV SingleQ reduce source
    gqa_split_kv_tiled_q.metal          Metal SplitKV TiledQ map/reduce source
    gqa_block_sdpa.metal         Metal dense block-SDPA partial-output source
    gqa_activation_gate.metal    Metal attention-output gate source
```

`crates/inference-executor-core` owns the backend-neutral GQA semantic metadata and replay shape.
`crates/inference-executor-metal` owns the Metal replay wiring and request page table.

The Metal GQA executor backend implements the executor `ReplayLayer` contract. Qwen model and layer code can use this
contract to append GQA work to a larger replay. A semantic layer input and output connect the work to a caller-owned
`Recorder`.

`request_page_table.rs` owns the executor request-slot KV page table. This table accumulates page IDs between reset
notifications. The runtime core owns physical page allocation and release. The model executor converts runtime cache
lanes into model-role and GQA-layer coordinates before it writes the table.

The gated, ungated, and DSpark GQA backends give each quantized projection to one adaptive affine operator.
The caller provides the fixed projection dimensions, quantization layout, and dtype when it creates the operator.
It provides the current active token count when it records the projection.
The affine operator selects QMV or a QMM tile.
GQA code does not store separate QMV/QMM kernels or a projection threshold.
This contract applies to fused QGKV/QKV projections, output projections, and DSpark context K/V projections.
The Metal backend registry derives legal SDPA kernel specializations from the static attention and KV-cache shape.
The Metal executor planner selects one complete Map/Reduce execution from the prepared batch shape.
Model plans do not specify Metal SDPA tiles or threadblock sizes.
An exact backend microbenchmark may construct a low-level SDPA component with explicit geometry.

SplitKV is the local KV-range map/reduce computation on one Metal device. It partitions each visible KV range into KV
segments. It maps the segments and reduces the partial outputs on the same device. This document does not call this
computation ContextParallel. ContextParallel is reserved for KV or context sharding across devices or ranks.

`SingleQ` and `TiledQ` describe the current concrete kernel families and Q tiling. These names are orthogonal to
SplitKV. The planner identifies the selected execution by its complete specialization. It does not expose a
SingleQ/TiledQ selector enum.

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

These equations describe tensor tiles inside one concrete kernel. They do not define `MapThreadBlockTask`,
`GQASDPAQTokenRange`, or `GQASDPAMapTaskTemplate`.

Only `model_layer_index` is per layer. Qwen validates the remaining fields once and uses them through the shared backend.

Qwen3 Main owns `Qwen3MainGQAState`, `UngatedGQA`, and `UngatedGQAScratch`.

Qwen3.5 owns `Qwen3xGQAState`. This state holds one shared gated `GQA` backend and one `GQAScratch` for compatible
invocations. It also holds one `Rc<GQARequestPageTable>` and reusable `GQAMetadataBuffers`.

Each `Qwen3xGQA` layer component retains clones of the backend, scratch, and page-table handles. It also owns its
weights and compact layer coordinate. Qwen3 Main uses its model-owned ungated leaf.

`Qwen35MainLayer` and `Qwen35MTPLayer` use the gated shared leaf in separate role-specific owners.

The gated and ungated executors share the lower-level norm/RoPE, KV update, SplitKV, and output-projection
components. Their projection, scratch, weights, and replay graphs remain concrete and separate.

This structure removes mode checks from the gated QGKV command sequence. It also makes the missing gate structural in
the ungated QKV path.

Qwen3 Main constructs `UngatedGQACore` and `UngatedGQA`.
Qwen3.5 Main and MTP construct `GQACore` and gated `GQA`.
Qwen3x DSpark constructs `UngatedDSparkGQACore` and `UngatedDSparkGQA`.
Its QKV attention graph is independent from Main and MTP.

Init-time component specialization supplies the head dimensions, head counts, RoPE constants, and page geometry. A
model-specific runtime branch does not supply these values.

The `RMSNormRope*` component operates on token-head rows. It does not own query grouping, KV grouping, or attention state.

`RopeScaling` selects unscaled default RoPE or Yarn RoPE.
Initialization resolves the correction range and the complete inverse-frequency table.
Yarn specialization blends extrapolated and interpolated inverse frequencies before Metal library compilation.
The RMSNorm/RoPE kernel reads the immutable inverse-frequency table for each rotary dimension.
It applies the configured Yarn attention factor to the rotated Q and K values.

`GQARequestPageTable` stores executor request-slot KV page IDs in a fixed-stride GPU buffer. It retains the IDs between
runtime reset and update notifications.

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

The runtime still owns physical page allocation and release.

Construction validates the complete flat table capacity. Entry updates and request-slot resets use direct flat-index
arithmetic after this proof. Debug bounds protect private invocation errors.

`GQAMetadataBuffers` stores the GPU arrays shared by every GQA layer in one model replay:

```text
req_slots[num_tokens]
flat_token_indices[num_tokens]
q_token_ranges[num_q_token_tiles][flat_q_token_begin/flat_q_token_end]  // TiledQ
sdpa_map_task_templates[num_total_sdpa_map_task_templates]
    [q_token_range_index/request_local_kv_token_begin/request_local_kv_token_end]
cu_sdpa_partial_outputs[num_tokens + 1]                                  // SingleQ
cu_sdpa_partial_outputs[num_q_token_tiles + 1]                           // TiledQ
```

`GQASDPAPlanner` validates cumulative-token monotonicity and request context ranges. It materializes the Q-token ranges,
Map task templates, cumulative partial-output offsets, replay extents, and plan metrics.
`GQAMetadataBuffers::update(...)` uploads that complete plan. The metadata owner does not select an execution or divide
KV work again.

Each three-`u32` `sdpa_map_task_templates` entry materializes one compact `SDPAMapTaskTemplate`. It contains a
Q-token-range index followed by the half-open request-local KV-token range. The shared replay shape retains the generic
TaskTemplate names because the DSpark composite contract can also reserve a block-bidirectional partial-output slot.

The grid supplies `kv_head_index` and `q_head_range_index`. These coordinates combine with the template to produce one
logical `MapThreadBlockTask`. One threadblock owns each task in a `1:1` relation. The buffer does not duplicate the
grid-derived coordinates.

The current one-Q specialization uses one-token Q ranges. The current tiled-Q specializations build request-local
Q-token ranges.

The planner assigns additional KV splits to the Q-token range with the most remaining KV-iteration work. KV splits for
one Q-token range are contiguous. This is the existing greedy allocation policy.

For a fixed Q-token/head output coordinate, adjacent `cu_sdpa_partial_outputs` values select the
`SDPAPartialOutput`s for the reducer.

`num_total_sdpa_map_task_templates` is the recorded replay capacity in the shared shape. The legacy exact-token
planner path retains its existing padded extent. The bucketed Qwen3.5 path uses the shared replay bucket policy.
Unused tail Map task templates contain an invalid Q-token-range index and do not write a map result.

The SplitKV `SingleQ` map also permits an invalid-Q-token-range `SDPAMapTaskTemplate` in one token's generic composite
range. This template does not write a history partial output for that slot.

A caller may populate the reserved max-logit, exp-sum, and normalized `SDPAPartialOutput` through
`GQABlockSDPAKernel`. It does this before it invokes the unchanged partial-output reducer.

This generic composition supports an attention connection that combines SplitKV history with a dense bidirectional
local block. The backend component does not own model-specific proposal or cache semantics.

Replay recording borrows `&GQAMetadataBuffers` directly. It does not use a duplicate bindings wrapper.

The CPU uses `cu_tokens`, per-request `req_slots`, and per-request starting `token_indices` to build these token-major
arrays. GQA kernels do not consume `cu_tokens`. Therefore, `GQAMetadataBuffers` does not retain a GPU copy.

The model-level GQA storage shape is:

```text
pages[num_cache_pages][page_bytes]

main_page_ids[num_req_slots][num_gqa_layers][num_blocks][num_page_ids_per_block]
optional_mtp_page_ids[num_req_slots][1][num_blocks][num_page_ids_per_block]
optional_dspark_page_ids[num_req_slots][num_dspark_layers][num_blocks][num_page_ids_per_block]

one KV page, viewed with the model KV dtype:
  [K/V][num_kv_heads][num_tokens_per_page][head_dim]
```

The Metal config stores `page_bytes` and the shared activation/KV dtype. It derives `num_tokens_per_page` from these
values, `num_kv_heads`, and `head_dim`. It does not store that derived value or a duplicate KV dtype.

For a flat token, the page lookup is exactly:

```text
req_slot -> gqa_layer_index -> block_index -> page_id_index -> page_id -> KV page
```

The main page-ID table uses a fixed compact main `gqa_layer_index`.
The optional MTP has an independent table with `K` GQA layer rows.
Runtime cache lane `step_index + 1` supplies the page IDs for MTP table row `step_index`.
`Qwen35MTP` performs this model-specific lane-to-row conversion through the generic page-table write API.

Qwen3 DSpark has a separate page table.
Runtime cache lane 0 contains one flat page-ID list for each block: `[Main IDs | DSpark IDs]` in DSpark mode.
The Qwen executor validates the exact combined length and splits this list once at the model prepare boundary.
Vanilla mode requires an empty DSpark remainder.
Both page tables keep their own layer and page-stride geometry.
The executor sends the same runtime request-slot reset notification to both tables before it reuses a slot.
Both tables use `GQARequestPageTable::reset_req_slots`.

`GQARequestPageTable` does not parse runtime cache lanes or model-role composition. It exposes symmetric per-entry
access:

```text
write_page_ids(req_slot, layer_index, block_index, page_ids)
read_page_ids(req_slot, layer_index, block_index)
```

Main and DSpark state owners accept one complete role-local block. They validate the exact role-local length and the
cache-page ID domain. They then write one table entry for each GQA layer.

The layout type uses generic `num_gqa_layers` for both tables. Each table instance owns its capacity and can use a
different GQA configuration.

The Qwen executor updates Main state once for each Main batch.
It updates optional MTP metadata once for the MTP stage.
It updates DSpark proposal metadata once for the DSpark Spec stage.

Main GQA layers borrow the Main state domain. The optional MTP owns its backend, scratch, and logical-lane page-ID table.

Layer owners retain immutable weights and a `gqa_layer_index`. A main layer does not retain model-level GQA
configuration or batch metadata.
The physical MTP layer binds its logical `gqa_layer_index` as a submission-time replay parameter.
MTP step `step_index` binds GQA layer row `step_index`.
Page preparation maps runtime cache lane `step_index + 1` to that row.
The step index does not enter the replay key.

## Backend specialization

The GQA benchmark uses these subcomponent names:

```text
qgkv-proj
split
q-norm-rope
k-norm-rope
kv-page-write
split-kv-single-q
split-kv-tiled-q
gate
output-proj
```

GQA owns KV page-table and cache interpretation inside the executor. The runtime core owns physical page allocation and
release. It also provides page IDs.

The replay SplitKV path reads the shared KV page arena through token metadata and the executor GPU page table. It does
not materialize a forward-local dense context window.

The path does not upload per-forward block tables before it launches the selected Metal attention kernels.

`SingleQ` and `TiledQ` map/reduce generate Metal source from the exact selected component geometry.
Immutable head, dtype, page, scale, and backend-selected specialization values become source constants.

Replay work determines the cached recorded variant. This work includes `num_tokens`, Q-token ranges, the total
KV-split extent, and the selected `max_q_heads` value.

SplitKV partial-output reduce also generates source for stable Q-head and head-dimension geometry. It keeps the active
token count as a replay argument.

The common kernel source-hash cache reuses identical generated pipelines. This specialization does not introduce
model-specific backend types or names.

Recording materializes the replay-specific `GQASplitKVSingleQKernels` or `GQASplitKVTiledQKernels`. The recorded invocation
retains its Metal pipelines. The GQA owner must not add a second pipeline cache.

SplitKV `SingleQ` exposes static geometry and tuning separately from dynamic replay work:

```text
GQASplitKVSingleQConfig              GQASplitKVSingleQShape
  num_q_heads                     num_total_tokens
  num_kv_heads                    num_total_sdpa_map_task_templates
  head_dim
  scale
  page_bytes
  page_table_layout
  kv_tokens_per_iteration
  required_threads
  max_q_heads
  dtype
```

`num_total_sdpa_map_task_templates` is the shared shape field for the padded KV-split extent. It is not the number of
KV iterations.

One KV split can cover several consecutive KV iterations. The replay bucket policy rounds up `num_kv_splits` to
produce the total replay dispatch and scratch extent. The replay field retains its established name.

The backend configuration is model-independent.
`model/qwen/v3/main/plan.rs` builds the Qwen3 Main ungated core.
Each `model/qwen/v3_x/dspark/attention.rs` layer derives its ungated core and Metal configuration from the normalized
DSpark config and its exact attention binding subtree.
`model/qwen/v3_5/plan.rs` builds gated Main and MTP cores.

Each DSpark layer owns its weight-dependent `UngatedDSparkGQA` and `UngatedDSparkGQAContextAppender`.
`UngatedDSparkGQAState` owns the shared page table, metadata, scratch, and SplitKV SingleQ history contract.
Quantization layout is not part of the shared-state compatibility contract.
The state receives static attention and KV-cache facts through `GQASDPAConfig`.
`GQASDPASpecializationRegistry::new_single_q_only(...)` provides the one legal history specialization. The DSpark metadata builder
keeps its component-specific history and block-partial composition.

Each concrete backend converts its core and `GQAMetalConfig` into a projection split. `GQAMetalConfig` contains only
model, quantization, storage, and RoPE facts. The backend derives Metal SDPA tuning from `head_dim`. It then constructs
the shared norm/RoPE, KV-update, SplitKV, and output-projection components.

The Qwen GQA weight owner loads one bounded `TensorMap` from its exact GQA binding subtree.
It retains the core and `GQAMetalConfig` values that created the backend.
Weight reload uses these retained values.
It removes Q/K/V, norm, and output tensors from that map.
It then materializes the fused QGKV or QKV buffers required by the selected backend ABI.
The map must be empty after construction.

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

KV page write uses the same model KV dtype as projection scratch and SplitKV. The component derives its page stride
and `num_tokens_per_page` with that dtype.

The Metal component selects the matching bf16/f32 update kernel. `GQAKVPageWriteConfig` owns the stable
`num_kv_heads`, `head_dim`, `page_bytes`, dtype, and derived tokens-per-page.

`GQAKVPageWriteKernelSpecialization` contains this compile-time config and a thread-block specialization that requires
256 threads. One thread owns one flattened `(token write, KV head, head-dimension)` value. The dispatch grid and active
token-write count remain invocation data.

`num_token_writes`, `gqa_layer_index`, and page-table coordinates remain invocation data.
The Metal replay core provides symmetric fixed-or-parameter scalar sources for `u32`, `u64`, `i32`, `i64`, and `f32`.
`ReplayArguments`, `CommandParameterLayoutBuilder`, and `CommandRecorder` support the same scalar set. GQA uses
`ReplayU32::Fixed(value)` or `ReplayU32::Parameter(key)` through the same kernel and invocation path. The GQA component
does not define a replay-indexed constructor, kernel variant, or model-specific flag.

The replay shape separates fixed page-table layout from active work and recorded capacity. It retains the established
Q-token-tile identifiers for replay ABI stability. These identifiers represent Q-token ranges:

```text
num_tokens                         active flat Q tokens in the microbatch
num_total_tokens                   recorded flat-Q-token capacity
num_q_token_tiles                  active request-local Q-token ranges
num_total_q_token_tiles            recorded Q-token-range capacity
num_sdpa_map_task_templates        active SDPA map TaskTemplates
num_total_sdpa_map_task_templates  recorded SDPA map TaskTemplate capacity
reduce_sdpa_partial_outputs        whether the active batch plan semantically requires partial reduction
```

For `SingleQ`, `num_q_token_tiles` equals `num_tokens`. The variant does not consume the active-Q-token-range replay
parameter.

`SingleQ` replay always records the reduce command. It records this command even when each token has only one map
KV split.

This rule lets both SplitKV variants share one recorded program for the same Q-token-range and KV-split geometry. The
flag remains batch-plan metadata and does not enter the replay key.

`GQASDPAPlanner::plan_exact(...)` keeps the legacy exact token and Q-token-range extents. It retains the existing padded
KV-split extent. `GQASDPAPlanner::plan_bucketed(...)` applies the shared capacity policy independently to tokens,
Q-token ranges, and KV splits. `GQAMetadataBuffers::update(...)` stores the selected execution and replay shape from the
complete plan.

`GQASDPAPlanner::plan_bucketed_with_token_capacity(...)` accepts a token capacity from a composite replay stage. GQA
does not apply its token bucket policy again on this path. GQA still selects Q-token-range and KV-split capacities with
its private policies.

The caller-owned token capacity must satisfy `num_tokens <= num_total_tokens <= max_tokens`. It must also preserve the
QGKV and output affine topologies selected for `num_tokens`. GQA validates these topologies during preparation and
recording. The recording check prevents a direct metadata update from bypassing the topology contract.

The GQA token policy includes topology boundaries from both affine projections. The Q-token-range and KV-split
policies use the shared default buckets. `GQA::replay_token_topology_boundaries()` exposes the union to composite
stage policies. `GQASDPAExecutionSpecialization` remains an explicit topology identity in the replay key.

The bucketed kernels consume these submission values:

```text
gqa.num_active_tokens                    qgkv/output affine, KV write, and attention token guards
gqa.num_active_q_token_tiles             TiledQ map and reduce guards
gqa.num_active_kv_splits                  SplitKV map guard
```

Projection split, norm/RoPE, KV write, activation gate, and the qgkv/output affine kernels return before an inactive
token reads input or metadata, mutates a page, or writes output. SplitKV SingleQ and TiledQ also return before inactive
KV splits or Q-token ranges read their metadata. All token-domain commands use the same active-token parameter key
and range.

The default bucketed API uses `gqa.num_active_tokens`. A composite stage can supply a stage-owned
`ReplayParameterKey` through `GQAReplayMode::BucketedWithTokenKey(...)` and `Qwen3xGQA::record_bucketed(...)`.
The supplied key must differ from the private Q-token-range and KV-split keys.

`add_gqa_replay_arguments(...)` supplies all default GQA arguments. A composite stage supplies its active-token
argument once. It then calls `add_gqa_private_replay_arguments(...)` for the Q-token-range and KV-split arguments.

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

For Qwen3.5 model replay, the Qwen executor validates runtime cache lane 0 and writes the Main page table. DSpark mode
also writes the independent DSpark page table. `Qwen35MTP` separately maps runtime cache lane `step_index + 1` to MTP
GQA layer row `step_index`. `GQA::prepare(...)` selects the SDPA execution and builds the complete batch plan once.
Every GQA layer reuses this plan.

`SingleQ` replay always records partial-output reduction. This rule also applies when each batch token has one
KV split.

The shared `Qwen35GQAReplayKey` contains the three recorded capacities. It also contains the complete
`GQASDPAExecutionSpecialization`, qgkv affine topology, and output affine topology. Active counts remain submission
values and do not enter this GQA subkey.

`Qwen35MainReplayKey` and `Qwen35MTPReplayKey` use this shared GQA subkey. Qwen3.5 Main selects one composite token
capacity and forces GQA metadata to use it. All Main token-row commands use the caller-owned Main active-token key.
The Q-token-range and KV-split counts remain private GQA replay dimensions. The Main key also contains the
non-optional GDN request-count subkey. Qwen3.5 MTP independently selects one composite body token capacity and forces
its GQA metadata to use it. All MTP body token-row commands use the caller-owned MTP active-token key. MTP does not
declare the component-local GQA active-token parameter.

MTP keeps its separate GQA and MLP composite key.
All MTP steps in one batch have the same token and attention shape, so they reuse one recorded program.
The replay argument selects the logical MTP GQA layer at execution time.
Main recording supplies each physical layer's fixed index through the same kernel ABI.

Qwen3 uses its separate ungated GQA implementation. DSpark keeps its existing exact replay key and submission ABI.
DSpark only uses the expanded shared replay-shape fields with equal active and total token/range values and its existing
TaskTemplate padding. DSpark retains this generic name because its composite map includes one block partial-output
slot.

### Execution strategy

See [GQA SDPA planning](gqa_sdpa_planner.md) for the reference execution, specialization, task, and planner hierarchy.

`GQASDPAConfig` contains static workload facts: I/O dtype, Q-head count, KV-head count, head dimension, and tokens per
KV page. `GQASDPASpecializationRegistry::new(...)` derives legal `GQASDPAExecutionSpecialization` values. Each value
contains one compatible Map and Reduce specialization. `supports(...)` checks only static capability and correctness
conditions.

`GQASDPAPlanner` owns the dynamic workload and replay capacity. It creates request-local Q-token ranges. It applies the
current greedy KV-segment allocation separately to each candidate specialization. It then computes complete plan
metrics and applies the current measured selection policy. The selected `GQASDPAPlan` contains the execution identity,
the materialized Q-token ranges and Map task templates, cumulative partial-output offsets, replay shape, and metrics.

GQA retains this planner and plan because these values form one coupled dynamic decision. A pure selector can return an
execution specialization, but it cannot also preserve the exact request-local work partition and replay extents that
were priced for that specialization. The plan is the atomic boundary between dynamic selection and metadata upload.

`GQAMetadataBuffers::update(...)` uploads the selected plan. Recording executes the stored specialization and does not
select again. Both current concrete kernel families partition a long visible KV range into independent KV segments.

DSpark uses `GQASDPASpecializationRegistry::new_single_q_only(...)`. Its SplitKV history map and block-bidirectional map
must produce the same one-Q partial ABI for one shared Reduce. DSpark keeps its component-specific metadata and
segmentation logic. It does not use the general GQA request planner.

The current concrete kernel families differ in the number of Q tokens and Q heads that one Map threadblock computes:

```text
shape: num_tokens grouped into request-local Q-token ranges
parallelism: KV split x KV head x Q-head range
input: normalized Q plus paged K/V selected through the request page table
output: partial attention states for one KV range, followed by one numerically stable reduce

SingleQ    one Q token per Q-token range; scalar dot/reduction work
TiledQ    several Q tokens/Q heads per range; SIMD-group matrix work
```

Both variants use GQA head sharing. If `G = Hq / Hkv`, KV head `k` supplies K/V to Q heads
`[k * G, (k + 1) * G)`.

A buffered Map task template supplies the Q-token-range index and half-open request-local KV-token range. The grid
supplies the regular head coordinates:

```text
Map task template                                grid coordinates
[q_token_range, kv_begin, kv_end)           +   [kv_head, q_head_range]
                    \                                /
                     +------ one Map threadblock

visible KV range [0, N)
  -> one or more KV splits
  -> independent partial output + max + exponential sum
  -> reduce to final [Q token, Q head, D]
```

`cu_sdpa_partial_outputs` selects the consecutive partials for each Q-token range. Padded replay Map task templates use
`q_token_range_index = u32::MAX`. Their threadblocks return without writing.

The registry always includes the current one-Q specialization. It includes tiled-Q specializations only for the
supported static shape. The explicit bf16 production profiles are
`(D=128, 8 KV tokens/page)`, `(D=256, 8 KV tokens/page)`, and `(D=256, 16 KV tokens/page)`.

All profiles support at most 8 Q heads per KV head. The gated backend currently reaches only the `D=256` profiles.

For the `(D=128, 8 KV tokens/page)` and `(D=256, 16 KV tokens/page)` profiles, the planner uses the average useful
tokens per request-local Q range. It does not use floating-point division:

```text
num_tokens < 2 * num_q_token_ranges       -> SingleQ
D=128 profile                             -> TiledQ, full Q/KV group
D=256 and num_tokens < 4 * num_q_token_ranges
                                          -> TiledQ, roughly half the Q/KV group
otherwise                                 -> TiledQ, full Q/KV group
```

The gated `(D=256, 8 KV tokens/page)` profile uses an additional measured policy. This policy applies only to complete
plans for legal backend specializations. It uses each request's Q-token count and history length. Each candidate plan
already contains the current greedy KV-split allocation at the metadata capacity. The policy does not change that
allocation.

The policy derives scheduled and active QK token pairs, Map threadblock and SIMDgroup-wave counts, active partial
states, and padded replay partial-state groups. It selects `TiledQ` only when active QK token pairs use at least half of
the scheduled QK token pairs. It also requires the existing selection-score crossover. This rule keeps `T1`, `T2`,
small `T25` workloads, and long one-token request tails on `SingleQ`. It selects `TiledQ` for the measured long-context
`T4` and larger workloads. The complete execution specialization remains part of the existing replay topology key. The
policy does not add a replay key.

`TiledQ` requires at most 256 threads. Its `max_q_tokens` and `max_q_heads` values determine this requirement. Current
reachable model variants are:

| Model | `Hq / Hkv / D` | KV tokens/page | Production variant |
| --- | --- | ---: | --- |
| Qwen3-14B | `40 / 8 / 128` | 8 | planner policy above, tiled `max_q_heads=5` |
| Qwen3.6/Qwen3.8-27B | `24 / 4 / 256` | 8 | measured planner policy above |
| Qwen3.6-35B-A3B | `16 / 2 / 256` | 16 | planner policy above |
| Qwen3 DSpark | checkpoint-derived | model-derived | SplitKV SingleQ history + block bidirectional |

For 35B, `TiledQ` uses `max_q_heads=4` below four useful tokens per Q-token range. It uses `max_q_heads=8` otherwise.

#### `SingleQ`

`SingleQ` always uses `max_q_tokens=1`. Qwen3-14B uses `kv_tokens_per_iteration=128`, `required_threads=128`, and
`max_q_heads=5`.

The Qwen3.5 profiles use `kv_tokens_per_iteration=256`, `required_threads=256`, and their model-derived
`max_q_heads` value:

```text
one block owns
  one Q token
  one KV head
  up to max_q_heads Q heads sharing that KV head
  one KV split's [kv_begin, kv_end) segment

Q[q_token, max_q_heads, D]             paged K/V
          |                                |
          +-- each thread scores K token --+
                           |
             threadgroup logits[max_q_heads, kv_tokens_per_iteration]
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
Map threadblock.

Its `logits[5, 128]` and `reduce_scratch[128]` use 3 KiB of threadgroup memory. A `max_q_heads` value of four requires
separate `4+1` head ranges.

A thread count of 64 doubles the output accumulators that each thread holds.

The Qwen3.5 `D=256` profiles use one active thread per output dimension. The 27B profile uses 7 KiB
(`max_q_heads=6`). The 35B profile uses 9 KiB (`max_q_heads=8`).

The kernels stream K/V from global memory instead of staging them as threadgroup tiles. Running statistics and owned
output dimensions are MSL thread-local values.

#### `TiledQ`

The common tiled specialization and its internal tensor regions are:

```text
max_q_tokens:              8
max_q_heads:               model- and workload-derived
kv_tokens_per_iteration:   16
grid:                      (Hkv * ceil((Hq/Hkv) / max_q_heads), TaskTemplates, 1)
required_threads:          (max_q_tokens / 8) * max_q_heads * 32
Q tensor region:           [up to max_q_tokens, max_q_heads, D]
K/V tensor region:         [kv_tokens_per_iteration, D] for one KV head

Qwen3-14B max_q_heads=5  -> 160 threads = 5 SIMD-groups
max_q_heads=4           -> 128 threads = 4 SIMD-groups
max_q_heads=8           -> 256 threads = 8 SIMD-groups
```

One 32-lane SIMD-group owns one Q head and one eight-token fragment. Its lanes collectively hold the Q rows in MSL
thread-local `q_fragments`.

Qwen3-14B has 16 dimension fragments per thread. The `D=256` profiles have 32. An incomplete request tail loads only
active rows.

```text
thread-local Q fragments stay resident for the KV split
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
                    the next KV iteration
                              |
                              v
partial output[up to max_q_tokens, max_q_heads, D] + statistics
```

For each K or V row, lanes load contiguous 16-byte segments. The two threadgroup tiles occupy 8.5 KiB for Qwen3-14B
(`2 * 16 * 136 * sizeof(bf16)`).

The tiles occupy 16.5 KiB for `D=256` (`2 * 16 * 264 * sizeof(bf16)`). Q, scores, running statistics, and output
fragments are MSL thread-local.

The current kernel has one K workspace and one V workspace. It does not double-buffer consecutive K/V tiles.

Both variants reduce partial attention states by rescaling them to one global maximum. `SingleQ` reduces flat
`[token, Q head, D]` elements.

`TiledQ` launches one Reduce block per `(Q head, Q-token range)`. The block strides over active `token x D` elements.

`SingleQ` records reduce even for one KV split per token. This rule keeps the replay topology stable.
`TiledQ` always records its tiled reduce.

Focused fixed, request-tail, multi-range, and ragged cases compare both variants with the CPU reference.

Qwen3.5 keeps reusable gated-GQA scratch in the directly owned `Qwen3xGQAState`. The `GQA` backend creates this scratch
from its planner limits and maximum registered Q-range width. Individual GQA layers do not own this scratch.

The executor owns one Main `GQAScratch`. The optional MTP owns one matching scratch because its GQA configuration can
differ. All logical MTP steps reuse this one MTP scratch owner.

This scratch contains the buffers for QGKV projection, the projected gate, gated output, norm/RoPE, and SplitKV. The fixed
gated graph requires these buffers.

Qwen3 Main owns `UngatedGQAScratch`. The `UngatedGQA` backend creates this scratch from the same planner limits. The
scratch contains the QKV projection, norm/RoPE, and SDPA buffers for the fixed ungated graph. It has no gate buffers.

Both scratch types expose matching borrowed replay bindings. The model stream serializes Main and MTP execution.
Therefore, submissions reuse their buffers without per-layer allocation.

Qwen3 DSpark owns `DSparkBlockScratch`.
This scratch contains proposal-local Q/K/V, SplitKV history partials, and block-bidirectional partials.
Its partial capacity is `next_power_of_two(2 * max_requests * num_spec_tokens)`.
It does not depend on `max_position_embeddings`.
`DSparkGQACapacity` owns this Metal resource rule.
The backend-neutral `DSparkBlockCapacity` contains only request and block geometry.

The bound for SDPA partial scratch is `max_tokens * maximum registered max_q_tokens * num_q_heads`.
It is independent of `max_position_embeddings`.

`GQAMetadataBuffers` owns the matching submission metadata. The owner receives its capacity once and updates its data
for each submission. Its buffers are read-only during a recorded GQA layer forward.

The buffer contract is:

```text
hidden_state / next_hidden_state     bf16 model boundary buffers shaped [num_tokens, hidden_dim]
req_slots                            request slot repeated per flat token
flat_token_indices                   request-absolute token index per flat token; used for RoPE, KV write address, and causal context length
q_token_ranges                      request-local flat-Q-token ranges consumed by SplitKV TiledQ
sdpa_map_task_templates             materialized Q-token-range index and request-local KV-token range for Map tasks
cu_sdpa_partial_outputs             cumulative partial-output counts selected per Q-token range by SplitKV Reduce
page_ids                             fixed-stride [req_slot, gqa_layer_index, block_index, page_id_index]
kv_pages                             shared runtime-provided KV arena backing
scratch                              caller-owned capacity buffers, used only up to current replay shape
weights                              immutable fused projection, q/k norm, and output projection buffers
```

Q and K norm weights keep the checkpoint BF16 storage type. The norm/RoPE kernel reads them directly. It preserves the
configured activation-type arithmetic and rounding order. RMS reduction and RoPE trigonometry use F32.
Default and Yarn RoPE use the same component and replay boundary.

The recording marks the KV arena as both a write and read resource. KV update writes the current tokens. SplitKV
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
  -> SplitKV map reads visible KV pages through the selected block lowering
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
  -> SplitKV map reads visible KV pages through the selected block lowering
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

KV page writes occur before SplitKV. Writes and reads use the same page-table interpretation. For each write token, the KV
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

SplitKV has map and reduce stages:

```text
map(map_task_template_index, kv_head, q_head_range)
  reads one Q-token range
  derives one MapThreadBlockTask from the template, grid coordinates, and specialization
  walks the task's KV-token range in fixed-size KV iterations
  merges iteration partials with online softmax
  resolves each KV token through page IDs + KV page arena
  writes partial_max_logits, partial_exp_sums, and SDPAPartialOutput

reduce(flat_token, q_head)
  uses cu_sdpa_partial_outputs to read that token's partial outputs
  combines stable online-softmax partials
  writes the final attention output token
```

Reduce uses per-partial-output max logits to combine exp sums and weighted outputs. It does not materialize a dense
context window.

`SingleQ` replay records this reduce even when each token has one KV split. This rule keeps the replay topology
stable.

Qwen3-14B `SingleQ` uses `kv_tokens_per_iteration=128`, `required_threads=128`, and `max_q_heads=5`.

Qwen3.5 retains `kv_tokens_per_iteration=256`, `required_threads=256`, and `max_q_heads <= 8`.

The layout groups Q heads by KV head. Each `MapThreadBlockTask` handles one KV head and one Q-head range.

The common resource dependencies are explicit:

```text
q/k norm+RoPE reads q/k scratch and flat_token_indices, writes normalized q/k scratch
KV update reads k/v scratch + flat token metadata + page IDs, writes KV pages
SplitKV map reads q scratch + KV pages + sdpa_map_task_templates + page IDs, writes SDPA scratch
SplitKV reduce reads SDPA scratch + cu_sdpa_partial_outputs, writes attention output
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

`gqa_split_kv_single_q_test.rs` compares SplitKV SingleQ with the CPU projected-GQA reference. It uses fixed input,
random input, and a
random ragged batch.

Another case uses one KV split that spans multiple KV iterations. The cases validate compact KV-split indexing,
online-softmax iteration merging, request slots, page-table lookup, and causal visibility.

`gqa_split_kv_tiled_q_test.rs` compares the BF16 SplitKV TiledQ map and reduce variant with the same CPU reference. One
bucketed replay
executes `5 -> 8 -> 5` active tokens. The test poisons inactive query, KV, request-slot, and token-index inputs. It checks
the active output and verifies that inactive partial-output, statistic, and final-output tails remain unchanged.

Metal backend component replay sanity lives in:

```text
cargo bench -p inference-backend-metal --bench gqa_split_kv -- --profile-time 1 --noplot

cargo bench -p inference-backend-metal --bench gqa_block_attn -- \
  --block-sizes 7 --num-requests 1,4 \
  --iters 1 --warmup-iters 0 --runs 1
```

The GQA backend bench records SplitKV SingleQ building blocks only in Metal replay/ICB paths. GQA Metal code does not
benchmark or expose direct-submit component or forward wiring.

`gqa_block_attn` records only `GQABlockSDPAKernel`.
It measures the dense block-bidirectional map contribution used by DSpark.
It does not measure history attention, partial reduction, projections, or a DSpark layer.

One block-SDPA Task owns one Q token and one Q head.
The backend uses one 32-thread SIMDgroup for the Task.
`GQABlockSDPAKernelSpecialization` contains the stable SDPA config and its thread-block specialization. The
specialization requires 32 threads and a 32-thread SIMDgroup. The generated source no longer declares an unused
`num_threads_per_threadblock` constant.
For `head_dim=128`, each thread keeps four F32 Q values and one dot-product accumulator.
The logical Q register payload is 16 bytes per thread.
The SIMDgroup computes each Q/K dot product across the head dimension.
It then computes four output dimensions per thread.

For the seven-token DSpark block, the kernel uses 28 bytes of static shared memory for seven F32 logits.
It does not allocate a shared reduction array.
Kernel construction checks the pipeline SIMD width, the pipeline thread limit, and the device shared-memory limit.
Metal does not expose the compiler register allocation.
The backend therefore uses the source-level live-value count and the production-shape benchmark to validate register
pressure.

Metal backend real full-forward replay bench lives in:

```text
cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 0,128,1024 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen3_dspark -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --cases dspark --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-split-kv-variants single_q \
  --iters 1 --warmup-iters 0 --runs 1
```

These benches use CLI arguments instead of environment variables.

`qwen3_dspark` runs the public Main and Spec executor hooks.
It measures the complete DSpark graph.
Use it for DSpark composition and lifecycle costs.
Do not compare its timing directly with `gqa_block_attn`.

`qwen3_gqa` loads the Qwen3 model config and first-layer ungated weights. It accepts explicit per-request token counts
and context lengths. It reports full-replay and SplitKV-only measurements.
It also reports exact QKV and output-projection measurements for QMV, QMM BM8/BN32, and QMM BM16/BN32.
These forced projection paths are benchmark-only.
Production GQA continues to use `AffineQuantizedMatmul` selection from the complete shape and dtype.

Its SplitKV SingleQ and TiledQ specialization arguments are configurable. `--validate` compares full SingleQ and
TiledQ outputs for a workload where production selects the TiledQ variant.

Use `--split-kv-single-q-kv-tokens-per-iteration`, `--split-kv-single-q-required-threads`, and
`--split-kv-single-q-max-q-heads` to configure SingleQ. Use `--split-kv-tiled-q-max-q-tokens`,
`--split-kv-tiled-q-kv-tokens-per-iteration`, and `--split-kv-tiled-q-max-q-heads` to configure TiledQ.

The validation also prints the derived threadgroup/register shape.

For `qwen35_gqa`, `--gqa-model 27b|35b` selects the real-weight layer profile. Pass the matching model directory with
`--model-dir`.

The bench uses the production 32 KiB physical page size. It derives the KV tokens per page from the selected model
profile and the bf16 K/V element size. The 27B profile uses 8 tokens per page. The 35B profile uses 16 tokens per page.

For GQA, `--tokens` is the total current flat-token count. `--num-reqs` is the number of request segments in that
microbatch. `--contexts` is the existing context length for each request before its measured tokens.

The bench distributes tokens as evenly as possible across requests. It builds `req_slots`, `flat_token_indices`, and a
fixed-stride request page table from these options.

Recommendation: For a single-request decode-style context sweep, use `--tokens 1 --num-reqs 1` and vary `--contexts`.

For a multi-request decode batch, you may use `--tokens 8 --num-reqs 8`.

For a prefill/suffix sweep, you may use `--tokens 64 --num-reqs 1 --contexts 0,2048,4096`.

Without an explicit context list, the bench uses existing context length zero. `--gqa-tokens-per-req` supplies explicit
ragged per-request token counts.

`--gqa-contexts-per-req` supplies the matching existing context length for each ragged request. It requires
`--gqa-tokens-per-req` and cannot be combined with `--contexts`.

`--max-tokens` fixes the token capacity, Map task-template capacity, and maximum active partial-state-group count for
both forced SplitKV candidates. The default is 128, which matches the server default. Each case reports Q-token
ranges, Map task templates, active and reserved partial-state groups, replay slots, and KV iterations per Map
task template.

The comparison replay reports `split_kv_variant=single_q` or `split_kv_variant=tiled_q`. Model execution uses the
automatic planner policy described above.

`--gqa-split-kv-single-q-kv-tokens-per-iteration`, `--gqa-split-kv-single-q-required-threads`, and
`--gqa-split-kv-single-q-max-q-heads` override the `SingleQ` defaults.

`--gqa-split-kv-tiled-q-max-q-tokens`, `--gqa-split-kv-tiled-q-kv-tokens-per-iteration`, and
`--gqa-split-kv-tiled-q-max-q-heads` configure the `TiledQ` comparison variant.

When the Q-head override is absent, the bench uses the production half/full Q/KV-group rule. The benchmark CLI and
output use the production specialization terms `max_q_tokens`, `kv_tokens_per_iteration`, `max_q_heads`, and
`required_threads`.

`--print-limits` prints the device threadblock-memory limit. It also prints the derived `SingleQ`
threadblock-memory footprint.

The current backend records explicit data-dependency barriers. The replay layer also infers hazards from declared
buffer usage. It does not add a conservative every-command fallback.

This bench loads real Qwen3.6 layer weights. It measures the full replay path: qgkv projection, projection split, q/k
norm+RoPE, KV page write, SplitKV, activation gate, and output projection.

Do not compare component-only SplitKV timings with full-forward numbers.

Subcomponent probes use the same request-slot/page-table capacity contract as full-forward replay. Multi-request
`kv-page-write` probes must pass the true `num_req_slots` through the page-table layout in `GQAKVPageWriteShape`.

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

During `unload_state`, each layer first drops its shared GQA backend, scratch, and request-page-table references.
The model state owner then releases the final references and its batch metadata.
State load rebuilds these transient resources before it restores the full `GQARequestPageTable` payload.
Each GQA state owner implements `FullStateIO` with `GQAStateSnapshotFiles`. `read_full_state` requires mutable state and
a unique `Rc<GQARequestPageTable>`. This contract prevents restore from overwriting a page table that is still attached
to an execution graph.

The core `scale` is part of both attention contracts. The executor passes it to SplitKV kernels. Kernels must not
silently substitute `1 / sqrt(head_dim)`.

Q and K norm/RoPE use separately specialized commands because their stable head counts differ.

Use typed indices for typed buffer offsets. Use byte offsets for raw Metal buffer bindings.

Use 64-bit Metal address math when page or stride arithmetic multiplies large strides. Resource usage marks must cover
KV page writes and reads exactly.

Record dynamic threadblock memory in both direct and ICB paths. Replay-cache shape keys must distinguish fixed
page-table stride, execution specialization, and replay extents.

Debug in this order:

1. Component primitive
2. Real-weight GQA wrapper
3. Attention slice in a layer
4. Layer ladder

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
