# Qwen Executor

This document describes the current Qwen3.5/Qwen3.6 Metal executor from normalized checkpoint configuration through
top-down component loading, state preparation, cached replay, sampling, and the supported single-module MTP path.

## Source layout

```text
crates/inference-executor-core/src/model/qwen/v3_5/
  batch.rs                  runtime batch normalization and sampled-decision contracts
  config.rs                 Qwen35ModelConfig parsing and one-time normalization
  dspark_config.rs          retained low-level DSpark configuration contract
  dspark_weight_layout.rs   retained exact DSpark tensor manifest
  pending_transactions.rs   sequence-ordered pending executor transactions
  weight_layout.rs          exact typed Main/unembed/MTP binding trees

crates/inference-executor-metal/src/
  replay.rs                 generic Replay<T> component/cache owner
  model/
    embed_unembed.rs        generic Embed and Unembed Metal owners
    gather.rs               generic row gather owner
    page_arena.rs           shared physical page buffer
    residual.rs             generic residual owner
    rms_norm.rs             generic RMS-normalization owner
    qwen/v3_5/
      executor/
        mod.rs              Qwen35Executor fields and ReplayableModelBatchExecutor integration
        load.rs             normalized config, aggregate count pass, and top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and common replay submission
        main.rs             MainEmbed, Main, and GatherUnembed orchestration
        sampling.rs         normal/draft/target/rejection orchestration and readback
        mtp.rs              MTP request, proposal-batch, and proposal flow
      layer/
        mod.rs              Qwen35Layer composition and variant selection
        scratch.rs          shared layer workspace
        gqa.rs              Qwen35GQA, private weights, load, and record
        gdn.rs              Qwen35GDN, private weights, load, and record
        dense_mlp.rs        Qwen35DenseMLP, private weights, load, and record
        moe.rs              Qwen35MoE, private weights, load, and record
      state/
        gqa.rs              Qwen35GQAState page/metadata/reset lifecycle
        gdn.rs              Qwen35GDNState prepare/restore/commit/publish/reset lifecycle
      model.rs              Qwen35MainEmbed, Qwen35Main, and Qwen35GatherUnembed
      mtp.rs                Qwen35MTPEmbed and the supported one-layer Qwen35MTP
      rejection_sampling.rs Qwen-specific rejection composition and result preparation
      plan.rs               direct geometry helpers and retained low-level DSpark plan
      weight.rs             shared Qwen checkpoint decoding/validation helpers only

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs         TopKSampling and TopKSamplingOutputBuffers
  top_k_replay.rs           Sampling/DraftSampling replay components
  rejection_sampling.rs     generic sparse rejection Metal owner
  spec_probs.rs             SpecProbsStore sparse draft/target probability workspace
```

Runtime core owns scheduling, request lifecycle, physical page allocation/free, and page IDs. The executor owns
model-specific interpretation of those IDs, trained tensors, backend state, replay caches, and submission ordering.
Metal kernels remain backend components.

## Semantic object tree

```text
Qwen35Executor
  main_gqa_state: Qwen35GQAState
  main_gdn_state: Qwen35GDNState
  mtp_gqa_state: Option<Qwen35GQAState>
  main_embed: Replay<Qwen35MainEmbed>
  main: Replay<Qwen35Main>
  gather_unembed: Replay<Qwen35GatherUnembed>
  sampling: Replay<Sampling>
  mtp_embed: Option<Replay<Qwen35MTPEmbed>>
  mtp: Option<Replay<Qwen35MTP>>
  draft_sampling: Replay<DraftSampling>
  rejection_sampling: Replay<RejectionSampling>
  pages: PageArena
```

Semantic components own weights, static configuration, and `load + record`. `Replay<T>` owns the corresponding replay
cache. `Qwen35Executor` owns dynamic workspaces, lifecycle ordering, and submissions.

`Qwen35Layer` owns two generic norms, residual composition, one selected attention owner, one selected MLP owner, and a
shared `Rc<Qwen35LayerScratch>`. `Qwen35GQA` and `Qwen35GDN` store compact per-kind layer indices, not model-layer
indices, for page-table and state-arena addressing.

## Configuration, bindings, and load

The checkpoint schema is `Qwen35ModelConfig`; runtime capacities are `Qwen35ExecutorConfig`. Configuration is normalized
once, then exact typed binding trees are resolved before real tensor reads:

```text
Qwen35ModelWeightBindings
  embed: QuantizedTensorBindings
  main:
    final_norm_weight
    layers: Vec<Qwen35LayerWeightBindings>
  unembed: QuantizedTensorBindings

Qwen35MTPWeightBindings
  embed:
    prev_hidden_norm_weight
    token_hidden_norm_weight
    projection: QuantizedTensorBindings
  body: Qwen35LayerWeightBindings
  final_norm_weight
```

Initialization is top-down:

1. Normalize Main and optional MTP configurations and validate the supported checkpoint contract.
2. Resolve exact binding trees.
3. Count Main GQA/GDN layers and Dense/MoE scratch requirements.
4. Construct Main GQA/GDN state domains and optional MTP GQA state before models that clone their handles.
5. Construct shared scratch and the shared token `Embed`.
6. Move each exact binding subtree to its semantic owner. Each owner reads and validates its own real weights.
7. Construct Main, GatherUnembed, optional MTPEmbed/MTP, sampling owners, workspaces, and `PageArena`.
8. Wrap every cached stage in `Replay<T>`.

There is no Main/MTP plan object tree or aggregate component-weight owner. `plan.rs` retains only direct geometry
conversion and the existing low-level DSpark plan.

## Replay ownership

`Replay<T>::record(runtime, input)` derives the component key, returns immediately on a hit, and records/builds/inserts
exactly once on a miss. It returns `(key, cache_hit)`. `Replay<T>::replay(key)` is a strict lookup and panics if record
did not establish the key. `Replay<T>` exposes `component()` explicitly and does not implement `Deref`.

The independent cached graphs are:

```text
Replay<Qwen35MainEmbed>      token embedding
Replay<Qwen35Main>           all Main layers -> final norm
Replay<Qwen35GatherUnembed>  gather -> unembed
Replay<Sampling>             ordinary target sampling
Replay<Qwen35MTPEmbed>       previous-hidden gather + token embed + input projection
Replay<Qwen35MTP>            one GQA body layer -> final norm
Replay<DraftSampling>        draft sampling + sparse draft distribution
Replay<RejectionSampling>    target sparse distribution + rejection
Replay<Rc<GDNRequestStateTable>>
                              snapshot restore into live GDN state
```

MainEmbed and MTPEmbed are the only newly separated replay boundaries. Existing key fields, cache behavior, dynamic
arguments, and composed submission order are preserved.

## Main data flow and workspace ownership

```text
token_ids
  -> MainEmbed
token_hidden_input
  -> Main layers using Qwen35LayerScratch::residual_stream[2] ping-pong
hidden_output
  -> GatherUnembed(row_indices)
unembed_hidden
  -> unembed
unembed_logits
  -> Sampling or RejectionSampling
```

`token_hidden_input` is the embedding destination and layer-0 input. `hidden_output` is the final RMSNorm destination.
They are executor-owned `Rc<Buffer>` workspace slots and are passed across runtime stages without `Option`, hidden
handles, or hidden-source enums. The final layer residual is only the local current ping-pong buffer; there is no
`final_residual` field, accessor, or allocation.

`Qwen35GatherUnembedArgs` is flat: it binds the final-normalized hidden source, row indices, gather destination, and
logits destination. Gathered hidden and logits remain executor workspaces.

## GQA/GDN lifecycle

`Qwen35GQAState` groups a backend, scratch, request page table, metadata buffers, and cache-lane information. Main and
MTP own distinct GQA state domains. Its lifecycle calls are:

```text
prepare_pages(core_batch)
prepare_metadata(stage_microbatch)
reset_req_slots(runtime_notification)
```

`Qwen35GDNState` groups a backend, scratch, request state table, metadata, cached restore replay, and one optional
asynchronous publish submission. Preparation is synchronous on the executor thread:

```text
Main GQA prepare_pages
Main GQA prepare_metadata
Main GDN prepare_states
Main GDN prepare_metadata       depends on prepared states
optional MTP GQA prepare_pages
optional GDN restore + wait
```

No prepare worker, channel, or receiver exists. GDN restore refreshes page-I/O staging on every batch, records only on
a replay miss, and waits before Main. Commit selects verified state versions and starts uncached publish when jobs
exist. Publish overlaps returning the response to runtime core; the next prepare/reset waits before shared page-I/O or
live-state resources are reused.

Whole-request reset enters through `Qwen35Executor::reset_req_slots` and fans out to sampling, Main GQA, optional MTP
GQA, debug speculative-probability metadata, and Main GDN. Inner state tables do not infer reset from token indices.
A state version ahead of its token index is a lifecycle invariant violation and panics.

## Supported MTP

The executor supports zero or one MTP module. The current checkpoint contract requires one GQA body layer, shared Main
token embedding, and no dedicated MTP embeddings.

`Qwen35MTPEmbed` owns previous-hidden gather, the shared `Rc<Embed>`, two checkpoint norms, concatenation, quantized
input projection, and its private temporaries. `Qwen35MTP` owns the single body `Qwen35Layer`, final residual
composition, final norm, and the MTP GQA page-table handle. There is no separate input-projector type or module loop.

The composed proposal sequence remains:

```text
MainEmbed -> Main -> GatherUnembed -> RejectionSampling
CPU target feedback
MTPEmbed -> MTP -> GatherUnembed -> DraftSampling
```

Normal non-MTP sampling remains:

```text
MainEmbed -> Main -> GatherUnembed -> Sampling
```

## DSpark scope

Qwen3.5 does not wire DSpark. `Qwen35Executor` has no DSpark field, replay, load/forward path, CLI option, Main residual
capture hook, or target-context path. Existing DSpark component source, configuration, exact bindings, low-level plan,
weights, and focused tests remain available for a future Qwen3 integration. This milestone verifies those low-level
contracts by compilation/tests only and makes no DSpark end-to-end claim.

## Verification

Unit coverage includes normalized config and exact bindings, GQA/GDN state, page overwrite/reset, GDN transactions and
snapshot I/O, generic replay idempotence/strict lookup, MTP rejection, generic sampling, and retained DSpark low-level
contracts. End-to-end verification must exercise Main and optional MTP through server/decode, inspect generated text,
and observe replay misses and hits. Performance evidence follows
[`executor_benchmarks.md`](executor_benchmarks.md) and is collected serially.
