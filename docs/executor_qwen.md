# Qwen Executor

This document describes the target-only Qwen3 Metal executor and the Qwen3.5/Qwen3.6 Metal executor from checkpoint
configuration through top-down component loading, state preparation, cached replay, and sampling. Qwen3.5 additionally
owns its single-module MTP path and the current dSpark component implementation. The `v3_x` directories contain only
version-neutral leaf components and utilities; each model owns its structural contracts and execution graph.

## Source layout

```text
crates/inference-executor-core/src/model/qwen/v3/
  config.rs                 Qwen3ModelConfig/Qwen3TextConfig, strict parsing, and EOS fallback
  batch.rs                  Qwen3Microbatch, request, response, and sampled-decision contracts
  weight_layout.rs          exact Qwen3 Main/unembed binding tree

crates/inference-executor-core/src/model/qwen/v3_x/
  config.rs                 shared quantization, RoPE, and tensor-path value utilities
  weight_layout.rs          shared GQA/GDN/dense-MLP/MoE leaf binding types and helpers

crates/inference-executor-core/src/model/qwen/v3_5/
  config.rs                 Qwen35ModelConfig/Qwen35TextConfig parsing and normalization
  batch.rs                  Qwen35Microbatch, request, response, and sampled-decision contracts
  pending_transactions.rs   Qwen35 sequence-ordered pending transactions
  weight_layout.rs          exact Qwen35 Main/unembed/MTP binding trees
  dspark_config.rs          Qwen35 dSpark configuration contract
  dspark_weight_layout.rs   exact Qwen35 dSpark tensor binding tree

crates/inference-executor-metal/src/
  replay.rs                 generic Replay<T> component/cache owner
  model/
    embedding.rs            shared weight-bearing Embed component
    unembedding.rs          shared weight-bearing Unembed component and QMV/QMM selection
    gather.rs               shared row-gather model component
    page_arena.rs           shared physical page buffer
    residual.rs             shared residual-add/capture model component
    rms_norm.rs             shared weight-bearing RMS-normalization component
    qwen/v3_x/
      layer/
        gqa.rs              shared Qwen3xGQA leaf load and record
        gdn.rs              shared Qwen3xGDN leaf load and record
        dense_mlp.rs        shared Qwen3xDenseMLP leaf load and record
        moe.rs              shared Qwen3xMoE leaf load and record
      state/
        gqa.rs              Qwen3xGQAState page/metadata/reset lifecycle
        gdn.rs              Qwen3xGDNState prepare/restore/commit/publish/reset lifecycle
      weight.rs             shared Qwen checkpoint decoding/validation helpers
    qwen/v3/
      main/
        mod.rs              Qwen3 Main owner, capture contract, and replay key
        embed.rs            Qwen3 Main embedding component and replay key
        gqa.rs              Qwen3 Main ungated GQA weights, state, load, and record
        layer.rs            fixed Qwen3MainLayer and Qwen3MainLayerScratch
        output.rs           Qwen3 gather/unembed component and replay key
        plan.rs             Qwen3 QKV GQA and dense-MLP core/Metal configuration
      executor/
        mod.rs              Qwen3Executor, private pending transactions, and runtime integration
        load.rs             target-only top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and Main replay submission
        main.rs             MainEmbed, Main, and GatherUnembed orchestration
        sampling.rs         ordinary target sampling and readback
    qwen/v3_5/
      main/
        mod.rs              Qwen35 Main owner, capture contract, and replay key
        embed.rs            Qwen35 Main embedding component and replay key
        layer.rs            Qwen35MainLayer variants and role-specific scratch
        output.rs           Qwen35 gather/unembed component and replay key
      mtp/
        mod.rs              supported one-layer Qwen35MTP owner and replay key
        embed.rs            Qwen35MTPEmbed and its replay key
        layer.rs            Qwen35MTPLayer and role-specific scratch
      rejection_sampling.rs Qwen35 rejection composition and result preparation
      plan.rs               Qwen35 component configuration, MTP validation, and dSpark plan
      dspark/                Qwen35 target/context/layer/Markov/speculator components
      executor/
        mod.rs              Qwen35Executor fields and ReplayableModelBatchExecutor integration
        load.rs             layer count pass and top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and common replay submission
        main.rs             MainEmbed, Main, and GatherUnembed orchestration
        sampling.rs         normal/draft/target/rejection orchestration and readback
        mtp.rs              MTP request, proposal-batch, and proposal flow

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs         TopKSampling and TopKSamplingOutputBuffers
  top_k_replay.rs           Sampling/DraftSampling replay components
  rejection_sampling.rs     generic sparse rejection Metal owner
  spec_probs.rs             SpecProbsStore sparse draft/target probability workspace
```

Runtime core owns scheduling, request lifecycle, physical page allocation/free, and page IDs. The executor owns
model-specific interpretation of those IDs, trained tensors, backend state, replay caches, and submission ordering.
Metal kernels remain backend components.

The sharing boundary is intentionally below the model execution graph. Qwen3 and Qwen3.5 each own a real model config,
batch and response contract, pending-transaction owner, model binding tree, layer and scratch type, plan/configuration
builder, Main components, executor, recorder, and replay keys. Structural APIs stay in their model directories.

`qwen/v3_x` is limited to true leaves: common quantization/RoPE/tensor-path values, per-component weight bindings and
checkpoint helpers, `Qwen3xGQA`/`Qwen3xGDN`/`Qwen3xDenseMLP`/`Qwen3xMoE`, and GQA/GDN state owners. Model-local layers
compose those leaves directly. Qwen3 therefore has no GDN transaction, state-page metadata, MTP lane, or Qwen3.5 replay
key.

## Semantic object tree

```text
Qwen3Executor
  main_gqa_state: Qwen3MainGQAState
  main_embed: Replay<Qwen3MainEmbed>
  main: Replay<Qwen3Main>
  gather_unembed: Replay<Qwen3GatherUnembed>
  sampling: Replay<Sampling>
  pages: PageArena

Qwen35Executor
  main_gqa_state: Qwen3xGQAState
  main_gdn_state: Qwen3xGDNState
  mtp_gqa_state: Option<Qwen3xGQAState>
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
cache. Each executor owns its dynamic workspaces, lifecycle ordering, and submissions.

Embedding, unembedding, row gather, residual add/capture, and RMS normalization remain shared model components. They
hide backend kernel and invocation details while concrete Main/MTP compositions retain ownership of graph ordering.

`Qwen3Microbatch` is target-only: it records which requests are decode requests and gathers exactly the last hidden
state from each. It rejects speculative input tokens while converting the shared runtime request and returns explicit
empty validated/speculative fields through the shared runtime response type.

Model role is also a structural boundary. `Qwen3MainLayer` owns Qwen3 Main's fixed-QKV `Qwen3MainGQA` plus
`Qwen3xDenseMLP` topology. `Qwen35MainLayer` owns Qwen3.5 Main's QGKV-GQA/GDN and dense-MLP/MoE variants, while
`Qwen35MTPLayer` independently owns the MTP decoder-layer graph. These role-specific layers may compose the same leaf
components, but they do not share a structural layer type. Qwen3 has no dSpark layer today; a future Qwen3 dSpark path
would likewise own a distinct role-specific type rather than extending `Qwen3MainLayer`.

`Qwen3xGQA` and `Qwen3xGDN` store compact per-kind layer indices, not model-layer indices, for page-table and
state-arena addressing.

## Configuration, bindings, and load

`Qwen3ModelConfig` strictly parses the flat Hugging Face Qwen3 schema and rejects unsupported GDN, MoE, MTP,
sliding-window, and RoPE-scaling variants. Its EOS token IDs provide a Qwen3-specific fallback when
`generation_config.json` supplies none. `Qwen35ModelConfig` independently parses and normalizes the Qwen3.5/Qwen3.6
schema, including layer-kind, MoE, MTP, and partial-RoPE fields. Runtime capacities are likewise model-specific in
`Qwen3ExecutorConfig` and `Qwen35ExecutorConfig`.

`Qwen35ExecutorConfig::max_requests` is the executor request-slot capacity,
not the scheduler per-batch request budget. The service initializes it from
`RuntimeConfig::max_running_requests`; GQA page tables, GDN request state,
sampling state, and request-indexed workspaces share that slot domain. Runtime
initialization separately requires `SchedulerConfig::max_requests` to be no
larger than this capacity.

Each model resolves its own exact typed binding tree before real tensor reads:

```text
Qwen3ModelWeightBindings
  embed: QuantizedTensorBindings
  main:
    final_norm_weight
    layers: Vec<Qwen3LayerWeightBindings>
      gqa: Qwen3xGQAWeightBindings
      mlp: Qwen3xDenseMLPWeightBindings
  unembed: QuantizedTensorBindings

Qwen35ModelWeightBindings
  embed: QuantizedTensorBindings
  main:
    final_norm_weight
    layers: Vec<Qwen35LayerWeightBindings>
      attention: Qwen35AttentionWeightBindings::{GQA,GDN}
      mlp: Qwen35MLPWeightBindings::{Dense,MoE}
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

1. Parse and validate the model-specific Main and optional MTP configurations.
2. Resolve exact binding trees.
3. Count Main GQA/GDN layers and Dense/MoE scratch requirements.
4. Construct Main GQA/GDN state domains and optional MTP GQA state before models that clone their handles.
5. Construct the model-local layer scratch, component scratch, and token `Embed`.
6. Move each exact binding subtree to its semantic owner. Each owner reads and validates its own real weights.
7. Construct Main, GatherUnembed, optional MTPEmbed/MTP, sampling owners, workspaces, and `PageArena`.
8. Wrap every cached stage in `Replay<T>`.

Qwen3 follows the same ownership order with a smaller graph: parse its flat config, resolve its Main binding
tree, construct one QKV GQA state domain and dense scratch, load Main/Embed/GatherUnembed, construct ordinary sampling,
and wrap the four model stages in `Replay<T>`. It has one cache lane and allocates no GDN state domain.

There is no Main/MTP plan object tree or aggregate component-weight owner. Qwen3 QKV GQA and dense-MLP
geometry/config conversion lives with its Main role in `qwen/v3/main/plan.rs`. Qwen3.5 owns its QGKV GQA, GDN,
dense-MLP, MoE, MTP validation, and low-level DSpark planning in `qwen/v3_5/plan.rs`. Shared leaf loaders receive
finalized core and Metal configurations rather than model configuration or default bags.

## Replay ownership

`Replay<T>::record(runtime, input)` derives the component key, returns immediately on a hit, and records/builds/inserts
exactly once on a miss. It returns `(key, cache_hit)`. `Replay<T>::replay(key)` is a strict lookup and panics if record
did not establish the key. `Replay<T>` exposes `component()` explicitly and does not implement `Deref`.

The independent cached graphs are:

```text
Replay<Qwen3MainEmbed>       Qwen3 token embedding
Replay<Qwen3Main>            Qwen3 dense full-attention layers -> final norm
Replay<Qwen3GatherUnembed>   Qwen3 gather -> unembed

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

MainEmbed and MTPEmbed are separate replay boundaries with their own keys.

Qwen3 defines separate replay keys for MainEmbed, Main, and GatherUnembed. Its Main key owns only the token count and
GQA replay topology. It never aliases a Qwen3.5 key or stores an optional GDN key.

## Main data flow and workspace ownership

```text
token_ids
  -> MainEmbed
token_hidden_input
  -> Main layers using model-local residual_stream[2] ping-pong
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

Qwen3 Main constructs `Qwen3MainLayerScratch`; Qwen3.5 Main constructs `Qwen35MainLayerScratch`, and Qwen3.5 MTP owns
its separate layer scratch. Similar workspace roles do not imply shared structural ownership.

`Qwen35Main` accepts an optional `Rc<dyn Qwen35MainResidualCapture>`, and `Qwen3Main` exposes the corresponding
model-specific `Qwen3MainResidualCapture` boundary. Immediately before each model layer's final
post-MLP residual add, Main asks the capture owner for an optional opaque `ResidualCaptureTarget`. The target selects a
stable, capture-owned BF16 column range; `None` records the ordinary residual add. The object-safe capture contract only
returns this descriptor and never receives a recorder; both Main record methods remain generic over
`Recorder<Operator = ReplayOp>`. The current target-only loaders supply no capture owner, so ordinary output buffers and
recorded operator sequences are unchanged.

`Qwen35GatherUnembedArgs` is flat: it binds the final-normalized hidden source, row indices, gather destination, and
logits destination. Gathered hidden and logits remain executor workspaces.

## GQA/GDN lifecycle

`Qwen3MainGQAState` groups Qwen3's ungated backend, scratch, request page table, metadata buffers, and cache-lane
information. It resets/prepares only KV page metadata. Qwen3 has zero state pages and does not construct, restore,
publish, commit, or reset a GDN state table. Qwen3.5 Main and MTP own distinct gated `Qwen3xGQAState` domains. Both
state types expose the same lifecycle concepts:

```text
prepare_pages(core_batch)
prepare_metadata(req_slots, token_indices, cu_tokens)
reset_req_slots(runtime_notification)
```

`Qwen3xGDNState` groups a backend, scratch, request state table, metadata, cached restore replay, and one optional
asynchronous publish submission. The current Qwen3.5 executor owns one `Qwen3xGDNState` mandatorily. Preparation is
synchronous on the executor thread:

```text
Main GQA prepare_pages
Main GQA prepare_metadata
Main GDN prepare_states(req_slots, block_indices, token_indices, cu_tokens, state_txns, state_page_ids_by_req)
Main GDN prepare_metadata(cu_tokens, prepared_states)
optional MTP GQA prepare_pages
optional GDN restore + wait
```

The shared GDN state leaf receives only these component inputs. Qwen3.5 extracts them from its model-owned microbatch
before calling the leaf; `Qwen3xGDNState` does not depend on the Qwen3.5 batch type.

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
input projection, and its private temporaries. `Qwen35MTP` owns the single `Qwen35MTPLayer`, final norm, and the MTP GQA
page-table handle. There is no separate input-projector type or module loop.

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

The Qwen3.5 subtree owns the current dSpark implementation: strict configuration and exact weight bindings in core,
model-specific geometry and Metal configuration in `v3_5/plan.rs`, and target residual projection, context append,
dSpark layers, Markov head, and speculator components under `v3_5/dspark/`.

Neither Qwen3 nor Qwen3.5 currently wires those components into its executor. There is no dSpark executor field, replay
stage, load/forward path, service option, or end-to-end claim. The focused component contract is documented in
[`dspark_design.md`](dspark_design.md).

## Verification

Unit coverage includes the strict flat Qwen3 adapter, model-specific target batch/replay keys, normalized Qwen3.5
config and exact bindings, GQA/GDN state, page overwrite/reset, GDN transactions and snapshot I/O, generic replay
idempotence/strict lookup, MTP rejection, generic sampling, and dSpark component contracts. End-to-end
verification exercises Qwen3 target-only and Qwen3.5 Main/optional MTP through server/decode and inspects generated
text. Performance evidence follows
[`executor_benchmarks.md`](executor_benchmarks.md) and is collected serially.
