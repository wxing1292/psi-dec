# Qwen Executor

This document describes the current Qwen3 and Qwen3.5/Qwen3.6 Metal executors.
The document covers checkpoint configuration, top-down loading, state preparation, cached replay, and sampling.
Qwen3 supports separate Vanilla and fixed-block DSpark modes.
Qwen3.5 supports separate Vanilla, reusable-layer MTP, and fixed-block DSpark modes.
The `v3_x` directories contain version-neutral leaf components, utilities, and the Qwen3x DSpark model.
Each model owns its structural contracts and execution graph.

## Source layout

```text
crates/inference-executor-core/src/model/qwen/v3/
  config.rs                 Qwen3ModelConfig/Qwen3TextConfig, strict parsing, and EOS fallback
  batch.rs                  Qwen3Microbatch, request, response, and sampled-decision contracts
  weight_layout.rs          exact Qwen3 Main/unembed binding tree

crates/inference-executor-core/src/model/qwen/v3_x/
  config.rs                 shared quantization, RoPE, and tensor-path value utilities
  weight_layout.rs          shared GQA/GDN/dense-MLP/MoE leaf binding types and helpers
  dspark/
    config.rs               official Qwen3 DSpark configuration contract
    weight_layout.rs        exact source and affine DSpark binding trees

crates/inference-executor-core/src/model/qwen/v3_5/
  config.rs                 Qwen35ModelConfig/Qwen35TextConfig parsing and normalization
  batch.rs                  Qwen35Microbatch, request, response, and sampled-decision contracts
  pending_transactions.rs   Qwen35 sequence-ordered pending transactions
  weight_layout.rs          exact Qwen35 Main/unembed/MTP binding trees

crates/inference-executor-metal/src/
  replay.rs                 generic Replay<T> component/cache owner
  model/
    embedding.rs            shared weight-bearing Embed component
    unembedding.rs          shared weight-bearing Unembed component and QMV/QMM selection
    gather.rs               shared row-gather model component
    main_residual_capture.rs
                            shared Main residual-capture contract
    page_arena.rs           shared physical page buffer
    residual_add.rs         shared residual-add/capture model component
    rms_norm.rs             shared weight-bearing RMS-normalization component
    qwen/v3_x/
      dspark/
        embed.rs            Qwen3x DSpark embedding replay
        execution.rs        shared execution resources and per-batch recording
        load.rs             shared checkpoint and resource load
        main_feature.rs     selected Main residual projection
        attention.rs        paged-history plus block-bidirectional attention
        layer.rs            independent Qwen3xDSparkLayer
        model.rs            context and body replay owners
        output.rs           gather/unembed and Markov sampling
        sampling.rs         Qwen3x Markov checkpoint weights and generic backend adapter
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
        mod.rs              Qwen3 Main owner and replay key
        embed.rs            Qwen3 Main embedding component and replay key
        gqa.rs              Qwen3 Main ungated GQA weights, state, load, and record
        layer.rs            fixed Qwen3MainLayer and Qwen3MainLayerScratch
        output.rs           Qwen3 gather/unembed component and replay key
        plan.rs             Qwen3 QKV GQA and dense-MLP core/Metal configuration
      executor/
        mod.rs              Qwen3Executor, private pending transactions, and runtime integration
        load.rs             separate Vanilla and DSpark top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and Main replay submission
        main.rs             MainEmbed, Main, and GatherUnembed orchestration
        sampling.rs         ordinary or rejection sampling and readback
        dspark.rs           DSpark Spec proposal orchestration
    qwen/v3_5/
      main/
        mod.rs              Qwen35 Main owner and replay key
        embed.rs            Qwen35 Main embedding component and replay key
        layer.rs            Qwen35MainLayer variants and role-specific scratch
        output.rs           Qwen35 gather/unembed component and replay key
      mtp/
        mod.rs              supported one-layer Qwen35MTP owner and replay key
        embed.rs            Qwen35MTPEmbed and its replay key
        layer.rs            Qwen35MTPLayer and role-specific scratch
      plan.rs               Qwen35 component configuration and MTP validation
      executor/
        mod.rs              Qwen35Executor fields and ReplayableModelBatchExecutor integration
        load.rs             layer count pass and separate Vanilla/MTP/DSpark top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and common replay submission
        main.rs             MainEmbed, Main, and GatherUnembed orchestration
        sampling.rs         Main/Spec/rejection orchestration and readback
        mtp.rs              MTP request, proposal-batch, and proposal flow
        dspark.rs           DSpark Spec proposal orchestration

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs         TopKSampling and TopKSamplingOutputBuffers
  top_k_replay.rs           Sampling/DraftSampling replay components
  rejection_replay.rs       generic sparse rejection replay owner
  dspark_markov.rs          DSpark Markov correction and sequential sampling
  spec_probs.rs             SpecProbsStore sparse Spec/Main probability workspace
```

Runtime core owns scheduling, request lifecycle, physical page allocation/free, and page IDs.
The executor owns model-specific interpretation of page IDs, trained tensors, backend state, replay caches, and
submission order.
Metal kernels remain backend components.

The sharing boundary is below the model execution graph.
Qwen3 and Qwen3.5 each own these model-level objects:

- A model configuration.
- Batch and response contracts.
- A pending-transaction owner.
- A model binding tree.
- Layer and scratch types.
- A plan and configuration builder.
- Main components.
- An executor and recorder.
- Replay keys.

Structural APIs stay in their model directories.

`qwen/v3_x` contains shared leaf components, values, and the reusable Qwen3x DSpark model.
These leaves include quantization, RoPE, tensor-path values, weight bindings, checkpoint helpers, and GQA/GDN state
owners.
They also include `Qwen3xGQA`, `Qwen3xGDN`, `Qwen3xDenseMLP`, and `Qwen3xMoE`.
The DSpark model composes the same generic leaf owners into an independent model role.
Qwen3 has no GDN transaction, state-page metadata, MTP lane, or Qwen3.5 replay key.

## Semantic object tree

```text
Qwen3Executor
  main_gqa_state: Qwen3MainGQAState
  main_embed: Replay<Qwen3MainEmbed>
  main: Replay<Qwen3Main>
  gather_unembed: Replay<Qwen3GatherUnembed>
  sampling: Replay<Sampling>
  speculator: Qwen3Speculator
    Vanilla
    DSpark
      execution: Qwen3xDSparkExecution
      rejection_sampling: Replay<RejectionSampling>
      spec_probs: SpecProbsStore
  pages: PageArena

Qwen35Executor
  main_gqa_state: Qwen3xGQAState
  main_gdn_state: Qwen3xGDNState
  main_embed: Replay<Qwen35MainEmbed>
  main: Replay<Qwen35Main>
  gather_unembed: Replay<Qwen35GatherUnembed>
  sampling: Replay<Sampling>
  speculator: Qwen35Speculator
    Vanilla
    MTP
      num_steps: usize
      gqa_state: Qwen3xGQAState
      embed: Replay<Qwen35MTPEmbed>
      body: Replay<Qwen35MTP>
      sampling: Replay<DraftSampling>
      execution: Qwen35MTPExecution
      common: Qwen35SpeculativeResources
    DSpark
      execution: Qwen3xDSparkExecution
      common: Qwen35SpeculativeResources
  pages: PageArena

Qwen35SpeculativeResources
  rejection_sampling: Replay<RejectionSampling>
  spec_probs: SpecProbsStore

Qwen3xDSparkExecution
  context: Replay<Qwen3xDSparkContext>
  gqa_state: UngatedDSparkGQAState
  embed: Replay<Qwen3xDSparkEmbed>
  body: Replay<Qwen3xDSparkBody>
  gather_unembed: Replay<Qwen3xDSparkGatherUnembed>
  sampling: Replay<Qwen3xDSparkSampling>
    markov: Qwen3xDSparkMarkov
  reusable hidden/logit workspaces
  page layout and block geometry

Qwen3ModelOpsRecorder / Qwen35ModelOpsRecorder
  dspark: Qwen3xDSparkRecording

Qwen3xDSparkRecording
  context/embed/body/gather/sampling replay keys
  sampling arguments and Markov replay shape
  request slots
```

Semantic components own weights, static configuration, and `load + record`.
`Replay<T>` owns the related replay cache.
Each executor owns its dynamic workspaces, lifecycle order, and submissions.
Each executor stores one closed speculator enum.
The enum cannot represent a partial resource set or simultaneous MTP and DSpark resources.
Vanilla executors do not allocate rejection or speculative-probability resources.
Per-batch recorders retain optional replay keys because a batch can omit output or Spec stages.
These lifecycle keys do not own initialized model resources.

Embedding, unembedding, row gather, residual add/capture, and RMS normalization remain shared model components.
These components hide backend kernel and invocation details.
Concrete Main and MTP compositions own graph order.

`Qwen3Microbatch` records decode requests and an optional speculative suffix.
It identifies the Main rows that require unembedding.
It implements the Metal `SpecMicrobatch` rejection-input contract.
The Qwen3 executor rejects speculative input when DSpark is disabled.

The model role is also a structural boundary.
`Qwen3MainLayer` owns the fixed-QKV `Qwen3MainGQA` and `Qwen3xDenseMLP` topology for Qwen3 Main.
`Qwen35MainLayer` owns the QGKV-GQA/GDN and dense-MLP/MoE variants for Qwen3.5 Main.
`Qwen35MTPLayer` independently owns the MTP decoder-layer graph.
The MTP embed, layer norms, and final norm load bounded tensor maps for their exact binding subtrees.
Each MTP owner removes its tensors before it creates Metal buffers.
`Qwen3xDSparkLayer` independently owns the DSpark decoder-layer graph.
These role-specific layers can compose the same leaf components.
They do not share a structural layer type.

`Qwen3xDSparkLayer` composes ungated DSpark GQA, RMSNorm, dense MLP, and residual components.
It does not extend `Qwen3MainLayer`.
It does not add a variant to `Qwen35MTPLayer`.

`Qwen3xGQA` and `Qwen3xGDN` store compact per-kind layer indices, not model-layer indices, for page-table and
state-arena addressing.

## Configuration, bindings, and load

`Qwen3ModelConfig` strictly parses the flat Hugging Face Qwen3 schema.
It rejects unsupported GDN, MoE, MTP, sliding-window, and RoPE-scaling variants.
Its EOS token IDs provide a Qwen3 fallback when `generation_config.json` supplies none.
`Qwen3xDSparkConfig` independently parses the official flat DSpark schema.
It validates Main compatibility, official `target_layer_ids`, fixed-block geometry, ungated GQA, and the `vanilla`
Markov head.
`Qwen35ModelConfig` independently parses and normalizes the Qwen3.5/Qwen3.6 schema.
That schema includes layer-kind, MoE, MTP, and partial-RoPE fields.
`Qwen3ExecutorConfig` and `Qwen35ExecutorConfig` keep runtime capacities model-specific.

CLI parser structs accept `Option` values for checkpoint arguments.
The service validates these values once and converts them to `Qwen3ModelMode` or `Qwen35ModelMode`.
The normalized service configuration stores only the closed model mode.
The executor public API has one initializer for each supported mode.
The private loader receives one closed init enum.
The constructed executor receives one closed speculator enum.

`Qwen35ExecutorConfig::max_requests` is the executor request-slot capacity.
`Qwen35Config` derives it from the same `--max-requests` value as the scheduler per-batch request capacity.
The service passes that value to `Qwen35ExecutorConfig::max_requests`, `RuntimeConfig::max_running_requests`, and
`SchedulerConfig::max_requests`.
GQA page tables, GDN request state, sampling state, and request-indexed workspaces share this slot domain.
Main, MTP, and DSpark use this same configuration flow.

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

Qwen3xDSparkWeightBindings
  optional embed: QuantizedTensorBindings
  main_feature:
    hidden_norm_weight
    fc: QuantizedTensorBindings
  layers: Vec<Qwen3xDSparkLayerWeightBindings>
    gqa: Qwen3xGQAWeightBindings
    mlp: Qwen3xDenseMLPWeightBindings
    input_layernorm_weight
    post_attention_layernorm_weight
  final_norm_weight
  optional unembed: QuantizedTensorBindings
  markov: Qwen3xDSparkMarkovWeightBindings
  confidence: Qwen3xDSparkConfidenceWeightBindings

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

1. Select one closed Vanilla, MTP, or DSpark init mode.
2. Parse and validate the Main configuration and the selected Spec configuration.
3. Count Main GQA/GDN layers and Dense/MoE scratch requirements.
4. Construct Main state domains and the selected Spec state domain.
5. Construct the model-local layer scratch, component scratch, and token `Embed`.
6. Move each exact binding subtree to its semantic owner. Each owner reads and validates its own real weights.
7. Construct Main and the selected Spec stages.
8. Aggregate all selected Spec resources into one enum variant.
9. Construct `PageArena` and wrap each cached stage in `Replay<T>`.

Qwen3 follows the same ownership order with separate Vanilla and DSpark graphs.
It parses its flat configuration and resolves its Main binding tree.
When configured, it parses the DSpark configuration and passes it to the shared DSpark loader.
It constructs one QKV GQA state domain and dense scratch.
It constructs a second ungated GQA state domain and `DSparkBlockScratch` when DSpark is enabled.
It loads Main and Main output for both modes.
The DSpark mode also loads the shared DSpark execution owner and rejection resources.

Qwen3.5 loads Vanilla, its MTP graph, or the reusable Qwen3x DSpark graph.
The loader cannot receive or construct both speculators.
When Main and MTP both use MoE, they reuse one model-owned `MoEScratch`.
The MTP compatibility check requires exact matches for `num_experts_per_tok`, `moe_intermediate_size`, and
`shared_expert_intermediate_size` before it allocates this scratch.
For `shared_expert_intermediate_size`, `0` means that the shared-expert branch is absent.
A positive value means that the branch is present, and both models must specify the same value.
`norm_topk_prob` does not determine scratch geometry and may differ.
If only Main or only MTP uses MoE, the unused model-side MoE geometry does not constrain the scratch allocation.
An incompatible shared geometry returns a recoverable model initialization error.
For DSpark, the loader validates Main hidden width, layer count, vocabulary, position limit, and RoPE values.
It permits the DSpark query projection width to differ from the Main hidden width.
The shared loader requires `block_size + 1 <= max_tokens_per_request` for both Main versions.
It adds DSpark context K/V pages to the Main cache lane and retains the Main GDN state domain.

Qwen3 has one runtime cache lane and allocates no GDN state domain.
The executor splits each runtime block between Main K/V and persistent DSpark context K/V.

There is no Main/MTP plan object tree or aggregate component-weight owner.
Qwen3 Main owns QKV GQA and dense-MLP geometry conversion in `qwen/v3/main/plan.rs`.
Qwen3.5 owns QGKV GQA, GDN, dense-MLP, MoE, and MTP validation in `qwen/v3_5/plan.rs`.
Qwen3x DSpark has no plan object or plan source file.
Each DSpark semantic owner derives its fixed geometry from `Qwen3xDSparkConfig` and resolves its affine layout from
the exact binding subtree that it consumes.
Each owner loads a bounded `TensorMap`, removes its tensors, performs its required fusion, and requires an empty map.
Each DSpark layer owns its weight-dependent GQA and dense-MLP backend.
The DSpark state domain shares only page tables, metadata, scratch, and geometry-dependent compute selection.

## Replay ownership

`Replay<T>::record(runtime, input)` derives the component key and returns immediately on a hit.
On a miss, it records, builds, and inserts exactly once.
It returns `(key, cache_hit)`.
`Replay<T>::replay(key)` is a strict lookup.
It panics if record did not establish the key.
`Replay<T>` exposes `component()` explicitly and does not implement `Deref`.

The independent cached graphs are:

```text
Replay<Qwen3MainEmbed>       Qwen3 token embedding
Replay<Qwen3Main>            Qwen3 dense full-attention layers -> final norm
Replay<Qwen3xDSparkContext>   selected Main residuals -> persistent DSpark context K/V
Replay<Qwen3GatherUnembed>   Qwen3 gather -> unembed
Replay<Sampling>             ordinary Main sampling
Replay<RejectionSampling>    Main sparse distributions -> speculative rejection
Replay<Qwen3xDSparkEmbed>     anchor + MASK block embedding
Replay<Qwen3xDSparkBody>      fixed DSpark layers -> final norm
Replay<Qwen3xDSparkGatherUnembed>
                             request-major hidden -> step-major logits
Replay<Qwen3xDSparkSampling>
                             Markov correction, sampling, and sparse draft storage

Replay<Qwen35MainEmbed>      token embedding
Replay<Qwen35Main>           all Main layers -> final norm
Replay<Qwen35GatherUnembed>  gather -> unembed
Replay<Sampling>             ordinary Main sampling
Replay<Qwen35MTPEmbed>       previous-hidden gather + token embed + input projection
Replay<Qwen35MTP>            one physical GQA body layer -> final norm
Replay<DraftSampling>        draft sampling + sparse draft distribution
Replay<RejectionSampling>    Main sparse distribution + rejection
Replay<Rc<GDNRequestStateTable>>
                              snapshot restore into live GDN state
```

MainEmbed and MTPEmbed are separate replay boundaries with their own keys.

The shared quantized embedding leaf supports exact and bucketed recording.
Exact recording fixes the active token count to `QuantizedEmbeddingShape::num_tokens` and declares no replay parameter.
Bucketed recording interprets `QuantizedEmbeddingShape::num_tokens` as the recorded capacity.
It validates buffers and dispatches the grid for that capacity.
It binds the caller-provided active-token key with the range `1..=capacity`.
The kernel checks the active token count before it reads `token_ids` or writes the output row.
Qwen3.5 MainEmbed reads the configured token-row capacity from `Embed::max_tokens()`.
It owns a base `ReplayBucketPolicy` capped by this capacity.
It records the bucket capacity in `Qwen35MainEmbedReplayKey` and never records the active token count in the key.
It uses the stage-owned `qwen3.5.main_embed.num_active_tokens` replay parameter for submission.
The executor stores this argument with the prepared key and submits both to the same replay program.
Qwen3 MainEmbed still selects exact recording.
Qwen3.5 MTPEmbed selects bucketed embedding recording as part of its composed replay.

The shared row-gather leaf supports exact and bucketed recording.
Exact recording fixes the active row count to `RowGatherShape::num_rows` and declares no replay parameter.
Bucketed recording interprets `RowGatherShape::num_rows` as the recorded capacity.
It validates the row-index and output buffers and dispatches the grid for that capacity.
It binds the caller-provided active-row key with the range `1..=capacity`.
The kernel checks the active row count before it reads an inactive row index or input value and before it writes an
inactive output value.
Qwen3.5 MTPEmbed selects bucketed row-gather recording.
Qwen3.5 GatherUnembed also selects bucketed row-gather recording.
Qwen3 and DSpark GatherUnembed still select exact row-gather recording.

The shared unembedding leaf supports exact and bucketed recording.
Exact recording fixes the active row count and declares no replay parameter.
Bucketed recording uses the caller-provided total row capacity and active-row key.
It validates that the total row capacity is in `1..=UnembedConfig::max_tokens`.
It validates the hidden input and logits output ranges against this total row capacity.
The affine replay parameter validates the submitted active row count in `1..=capacity`.
The leaf exposes the affine kernel topology for a row capacity and every row count that changes this topology.
The stage bucket policy must include these topology boundaries.
This rule lets Gather and Unembed use one active-row key without padding across an affine kernel change.
Qwen3.5 GatherUnembed selects bucketed unembedding recording.
Qwen3 and DSpark GatherUnembed still select exact unembedding recording.

The shared BF16 row-concat leaf supports exact and bucketed recording.
Exact recording fixes the active row count to `Bf16ConcatRowsShape::num_rows` and declares no replay parameter.
Bucketed recording interprets `Bf16ConcatRowsShape::num_rows` as the recorded capacity.
It validates both input buffers and the output buffer and dispatches the grid for that capacity.
It binds the caller-provided active-row key with the range `1..=capacity`.
The kernel checks the active row count before it reads an input value or writes an output value.
The row-concat leaf has one fixed topology and adds no replay bucket boundary.
Qwen3.5 MTPEmbed selects bucketed row-concat recording.

The shared RMS-normalization leaf supports exact and bucketed recording.
Exact recording fixes the token count and declares no replay parameter.
Bucketed recording dispatches the recorded token capacity and binds the caller-provided active-token key with the
range `1..=capacity`.
The RMS-normalization kernel checks the active token count before it reads or writes a row.

Replay recording can fuse a residual add with the immediately following RMS normalization.
The required-fusion residual path makes this adjacency and buffer-identity contract mandatory.
Replay construction fails if another operator occurs first or if the RMS normalization consumes a different buffer.
The residual shape contains `capacity * hidden_dim` values.
The fused command inherits the active-token key and token capacity from the RMS normalization.
It does not declare a separate active-value parameter.
The ordinary and capture fused kernels check the active token count before they read or write a row.
The capture variant validates its destination for the recorded capacity and writes only active rows.

Standalone residual-add recording remains exact.
A bucketed Qwen stage must use required residual/RMS-normalization fusion instead of a standalone residual dispatch.
RMS normalization and residual/RMS-normalization fusion have fixed token-count topology and add no replay bucket
boundary.
Qwen3.5 MTPEmbed selects bucketed normalization recording.
The current Qwen3.5 body stages still select exact normalization and residual recording.

Qwen3.5 MTPEmbed owns one token-capacity replay domain.
Its replay policy uses the shared base bucket ladder and the input-projection FC topology boundaries.
The policy is capped by the executor token workspace capacity.
The selected capacity and FC topology identify a production replay key.
The active token count does not enter this key.
The source-compatible `Qwen35MTPEmbedReplayKey::new(...)` constructor creates only a legacy exact/manual identity.
Production recording uses `Qwen35MTPEmbed::prepare_replay(...)`.
The composed Gather, Embed, both RMS normalizations, BF16 concat, and FC commands use the same
`qwen3.5.mtp_embed.num_active_tokens` parameter.
The composed replay declares exactly one parameter.
The executor stores this argument with the MTPEmbed key and reuses it for every MTP step.

Qwen3.5 GatherUnembed owns one output-row-capacity replay domain.
Its replay policy combines the shared base bucket ladder with every unembed affine topology boundary.
`UnembedConfig::max_tokens` caps this policy.
The loader requires this cap to equal the executor `max_tokens` workspace capacity.
This cap counts output rows, not requests.
One speculative request can produce more than one output row.
`Qwen35GatherUnembed::prepare_replay(...)` maps each nonzero active row count to one recorded capacity.
The production replay key records `num_total_rows` and the categorical unembed topology.
The active row count does not enter this key.
The source-compatible `Qwen35GatherUnembedReplayKey::from_microbatch(...)` constructor creates only a legacy
exact/manual identity.
The composed Gather and Unembed commands use the same `qwen3.5.gather_unembed.num_active_rows` parameter.
The composed replay declares exactly one parameter.
Main and MTP use the same replay cache because they bind the same stable buffers.
The recorder stores separate Main and MTP `ReplayArguments` because their active row counts can differ.
An active row count of zero omits GatherUnembed replay.

Qwen3 defines separate replay keys for MainEmbed, Main, and GatherUnembed.
Its Main key owns only the token count and GQA replay topology.
It never aliases a Qwen3.5 key or stores an optional GDN key.

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

`token_hidden_input` is the embedding destination and layer-0 input.
`hidden_output` is the final RMSNorm destination.
The executor owns both `Rc<Buffer>` workspace slots.
Runtime stages pass them without `Option`, hidden handles, or hidden-source enums.
The final layer residual is only the current local ping-pong buffer.
There is no `final_residual` field, accessor, or allocation.

Qwen3 Main constructs `Qwen3MainLayerScratch`.
Qwen3.5 Main constructs `Qwen35MainLayerScratch`.
Qwen3.5 MTP owns separate layer scratch.
All configured logical MTP steps reuse this scratch and the same physical weights.
Similar workspace roles do not imply shared structural ownership.

`Qwen3Main` and `Qwen35Main` accept the shared `MainResidualCapture` boundary.
Main queries the capture owner immediately before each layer's final post-MLP residual add.
The capture owner returns an optional opaque `ResidualAddCaptureTarget`.
The destination selects a stable BF16 column range that the capture owner owns.
`None` records the ordinary residual add.

The object-safe capture contract returns only this descriptor.
It never receives a recorder.
Both Main record methods remain generic over `Recorder<Operator = ReplayOp>`.

The Qwen3 and Qwen3.5 loaders supply no capture owner when DSpark is disabled.
When DSpark is enabled for Qwen3 or Qwen3.5, `Qwen3xDSparkMainFeatureProjector` owns the capture destinations.
`Qwen3xDSparkContext` records Main-feature projection and context append after Main.
Main does not depend on a concrete DSpark type.

`Qwen35GatherUnembedArgs` has a flat structure.
It binds the final-normalized hidden source, row indices, gather destination, and logits destination.
Gathered hidden and logits remain executor workspaces.

## Batch execution lifecycle

The service calls the model hooks in a fixed order.
The service records the Main components first.
The service then owns the Main `submit`, `wait`, and CPU read boundaries.
When `run_spec` returns true, the service records the MTP or DSpark components after the Main CPU read.
The service then owns the Spec `submit`, `wait`, and CPU read boundaries.
An empty component input omits that component from its model sequence.
The executor does not store a separate submitted-state flag.

`embed_main` materializes MainEmbed.
`forward_main` materializes Main.
For Qwen3 or Qwen3.5 DSpark, it also materializes `Qwen3xDSparkContext`.
It registers the pending model transaction.
It does not submit backend work.

`unembed_main` materializes GatherUnembed when the batch has sampled rows.
It returns immediately when the batch has no sampled rows.

`sample_main` materializes Sampling or RejectionSampling when the batch has sampled rows.
It returns immediately when the batch has no sampled rows.
It does not submit backend work or read backend output.

`submit_main` submits one Main sequence.
For a batch with no sampled rows, the sequence is:

```text
MainEmbed -> Main
```

When DSpark is enabled, the same sequence appends `DSparkContext` after Main.

For ordinary sampling, the sequence is:

```text
MainEmbed -> Main -> GatherUnembed -> Sampling
```

When a speculator is enabled, `sample_main` materializes RejectionSampling for both initial and speculative input.
RejectionSampling supports a ragged `0..N` speculative-token count per request.
For MTP, `submit_main` submits this sequence:

```text
MainEmbed -> Main -> GatherUnembed -> RejectionSampling
```

For DSpark, it submits this sequence:

```text
MainEmbed -> Main -> DSparkContext -> GatherUnembed -> RejectionSampling
```

The service waits for the Main submission.
It then calls `read_main`.
`read_main` reads the sampled or rejection results on the CPU.

The service calls `run_spec` after `read_main`.
Qwen3 requires a configured DSpark model and at least one Main decode result.
Qwen3.5 uses its configured MTP or DSpark capability.
Qwen3.5 also records MTP during prefill because MTP owns a persistent KV lane.
Qwen3 and Qwen3.5 record DSpark context in the Main submission.
They record DSpark Spec only after Main returns a sampled anchor.
The lifecycle does not use a per-batch submitted-state flag.
When the gate is true, the service calls `embed_spec`.
`embed_spec` consumes the completed Main output and sampled results.

For MTP, these hooks materialize MTPEmbed, MTP, GatherUnembed, and DraftSampling once.
For DSpark, they materialize DSparkEmbed, DSpark, DSparkGatherUnembed, and DSparkSampling.
These hooks do not submit backend work or read backend output.

`submit_spec` starts one model-specific Spec transaction.
For `--num-mtp-steps K`, the Qwen3.5 MTP owner executes this dependent sequence K times:

```text
for step_index in 0..K:
    MTPEmbed -> MTP -> GatherUnembed -> DraftSampling
    sampled token becomes the next step's token input
```

The physical weights, scratch, buffer bindings, and replay programs remain the same for every step.
The MTP output overwrites the stable previous-hidden source buffer, so the next replay consumes the prior MTP output
without changing a buffer binding.
The step index is a replay argument for GQA page selection and distribution-row selection.
It is not part of an MTP replay key.

The current implementation waits and reads after each non-final MTP pass.
`submit_spec` returns the final submission, and the service performs the final wait and `read_spec` call.
This preserves one external Spec lifecycle while the MTP owner controls K internal passes.

The Qwen3x DSpark sequence is:

```text
DSparkEmbed -> DSpark -> DSparkGatherUnembed -> DSparkSampling
```

The service waits for the Spec submission.
It then calls `read_spec`.
`read_spec` reads draft tokens and probabilities on the CPU.

The complete service order is:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> wait -> read_main
if run_spec(model_batch_req, sampled_output):
    embed_spec -> forward_spec -> unembed_spec -> sample_spec
    submit_spec -> wait -> read_spec
commit
```

## GQA/GDN lifecycle

`Qwen3MainGQAState` groups the Qwen3 ungated backend, scratch, request page table, metadata buffers, and cache-lane
information.
It resets and prepares only KV page metadata.
Qwen3 has zero state pages.
It does not construct, restore, publish, commit, or reset a GDN state table.
`UngatedDSparkGQAState` owns a separate DSpark page table, metadata buffers, backend, and block scratch.
Both Qwen3 GQA states consume spans from the same runtime cache block.
`prepare_batch` splits each flat runtime page-ID span before it updates these tables.
Qwen3.5 Main and MTP own distinct gated `Qwen3xGQAState` domains.
Both state types expose the same lifecycle concepts:

```text
prepare_pages(core_batch)
prepare_metadata(req_slots, token_indices, cu_tokens)
reset_req_slots(runtime_notification)
```

`Qwen3xGDNState` groups a backend, scratch, request state table, metadata, cached restore replay, and one optional
asynchronous publish.
The current Qwen3.5 executor must own one `Qwen3xGDNState`.
The executor thread prepares it synchronously:

```text
Main GQA prepare_pages
Main GQA prepare_metadata
Main GDN prepare_states(req_slots, block_indices, token_indices, cu_tokens, state_txns, state_page_ids_by_req)
Main GDN prepare_metadata(cu_tokens, prepared_states)
Qwen3.5 MTP page-table prepare
optional GDN restore + wait
```

The shared GDN state leaf receives only these component inputs.
Qwen3.5 extracts them from its model-owned microbatch before it calls the leaf.
`Qwen3xGDNState` does not depend on the Qwen3.5 batch type.

No prepare worker, channel, or receiver exists.
GDN restore refreshes page-I/O staging on every batch.
It records only on a replay miss and waits before Main.
Commit selects verified state versions.
It starts an uncached publish when jobs exist.

MTP keeps the GDN current version aligned with the Main runtime cache frontier.
For one request, Main calculates `num_fixed_tokens = q_len - num_spec_tokens`.
It commits `input_state_version + num_fixed_tokens + num_accepted_tokens`.
`num_mtp_steps` does not directly adjust this state version.
MTP decode replays K - 1 verified tail tokens in the next Main call.
Qwen verification keeps the verified state version unchanged and calculates
`replay_source_state_version = verified_state_version - (K - 1)`. It passes this physical source to GDN commit as the
state version that becomes current.
The transaction materializes the union of all verified and replay-source choices.
The next newly sampled token index remains the verified state version.
Runtime can represent a full prompt as `QueryTokens::Decode` when the prompt fits the token budget, so this rule also
applies to that zero-spec warm-up.
`QueryTokens::Prefill` commits its full window without the replay shift.

Runtime core can receive the response while publish continues.
The next prepare or reset waits before it reuses shared page-I/O or live-state resources.

Whole-request reset enters through `Qwen35Executor::reset_req_slots`.
It fans out to sampling, Main GQA, the selected Spec owner, and Main GDN.
Inner state tables do not infer reset from token indices.
A state version ahead of its token index is a lifecycle invariant violation and panics.

## Supported MTP

The executor supports zero or more logical MTP steps.
The current checkpoint contract requires exactly one physical GQA body layer and shared Main token embedding.
It does not permit dedicated MTP embeddings.
`num_mtp_steps = K` chains that one physical layer K times.
The logical model has K+1 token and cache lanes: Main plus one MTP lane for each dependent step.

`Qwen35MTPEmbed` owns previous-hidden gather, the shared `Rc<Embed>`, two checkpoint norms, concatenation, and quantized
input projection.
It also owns its private temporary buffers.
`Qwen35MTP` owns the single `Qwen35MTPLayer`, final norm, and MTP GQA page-table handle.
It maps runtime cache lanes `1..=K` to the matching MTP page-table rows.
`Qwen35MTPSpeculator` owns the internal step loop and one execution accumulator.
There is no physical layer vector and no duplicated weight owner.

The composed proposal sequence remains:

```text
Main batch submission:
    MainEmbed -> Main -> GatherUnembed -> RejectionSampling
CPU sampling feedback
MTP internal passes:
    K * (MTPEmbed -> MTP -> GatherUnembed -> DraftSampling)
```

Normal non-MTP sampling remains:

```text
Main batch submission:
    MainEmbed -> Main -> GatherUnembed -> Sampling
```

## Supported DSpark

DSpark support is experimental.
The Qwen3 and Qwen3.5 DSpark modes support one fixed-block DSpark checkpoint.
Qwen3.5 MTP and DSpark are mutually exclusive.

Each DSpark-enabled executor records persistent context updates in the Main submission:

```text
MainEmbed -> Main -> DSparkContext -> GatherUnembed -> RejectionSampling
```

The CPU reads the Main result before it constructs the anchor block.
The executor then records one Spec submission:

```text
DSparkEmbed -> DSpark -> DSparkGatherUnembed -> DSparkSampling
```

DSpark input is request-major.
`DSparkGatherUnembed` converts body output to the step-major order required by sequential Markov sampling.
The draft probability store uses request-slot identity because these rows cross a batch boundary.
Main verification distributions use compact active-row identity because they exist only in one submission.

Main K/V and persistent DSpark context K/V share one runtime cache-block lifecycle.
The executor owns separate page tables and splits each runtime page span.
Proposal-local Q/K/V and attention partials remain in executor-owned `DSparkBlockScratch`.

Qwen3.5 GDN keeps one current state and `block_size + 1` decision candidates for each DSpark request slot.
It also reserves cache-block boundary candidates.
The Qwen3.5 service sets the running-slot capacity from `--max-requests` for Main, MTP, and DSpark.
These state buffers remain allocated, reusable, and resident with the cached replay resources.

[`dspark_design.md`](dspark_design.md) documents the complete current component contract.

## Verification

Unit tests cover:

- The strict flat Qwen3 adapter.
- Model-specific Main batch and replay keys.
- Normalized Qwen3.5 configuration and exact bindings.
- GQA/GDN state and page overwrite/reset.
- GDN transactions and snapshot I/O.
- Generic replay idempotence and strict lookup.
- MTP and DSpark sparse rejection.
- DSpark configuration, bindings, attention, Markov sampling, and page splitting.

End-to-end tests exercise Qwen3 Main-only and Qwen3 DSpark through server/decode.
They also exercise Qwen3.5 Vanilla, MTP, and DSpark modes.
The tests inspect generated text.
Performance evidence follows [`executor_benchmarks.md`](executor_benchmarks.md).
Collect that evidence serially.
