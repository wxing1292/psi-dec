# Qwen3 DSpark Integration Design (Temporary Draft)

Temporary status: This document records the current design discussion.
It is not a current-source contract.

This document is the single draft record for planned Qwen3 DSpark integration.
It also owns the related deferred executor-ownership, confidence-scheduling, and backend-neutral replay work.
Current-component documents continue to describe current `src`.

Current status:

- Qwen3 is target-only.
- Qwen3.5 supports zero or one MTP module.
- Qwen3.5 retains low-level DSpark-era components, configuration, bindings, conversion, and tests.
- No executor or service wires DSpark.
- No end-to-end DSpark correctness or performance claim exists.

The retained Qwen3.5 implementation is reference input.
It is not the target Qwen3 design.

## Confirmed scope

The first Qwen3 DSpark milestone must use the official DeepSeek Qwen3 DSpark model contract.
It must not directly wire the retained Qwen3.5-era low-level graph into `Qwen3Executor`.

The milestone must preserve these boundaries:

- `Qwen3MainLayer`, `Qwen35MTPLayer`, and `Qwen3DSparkLayer` are separate role types.
- DSpark must not extend or add variants to `Qwen3MainLayer`.
- DSpark must not add fields or behavior to `Qwen35MTPLayer`.
- `Qwen3DSparkLayer` may compose shared GQA, norm, dense-MLP, and other leaf components.
- `Qwen3DSparkLayer` may own DSpark-specific operations and components when the model requires them.
- The service must continue to use the generic Main and Spec lifecycle hooks.
- Main and Spec must expose the same `embed -> forward -> unembed -> sample -> submit -> wait -> read` lifecycle shape.
- `forward_spec` must not also record Spec embedding, unembedding, or sampling.
- The service must not add DSpark-specific prepare, sample, commit, or replay hooks.

The first milestone uses one in-flight batch per executor.
`prepare_batch` and batch execution form one exclusive interval.
Shared scratch and output buffers may be reused only after that interval ends.
The design does not permit overlapping batch preparation or execution.

GPU execution failures are terminal internal failures.
The design does not provide recoverable GPU rollback.

## Current retained source

The repository currently retains these Qwen3.5-era files:

```text
crates/inference-executor-core/src/model/qwen/v3_5/
  dspark_config.rs          retained nested DFlash-era configuration
  dspark_weight_layout.rs   retained tensor binding tree

crates/inference-executor-core/src/bin/
  qwen35_dspark_quantize.rs retained checkpoint converter

crates/inference-executor-metal/src/model/qwen/v3_5/
  plan.rs                   retained Qwen35DSparkPlan
  dspark/
    attention.rs
    block_request.rs
    context.rs
    layer.rs
    markov.rs
    speculator.rs
    target.rs
    weights.rs
```

The Qwen3.5 executor does not reference these components from its load or forward path.
The Qwen3.5 loader supplies no DSpark capture owner.
MTP remains the only speculator that Qwen3.5 executes.

The retained components provide implementation evidence for:

- Tensor names and affine layouts
- Target-feature geometry
- Local block metadata
- GQA context append
- Dense layer and scratch geometry
- Markov correction
- Metal buffer and kernel contracts

They do not define the new Qwen3 config, layer, batch, transaction, or executor contract.

## Official configuration contract

The Qwen3 DSpark checkpoint uses a flat Hugging Face configuration.
The new core type must be `Qwen3DSparkConfig`.
It must parse the official field names directly.
It must not contain `dflash_config` or `DSparkDFlashConfig`.

The initial contract includes these fields when the checkpoint provides them:

```text
architectures
block_size
mask_token_id
target_layer_ids
num_target_layers
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
rms_norm_eps
rope_parameters
markov_head_type
markov_rank
enable_confidence_head
confidence_head_with_markov
vocab_size
```

The implementation must follow the official Qwen3 DSpark checkpoint schema.
It must remove the retained nested DFlash-era parser, validation, plan wiring, and tests when the new Qwen3 contract
replaces them.
It must not keep a compatibility alias for `dflash_config`.

`target_layer_ids` selects target decoder-layer outputs.
The first milestone applies these requirements:

- The list must not be empty.
- The values must be strictly increasing.
- Each value must satisfy `0 <= layer_id < num_target_layers`.
- The value `-1` is unsupported.
- Embedding outputs and final-norm sentinels are unsupported.

If the current residual-capture location does not represent the official decoder-layer output, the Qwen3 capture owner
must change.
The implementation must not redefine the checkpoint field to match a retained seam.

## Weight contract

The Qwen3 DSpark binding tree must identify these semantic groups:

- Optional DSpark token embedding
- Target-feature projection and combination
- DSpark decoder layers
- DSpark final norm
- Optional DSpark language-model head
- Markov embedding and projection weights
- Optional draft-to-target vocabulary mapping
- Confidence-head weights when present

The checkpoint decides embedding and language-model-head ownership.
When the checkpoint includes a DSpark-owned weight, DSpark must load and use it.
When the checkpoint omits that weight, initialization may alias the compatible target weight.
The loader must validate shape, dtype, vocabulary, and sharing requirements before it creates the alias.
It must not select ownership per batch.

The first milestone must recognize confidence fields and weights.
It must not materialize or execute confidence-head weights.
The loader must report this limitation clearly.

## Model semantics

### Target features

Qwen3 Main produces the selected decoder-layer outputs.
DSpark consumes only committed target history before the proposal anchor.
Target-feature capture belongs to the Qwen3 model graph.
It must not expose `ReplayRecorder` through the semantic capture contract.

Persistent target context may contain only data produced by target Main.
DSpark proposal-local attention data must never become persistent target context.

### Proposal block

Let `N = block_size`.
The official anchor-as-first-prediction layout has exactly `N` local query rows:

```text
row 0       anchor token
row 1..N-1  MASK tokens
```

Every local query row produces one draft distribution.
The proposal therefore contains exactly `N` draft tokens:

```text
anchor@P
MASK@P+1
...
MASK@P+N-1

produces:

draft@P+1
draft@P+2
...
draft@P+N
```

For `block_size = 7`, the local block contains one anchor and six MASK rows.
It produces seven draft tokens.
It does not contain seven MASK rows.

The first milestone supports only this anchor-as-first-prediction layout.
It does not support a DFlash-compatible `1 + N` query layout.

### Attention

Attention inside the local proposal block is bidirectional.
Each local query can attend to:

- All committed target history before the anchor
- The anchor row
- Every MASK row in the same local block

No local query can attend to unverified target context after the anchor.

The proposal pass computes local K/V for the anchor and all MASK rows.
This K/V is temporary scratch.
It serves only the current bidirectional block.
It must not be committed, retained, or reused by the next proposal.

The newly sampled target token has no target Main K/V in the batch that sampled it.
It becomes the next proposal anchor.
The next DSpark proposal computes that anchor's local K/V as part of the new block.

### Markov correction and sampling

The DSpark body produces one draft logit row for each proposal position.
The Markov head corrects later draft logits from previously sampled draft tokens.
Draft sampling therefore proceeds from left to right across the block.

The first milestone produces these request-major outputs:

```text
draft_tokens[request][position]
draft_probs[request][position]
```

`draft_probs` must contain the sparse distributions required by target rejection.
Confidence output is deferred.

### Target verification

The next target batch validates the proposal with the ordinary target Main and rejection path.
The target input contains the anchor followed by the selected draft prefix.
The rejection result can accept a different number of draft tokens for each request.
The target then samples one new token.
That sampled token becomes the next proposal anchor.

Only target Main can publish persistent target K/V and target features.
Proposal-local K/V is not part of target verification or commit.

## Batch and transaction contract

The first milestone uses a fixed proposal capacity of `block_size` for every active decode request.
The DSpark module batch must preserve request identity, request slot, target-history extent, anchor token, local-block
offsets, sampler configuration, and sample positions.
It must preserve ragged target histories.

The executor must store draft tokens and sparse draft probabilities by request slot.
The next target-rejection batch consumes those values.
The existing generic `SpecProbsStore` contract may be reused when its shape and lifecycle fit the official DSpark
contract.

The Main transaction follows the existing Qwen batch lifecycle:

```text
prepare Main metadata and state
embed_main
forward_main
unembed_main
sample_main
outer submit_main
outer wait
read_main
if the executor has DSpark:
    embed_spec
    forward_spec
    unembed_spec
    sample_spec
    outer submit_spec
    outer wait
    read_spec
commit Main request and model state
publish runtime response
```

The service uses one fixed hook order for prefill, decode, and mixed batches.
The executor omits unembed and sampling components when their input row count is zero.
Empty component input does not change the executor lifecycle.
The static DSpark capability check selects whether the proposal sequence exists.
It is not a per-batch execution-state flag.
The lifecycle does not use `main_stage_submitted` or `read_sampling_output`.

For a prefill-only batch:

```text
MainEmbed -> Main + selected target-feature capture
empty GatherUnembed and sampling input
the DSpark capability branch records no proposal rows
```

For a decode batch with DSpark:

```text
Main batch submission:
    MainEmbed -> Main -> GatherUnembed -> Sampling/RejectionSampling
CPU sampling result
DSpark batch submission:
    DSparkEmbed -> DSpark -> GatherUnembed -> DSparkDraftSampling
CPU draft proposal
```

Main, MTP, and DSpark must each use one batch submission.
Each sequence contains its model body and its optional unembed and sampling suffix.
The service owns each `submit` and `wait`.
Model hooks only record a sequence or read completed CPU-visible output.
DSpark must use the same empty-component behavior as Main and MTP.
This rule does not require a new sequence wrapper or replay-owner type.

### Main and MTP submission evidence

The 2026-07-28 comparison used base commit `3be69962c392d6ae75f33b2bef65e9403b680b30`.
The split source was clean.
The one-sequence candidate had uncommitted executor and documentation changes.
The machine was an Apple M3 Max with 40 GPU cores, macOS 27.0, and `arm64`.
The service used `max_running_requests=8`, seed `42`, and these checkpoints:

- `Qwen3.6-35B-A3B-4bit`
- `Qwen3.6-35B-A3B-MTP-4bit`

The command used `scripts/qwen35_e2e_decode_perf.sh`.
It ran one GPU service at a time.
The split results came from the supplied three-run output.
The one-sequence non-MTP result used a seven-run warm repeat.
The one-sequence MTP result used three runs.

| Case | Tokens | Split median | One-sequence median | Change |
|---|---:|---:|---:|---:|
| `35b_off` | 256 | 93.414 tok/s | 96.395 tok/s | +3.2% |
| `35b_off` | 1024 | 90.801 tok/s | 93.658 tok/s | +3.1% |
| `35b_on` | 256 | 138.262 tok/s | 145.307 tok/s | +5.1% |
| `35b_on` | 1024 | 122.080 tok/s | 127.684 tok/s | +4.6% |

The MTP trajectories were unchanged.
The 256-token case proposed 135 tokens and accepted 121 tokens.
The 1024-token case proposed 591 tokens and accepted 432 tokens.
The checked-in historical baseline used `max_running_requests=4`.
It is not a strict comparison for these results.

Verdict: Main and MTP must not add a submit-and-wait boundary between the model body and the unembed/sampling suffix.
Each model entity keeps one batch submission.

Pushing an executor pending transaction after Main recording is valid under the current synchronous and terminal-failure
model.
Commit remains the publication boundary.
The design does not add a second rollback transaction for replay submission.

## Executor roles

The following list identifies semantic roles.
It is not the final Rust field layout:

```text
Qwen3Executor
  target roles
    main_embed
    main
    gather_unembed
    sampling
    rejection_sampling

  DSpark roles
    dspark_embed
    dspark
    draft_unembed
    draft_sampling

  shared execution state
    request sampling state
    target and draft sparse probability store
    Qwen3 GQA state
    DSpark target-context state
    page arena
    pending target transactions
```

`dspark` means the DSpark model body.
It does not include embedding, unembedding, Markov correction, or sampling.

`Qwen3DSparkEmbed` is a distinct semantic stage even when it aliases the target embedding weight.
It constructs the anchor-and-MASK input block.

The draft-unembed stage may reuse a common unembedding leaf.
Its replay owner may be shared with target GatherUnembed only if the weight and replay contracts fit.
The design must not force this sharing for structural symmetry.

`Qwen3DSparkDraftSampling` owns the DSpark proposal-sampling lifecycle.
It composes Markov correction, sequential sampling, and sparse draft-probability output.
It may reuse generic sampling leaves.
It is not identical to the current MTP `DraftSampling` graph.

Each stage must use the same `Replay<T>` cache pattern as Main, GatherUnembed, MTP, and sampling.
The executor must not add separate “context replay owner” or “proposal replay owner” abstractions.
DSpark context still requires a buffer and state owner.
That data owner is not a second replay framework.

## Executor-core and executor-metal boundary

`inference-executor-core` owns the backend-neutral model contract:

- `Qwen3DSparkConfig`
- Official config parsing and validation
- Exact semantic weight bindings
- Model and proposal dimensions
- Target-layer selection
- DSpark batch and transaction metadata
- Proposal token and sparse-probability contracts
- CPU reference logic
- Backend-neutral shape and lifecycle invariants

`inference-executor-metal` owns the Metal realization:

- Metal buffers and tensor views
- Immutable Metal weight materialization
- DSpark kernels and component adapters
- Scratch and local proposal K/V
- Replay keys, replay caches, and replay materialization
- Replay programs and submission arguments
- Fusion and command ordering
- Kernel resource binding
- Metal profiling and benchmarks

Runtime core continues to own scheduling, request lifecycle, page allocation and ownership, and cache/state
notifications.
It supplies batch metadata and page IDs.
It must not parse DSpark tensor layouts.

## Confirmed tradeoffs

### Independent DSpark model role

Decision: Use `Qwen3DSparkLayer` and DSpark-specific stage owners.

Rejected direction: Extend `Qwen3MainLayer` or reuse `Qwen35MTPLayer`.

Reason: Main, MTP, and DSpark have different attention, input, state, and sampling semantics.

### Fixed block before confidence scheduling

Decision: Implement fixed `block_size` proposals first.

Deferred direction: Variable per-request proposal lengths selected from confidence.

Reason: Confidence scheduling is not required to validate the official DSpark backbone, Markov correction, target
rejection, or executor lifecycle.

### Existing service lifecycle

Decision: Reuse the generic Main and Spec lifecycle hooks.

Rejected direction: Add DSpark-specific service hooks.

Reason: The service lifecycle already provides the target-decision-to-proposal dependency.

### Replay abstraction

Decision: Use the backend-neutral submission lifecycle and the existing executor-side `Replay<T>` composition for the
first milestone.

The common lifecycle exposes submission and wait boundaries.
The service owns stage ordering.
Executor-side adapters materialize and submit backend programs.

Deferred direction: Redesign backend-neutral recorder, operator, replay-program, or fusion APIs before DSpark
integration.

Reason: Model semantics and CPU/GPU lifecycle must be stable before a replay abstraction review.

## Implementation order

1. Add the official `Qwen3DSparkConfig`, validation, and exact weight-binding contract to
   `inference-executor-core`.
2. Remove the retained nested DFlash-era configuration and its compatibility tests.
3. Define `Qwen3DSparkLayer` and its Qwen3-specific Metal components.
4. Define the DSpark block batch, target-feature context, and proposal transaction.
5. Review the exact `Qwen3Executor` fields, target-capture owner, and draft unembedding owner before wiring them.
6. Wire target capture and DSpark proposal through the generic Qwen executor lifecycle.
7. Add checkpoint loading, deterministic component parity, and end-to-end target-rejection validation.
8. Measure performance only after correctness and replay-cache behavior are stable.

The implementation must not start with a backend-neutral replay redesign.

## Deferred work

### Executor ownership and access points

Future work: Review the exact `Qwen3Executor` field layout before implementation step 6.
Resolve these items:

- The target-feature capture and persistent-context owner
- The own-or-shared DSpark embedding owner
- The own-or-shared draft-unembedding owner
- The final `Qwen3DSparkDraftSampling` composition
- The request-slot probability-store layout
- The target-context commit and reset entry points

The review must use the current Qwen3.5 MTP layout as evidence.
It must not make DSpark an MTP variant.

### Confidence head and global scheduling

Future work: Materialize and execute the confidence head after fixed-block DSpark is correct.

The DSpark executor computes confidence per request and proposal position:

```text
confidence[request][position]
```

Runtime scheduling owns the global verification budget and cross-request selection policy.
It may rank proposal positions across all active requests and return a selected proposal length per request.
The executor must not become the global scheduler.

The first milestone must parse and recognize `enable_confidence_head` and related official fields.
It must document that confidence execution and variable-length scheduling are unavailable.

### Backend-neutral replay boundary

Current decision: Runtime core owns only the common execution lifecycle.
`ExecutionSubmission` provides the backend-neutral `wait` operation.
The service calls executor-side `submit_main` and `submit_spec`.
It waits before each CPU read.

Future work: Review the backend-neutral `Runtime` and `Recorder` boundary only after the Qwen3 DSpark lifecycle and
executor ownership are stable.

The review may move common execution lifecycle contracts.
It must keep these items in the Metal backend:

- Metal replay programs
- Command fusion
- Kernel and resource binding
- Metal scratch and residency
- Metal submission arguments

The review must not mirror two recorder/operator APIs only for formal symmetry.
It must not move model semantics into runtime core.
It must not redesign replay before DSpark integration.

### Additional checkpoint variants

Future work: Evaluate non-anchor DSpark or DFlash-compatible `1 + N` layouts only when an official supported checkpoint
requires them.
The first milestone rejects those layouts.

### Overlap and asynchronous recovery

Future work: Revisit overlapping prepared batches only after every shared scratch, output, replay-argument, probability,
and state owner has a bounded in-flight domain.

Recoverable GPU failure remains out of scope unless the runtime adopts a recoverable device-execution contract.

## Verification gates

The first integrated implementation must provide:

- Official config parsing tests
- Strict `target_layer_ids` validation tests
- Exact weight-binding and own-or-shared weight tests
- Anchor-plus-`N-1`-MASK block tests
- Bidirectional local-attention reference parity
- Temporary local-K/V lifecycle tests
- Markov correction and sequential-sampling parity
- Ragged target-rejection tests
- Target-only Qwen3 parity when DSpark is disabled
- Qwen3.5 target and MTP parity
- End-to-end greedy and probabilistic DSpark decoding
- Replay cache hit and replay-key coverage

Performance evidence must follow `docs/executor_benchmarks.md`.
It must separate replay build, normal replay, forced synchronization, and end-to-end wall-clock throughput.
