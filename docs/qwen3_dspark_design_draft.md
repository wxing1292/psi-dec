# Qwen3 DSpark Integration Record

Temporary status: This document records the active Qwen3 DSpark implementation and its design audit.
It is not a stable current-source contract.
Current-component documents must describe the checked-in `src`.

Model roles use `Main`, `MTP`, and `DSpark` or `Spec`.
They do not use `Target` or `Draft`.
The official checkpoint keys `target_layer_ids` and `num_target_layers` keep their upstream names.
The generic rejection-sampling API keeps `target_*` and `draft_*`.
In that API, these names identify probability distributions.
The Metal replay `residual_add::CaptureTarget` uses `Target` to mean a write destination.
The tracing `target:` field identifies a log category.
None of these names identify model roles.

## Milestone status

The active `dev` worktree implements the first fixed-block Qwen3 DSpark milestone.
The semantic, correctness, and performance verification is complete.
The fixed-block path is functionally valid.
It does not improve throughput for the measured Qwen3-14B workload on the tested Apple M3 Max.
This result prevents a performance-benefit claim.
It does not block the fixed-block functional milestone.

The current worktree contains these new areas:

```text
crates/inference-executor-core/src/
  attn/gqa/block_spec_core.rs
  model/qwen/v3_x/dspark/
  bin/qwen3_dspark_quantize.rs

crates/inference-backend-metal/src/components/
  dspark_markov_sampling.rs
  gqa/block_sdpa.rs
  metal/dspark_markov_sampling.metal
  metal/gqa_block_sdpa.metal

crates/inference-executor-metal/src/
  attn/block_spec/*.rs
  model/qwen/v3_x/dspark/
  sampling/dspark_markov.rs
  sampling/rejection_replay.rs
  model/qwen/v3/executor/dspark.rs
```

The Qwen3 executor can load an optional affine DSpark checkpoint.
The Qwen3 service exposes `--hf-dspark-model-dir`.
The service keeps the common Main and Spec lifecycle.

The first milestone has these limits:

- It supports the official anchor-first fixed-block proposal layout.
- It supports ungated Qwen3 GQA.
- It supports the `vanilla` Markov head.
- It does not execute the confidence head.
- It does not schedule variable proposal lengths.
- It permits one in-flight batch for each executor.
- It treats GPU execution failures as terminal internal failures.

The end-to-end Qwen3 path passed greedy and probabilistic decode.
The repository no longer contains the earlier Qwen3.5-era DSpark source.
The current implementation does not keep a compatibility path to that source.

## Confirmed structure

The model roles are independent:

```text
Qwen3MainLayer
Qwen35MTPLayer
Qwen3xDSparkLayer
```

`Qwen3xDSparkLayer` composes generic GQA, RMSNorm, dense MLP, residual, and Metal leaf components.
It does not extend `Qwen3MainLayer`.
It does not add a variant to `Qwen35MTPLayer`.

The Qwen3 executor owns flat semantic stage roles:

```text
Qwen3Executor
  Main
    main_embed
    main
    gather_unembed
    sampling
    rejection_sampling

  DSpark Prefill
    dspark_prefill

  DSpark Decode
    dspark_embed
    dspark
    dspark_gather_unembed
    dspark_sampling

  shared execution data
    Main GQA state
    DSpark GQA state
    request sampling state
    sparse probability store
    Main and DSpark page-table layouts
    BlockSpecScratch
    PageArena
    pending Main transactions
```

`dspark_prefill` is a normal `Replay<Qwen3xDSparkPrefill>` component.
It runs in the Spec submission after the Main CPU read.
This component projects selected Main residuals and appends persistent DSpark context K/V.
This boundary keeps `Qwen3Main` independent from concrete DSpark types.

`dspark` means the DSpark model body.
It does not include embedding, unembedding, Markov correction, or sampling.

## Canonical configuration contract

The repository defines one flat canonical checkpoint schema.
`Qwen3xDSparkConfig` parses and validates only that schema.
The checkpoint boundary selects an adapter from `CHECKPOINT_CONFIG_ADAPTERS` before canonical deserialization.
The `DSparkDraftModel` adapter maps the published nested `dflash_config` fields to the canonical flat fields.
It rejects different flat and nested values.
The canonical parser and executor do not branch on the external architecture name.

The first milestone parses these official fields:

```text
architectures
attention_bias
attention_dropout
block_size
confidence_head_with_markov
dtype
enable_confidence_head
head_dim
hidden_act
hidden_size
intermediate_size
layer_types
markov_head_type
markov_rank
mask_token_id
max_position_embeddings
model_type
num_attention_heads
num_hidden_layers
num_key_value_heads
num_target_layers
quantization
rms_norm_eps
rope_parameters
sliding_window
target_layer_ids
tie_word_embeddings
torch_dtype
use_cache
use_sliding_window
vocab_size
```

The canonical semantic validator requires these values:

- `model_type = "qwen3"`
- `attention_bias = false`
- `attention_dropout = 0`
- `hidden_act = "silu"`
- `layer_types` is empty or contains only `full_attention`
- `markov_head_type = "vanilla"`
- `tie_word_embeddings = false`
- `use_cache = true`
- Sliding-window attention is disabled.
- RoPE uses the unscaled `default` form or Yarn scaling with full-head rotation.
- The execution dtype is BF16.

`target_layer_ids` selects raw decoder-layer outputs.
The first milestone requires these properties:

- The list must not be empty.
- The values must be strictly increasing.
- Each value must satisfy `0 <= layer_id < num_target_layers - 1`.
- The value `-1` is unsupported.
- The final Main decoder layer is unsupported in this milestone.

The loader compares the DSpark config with the Main config.
It validates these shared dimensions:

```text
hidden_size
num_target_layers == Main num_hidden_layers
vocab_size
max_position_embeddings
rope_theta
```

## Weight contract

The Qwen3 DSpark binding tree contains these semantic groups:

```text
optional embed_tokens
Main-feature projection
  fc
  hidden_norm
DSpark decoder layers
  input norm
  ungated GQA
  post-attention norm
  dense MLP
final norm
optional lm_head
Markov head
  markov_w1
  markov_w2
optional confidence head
```

The official source resolver requires an exact BF16 tensor manifest.
It recognizes `confidence_head.proj.weight` and `confidence_head.proj.bias`.

The affine converter creates an exact executor checkpoint.
It preserves DSpark-owned `embed_tokens` and `lm_head` when the source provides them.
It omits confidence-head tensors for the first milestone.

The executor uses a DSpark-owned embedding or language-model head when the affine checkpoint provides it.
The executor aliases the compatible Main component when the DSpark checkpoint omits that complete group.
The loader selects ownership once during initialization.

The loader must reject a partial optional affine group.
It must reject extra affine tensors.

## Model semantics

### Main features

Qwen3 Main captures the raw output of each selected decoder layer.
The capture owner preserves `target_layer_ids` order.
Each selected layer writes directly into its assigned hidden-dimension columns.
The implementation does not use a concatenate kernel or copy.

The DSpark Main-feature stage applies this computation:

```text
selected Main residuals
  -> fc
  -> hidden_norm
  -> projected Main feature
```

Each DSpark layer projects the same normalized Main feature through its K and V weights.
It applies K RMSNorm and RoPE.
It appends the result to that DSpark layer's persistent context pages.

The Main submission contains this order:

```text
MainEmbed
  -> Main
       selected residual capture
```

The Spec Prefill invocation contains this order:

```text
DSparkPrefill
  -> Main-feature projection
  -> per-layer context K/V append
```

The capture interface returns only an opaque `residual_add::CaptureTarget`.
It does not expose `ReplayRecorder`.
`Qwen3Main` does not reference `Qwen3xDSparkModel`.

The context append runs for prefill, decode, and verification rows.
Candidate verification rows can reach persistent pages before CPU rejection completes.
The accepted request extent controls later visibility.
The next Main batch overwrites rejected positions.
The design does not require a GPU rollback copy.

### Proposal block

Let `N = block_size`.
One request has exactly `N` local query rows:

```text
row 0       anchor token
row 1..N-1  MASK tokens
```

These rows produce exactly `N` proposal distributions:

```text
anchor@P       -> draft@P+1
MASK@P+1      -> draft@P+2
...
MASK@P+N-1    -> draft@P+N
```

For `block_size = 7`, the local block contains one anchor and six MASK rows.
It produces seven proposal tokens.

The first milestone does not support a `1 + N` local-query layout.

### Attention

The official Qwen3 DSpark checkpoint uses ungated GQA.
It does not use MLA.
It does not use an attention-output gate.

The known Qwen3-14B checkpoint has this geometry:

```text
hidden_size          = 5120
num_attention_heads  = 40
num_key_value_heads  = 8
head_dim             = 128
Q heads per KV head  = 5
```

The executor derives all values from the checkpoint.

`BlockSpecGQA` owns two K/V domains:

```text
persistent history K/V  runtime pages
proposal-local K/V      BlockSpecScratch
```

The attention component records this sequence:

```text
history_causal_sdpa_map = split_kv_single_q.invoke_map(...)
block_bidi_sdpa_map = block_sdpa.invoke(...)
sdpa_reduce = split_kv_single_q.invoke_reduce(...)
```

The paged map processes accepted history in `[0, anchor_position)`.
CPU metadata defines this causal extent for every local query.

`gqa::block_sdpa::Compute` processes one request-local block.
Every local query sees the anchor and all MASK positions in the same block.
It cannot see another request's block.

Each history task and block task writes one online-softmax partial:

```text
partial maximum
partial exponential sum
normalized partial output
```

The existing `gqa::split_kv::single_q::ReduceInvocation` combines all partials.
The block kernel does not accept a configurable mask.

The proposal pass computes local Q/K/V for the current block.
These tensors are temporary.
The executor must not commit, retain, or reuse them.

The newly sampled Main token has no Main K/V in the batch that sampled it.
It becomes the next proposal anchor.
The DSpark block computes that anchor's local K/V.

### Markov correction

The DSpark body produces one base logit row for each proposal position.
The vanilla Markov head applies this correction:

```text
bias(previous_token) = markov_w2(markov_w1(previous_token))
corrected_logits = base_logits + bias(previous_token)
```

Sampling proceeds from left to right:

```text
step 0 uses the Main sampled anchor
step i uses the proposal token sampled at step i - 1
```

Markov correction and sampling form one `Qwen3xDSparkSampling` component.
The implementation does not introduce a CPU loop or a submit boundary between steps.

### Main verification

The next Main batch validates the proposal through ordinary Main and sparse rejection.
Each request can accept a different proposal prefix.
Rejection then samples one continuation token.
That token becomes the next DSpark anchor.

Only Main publishes persistent Main K/V and Main features.
The DSpark local block never becomes persistent history.

## Distribution ownership

Draft and Main verification distributions have intentionally different identities.
They must not use one symmetric indexing rule.

Draft distributions cross a batch boundary:

```text
Spec batch writes proposal distributions
next Main batch reads proposal distributions
```

The store indexes draft distributions by:

```text
request_slot * block_size + proposal_position
```

This identity remains stable when a later batch contains different request order or non-contiguous request slots.

Main verification distributions are temporary within one Main submission:

```text
GatherUnembed compact rows
  -> sparse Main distributions
  -> sparse rejection
```

The store writes these rows with compact identity indices:

```text
0, 1, 2, ..., num_active_target_distributions - 1
```

`cu_target_distributions` uses the same compact row domain.
It must not use request-slot indices.

This asymmetry follows data lifetime.
It is not an optimization exception.

## Page ownership

Main K/V and persistent DSpark context K/V share one runtime cache lane.
The runtime core allocates one flat page-ID list for each logical cache block:

```text
block_page_ids:
  [0 .. main_page_count)           Main K/V
  [main_page_count .. total_count) DSpark context K/V
```

The Qwen3 batch adapter splits this list.
It updates separate Main and DSpark page tables.
The runtime core does not parse either table.

The service calculates:

```text
num_pages_per_kv_block =
    main_pages_per_block
  + dspark_pages_per_block
```

The runtime core allocates, caches, restores, evicts, and releases both spans together.
DSpark-disabled Qwen3 retains its Main-only page count.

## Scratch ownership

`BlockSpecScratch` is an executor-owned fixed-capacity resource.
It contains:

```text
local Q/K/V
Q/K norm and RoPE outputs
history and block attention partials
reduced attention output
```

The runtime page arena does not own this scratch.
The scheduler does not allocate it for each request.

Define:

```text
T_capacity = max_requests * block_size
P_capacity = next_power_of_two(2 * T_capacity)
```

The first milestone uses single-Q history map tasks.
Each local query requires at least one history partial and one block partial.
The metadata builder can divide long history across spare history tasks.
It cannot exceed `P_capacity`.

The scratch allocates:

```text
partial_max_logits[P_capacity, num_q_heads]       f32
partial_exp_sums[P_capacity, num_q_heads]         f32
partial_output[P_capacity, num_q_heads, head_dim] model dtype
```

Context length does not set this capacity.
One history task can process many K/V tiles with online softmax.

## Execution lifecycle

The service owns `submit` and `wait`.
Model components only record work or read completed output.

The common outer flow is:

```text
model_batch_req = prepare_batch(batch)
recorder = begin_ops_recording(model_batch_req)

main_hidden = embed_main(recorder, model_batch_req)
main_hidden = forward_main(recorder, model_batch_req, main_hidden)
main_response = unembed_main(recorder, model_batch_req, main_hidden)
sample_main(recorder, model_batch_req, main_response)

main_submission = submit_main(recorder)
main_submission.wait()
sampled_output = read_main(recorder, model_batch_req)

if run_spec_prefill(model_batch_req):
    prefill_spec(recorder, model_batch_req)
if run_spec_decode(model_batch_req, sampled_output):
    decode_spec(recorder, model_batch_req, sampled_output)
if Spec Prefill or Spec Decode was recorded:
    spec_submission = submit_spec(recorder)
    spec_submission.wait()
if Spec Decode was recorded:
    sampled_output = read_spec(recorder, model_batch_req, sampled_output)

response = commit_batch(batch, sampled_output)
```

The Main submission is:

```text
MainEmbed
  -> Main
  -> GatherUnembed                         when sample rows exist
  -> Sampling or RejectionSampling         when sample rows exist
```

The prefill-only DSpark Spec submission is:

```text
DSparkPrefill
```

The decode-ready DSpark Spec submission is:

```text
DSparkPrefill
  -> DSparkEmbed
  -> DSpark
  -> DSparkGatherUnembed
  -> DSparkSampling
```

The CPU must read Main output before it can construct the DSpark anchor block.
This dependency requires two submissions.
There is no submit boundary inside Main or inside DSpark Spec.

Prefill, decode, and mixed batches use the same Main hook order.
An empty unembed or sampling stage records no component.

`run_spec_prefill` is true when both conditions are true:

```text
the executor has DSpark
Main produced at least one capture row
```

`run_spec_decode` also requires at least one Main decode result.
A prefill-only batch therefore records Spec Prefill and omits Spec Decode.
These conditions are semantic data-availability gates.
They are not execution-state flags.

The lifecycle does not use `main_stage_submitted`.
It does not use `read_sampling_output`.
It does not return a dummy completed submission.

The executor pushes one pending Main transaction after Main recording.
It commits that transaction after the optional Spec submission completes.
The current synchronous terminal-failure model does not require a second rollback transaction.

## Core and Metal boundary

`inference-executor-core` owns backend-neutral contracts:

- Official config parsing and validation
- Exact semantic weight bindings
- GQA geometry and block capacity
- Proposal metadata
- Speculative microbatch access
- Proposal token and probability contracts
- Backend-neutral lifecycle invariants

`inference-backend-metal` owns reusable Metal kernels:

- Paged GQA map and reduce
- `gqa::block_sdpa::Compute`
- Kernel validation
- Metal resource binding
- Dispatch

`inference-executor-metal` owns the model realization:

- Weight buffers and tensor views
- `BlockSpecGQA`
- Persistent DSpark page interpretation
- `BlockSpecScratch`
- `Qwen3xDSparkLayer`
- Main-feature projection and context append
- Markov sampling
- Replay keys and replay caches
- Replay sequence composition
- Metal profiling and benchmarks

The runtime core owns:

- Scheduling
- Request lifecycle
- Token and block metadata
- Page allocation and ownership
- Cache lifecycle notifications

The runtime core must not parse DSpark tensor layout.
The executor must not implement global scheduling policy.

## Design audit

The first-principles audit used these questions:

| Question                                                   | Result                                                                |
| ---------------------------------------------------------- | --------------------------------------------------------------------- |
| Does each persistent value follow accepted Main history?   | Yes. Main K/V and DSpark context K/V share one cache-block lifecycle. |
| Can proposal-local state escape its batch?                 | No. Local Q/K/V and attention partials live in `BlockSpecScratch`.  |
| Does every CPU dependency create one clear wait boundary?  | Yes. Main output is read before Spec block construction.              |
| Does any component submit or wait internally?              | No. The service owns both boundaries.                                 |
| Does Main depend on a concrete DSpark model?               | No. Main exposes only the residual-capture seam.                      |
| Does Spec use the official anchor-first layout?            | Yes. `N` rows produce `N` proposal tokens.                            |
| Can ragged requests keep identity across batches?          | Yes. Draft distributions use request-slot identity.                   |
| Can sparse rejection read compact Main rows correctly?     | Yes. Main verification distributions use compact identity indices.    |
| Does context length increase static attention scratch?     | No. Metadata divides history across a fixed task capacity.            |
| Does replay caching include all command-topology inputs?   | Yes. Keys include active row counts and SDPA task capacity.           |
| Does a prefill-only batch run DSpark Decode?               | No. It records Spec Prefill without a sampled anchor.                 |
| Does the design require a backend-neutral replay redesign? | No. Existing lifecycle and Metal replay composition are sufficient.   |

No unresolved design item blocks the fixed-block milestone.
The remaining items are implementation verification or explicitly deferred features.

## Confirmed tradeoffs

### Independent DSpark role

Decision: Use `Qwen3xDSparkLayer` and DSpark-specific stage owners.

Rejected direction: Extend `Qwen3MainLayer` or reuse `Qwen35MTPLayer`.

Reason: Main, MTP, and DSpark have different attention, input, state, and sampling semantics.

### Shared cache-block lifecycle

Decision: Allocate Main K/V and DSpark context K/V in one logical cache block.

Rejected direction: Add a second DSpark cache lane.

Reason: Both persistent values follow accepted Main history.

### Dedicated block attention

Decision: Use the paged map for history and `gqa::block_sdpa::Compute` for local bidirectional attention.

Rejected direction: Add a configurable mask to the paged kernel.

Reason: History pages and local dense scratch have different storage and visibility contracts.

### Fixed block before confidence scheduling

Decision: Implement `block_size` proposal tokens for every active decode request.

Deferred direction: Use confidence to select variable proposal lengths.

Reason: Confidence scheduling is not required to validate the DSpark backbone, Markov correction, rejection, or
lifecycle.

### Existing replay boundary

Decision: Use the common submission lifecycle and executor-side `Replay<T>` composition.

Deferred direction: Redesign backend-neutral recorder and operator APIs before DSpark.

Reason: The current boundary expresses all required CPU and GPU dependencies.

## Deferred work

### Follow-up order

Use this order for the marked follow-up work:

| Order | Work                                       | Completion condition                                                                                                                           |
| ----: | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
|     1 | Main multi-row verification investigation  | Separate Main body, GatherUnembed, and sparse rejection evidence without adding a submission boundary.                                         |
|     2 | Deterministic DSpark end-to-end validation | Re-run throughput, proposal count, accepted-token count, acceptance efficiency, and stage timing after the retained proposal and Main changes. |
|     3 | Confidence and global scheduling           | Execute the confidence head first. Add variable proposal lengths only when runtime scheduling owns the cross-request budget.                   |
|     4 | Checkpoint-triggered DSpark variants       | Add gated attention or other layout/head variants only for a real supported checkpoint.                                                        |
|     5 | Replay and overlap evolution               | Review these boundaries only after the fixed-block lifecycle and all in-flight owners are stable.                                              |

Items in a later row must not block an earlier row unless new correctness evidence identifies a dependency.

### Main multi-row verification and end-to-end validation

Future work: Continue the Main multi-row verification investigation.

Keep one Main submission for each batch.
Do not add a submit-and-wait boundary between the Main body and its GatherUnembed/rejection suffix.
Measure these regions separately:

```text
Main body
GatherUnembed
sparse rejection
```

The investigation must compare the same batch shape and deterministic token trajectory.
It must report proposal count, accepted-token count, acceptance efficiency, and stage timing.
After a retained change, re-run the deterministic service comparison.
Do not replace the current deployment verdict with isolated component timing.

### GDN adaptive affine ownership

Completed: GDN uses the adaptive affine owner for both projections.

The GDN `qkvabz` projection uses BF16 input, BF16 affine parameters, and F32 output.
The GDN output projection uses F32 input, BF16 affine parameters, and BF16 output.
Both projections perform the boundary conversion within `affine_quantized::Matmul`. GDN does not record a separate
buffer cast.

`affine_quantized::Config` must provide these fixed workload facts:

```text
n
k
group_size
bits
input_dtype
scale_bias_dtype
output_dtype
```

`affine_quantized::Invocation` must provide `m`, buffers, and byte offsets.
Metal buffers do not carry a tensor dtype.
The model executor must provide each dtype.
It must not provide a QMV/QMM family or tile selection.
The backend derives the supported kernel set from the dtype signature.
It must select the kernel family and tile from the complete workload facts.

The shared adaptive affine owner supports all 27 combinations of F32, F16, and BF16 input, scale/bias, and output
data types.
QMV BN8/BK32 and QMM BM8/BN32, BM16/BN32, and BM32/BN32 provide this capability.
QMV Quad BN64 remains a same-dtype specialization for its supported shapes.
The adaptive owner falls back to QMV BN8/BK32 when QMV Quad BN64 is not valid.
Do not add an executor-visible mixed-dtype mode.
Do not weaken GDN precision.

The backend extension and the GDN migration remain separate reviewable changes.
The backend extension includes exhaustive dtype-combination reference coverage and exact-path benchmark controls.
The GDN migration removes its manual QMV/QMM selection.

The gather and ragged expert-affine templates currently require the input, scale/bias, and output data types to match.
Their config still provides the complete dtype facts.
The backend must reject unsupported mixed-dtype expert configurations during initialization.
Future work: Generalize these templates only when a supported model requires a mixed-dtype expert projection.
Do not add an executor-visible mixed-dtype mode.

An alternating full-replay benchmark compared the retained manual QMV/BM32 policy with adaptive selection.
Both runs used one request, context length 0, 30 warmup iterations, 100 measured iterations, and five runs.

| Rows | Manual QMV/BM32 |   Adaptive | Change |
| ---: | --------------: | ---------: | -----: |
|    1 |      324.055 us | 323.548 us |  -0.2% |
|    6 |      463.776 us | 426.542 us |  -8.0% |
|    8 |      520.372 us | 443.192 us | -14.8% |
|   10 |      576.680 us | 515.592 us | -10.6% |
|   12 |      660.710 us | 575.605 us | -12.9% |
|   16 |      707.483 us | 575.712 us | -18.6% |
|   18 |      733.124 us | 716.686 us |  -2.2% |
|   32 |      809.142 us | 813.413 us |  +0.5% |

Rows 1, 18, and 32 use the same kernel family in both policies and remain within run-to-run variance.
Rows 6 through 16 benefit from the BM8/BN32 and BM16/BN32 candidates.

The fused DSpark Markov map combines W1 lookup, W2 projection, corrected-logit addition, and vocabulary-tile Top-K.
Do not replace its W2 stage mechanically with a standalone affine dispatch.
Treat this fused map as a separate backend operator.
Compare alternative map geometry and W2 implementations inside the complete Markov replay.
Benchmark-only code may force an exact implementation for comparison.
Production model and executor APIs must not expose that force control.

### DSpark Markov numerical contract

Decision: Retain the current BF16 corrected-logit contract.

The retained path is:

```text
BF16 latent -> F32 W2 accumulation -> BF16 correction
  -> F32 add -> BF16 corrected logit -> F32 Top-K
```

The rejected candidate was:

```text
BF16 latent -> F32 W2 accumulation
  -> F32 add -> F32 Top-K
```

The candidate had no isolated Markov replay performance gain.
It changed sparse draft probabilities and the deterministic end-to-end acceptance trajectory.
The current contract also matches the Qwen3 tensor-dtype behavior in vLLM and SGLang.
[`executor_sampling.md`](executor_sampling.md) records the upstream, resource, correctness, performance, and
end-to-end evidence.

### Confidence and global scheduling

Future work: Materialize and execute the official confidence head.

The DSpark executor will produce:

```text
confidence[request][proposal_position]
```

Runtime scheduling owns the global verification budget.
It can rank proposals across active requests.
It returns the selected proposal length for each request.

The executor must not become the global scheduler.

### Gated DSpark GQA

Future work: Add a separate `GatedBlockSpecGQA` when a supported checkpoint requires it.

The implementation must not add a `gated` runtime flag to `BlockSpecGQA`.
The history map, block map, reducer, page layout, and scratch contracts must remain gate-neutral.

### Backend-neutral replay boundary

Future work: Review the backend-neutral `Runtime` and `Recorder` boundary after the DSpark lifecycle is stable.

The review may move common execution lifecycle contracts.
It must keep these items in the Metal backend:

- Metal replay programs
- Fusion
- Kernel resource binding
- Metal scratch and residency
- Metal submission arguments

The review must not mirror two recorder/operator APIs for formal symmetry.

### Additional checkpoint variants

Future work: Support `-1`, the final Main layer, non-anchor layouts, or other Markov heads only when a supported
checkpoint requires them.

### Overlap and recovery

Future work: Permit overlapping batches only after every scratch, output, replay-argument, probability, and state owner
has a bounded in-flight domain.

Recoverable GPU failure remains out of scope.

## Verification gates

The fixed-block milestone requires:

- Official config parsing tests
- Strict `target_layer_ids` tests
- Exact source and affine binding tests
- Real official-to-affine conversion
- DSpark-owned embedding and unembedding load
- Anchor plus `N - 1` MASK tests
- Bidirectional block-attention Metal parity
- Combined history and block reduction parity
- Flat Main and DSpark page split tests
- Static partial-scratch capacity tests
- Markov sequential-sampling parity
- Ragged sparse-rejection tests
- Non-contiguous request-slot rejection coverage
- Main-only Qwen3 regression tests
- Qwen3.5 Main and MTP regression tests
- End-to-end greedy DSpark decode
- End-to-end probabilistic DSpark decode
- Replay cache and replay-key coverage
- Standalone `gqa_block_attn` block-bidirectional component benchmark
- Real-checkpoint `qwen3_dspark` Main/DSpark executor benchmark
- End-to-end performance evidence

Performance evidence must follow `docs/executor_benchmarks.md`.
It must separate replay build, normal replay, forced synchronization, and end-to-end wall-clock throughput.

## Qwen3 DSpark end-to-end evidence

The 2026-07-29 comparison used base commit `91b65fbc8f98cb75ee29a6c1765ef17ce6a10192`.
The final verification worktree had 92 changed or untracked paths.
The implementation was not committed.
The machine was an Apple M3 Max with 40 GPU cores and 48 GB of memory.
It used macOS 27.0 build `26A5388g` on `arm64`.

The comparison used these checkpoints:

- Main: `/Users/wenquanxing/Workspace/models/Qwen3-14B-4bit`
- DSpark affine: `/tmp/qwen3-dspark-affine-e2e`

The affine checkpoint came from the official Qwen3 DSpark source checkpoint.
It contains 142 tensors.
It owns its embedding and unembedding weights.
It omits the deferred confidence head.

Each service used these arguments:

```text
--num-cache-pages 4096
--max-requests 4
--max-tokens 128
--max-tokens-per-request 64
```

The Main-only service used 80 K/V pages for each cache block.
It had 51 cache blocks.
The DSpark service used 80 Main pages and 10 DSpark pages for each cache block.
It had 45 cache blocks.
Only one GPU service ran at a time.

The decode command was:

```sh
target/release/decode \
  --server-url http://127.0.0.1:50151 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --prompt-str 'Explain in concise technical terms why the sky appears blue during the day.' \
  --disable-thinking \
  --max-sampled-tokens 256 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 42 \
  --no-output-str \
  --show-stats
```

Both paths generated the same 98-token deterministic trajectory and then reached EOS.
The first sample warmed the replay and device state.
The median uses the last three samples.

| Path      | Stable samples               |       Median | Output chunks |
| --------- | ---------------------------- | -----------: | ------------: |
| Main-only | 41.180, 41.187, 41.152 tok/s | 41.180 tok/s |            98 |
| DSpark    | 33.611, 33.581, 33.655 tok/s | 33.611 tok/s |            34 |

The DSpark path was `18.4%` slower than Main-only.
It submitted 33 Spec verification batches after the initial Main token.
It proposed 231 tokens and accepted 64 proposal tokens.
The proposal acceptance rate was `27.7%`.
Each output chunk contained an average of `2.88` tokens.

The public executor benchmark reproduced the same cost direction.
It used base commit `91b65fbc8f98cb75ee29a6c1765ef17ce6a10192` with 100 changed or untracked paths.
It ran on the same Apple M3 Max.
No other GPU command ran concurrently.
The Main checkpoint was `/Users/wenquanxing/Workspace/models/Qwen3-14B-4bit`.
The DSpark checkpoint was
`/Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine`.

The command was:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --cases main,dspark \
  --num-requests 1 \
  --warmup-iters 2 \
  --iters 10 \
  --runs 3
```

The median phase times were:

| Case and phase                                   |     Median |
| ------------------------------------------------ | ---------: |
| Main-only, one-token full Main submission        |  27.345 ms |
| DSpark, eight-token Main verification submission |  86.922 ms |
| DSpark proposal submission                       |  14.436 ms |
| DSpark complete cycle                            | 101.193 ms |

Steady record, read, and commit work contributed less than `0.04 ms` to one DSpark cycle.
The Main verification submission contributed approximately `85.9%` of the cycle.
The DSpark proposal submission contributed approximately `14.3%`.

A matching synthetic sparse-rejection benchmark used one request, seven proposal tokens, `top_k=1`, and
`vocab=151936`.
Its median was `0.351 ms`.
Therefore, the sparse-rejection kernel is not the source of the `86.922 ms` Main verification time.

The end-to-end run accepted an average of `64 / 33 = 1.94` proposal tokens for each verification batch.
Including the continuation token, one verification cycle committed an average of `2.94` tokens.
At the executor benchmark phase times, DSpark needs approximately `2.70` accepted proposal tokens per cycle to match
the Main-only rate.
This value is a `38.6%` proposal-token acceptance rate.
The measured `27.7%` rate is below that break-even point.

Even a zero-cost DSpark proposal would leave the measured Main verification submission at `86.922 ms`.
At the measured acceptance rate, that limit is approximately `33.8 tok/s`.
The corresponding Main-only executor benchmark rate is approximately `36.5 tok/s`.
Thus, proposal-only optimization cannot recover the regression.
The implementation needs higher acceptance and lower multi-row Main verification cost.

An 8-bit DSpark affine checkpoint tested whether 4-bit quantization caused the low acceptance rate.
The 8-bit checkpoint was 3.4 GB.
It produced the same 98 tokens, 34 output chunks, and proposal acceptance trajectory.
Its stable samples were 31.900, 31.917, and 31.964 tok/s.
Its median was `31.917 tok/s`.
The 8-bit setup was `5.0%` slower than the 4-bit DSpark setup and did not improve acceptance.

A separate 64-token greedy check produced the same decoded text for Main-only and DSpark.
Probabilistic DSpark decode also completed successfully.
The probabilistic Main-only and DSpark trajectories differed after seeded sampling.
Therefore, that result is not used as a pure throughput comparison.

Verdict: The end-to-end semantic path passes.
The performance verification is complete, and this setup has a negative deployment verdict.
Profiling attributes approximately 75.3 ms of each stable 8-row batch to Main verification and rejection.
It attributes approximately 12.7 ms to the complete DSpark proposal submission.
The low acceptance rate makes Main verification the dominant cost.

### Proposal stage breakdown

The proposal execution order is fixed:

```text
DSparkEmbed
  -> five-layer DSpark forward
  -> final norm
  -> GatherUnembed
  -> sequential Markov correction and sampling
```

The forward stage does not include `GatherUnembed`.
The `GatherUnembed` stage does not include Markov correction or sampling.

The 2026-07-29 stage comparison used base commit `a3ba1bfc01e1b5aaf6ae7355cd52b66e1dc188f5`.
The worktree was dirty.
It contained 11 changed or untracked paths before this evidence section changed.
The machine was an Apple M3 Max with 40 GPU cores and 48 GB of memory.
It used macOS 27.0 build `26A5388g` on `arm64`.
No other GPU command ran concurrently.

The forward command was:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_forward -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 \
  --context 128 \
  --warmup-iters 5 \
  --iters 20 \
  --runs 3
```

The Main case recorded `MainEmbed`, all 40 Main layers, and final norm for seven prefill rows.
Its empty `GatherUnembed` and sampling stages recorded no replay.
The DSpark case recorded `DSparkEmbed`, all five DSpark layers, and final norm for seven proposal rows.
Both cases measured submit and wait only.

| Forward case |     Total | Per layer |
| ------------ | --------: | --------: |
| Main         | 73.547 ms |  1.839 ms |
| DSpark       |  9.809 ms |  1.962 ms |

The DSpark forward took `13.3%` of the complete Main forward time.
The DSpark per-layer time was `6.7%` higher.
Thus, the DSpark backbone cost follows its five-layer depth.
It is not the abnormal part of the proposal cost.

### Block-bidirectional attention geometry

A 2026-07-29 follow-up used base commit `9fc8a875d65c7d909735d78c433a7d976e4ea015`.
The worktree was dirty with the DSpark sampling optimization and this forward investigation.
The machine was an Apple M3 Max with 40 GPU cores and 48 GB of memory.
It used macOS 27.0 build `26A5388g` on `arm64`.
No other GPU command ran concurrently.

The original block kernel used one threadblock for each Q token and Q head.
It used 128 threads.
Seven threads computed the seven Q/K dot products serially across `head_dim = 128`.
The remaining threads waited for the reduction and then produced the output dimensions.

The thread-count sweep for that original kernel was:

| Threads | Shared memory |     Median |
| ------: | ------------: | ---------: |
|      32 |     156 bytes | 290.676 µs |
|      64 |     284 bytes | 287.702 µs |
|     128 |     540 bytes | 283.614 µs |
|     256 |   1,052 bytes | 288.918 µs |

The 128-thread case was the best original configuration.
It gave one thread to each output dimension.
The 256-thread case left half of the threads idle during output generation.
The 32-thread and 64-thread cases required four and two output iterations per thread.

A rejected alternative grouped all five Q heads for one KV head into one threadblock.
It reduced the grid from 280 threadblocks to 56 threadblocks.
Its 32/64/128/256-thread medians were `318.279`, `314.051`, `305.660`, and `298.941 µs`.
The smaller grid reduced device-level parallelism.
This alternative is not retained.

The retained kernel keeps one Q-token/Q-head Task per threadblock.
It uses one 32-thread SIMDgroup.
Each thread keeps four F32 Q values, for 16 bytes of logical Q register payload.
The threadblock keeps seven F32 logits in 28 bytes of shared memory.
The SIMDgroup computes each Q/K dot product across 32 lanes.
Each lane computes four output dimensions.

The retained component median was `277.502 µs`.
The original component median was `283.614 µs`.
The component improved by `2.2%`.

Commit `d99ca7913eb483083a8e265a29f1054ba1e56a9f` was clean.
A later operability run used this command:

```sh
cargo bench -p inference-backend-metal --bench gqa_block_attn -- \
  --block-sizes 7 --num-requests 1 \
  --num-q-heads 40 --num-kv-heads 8 --head-dim 128 --dtypes bf16 \
  --max-q-tokens 1 \
  --warmup-iters 20 --iters 100 --runs 7
```

It used the same machine and measured `296.311 µs`.
That run did not include a same-process baseline.
Device-frequency state differed from the earlier paired comparison.
Therefore, the later run does not replace the paired `2.2%` verdict.

The forward benchmark now reports DSpark embed, forward body, and combined embed-forward separately.
For one request, seven rows, and context 128, the retained forward-body median was `9.268 ms`.
The immediately preceding forward-body median was `9.301 ms`.
This is a `0.35%` stage improvement.
The change does not establish an end-to-end throughput gain.

The production `GatherUnembed` command was:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_unembedding -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 \
  --warmup-iters 20 \
  --iters 100 \
  --runs 7
```

This DSpark checkpoint does not contain `lm_head`.
The fixture therefore used the Main unembed weights, as the production executor does.
The shape was `7 x 5120 -> 7 x 151936`.
The fixture recorded the production `Qwen3xDSparkGatherUnembed` component.
It included row gather and unembed.
It did not include DSpark forward, Markov correction, or sampling.

An exact production-path A/B comparison changed only the QMV-to-QMM crossover during the baseline run.
The retained policy selects QMM BM16/BN32 for this seven-row shape.

| `GatherUnembed` policy |   Median |
| ---------------------- | -------: |
| QMV baseline           | 4.166 ms |
| QMM BM16/BN32          | 3.056 ms |

QMM BM16/BN32 reduced this stage by `26.6%`.
The backend kernel has a CPU-reference correctness test for Q4 BF16 input.
The production unembed owner requires BF16 input, affine parameters, and output.
The selector uses QMM BM16/BN32 for this large-vocabulary shape through 16 rows.
It uses QMM BM32/BN32 above 16 rows.
It retains the general policy for smaller-output shapes.

The row-count policy used representative DSpark batch shapes:

| Rows | QMM BM16/BN32 | QMM BM32/BN32 | Selected      |
| ---: | ------------: | ------------: | ------------- |
|   14 |      2.938 ms |      5.443 ms | QMM BM16/BN32 |
|   16 |      3.431 ms |      5.674 ms | QMM BM16/BN32 |
|   21 |      5.741 ms |      5.275 ms | QMM BM32/BN32 |
|   28 |      6.101 ms |      5.761 ms | QMM BM32/BN32 |

The Markov and sampling command was:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_sampling -- \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 \
  --top-k 1 \
  --warmup-iters 10 \
  --iters 50 \
  --runs 5
```

The production component used `block_size=7`, `vocab_size=151936`, and `markov_rank=256`.
It measured the complete sequential Markov correction, sampling, and write-distribution replay.
The earlier top-k 1 measurement had a `1.398 ms` median.

The optimized Markov path records two commands for each proposal position:

```text
fused W1 -> W2 -> base-logit add -> 64-token tile Top-K
  -> generic global Top-K/top-p/sample/write-distribution reducer
```

The seven-position block uses 14 commands in one Spec submission.
It does not materialize full-vocabulary bias or corrected-logit tensors.
The CPU reference preserves the earlier BF16 rounding points.
The Metal parity test covers `3` active requests in a replay bucket of `4`, a maximum capacity of `6`, and
non-contiguous request slots.

For the official `markov_rank = 256` shape, the retained fused map uses 128 threads and 1,024 bytes of shared memory per
threadblock.
Each thread holds eight F32 latent values.
It uses one sequential W2 dot accumulator.
The 128-thread threadblock contains four SIMDgroups.
Metal does not expose register allocation through `MTLComputePipelineState`.
The tile choice therefore used both the source-level live-value count and real-weight measurements.

The real-weight tile sweep used the same Qwen3-14B DSpark checkpoint and one request:

| Map tile, threads, and implementation                              | Complete Markov sampling median |
| ------------------------------------------------------------------ | ------------------------------: |
| Original five-command step                                         |                     1.729050 ms |
| 256-token tile, 256 threads, before affine reuse                   |                     2.499804 ms |
| 64-token tile, 256 threads, before affine reuse                    |                     2.081029 ms |
| 32-token tile, 256 threads, before affine reuse                    |                     2.399168 ms |
| 64-token tile, 256 threads, with per-lane affine reuse, tuning run |                     1.419841 ms |
| 64-token tile, 256 threads, with per-lane affine reuse, later run  |                     1.478628 ms |
| 64-token tile, 64 threads, with per-lane affine reuse              |                     1.397572 ms |
| 64-token tile, 128 threads, geometry sweep                         |                     1.393562 ms |
| 64-token tile, 128 threads, final current run A                    |                     1.461522 ms |
| 64-token tile, 128 threads, final current run B                    |                     1.389819 ms |

The 32-token tile doubled the threadblock count and repeated W1 and latent setup.
The smaller tile did not recover enough occupancy to offset that work.
The 64-thread and 128-thread cases are within `0.3%`.
The retained 128-thread case uses four SIMDgroups and four W2 waves for each tile.
The two final current runs reduced the complete Markov replay median by `15.5%` and `19.6%` from the original path.
This range reflects the observed device-frequency variation.
This result is a proposal-stage improvement.
It does not change the earlier end-to-end acceptance verdict.

The final comparison used commit `9fc8a875d65c7d909735d78c433a7d976e4ea015`.
The baseline was clean.
The current worktree was dirty with only this Markov optimization.
Both cases used macOS 27.0 on an Apple M3 Max and this command:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_sampling -- \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 --top-k 20 \
  --warmup-iters 20 --iters 100 --runs 7
```

The baseline median was `1.729050 ms` per complete seven-step replay.
The two final current medians were `1.461522 ms` and `1.389819 ms`.

Do not add the three isolated stage times and compare the sum with one complete proposal submission.
Each isolated target has its own submit-and-wait boundary.
The measurements also ran under different device-frequency states.
The complete executor submits the three stages as one replay sequence.

A compiled full-proposal A/B normalized proposal time against the Main time from the same process:

| Full executor build |      Main |  Proposal | Proposal/Main |
| ------------------- | --------: | --------: | ------------: |
| QMM BM16/BN32 A     | 86.368 ms | 13.607 ms |       0.15755 |
| QMV baseline        | 85.686 ms | 14.383 ms |       0.16786 |
| QMM BM16/BN32 B     | 79.222 ms | 12.520 ms |       0.15804 |

The absolute times changed with device frequency.
The normalized proposal ratio improved by `5.8%` to `6.1%`.
This result confirms a complete-proposal gain.
It does not change the earlier deployment verdict because Main multi-row verification remains the dominant cost.

The earlier dense-MLP diagnostic compared QMV with BM32/BN32 QMM.
It did not test a QMM row tile that matched the eight-row verification shape.
Therefore, its conclusion that QMV was optimal was incomplete.

A 2026-07-29 follow-up used base commit `9fc8a875d65c7d909735d78c433a7d976e4ea015`.
The worktree was dirty with the DSpark Markov and block-attention changes.
The real-weight checkpoint was `/Users/wenquanxing/Workspace/models/Qwen3.6-27B-4bit`.
It has the same `5120 -> 17408 -> 5120` dense-MLP geometry and 4-bit affine format as Qwen3-14B.

The same-process eight-row comparison measured:

| Dense-MLP case                  |      Median |
| ------------------------------- | ----------: |
| Previous production QMV         | 2075.433 µs |
| QMM BM8/BN32                    | 1406.506 µs |
| Previous production gate/up QMV | 1421.795 µs |
| Gate/up QMM BM8/BN32            |  934.661 µs |
| Previous production down QMV    |  900.451 µs |
| Down QMM BM8/BN32               |  930.766 µs |

The complete BM8/BN32 replay was `32.2%` faster.
The gate/up projection supplied the primary gain.
The complete replay still selected BM8/BN32 for both projections because that composition retained one QMM pipeline.
A mixed BM8/QMV composition retained one more pipeline and did not improve the eight-row full replay.

The row sweep selected these production ranges for large dense MLPs:

| Active rows | Selected path |
| ----------: | ------------- |
|         1–5 | QMV           |
|         6–8 | QMM BM8/BN32  |
|        9–16 | QMM BM16/BN32 |
|  17 or more | QMM BM32/BN32 |

The backend owns this selection.
Main, MTP, and DSpark pass only the active row count and complete matrix shape.
The production model API does not contain QMV/QMM or tile controls.

The eight-row command was:

```sh
cargo bench -p inference-executor-metal --bench qwen35_dense_mlp -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3.6-27B-4bit \
  --tokens 8 \
  --cases full_auto,full_qmm_bm8_bn32 \
  --iters 50 \
  --warmup-iters 20 \
  --runs 7
```

The BM8/BN32 kernel uses 64 threads and 3200 bytes of static threadblock memory for BF16.
Initialization checks the SIMD width, pipeline thread limit, calculated threadblock memory, reported static
threadblock memory, and device threadblock-memory limit.
The backend reference test covers BM8/BN32 affine output.
The complete dense-MLP CPU/GPU test covers the seven-row production selection.

A full executor control compared the retained BM8/BN32 selector with a temporary QMV selector.
Both cases proposed 210 tokens, accepted zero tokens, and sampled 30 continuation tokens.
Thus, the fixed synthetic trajectory was identical.
The BM8/BN32 run measured `53.321 ms` for Main verification, `9.655 ms` for proposal, and `63.001 ms` for the
complete cycle.
The QMV control measured `73.640 ms`, `11.373 ms`, and `85.042 ms`.
The separate processes had different device-frequency states, so these absolute values are directional evidence.
The identical trajectory removes acceptance as the source of the measured difference.

A Qwen3-14B real-weight GQA diagnostic used eight rows and 128 history tokens.
The production tiled full replay measured `0.695479 ms` for one layer.
The single-Q full replay measured `0.725750 ms`.
Thus, the production tiled-attention selection is also correct for this shape.

The GQA diagnostic command was:

```sh
cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --tokens-per-req 8 \
  --contexts 128 \
  --iters 50 \
  --warmup-iters 20 \
  --runs 5 \
  --validate
```

The measured Main cost is real multi-row transformer work.
It is not evidence of an incorrect submission boundary or replay-state flag.
The BM8/BN32 dense-MLP path reduces this cost without changing executor orchestration.
Confidence-guided verification lengths remain deferred.

### Final Main verification audit

The 2026-07-29 follow-up used base commit `b91b4464`.
The production source was unchanged.
The worktree contained two benchmark-only changes during the final measurements.
The machine was an Apple M3 Max with 40 GPU cores and 48 GB of memory.
No other GPU command ran concurrently.

The Main checkpoint was `/Users/wenquanxing/Workspace/models/Qwen3-14B-4bit`.
The DSpark checkpoint was `/Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine`.
The full executor command used one request, context 128, five warmup iterations, 20 measured iterations, and five runs.

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --cases main,dspark \
  --num-requests 1 \
  --start-context 128 \
  --warmup-iters 5 \
  --iters 20 \
  --runs 5
```

The full executor medians were:

| Stage                                  |    Median |
| -------------------------------------- | --------: |
| One-row Main submission                | 30.949 ms |
| Eight-row Main verification submission | 58.760 ms |
| Seven-row DSpark proposal submission   | 11.184 ms |
| Complete DSpark cycle                  | 70.048 ms |

The synthetic executor trajectory proposed 700 tokens and accepted zero tokens.
Thus, it fixed the Main and DSpark execution shapes but did not measure deployment acceptance.

The verification-body command used `qwen3_dspark_forward` with the same checkpoints, request count, and context.
The new `main-verification` case recorded `MainEmbed`, all 40 Main layers, Main residual capture, and DSpark context
projection for eight rows.
It did not record `GatherUnembed` or rejection sampling.
Its median was `54.567 ms`.

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_forward -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 \
  --context 128 \
  --warmup-iters 5 \
  --iters 20 \
  --runs 5
```

The production DSpark `GatherUnembed` proxy used seven rows and the Main unembed weights.
Its median was `2.930 ms`.
The matching sparse-rejection benchmark used one request, seven proposal tokens, `top_k=1`, and vocabulary size 151936.
Its median was `0.283 ms`.

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark_unembedding -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 \
  --warmup-iters 20 \
  --iters 100 \
  --runs 7

cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode rejection-sparse \
  --num-reqs 1 \
  --spec-tokens 7 \
  --top-k 1 \
  --vocab 151936 \
  --warmup-iters 50 \
  --iters 200 \
  --runs 7
```

The independent measurements identify the principal contributors to the Main verification stage:

```text
MainEmbed + Main + residual/context capture   54.567 ms
seven-row GatherUnembed proxy                  2.930 ms
sparse rejection                               0.283 ms
full eight-row Main stage                     58.760 ms
```

Do not add the independent medians as a substitute for the full-stage time.
Each independent measurement has its own submit-and-wait boundary.
The values show attribution only.

The eight-row Qwen3 GQA benchmark measured:

| GQA operation        |   Median |
| -------------------- | -------: |
| Full tiled GQA       | 0.649 ms |
| SplitKV TiledQ only  | 0.319 ms |
| QKV QMV BN8/BK32     | 0.498 ms |
| QKV QMM BM8/BN32     | 0.436 ms |
| QKV QMM BM16/BN32    | 0.485 ms |
| Output QMV BN8/BK32  | 0.433 ms |
| Output QMM BM8/BN32  | 0.416 ms |
| Output QMM BM16/BN32 | 0.445 ms |

The existing QMM BM8/BN32 selection is correct for both eight-row GQA projections.
The existing SplitKV TiledQ `q_head_tile=5` also remained the best tested geometry.
The eight-row dense-MLP replay measured `1.327 ms` with the existing QMM BM8/BN32 selection.

```sh
cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --tokens-per-req 8 \
  --contexts 128 \
  --iters 50 \
  --warmup-iters 20 \
  --runs 7 \
  --validate

cargo bench -p inference-executor-metal --bench qwen35_dense_mlp -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3.6-27B-4bit \
  --tokens 8 \
  --cases full_auto,gate_up_auto,activation,down_auto \
  --iters 50 \
  --warmup-iters 20 \
  --runs 7
```

Verdict: The remaining Main verification time is the real 40-layer transformer workload.
It is not replay orchestration, `GatherUnembed`, or sparse-rejection overhead.
No additional production kernel or submission change passed the evidence threshold.

### Final deterministic end-to-end retest

The final end-to-end comparison used the same base commit, machine, and checkpoints as the final Main audit.
The worktree changes were benchmark and documentation changes only.
Each service used:

```sh
target/release/qwen3 \
  --grpc-listen-addr 127.0.0.1:50151 \
  --http-listen-addr 127.0.0.1:8011 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --num-cache-pages 4096 \
  --max-requests 4 \
  --max-tokens 128 \
  --max-tokens-per-request 64 \
  --logging info

target/release/qwen3 \
  --grpc-listen-addr 127.0.0.1:50151 \
  --http-listen-addr 127.0.0.1:8011 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --hf-dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-cache-pages 4096 \
  --max-requests 4 \
  --max-tokens 128 \
  --max-tokens-per-request 64 \
  --logging info
```

The two service commands ran separately.

Each client used:

```sh
target/release/decode \
  --server-url http://127.0.0.1:50151 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --prompt-str 'Explain in concise technical terms why the sky appears blue during the day.' \
  --disable-thinking \
  --max-sampled-tokens 256 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 42 \
  --show-stats
```

The common service limits were:

```text
--num-cache-pages 4096
--max-requests 4
--max-tokens 128
--max-tokens-per-request 64
```

The request used the prompt:

```text
Explain in concise technical terms why the sky appears blue during the day.
```

It used `temperature=0`, `top_k=1`, `top_p=1`, `seed=42`, and a 256-token output limit.
Both paths stopped after 98 sampled tokens.
Both paths produced the same final text.

The last three warmed samples were:

| Path      | Samples                      |       Median | Output chunks |
| --------- | ---------------------------- | -----------: | ------------: |
| Main-only | 37.154, 36.565, 36.272 tok/s | 36.565 tok/s |            98 |
| DSpark    | 42.788, 43.023, 42.786 tok/s | 42.788 tok/s |            35 |

DSpark was `17.0%` faster than Main-only.
The deterministic DSpark request used 34 verification batches.
It proposed 238 tokens and accepted 65 proposal tokens.
Its proposal-token acceptance rate was `27.31%`.

Verdict: The final deterministic output and trajectory are stable.
The optimized fixed-block DSpark path now has a positive deployment-performance result for this one-request workload.
Confidence-guided proposal length remains deferred.

### Acceptance correctness audit

The 2026-07-29 audit used clean commit `dad01342fb8eb4fc828b4fdd00db51d8db65fe9f`.
It ran on an Apple M3 Max with a 40-core GPU.
It used `Qwen3-14B-4bit` and `dspark_qwen3_14b_block7-affine`.
All GPU commands ran serially.

Each request used:

```text
temperature=0
top_k=1
top_p=1
seed=1
```

The four-prompt result was:

| Workload             | Output tokens | Verification rounds | Proposed | Accepted | Proposal acceptance | Accepted length |
| -------------------- | ------------: | ------------------: | -------: | -------: | ------------------: | --------------: |
| Sky explanation      |            98 |                  34 |      238 |       65 |              27.31% |            2.91 |
| Algebra              |           128 |                  20 |      140 |      113 |              80.71% |            6.65 |
| Python code          |           128 |                  20 |      140 |      109 |              77.86% |            6.45 |
| Database engineering |           128 |                  40 |      280 |       93 |              33.21% |            3.33 |
| Aggregate            |           482 |                 114 |      798 |      380 |              47.62% |            4.33 |

`Accepted length` includes the Main continuation token from each verification round.
Proposal acceptance uses only the seven draft slots.

The aggregate position-conditional acceptance was:

```text
position:                 1      2      3      4      5      6      7
conditional acceptance: 82.5%  79.8%  74.7%  87.5%  83.7%  85.4%  85.7%
```

This curve rules out an anchor or proposal-position shift.
It also does not show suffix collapse.
The workload spread follows the expected DSpark domain behavior.

The audit compared all four DSpark greedy outputs with Main-only outputs.
The sky, algebra, and code outputs matched byte for byte.
The database-engineering output diverged at output token 51.
The divergence repeated deterministically in each path.
The DSpark proposal token at that position was `235`.
Sparse rejection accepted zero draft tokens.
The eight-row Main verification sampled token `117`.
The one-row Main path sampled token `223`.

Verdict: The divergence is not an incorrect draft acceptance.
The audit found no DSpark proposal, Markov, attention, or sparse-rejection defect.
Strict one-row and multi-row Main numerical parity remains a separate investigation.

## Main and MTP submission evidence

The 2026-07-28 comparison used base commit `3be69962c392d6ae75f33b2bef65e9403b680b30`.
The split source was clean.
The one-sequence candidate had uncommitted executor and documentation changes.
The machine was an Apple M3 Max with 40 GPU cores, macOS 27.0, and `arm64`.
The service used `max_running_requests=8`, seed `42`, and these checkpoints:

- `Qwen3.6-35B-A3B-4bit`
- `Qwen3.6-35B-A3B-MTP-4bit`

The command used `scripts/qwen35_e2e_decode_perf.sh`.
It ran one GPU service at a time.

| Case      | Tokens |  Split median | One-sequence median | Change |
| --------- | -----: | ------------: | ------------------: | -----: |
| `35b_off` |    256 |  93.414 tok/s |        96.395 tok/s |  +3.2% |
| `35b_off` |   1024 |  90.801 tok/s |        93.658 tok/s |  +3.1% |
| `35b_on`  |    256 | 138.262 tok/s |       145.307 tok/s |  +5.1% |
| `35b_on`  |   1024 | 122.080 tok/s |       127.684 tok/s |  +4.6% |

The MTP trajectories were unchanged.
The 256-token MTP case proposed 135 tokens and accepted 121 tokens.
The 1024-token MTP case proposed 591 tokens and accepted 432 tokens.

Verdict: Main, MTP, and DSpark must keep one submission for each model entity.
They must not add a submit-and-wait boundary between a model body and its unembed/sampling suffix.
