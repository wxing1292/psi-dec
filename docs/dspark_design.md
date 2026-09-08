# Qwen3x DSpark Design

DSpark produces a fixed block of proposals with one transformer forward, then samples those proposals in sequence.
The Markov head conditions each sample on the preceding token. The confidence head returns a score for each proposal.
This document describes the current Qwen3 and Qwen3.5 implementation.
DSpark is experimental, and its checkpoint contract and proposal policy may change.

Use [Proposal block and attention](#proposal-block-and-attention) for row counts and [End-to-end flow](#end-to-end-flow) for execution order.
[`executor_gqa.md`](executor_gqa.md) owns shared GQA details. [`executor_sampling.md`](executor_sampling.md) owns shared sampling contracts.

## Scope and ownership

Use `Main`, `MTP`, and `DSpark` or `Spec` for roles.
Checkpoint fields such as `target_layer_ids` retain upstream names.

| Owner | Responsibility |
| --- | --- |
| Runtime core | Scheduling, request lifecycle, physical pages, and page IDs |
| Main | Token embedding, transformer execution, selected-layer residual capture, unembedding, Main sampling, and rejection |
| DSpark | Checkpoint, history page table, replay caches, workspaces, Markov head, and confidence head |
| Qwen executor | Compose Main and DSpark recordings, submit the ordered sequence, wait, and adapt results |

DSpark supports ungated GQA, the `vanilla` Markov head, Markov-conditioned confidence, `default` RoPE, and Yarn RoPE.
Qwen3.5 MTP and DSpark are mutually exclusive.

Current limits:

- Each Decode request produces exactly `N` proposals.
- Confidence does not change the DSpark proposal count or its recorded graph.
- Gated GQA is unsupported.
- Each executor supports one in-flight batch.

## Proposal block and attention

The runtime proposal length `N` is fixed at startup.
The service uses checkpoint `block_size` as the default. `--num-spec-tokens N` overrides this default.
The executor keeps the checkpoint unchanged and uses `N` for query rows, scratch, replay, and draft distributions.

| Symbol | Meaning in this document |
| --- | --- |
| `N` | Proposals per Decode request and query rows per DSpark block |
| `p` | Absolute position of the newly sampled Main anchor |
| `t` | DSpark row and proposal index, from `0` through `N - 1` |
| `x_{-1}` | Main anchor at position `p` |
| `x_t` | Proposal token at position `p + t + 1` |
| `h_t` | Final DSpark hidden output from query row `t` |

The proposal block has one anchor and `N - 1` MASK rows.
All `N` rows produce proposals in one transformer forward.
For `N = 3`, the row-to-token relation is:

```text
+--------------------+----------+----------+----------+
| Query row t        | 0        | 1        | 2        |
+--------------------+----------+----------+----------+
| Query position     | p        | p + 1    | p + 2    |
| Input token        | x_{-1}   | MASK     | MASK     |
| Final hidden       | h_0      | h_1      | h_2      |
| Produced proposal  | x_0      | x_1      | x_2      |
| Proposal position  | p + 1    | p + 2    | p + 3    |
+--------------------+----------+----------+----------+
```

The anchor row produces the first proposal.
[DFlash2](dflash2_design.md#proposal-block-and-attention) instead uses `N + 1` query rows and gathers only its `N` MASK rows.

Each local row reads history range `[0, p)` and attends to all `N` local rows.
Each layer reduces SplitKV history partials with bidirectional local-block SDPA partials.
Proposal-local K/V is temporary.

The Spec Decode replay key contains padded history TaskTemplate capacity.
The active count remains a submission argument, so matching padded capacities reuse one replay.

## End-to-end flow

Spec Prefill converts captured Main features into persistent history K/V.
Spec Decode uses that history and the anchor-plus-MASK block to produce new proposals.
The two stages use independent replay recordings within one ordered GPU submission:

```text
Main Embed -> Main forward and selected-layer capture
  -> GatherUnembed -> RejectionSampling
  -> Spec Decode prepare: write anchor, positions, and visible history ranges
  -> DSpark Prefill: project captured Main rows -> history K/V pages
  -> DSpark Embed -> all DSpark layers -> final RMSNorm
  -> GatherUnembed -> sequential Markov sampling + confidence

CPU boundary: submit sequence -> wait for completion -> read decision + proposals

Prefill-only: Main Embed -> Main forward/capture -> DSpark Prefill
```

The Qwen executor owns recording, submission, and wait boundaries.
There is no CPU rejection read between Main verification and Spec Decode.
A prefill-only batch creates history but has no sampled anchor and produces no proposals.

Each selected Main layer writes every Main row directly into its assigned capture columns.
Spec Prefill borrows the active token count, request slots, and flat token indices from the current Main GQA metadata.
It persists every captured Main row, including the rejected physical suffix.
Logical commit exposes only fixed Main rows and the accepted speculative prefix.

For example, Main verifies one known token `w` and three old drafts `d0, d1, d2` at positions `b` through `b + 3`.
Rejection sampling returns a new anchor `y`:

| Accepted old drafts | New anchor position `p` | Newly visible history rows |
| --- | --- | --- |
| None | `b + 1` | `w` |
| `d0` | `b + 2` | `w, d0` |
| `d0, d1, d2` | `b + 4` | `w, d0, d1, d2` |

In each case, DSpark starts its new block with `x_{-1} = y` at `p`.
The history bound `[0, p)` excludes the rejected suffix even when physical pages contain its captured rows.

Each DSpark layer applies two residual branches in order:

```text
residual -> RMSNorm -> attention -> add residual
         -> RMSNorm -> dense MLP -> add residual -> next layer
```

## Markov sampling and confidence

At step `t`, `x_{t-1}` is the preceding sample.
For `t = 0`, `x_{-1}` is the Main anchor.
All final hidden rows and base logits exist before this sequential sampling starts.
Sampling does not run the DSpark transformer again for each token.

| Symbol | Value |
| --- | --- |
| `U_t` | Base vocabulary-logit vector from unembedding `h_t` |
| `l_t` | Markov latent row `W_1[x_{t-1}]` |
| `M_t` | Complete vocabulary-sized Markov bias vector |
| `Z_t` | Corrected vocabulary-logit vector |
| `q_t(. \| x_{t-1})` | Proposal distribution after top-k, temperature, and top-p |
| `c_t` | Confidence returned with proposal `x_t` |

The Markov equations are:

```text
M_t = W_2 W_1[x_{t-1}]
Z_t = U_t + M_t
```

Each sample supplies the next step's Markov input:

```text
x_{-1} --[U_0 + Markov bias]--> sample x_0
x_0    --[U_1 + Markov bias]--> sample x_1
x_1    --[U_2 + Markov bias]--> sample x_2
```

Confidence uses the same preceding-token latent as that step's Markov bias:

```text
h_t -------------------+
                       +-> concat -> confidence projection + bias
W_1[x_{t-1}] = l_t -----+           -> sigmoid temperature 1.0 -> c_t
```

Runtime core stores `c_t` with each proposal.
Its [token budget allocator](token_budget_allocator.md) ranks speculative prefixes by cumulative confidence.
When the batch budget cannot contain every proposal row, the scheduler can select a shorter prefix for Main verification.
DSpark still produces exactly `N` proposals. The current policy uses no absolute confidence threshold.

The fused Metal map computes `l_t`, `M_t`, tile-local Top-K, and confidence.
The reducer performs global Top-K, top-p sampling, and sparse writes.
No full latent, bias, or corrected-logit buffer is materialized.

## Cache and lifecycle

Runtime core supplies one page-ID list for each logical cache block.
Main and DSpark history share this block, which the executor splits once:

```text
[Main page IDs | DSpark history page IDs]
```

Main and DSpark use separate page tables with one request-slot lifecycle.
A reset clears both bindings, while runtime core retains physical-page ownership.

Draft-distribution identity remains stable across submissions:

```text
draft_distribution_index = req_slot * N + proposal_position
```

Here, `proposal_position` is the zero-based proposal index within the request.
Persistent state contains the DSpark page table and history K/V pages.
Local Q/K/V, attention partials, logits, Markov scratch, and output are ephemeral.

## Checkpoint contract

The checkpoint boundary adapts supported schemas to flat `Qwen3xDSparkConfig`.
It validates Main compatibility, selected layers, attention geometry, RoPE, Markov shape, confidence, and dtype.
It rejects unknown architectures and conflicting fields.

The canonical schema supports `Qwen3DSparkModel` and `DSparkDraftModel`.
Yarn requires `factor` and `original_max_position_embeddings`.
The loader requires `enable_confidence_head = true` and `confidence_head_with_markov = true`.
Exact source and affine tensor manifests are mandatory.

The shared Spec converter writes packed `U32` matrices and BF16 affine parameters.
It preserves DSpark-owned embedding, unembedding, and confidence tensors when present.
Use [`service.md`](service.md) for conversion commands.

## Key source layout

```text
crates/inference-executor-core/src/model/qwen/v3_x/dspark/
  config.rs                  checkpoint schema and validation
  weight_layout.rs           exact source and affine manifests

crates/inference-executor-core/src/sampling/
  dspark.rs                  CPU Markov and confidence reference

crates/inference-executor-core/src/bin/qwen3x_spec_quantize/
  dspark.rs                  DSpark conversion policy

crates/inference-executor-metal/src/model/qwen/v3_x/dspark/
  execution.rs               Prefill and Decode orchestration
  main_feature.rs            all-row projection from selected Main layers
  attention.rs               history-plus-block attention
  layer.rs                   DSpark layer composition
  model.rs                   model and replay owners
  output.rs                  gather, unembed, and sampling
  sampling.rs                checkpoint weight adapter

crates/inference-executor-metal/src/sampling/
  dspark_markov.rs           sequential Markov replay

crates/inference-executor-metal/src/model/qwen/v3/executor/dspark.rs
crates/inference-executor-metal/src/model/qwen/v3_5/executor/dspark.rs

crates/inference-backend-metal/src/components/
  sampling/dspark_markov.rs  fused Markov and confidence map
```

Qwen3 and Qwen3.5 share `qwen/v3_x/dspark/` owners.
Their executors own model-specific batches, transactions, and result adaptation.

## Verification

Focused tests cover config adaptation, manifests, conversion, block construction, page splitting, attention, Markov and
confidence parity, sequential sampling, sparse distributions, replay active counts, request slots, and rejection.

Use [`service.md`](service.md) for end-to-end commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before a performance claim.
[`future_work.md`](future_work.md) owns remaining confidence calibration and scheduling work, gated GQA, and additional checkpoint variants.
