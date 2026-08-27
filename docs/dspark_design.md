# Qwen3x DSpark Design

This document describes DSpark-specific composition, state, Markov sampling, and confidence for Qwen3 and Qwen3.5.
DSpark is experimental, and its checkpoint contract and proposal policy may change.
See [`executor_gqa.md`](executor_gqa.md) for shared GQA and [`executor_sampling.md`](executor_sampling.md) for sampling.

## Scope and ownership

Use `Main`, `MTP`, and `DSpark` or `Spec` for roles.
Checkpoint fields such as `target_layer_ids` retain upstream names.

Main owns token embedding, the transformer, residual capture from selected Main layers, unembedding, Main sampling,
and rejection.
The DSpark owner owns its checkpoint, history page table, replay caches, workspaces, Markov head, and confidence head.
Runtime core owns scheduling, request lifecycle, physical pages, and page IDs.

DSpark supports ungated GQA, the `vanilla` Markov head, Markov-conditioned confidence, `default` RoPE, and Yarn RoPE.
Checkpoint `block_size` fixes proposal length `N` at startup.
Qwen3.5 MTP and DSpark are mutually exclusive.

Current limits:

- Each Decode request produces exactly `N` proposals.
- Confidence does not change verification length.
- Gated GQA is unsupported.
- Each executor supports one in-flight batch.

## End-to-end flow

```text
DSpark
------

selected Main-layer hidden states
                  │
                  ▼
        capture / projection
                  │
                  ▼
        Main context H
       (Attention only)

      [ANCHOR] [MASK] [MASK] [MASK] ...
         t=0     t=1    t=2    t=3
                  │
                  ▼
        embedding / proposal input
                  │
                  ▼

┌───────────────────────────────────────────────────────────────┐
│                     DSpark Layer × L                          │
│                                                               │
│                    draft hidden h_t                           │
│                           │                                   │
│                           ▼                                   │
│                  ┌─────────────────┐                          │
│                  │ Attention / MLP │                          │
│                  └────────┬────────┘                          │
│                           ▼                                   │
│                   residual / RMSNorm                          │
│                           │                                   │
│                           ▼                                   │
│                   next-layer hidden                           │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
          final draft hidden h_t
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
 Main unembedding    confidence branch
        │                  └──────────────► CONFIDENCE HEAD below
        ▼
 base logits U_t
        │
        └───────────────────────────────► MARKOV SAMPLING below

MARKOV SAMPLING
---------------

previous sampled token x_{t-1}
                 │
                 ▼
          W_1[x_{t-1}] = l_t
                 │
                 ▼
                W_2
                 │
                 ▼
      Markov bias vector M_t
                 │
       base logits U_t
                 │
                 ▼
           Z_t = U_t + M_t
                 │
                 ▼
     top-k / temperature / top-p
                 │
                 ▼
          q_t(. | x_{t-1})
                 │
                 ▼
             sample x_t
                 │
                 └──────────────► next proposal position
                                   x_t becomes x_{t-1}

M_t = W_2 W_1[x_{t-1}]
Z_t = U_t + M_t


CONFIDENCE HEAD
---------------

          draft hidden h_t
                 │
                 ├───────────────┐
                 │               │
                 │      W_1[x_{t-1}] = l_t
                 │               │
                 └───────┬───────┘
                         ▼
                  concat(h_t, l_t)
                         │
                         ▼
            confidence projection + bias
                         │
                         ▼
                sigmoid temperature 1.0
                         │
                         ▼
                        c_t
                         │
                         ▼
             returned with proposal token

N = block_size proposal tokens
```

Spec Prefill and Spec Decode use independent replay recordings.
The Qwen executor records them with Main and submits one ordered sequence.
Spec Decode prepare follows rejection sampling. Prefill follows Main capture.
When both exist, the serial sequence emits Spec Decode prepare first, then Prefill, and then the remaining Spec Decode
work.
Each selected Main layer writes every Main row directly into its assigned capture columns.
Spec Prefill persists every captured Main row, including the rejected physical suffix.
Logical commit exposes only fixed Main rows and the accepted speculative prefix.
Spec Prefill borrows the active token count, request slots, and flat token indices from the current Main GQA metadata.

## Proposal block and attention

The proposal block has one anchor and `N - 1` MASK rows.
All `N` rows produce proposals in one transformer forward.

Each layer reduces SplitKV history partials with bidirectional local-block SDPA partials.

For anchor position `p`, each local row reads history range `[0, p)` and attends to all `N` local rows.
Proposal-local K/V is temporary.

The template runs once for attention and once for MLP.

The Spec Decode replay key contains padded history TaskTemplate capacity.
The active count remains a submission argument, so matching padded capacities reuse one replay.

## Markov sampling and confidence

At step `t`, `x_{t-1}` is the preceding sample.
For `t = 0`, `x_{-1}` is the Main anchor.

`U_t` is the base vocabulary-logit vector.
`M_t` is the complete vocabulary-sized Markov bias vector.
`Z_t` is the corrected vocabulary-logit vector.

Confidence reuses `l_t` and `h_t`.
Its sigmoid uses temperature `1.0`.
Runtime returns `c_t` but does not use it to change verification length.

The fused Metal map computes `l_t`, `M_t`, tile-local Top-K, and confidence.
The reducer performs global Top-K, top-p sampling, and sparse writes.
Each sample becomes the next Markov input.
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

The service owns submission and wait boundaries.
DSpark uses one combined Main submission:

```text
Main Embed -> Main -> GatherUnembed -> RejectionSampling
  -> Spec Decode prepare -> DSpark Prefill
  -> DSpark Embed -> DSpark -> gather/unembed -> Markov sampling

Prefill-only: Main Embed -> Main -> DSpark Prefill
```

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
[`future_work.md`](future_work.md) owns confidence-guided scheduling, gated GQA, and additional checkpoint variants.
