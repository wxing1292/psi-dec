# Qwen3x DFlash2 Design

This document describes DFlash2-specific composition, state, and proposal selection for Qwen3.5.
See [`executor_gqa.md`](executor_gqa.md) for shared GQA and [`executor_sampling.md`](executor_sampling.md) for sampling.

## Scope and ownership

DFlash2 is a model role and a peer of DSpark, not a DSpark mode.

Main owns token embedding, the transformer, selected residual capture, unembedding, Main sampling, and rejection.
The DFlash2 owner owns its checkpoint, history page table, replay caches, workspaces, dynamic convolution, and selector.
Runtime core owns scheduling, request lifecycle, physical pages, and page IDs.

DFlash2 reuses Main Embed, Main Unembed, GQA, dense MLP, raw Top-K, and sparse rejection.
It has no confidence head, so the response adapter returns `1.0` for each proposal token.

## End-to-end flow

```text
DFlash2
-------

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
│                    DFlash2 Layer × L                          │
│                                                               │
│                    draft hidden h_t                           │
│                            │                                  │
│              ┌─────────────┴─────────────┐                    │
│              │                           │                    │
│              │ data           kernel projection               │
│              │                           │                    │
│              │                 ┌─────────┴─────────┐          │
│              │                 ▼                   ▼          │
│              │              ΔK_pre              ΔK_post       │
│              │                 │                   │          │
│              │                 ▼                   ▼          │
│              │        K_base_pre+ΔK_pre   K_base_post+ΔK_post │
│              │                 │                   │          │
│              ▼                 │                   │          │
│         ┌──────────┐           │                   │          │
│ h_t ───►│ PRE-CONV │◄──────────┘                   │          │
│ h_t-1 ─►│          │                               │          │
│         └────┬─────┘                               │          │
│              ▼                                     │          │
│             h'_t                                   │          │
│              │                                     │          │
│              ▼                                     │          │
│      ┌─────────────────┐                           │          │
│      │ Attention / MLP │                           │          │
│      └────────┬────────┘                           │          │
│               ▼                                    │          │
│              r_t                                   │          │
│                                                    │          │
│         ┌───────────┐                              │          │
│ r_t ───►│ POST-CONV │◄─────────────────────────────┘          │
│ r_t-1 ─►│           │                                         │
│         └────┬──────┘                                         │
│              ▼                                                │
│       residual / RMSNorm                                      │
│              │                                                │
│              ▼                                                │
│      next-layer hidden                                        │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
          final draft hidden
                 │
                 ▼
        Main unembedding / LM head
                 │
                 ▼
       unary vocabulary logits U_t
                 │
                 ▼
          top-k candidates C_t
                 │
                 └──────────────► CANDIDATE SELECTOR below


CANDIDATE SELECTOR
------------------

At proposal position t:

previous selected token a
current top-k candidate b
current draft hidden h_t

 h_t ──► H(h_t) ──┐
                   ▼
 a ───► A(a) ───► elementwise product ──┐
                                         ▼
 b ───► B(b) ─────────────────────────► dot ──┐
                                               ▼
 U_t(b) ─────────────────────────────────────► add
                                               │
                                               ▼
                                           S_t(a,b)

S_t(a,b) = U_t(b) + <A(a) * H(h_t), B(b)>


anchor x_0
    │
    │ sample q_1(. | x_0) from C_1
    ▼
   x_1
    │
    │ sample q_2(. | x_1) from C_2
    ▼
   x_2
    │
    │ sample q_3(. | x_2) from C_3
    ▼
   x_3
    │
   ...

N = block_size - 1 proposal tokens
```

Spec Prefill and Spec Decode use independent replay recordings.
The outer owner may submit either replay or both after Main completes.
Prefill keeps fixed Main rows and the accepted speculative prefix.
It excludes the rejected suffix and the new anchor.
Main writes selected residuals directly into assigned capture columns.

## Proposal block and attention

The query block has one anchor row and `block_size - 1` MASK rows.
Only MASK rows produce proposals.
Each layer reduces SplitKV history partials with bidirectional block-SDPA partials.

DFlash2 stores all committed history and reads this half-open range for each query:

```text
[max(0, query_position + 1 - sliding_window), anchor_position)
```

The local block contains the anchor and all MASK K/V with bidirectional attention.
Its K/V is temporary.

The Decode replay key contains padded history TaskTemplate capacity.
The active count remains a submission argument, so matching padded capacities reuse one replay.

## Layer composition

The template runs once for attention and once for MLP.
Each branch has one kernel projection and BF16 base kernel.

For branch side `s` in `{pre, post}`, the effective kernel is:

```text
K_s(t) = K_base_s + Delta K_s(h_t)
```

The convolution reads current and earlier rows in one request-local block.
It never combines requests.

## Candidate selection

Main Unembed produces unary logits for each MASK row.
Raw Top-K selects `selector_top_k` candidates `C_t` per position.

`A` and `B` are the predecessor and successor codebooks.
`H` is the hidden projection.
`U_t(b)` is the unary logit for candidate `b`.

The selector scores all top-k-to-top-k edges between adjacent positions.
The path starts from the anchor and follows each selected predecessor.
It applies request temperature but no post-selection top-p.

## Cache and lifecycle

Runtime core supplies one page-ID list for each logical cache block.
Main and DFlash2 history share cache lane 0, which the executor splits once:

```text
[Main page IDs | DFlash2 history page IDs]
```

Main and DFlash2 use separate page tables with one request-slot lifecycle.
A reset clears both bindings, while runtime core retains physical-page ownership.

Persistent state contains the request page table and history K/V pages.
Snapshots use `dflash2-gqa-request-page-table`.
Local Q/K/V, attention partials, convolution coefficients, candidates, and selector output are ephemeral.

## Checkpoint contract

`Qwen3xDFlash2Config` adapts `DFlash2DraftModel` to the flat canonical schema.
It validates Main compatibility, selected layers, block and window sizes, attention and convolution geometry, selector
rank, and candidate count.
The checkpoint boundary rejects unknown architectures and nested fields.

`Qwen3xDFlash2WeightBindings` accepts only the exact source or affine manifest.
The manifest cannot replace reused Main embedding or unembedding.
Each projection resolves its exact affine layout.

The shared Spec converter writes packed `U32` matrices and BF16 affine parameters.
It keeps RMSNorm weights and convolution base kernels as BF16.
The loader also accepts uniform F32 affine scales and biases.
The default conversion uses group size 64 and 4-bit matrices.
Layer 2 and layer 4 `v_proj` and `down_proj` use 6-bit matrices.
Use [`service.md`](service.md) for conversion commands.

## Key source layout

```text
crates/inference-executor-core/src/model/qwen/v3_x/dflash2/
  config.rs                  checkpoint schema and validation
  weight_layout.rs           exact source and affine manifests

crates/inference-executor-core/src/bin/qwen3x_spec_quantize/
  dflash2.rs                 DFlash2 conversion policy

crates/inference-executor-metal/src/model/qwen/v3_x/dflash2/
  execution.rs               Prefill and Decode orchestration
  main_feature.rs            selected Main residual projection
  attention.rs               history-plus-block attention
  conv.rs                    dynamic grouped convolution
  layer.rs                   DFlash2 layer composition
  model.rs                   model and replay owners
  output.rs                  Top-K, selector, and draft distributions

crates/inference-executor-metal/src/model/qwen/v3_5/executor/
  dflash2.rs                 Qwen3.5 outer integration

crates/inference-backend-metal/src/components/
  dynamic_grouped_conv.rs
  sampling/dflash2_selector.rs
```

## Verification

Focused tests cover config adaptation, manifests, affine layouts, sliding ranges, convolution and selector parity,
replay active counts, service modes, and speculator mutual exclusion.

Use [`service.md`](service.md) for end-to-end commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before a performance claim.
