# Qwen3.5 MTP Design

This document describes MTP-specific composition, state, sequential proposal generation, and sampling.
See [`executor_gqa.md`](executor_gqa.md) for shared GQA and [`executor_sampling.md`](executor_sampling.md) for sampling.

## Scope and ownership

MTP is a Qwen3.5-family model role.
It is mutually exclusive with DSpark and DFlash2.

Main owns token embedding, the transformer, unembedding, Main sampling, and rejection.
The MTP owner owns its checkpoint, input projection, physical body layer, GQA page table, replay caches, scratch, and
proposal loop.
Runtime core owns scheduling, request lifecycle, physical pages, and page IDs.

MTP reuses Main embedding, Main unembedding, draft sampling, sparse distributions, and rejection.
It has no confidence head, so the response adapter returns `1.0` for each proposal token.

## End-to-end flow

```text
MTP
---

previous hidden state h_{t-1} ── LM head / sampling ──► token x_t
             │                                             │
             ▼                                             ▼
      gather / RMSNorm                            Main embedding / RMSNorm
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
                           concat / projection
                                    │
                                    ▼
                           MTP proposal input
                                    │
                                    ▼

┌───────────────────────────────────────────────────────────────┐
│                      MTP Layer × 1                            │
│                                                               │
│                    draft hidden h_t                           │
│                            │                                  │
│                            ▼                                  │
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
                    MTP output hidden h_t
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       │
      Main unembedding / LM head         │
                 │                       │
                 ▼                       │
       vocabulary logits U_{t+1}         │
                 │                       │
                 ▼                       │
 top-k / temperature / top-p sampling    │
                 │                       │
                 ▼                       │
            sample x_{t+1}               │
                 │                       │
                 └───────────┬───────────┘
                             ▼
                    next logical step
                 previous hidden: h_t
                 token input: x_{t+1}

K = num_spec_tokens sequential proposals
```

## Sequential proposals

MTP uses one physical body layer for `K` dependent logical steps.
At logical step `t`, MTP combines `h_{t-1}` with `x_t` and samples `x_{t+1}`.
At step 0, the sampled Main token is the token input, and Main supplies the previous hidden state.
Each later step uses the preceding draft token and MTP hidden output.

Each step samples one token for each active Decode request.
The sampler writes the exact sparse distribution that produced that token.
The distribution index is stable across submissions:

```text
draft_distribution_index = req_slot * K + step_index
```

The current implementation waits and reads after each non-final step because the next step needs the sampled token.
The public lifecycle remains one `submit_spec -> wait -> read_spec` transaction.
MTP does not use the fixed-block Spec Prefill or Decode lifecycle.

## Replay ownership

MTP uses separate replays for MTPEmbed, the body, GatherUnembed, and DraftSampling.
All `K` steps reuse the same weights, scratch, stable buffers, and recorded programs.

MTPEmbed gathers previous hidden rows and embeds the shifted token rows.
It normalizes both inputs, concatenates them, and applies the input projection.

The body replay key contains token capacity, GQA capacity and topology, and MLP topology.
Active counts and the logical GQA layer index remain submission parameters.

## Cache and lifecycle

The logical model has one Main cache lane and `K` MTP cache lanes:

```text
lane 0       Main
lanes 1..=K  MTP logical steps 0..K-1
```

The MTP owner maps each MTP lane to one row in its GQA page table.
The physical body layer is reused for every row.
Main and MTP use separate GQA state domains with one request-slot lifecycle.

A reset clears both page-table bindings, while runtime core retains physical-page ownership.
Snapshots use `mtp-gqa-request-page-table`.
Token, hidden, logits, and sampling workspaces are ephemeral.

The MTP owner allocates one reusable BF16 hidden-state cache.
Logical module `m` stores `m + 1` rows for each request slot.
The final module stores no rows because no later module consumes its output.
For `R` request slots and `K` MTP modules, the cache contains this number of rows:

```text
R * K * (K - 1) / 2
```

The cache uses a module-major layout.
Each request slot owns `Qwen35MTPCacheState`: `Empty`, `Prefill`, or `Decode`.
Each nonempty variant stores the pending Main index and K cached tail token IDs.
For K=3, Decode stores `[w, x1, x2]`. The final draft has no cached KV slot.
The enum distinguishes canonical Prefill lookahead from Decode metadata with retained hidden rows.
Request-slot reset selects `Empty`. The input preparation commit connects this metadata to execution.

The next Main call replays `K - 1` verified tail tokens.
MTP shifts the related GDN decision-candidate versions by `K - 1`.
See [`executor_gdn.md`](executor_gdn.md) for the state-version contract.

## Checkpoint contract

The checkpoint must contain exactly one physical GQA body layer.
It must share the Main token embedding.
It must not contain a dedicated MTP embedding.
The physical layer can use dense MLP or MoE.

The validator checks Main-compatible hidden, attention, and expert geometry.
It also checks shared dense-MLP or MoE scratch geometry when applicable.
The loader requires an exact tensor manifest and quantized MTPEmbed projection weights.

`--num-spec-tokens K` sets the number of logical proposal steps.
`K` must be positive.
`--max-tokens-per-request` must be at least `K`.
Use [`service.md`](service.md) for startup and validation commands.

## Key source layout

```text
crates/inference-executor-core/src/model/qwen/v3_5/
  config.rs                  MTP checkpoint fields
  weight_layout.rs           exact MTP tensor bindings

crates/inference-executor-metal/src/model/qwen/v3_5/
  component_config.rs        Main/MTP compatibility validation
  mtp/
    embed.rs                 hidden and token input composition
    hidden_state_cache.rs          per-request cross-step BF16 hidden-state cache
    layer.rs                 physical GQA and MLP layer
    mod.rs                   body, final norm, and replay key
  executor/
    mtp.rs                   logical-step loop and proposal output

crates/inference-executor-metal/src/sampling/
  top_k_replay.rs            DraftSampling replay
  spec_probs.rs              sparse draft-distribution storage
```

## Verification

Focused tests cover checkpoint validation, exact bindings, input composition, cache-lane mapping, replay parameters,
sequential sampling, sparse distributions, GDN state shifts, and rejection.

Use [`service.md`](service.md) for end-to-end commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before a performance claim.
