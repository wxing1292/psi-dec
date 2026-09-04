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

MTP uses separate replays for HiddenStateTransfer, MTPEmbed, the body, GatherUnembed, and DraftSampling.
All `K` steps reuse the same weights, scratch, stable buffers, and recorded programs.

HiddenStateTransfer gathers previous hidden rows and writes the retained producer tail.
MTPEmbed embeds the shifted token rows.
It normalizes the prepared hidden input and token input, concatenates them, and applies the input projection.

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
Token, per-wave hidden, logits, and sampling workspaces are ephemeral.

The MTP owner allocates one reusable BF16 hidden-state cache.
Logical module `m` stores `m + 1` rows for each request slot.
The final module stores no rows because no later module consumes its output.
For `R` request slots and `K` MTP modules, the cache contains this number of rows:

```text
R * K * (K - 1) / 2
```

The cache uses a module-major layout.
Each request slot owns `Qwen35MTPCacheState`: `Empty`, `Prefill`, or `Decode`.
Each nonempty variant stores the pending Main cache-local index and exactly `K` tail token IDs.
`Prefill` identifies canonical lookahead. `Decode` identifies the cached speculative tail and valid hidden rows.
For `K = 3`, Decode stores `[w, x1, x2]`. The sampled `x3` has not entered any MTP KV cache.
The complete proposal and rejection distributions still contain all `K` drafts.
Request-slot reset sets this metadata to `Empty`. Turn completion does not reset it.

The next Main call starts at the pending index. It does not replay verified tail tokens.
At steady-state Decode, Main consumes one anchor plus the submitted drafts.
For `K = 3` with a full proposal, Main consumes `[anchor, draft 1, draft 2, draft 3]`.
Core can submit fewer drafts for a token budget or a stop boundary.
The executor uses that actual count, not `K`, to separate the known input from the speculative suffix.
The runtime Main token budget remains `B`.
The MTP initializer sizes shared workspaces for `B + min(R, B) * (K - 1)` rows, where `R` is request capacity.
Each Decode request can require `K - 1` extra MTP rows.
A Prefill that replaces an old Decode tail can also require `K - 1` extra rows.
The same bound covers mixed batches.
GDN uses the Main state-version domain without an MTP shift.
See [`executor_gdn.md`](executor_gdn.md) for the state-version contract.

## Decode hidden cache and routing

Status: The model executor implements the acceptance-aware plan in `decode_plan.rs`.
It implements the persistent BF16 hidden-state cache in `hidden_state_cache.rs`.
It implements cache routing and writeback in `hidden_state_transfer.rs`.
Runtime core implements the matching token reconciliation contract in [`core.md`](core.md).
Main does not replay old rows. The complete transaction has a shared cached end across lanes.

New canonical tokens can replace an existing MTP speculative tail in either incoming query variant.
The executor compares incoming token identities with the last completed wave before it prepares MTP input.
The pending index and anchor must match. Prefill metadata also requires matching canonical lookahead.
Decode metadata compares only the overlapping cached prefix. A missing suffix is not a mismatch.
Changing only the last, uncached draft is not a mismatch.
Any mismatch in the cached prefix disables reuse from old `x1` onward in every MTP module.
The executor preserves old Decode metadata and hidden rows until their consumers finish.
It replaces metadata only after the complete MTP wave finishes.
This contract applies to continued Decode, guided canonical insertion, and same-request turn resume.

Each cache lane starts at index `0`.
The token labels differ because each MTP lane is shifted by one token.

The following example uses `K = 3`.
The final column is the pending diagonal at cache-local index `P = 3`.
Each earlier token has executed in that lane.
The pending token has not executed in that lane.

```text
+-------+----+----+----+----+
| Index | 0  | 1  | 2  | 3  |
+-------+----+----+----+----+
| Main  | t0 | t1 | t2 | w  |
| MTP0  | t1 | t2 | w  | x1 |
| MTP1  | t2 | w  | x1 | x2 |
| MTP2  | w  | x1 | x2 | x3 |
+-------+----+----+----+----+
                        P
```

The hidden dependency between adjacent lanes uses the same cache-local index.
For example, `MTP0` produces `H0(w)` at index `2`.
`MTP1` consumes `H0(w)` when it executes `x1` at index `2`.

Runtime core reconciles mutable token metadata after execution.
It does not repeat MTP input planning or request extra Main computation.
See [the runtime contract](core.md) and `TrieDecoderBlocks::commit` for block-column publication and cross-block copies.

### Hidden-cache contents

Logical MTP module `m` stores `m + 1` hidden rows.
The final module stores no hidden rows because no later module consumes its output.
For pending index `P`, module `m` stores this cache-local range:

```text
[P - (m + 1), P)
```

The `K = 3` example stores these rows:

```text
MTP0: [H0(w)]            cache-local indices [2, 3)
MTP1: [H1(w), H1(x1)]    cache-local indices [1, 3)
MTP2: no hidden cache
```

One reusable BF16 flat buffer stores all requests and all non-final modules.
The buffer uses a module-major layout.
For `R` request slots, it contains this number of hidden rows:

```text
R * (1 + 2 + ... + K - 1) = R * K * (K - 1) / 2
```

The following example uses `K = 3` and `R = 2`:

```text
+----------+--------+---------+--------------+
| Flat row | Module | Request | State offset |
+----------+--------+---------+--------------+
| 0        | MTP0   | 0       | 0            |
| 1        | MTP0   | 1       | 0            |
| 2        | MTP1   | 0       | 0            |
| 3        | MTP1   | 0       | 1            |
| 4        | MTP1   | 1       | 0            |
| 5        | MTP1   | 1       | 1            |
+----------+--------+---------+--------------+
```

A standalone layout function maps a request slot and module to flat-buffer row indices.
The indices are contiguous, so the function can return `Range<u32>`.
The batch metadata builder can extend its reusable index vector from this range.

```rust
fn mtp_hidden_state_cache_row_range(
    max_request_slots: usize,
    req_slot: RawRequestSlot,
    module_index: usize,
) -> Range<u32> {
    let num_module_rows = module_index + 1;
    let req_slot = req_slot as usize;
    let module_base = max_request_slots * module_index * (module_index + 1) / 2;
    let req_base = module_base + req_slot * num_module_rows;

    req_base as u32..(req_base + num_module_rows) as u32
}
```

This function returns physical flat-buffer rows.
It does not return cache-local token indices.
The planner derives cache-local indices from `P` and the module index.

### Decode ranges

Let `V` be the number of committed continuation tokens after the first Main input token.
The continuation contains the remaining Main input tokens and the validated speculative prefix:

```text
continuation_tokens = main_input_tokens.skip(1) + validated_tokens
```

For the three examples below, Main has one input token before the speculative suffix.
Therefore, `V` equals the number of validated speculative tokens.
The next pending index is:

```text
P' = P + V + 1
```

Logical MTP module `m` executes this fresh range:

```text
start = P - max(m - V, 0)
end   = P'

fresh range = [start, P')
num rows    = max(m, V) + 1
```

The modules can have different fresh ranges.
The hidden dependencies remain aligned by cache-local index.

If Main input is `[p0, p1, w, x1, x2, x3]` and Main validates `x1`, then:

```text
main_input_tokens  = [p0, p1, w]
validated_tokens   = [x1]
continuation_tokens = [p1, w, x1]
V = 3
```

The same formulas apply when the cached prefix matches, or when no Decode tail exists.
The executor does not need a prompt-tail branch.

### Replacing a cached Decode tail

For a mismatch, module `m` starts at `P - m` and reuses no continuation tokens.
Decode input is `continuation_tokens + [sampled_token] + drafts.take(m)`.
It has `V + 1 + m` rows. Every lane still ends at `P + V + 1`.

For `K = 3`, new canonical input `[w, a, b, c]` replaces `[w, x1, x2, x3]`:

```text
+-------+----+----+----+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
+-------+----+----+----+----+----+----+----+----+
| Main  | t0 | t1 | t2 | w  | a  | b  | c  | y  |
| MTP0  | t1 | t2 | w  | a  | b  | c  | y  | z1 |
| MTP1  | t2 | w  | a  | b  | c  | y  | z1 | z2 |
| MTP2  | w  | a  | b  | c  | y  | z1 | z2 | z3 |
+-------+----+----+----+----+----+----+----+----+
                                             pending

Main @3: w, a, b, c
MTP0 @3: a, b, c, y
MTP1 @2: a, b, c, y, z1
MTP2 @1: a, b, c, y, z1, z2
```

For incoming Prefill with one Main row, the input has known lookahead `[w, a, b, c]`:

```text
Main @3: w
MTP0 @3: a
MTP1 @2: a, b
MTP2 @1: a, b, c
```

Every lane ends at index `4`. The next metadata is `Prefill { token_index: 4, token_ids: [a, b, c] }`.
For a window of `W` Main rows, module `m` executes `W + m` rows from `P - m`.
It uses the known source slice `[P + 1, P + W + m + 1)`.
Main still starts at `P`. Neither case changes Main GDN state selection or requires GDN rollback.
The first hidden input of each later module is old `H(w)` from the previous module's cache.
Subsequent hidden inputs come from that module's current output.

Ordinary Prefill remains rectangular and uses only current-wave hidden rows.
The first Decode after Prefill receives at least `K` known Main tokens.
For example, Prefill can leave `[t3, t4, t5]` at `P = 3`:

```text
Main @3: t3, t4, t5
MTP0 @3: t4, t5, y
MTP1 @3: t5, y, z1
MTP2 @3: y, z1, z2
```

These hidden inputs are all current-wave outputs. No Prefill hidden cache write is needed.

### Matching Decode examples

#### All reject

For `V = 0`, the new pending index is `P' = 4`:

```text
+-------+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  |
+-------+----+----+----+----+----+
| Main  | t0 | t1 | t2 | w  | y  |
| MTP0  | t1 | t2 | w  | y  | z1 |
| MTP1  | t2 | w  | y  | z1 | z2 |
| MTP2  | w  | y  | z1 | z2 | z3 |
+-------+----+----+----+----+----+
                             P'
```

The fresh MTP ranges are:

```text
MTP0: [3, 4)    y
MTP1: [2, 4)    y, z1
MTP2: [1, 4)    y, z1, z2
```

The first hidden input of each later module comes from the old hidden cache:

```text
index 3: Main H(w)       -> MTP0 executes y
index 2: cached H0(w)    -> MTP1 executes y
index 1: cached H1(w)    -> MTP2 executes y
```

The new hidden-cache contents are:

```text
MTP0: [H0(y)]            cache-local indices [3, 4)
MTP1: [H1(y), H1(z1)]    cache-local indices [2, 4)
```

#### Partial accept

For `V = 1`, Main accepts `x1` and rejects `x2` and `x3`.
The new pending index is `P' = 5`:

```text
+-------+----+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  | 5  |
+-------+----+----+----+----+----+----+
| Main  | t0 | t1 | t2 | w  | x1 | y  |
| MTP0  | t1 | t2 | w  | x1 | y  | z1 |
| MTP1  | t2 | w  | x1 | y  | z1 | z2 |
| MTP2  | w  | x1 | y  | z1 | z2 | z3 |
+-------+----+----+----+----+----+----+
                                  P'
```

The fresh MTP ranges are:

```text
MTP0: [3, 5)    x1, y
MTP1: [3, 5)    y, z1
MTP2: [2, 5)    y, z1, z2
```

The hidden dependencies are:

```text
index 3: Main H(w)        -> MTP0 executes x1
index 4: Main H(x1)       -> MTP0 executes y

index 3: fresh H0(x1)     -> MTP1 executes y
index 4: fresh H0(y)      -> MTP1 executes z1

index 2: cached H1(x1)    -> MTP2 executes y
index 3: fresh H1(y)      -> MTP2 executes z1
index 4: fresh H1(z1)     -> MTP2 executes z2
```

The new hidden-cache contents are:

```text
MTP0: [H0(y)]            cache-local indices [4, 5)
MTP1: [H1(y), H1(z1)]    cache-local indices [3, 5)
```

#### All accept

For `V = 3`, the new pending index is `P' = 7`:

```text
+-------+----+----+----+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
+-------+----+----+----+----+----+----+----+----+
| Main  | t0 | t1 | t2 | w  | x1 | x2 | x3 | y  |
| MTP0  | t1 | t2 | w  | x1 | x2 | x3 | y  | z1 |
| MTP1  | t2 | w  | x1 | x2 | x3 | y  | z1 | z2 |
| MTP2  | w  | x1 | x2 | x3 | y  | z1 | z2 | z3 |
+-------+----+----+----+----+----+----+----+----+
                                            P'
```

All MTP modules execute `[3, 7)`.
All previous-hidden inputs come from the current wave.
The new hidden-cache contents are:

```text
MTP0: [H0(y)]            cache-local indices [6, 7)
MTP1: [H1(y), H1(z1)]    cache-local indices [5, 7)
```

### Hidden-cache read and write

The executor reads an old cache row before it overwrites that row.
It gathers old and fresh hidden rows into the contiguous `previous_hidden` scratch.
MTPEmbed combines this scratch with the token embedding input.

```text
previous-module old cache --+
                            +-- gather --> previous_hidden --+
previous-module output -----+                                |
                                                             +--> MTPEmbed projection
token IDs --------------------- embed ---> token_hidden -----+

After the gather:
previous-module output tail --- scatter --> previous-module cache
```

For each Decode request, module `m` retains its latest `m + 1` hidden rows.
Prefill does not write persistent hidden rows. A repair Prefill can read the preceding Decode's hidden rows.
The first Decode after Prefill keeps at least `K` known Main tokens, so it uses only fresh hidden inputs.
That Decode writes each complete retained tail before later Decode calls can read cached rows.
This also permits a first Prefill or Decode at a nonzero index after a prefix-cache hit.
Without repair, module `m + 1` reads old hidden-cache slot `V` only when `m >= V`.
With repair, each later module reads slot `0` before it consumes fresh output.

```text
V = 0: MTP1 reads MTP0 slot 0. MTP2 reads MTP1 slot 0.
V = 1: MTP1 reads no slot. MTP2 reads MTP1 slot 1.
V = 3: MTP1 and MTP2 read no old cache slot.
```

The hidden-state transfer gathers previous hidden rows before it scatters the retained tail.
The replay sequence then runs MTPEmbed and the next MTP body.
This read-before-write order permits one hidden-cache buffer.
It does not require ping-pong storage.

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
    decode_plan.rs            Decode token and cache-local index plan
    embed.rs                 hidden and token input composition
    hidden_state_cache.rs     per-request cross-step BF16 hidden-state cache
    hidden_state_transfer.rs  previous-hidden routing and retained-tail writeback
    layer.rs                 physical GQA and MLP layer
    mod.rs                   body, final norm, and replay key
  executor/
    mtp.rs                   logical-step loop and proposal output

crates/inference-backend-metal/src/operators/
  row_route.rs               BF16 row gather from Main, MTP, and hidden cache
  row_scatter.rs             BF16 retained-tail writes to the hidden cache

crates/inference-executor-metal/src/sampling/
  top_k_replay.rs            DraftSampling replay
  spec_probs.rs              sparse draft-distribution storage
```

## Verification

Focused tests cover checkpoint validation, exact bindings, input composition, cache-lane mapping, replay parameters,
sequential sampling, sparse distributions, unshifted GDN transactions, and rejection.
Tail tests cover missing drafts, changed cached IDs, the uncached final draft, slot reset, and cross-block repair.
`inference-runtime-service/tests/generation_resume.rs` compares resumed model output with a fresh canonical history.

Use [`service.md`](service.md) for end-to-end commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before a performance claim.
