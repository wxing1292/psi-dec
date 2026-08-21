# Spec Prefill and Decode Design Draft

Temporary status: This document records the active DSpark Prefill and Decode implementation decision.
It also records the corresponding DFlash2 target.
Current-component documents describe the checked-in `src`.

## Decision

DSpark and DFlash2 use two independent Spec invocations:

```text
Spec Prefill
Spec Decode
```

`Spec Prefill` creates persistent Spec history K/V from Main outputs.
`Spec Decode` runs one anchor-and-MASK proposal block.

Main must not record or own `Spec Prefill`.
Main only writes the selected residual capture that `Spec Prefill` consumes.

## Terms

This document uses these terms:

| Term | Meaning |
| --- | --- |
| Main capture | Selected Main residual outputs in one prearranged output buffer. |
| Spec Prefill | The Spec invocation that creates persistent history K/V. |
| Spec Decode | The Spec invocation that creates proposals from an anchor-and-MASK block. |
| persistent history K/V | Per-layer Spec K/V that later proposal blocks can read. |
| proposal-local K/V | Temporary K/V for one anchor-and-MASK block. |

`DSpark` and `DFlash2` are different high-level model owners.
They use the same Spec Prefill and Spec Decode vocabulary.
Vanilla and MTP have separate outer model owners.
MTP keeps one combined Spec invocation and does not use the Prefill and Decode hooks in this document.

The outer composition must select one peer wrapper:

```text
Qwen model executor
  |- Vanilla
  |- MTP
  |- DSpark
  |    `- Qwen3xDSparkExecution
  `- DFlash2
       `- DFlashExecution
```

`DFlashExecution` must not be a mode of `Qwen3xDSparkExecution`.

## Ownership

Main owns its normal model forward and selected residual capture points.
Main writes each selected residual directly into its assigned output columns.
Main does not own Spec projection weights, Spec K/V pages, or Spec proposal logic.

The outer DSpark or DFlash2 model owner owns these resources and operations:

- The Main capture destination and its selected-layer layout.
- The Main-feature projection and normalization.
- Per-layer Spec K/V projection weights.
- Persistent Spec K/V interpretation and writes.
- Proposal-block embedding, layers, output, and sampling.
- Replay composition for Spec Prefill and Spec Decode.

Each outer model owner selects one lifecycle contract.
It must not route MTP through DSpark Prefill or Decode.
Shared Main implementation does not make the high-level owners interchangeable.

The runtime core owns page allocation, page IDs, request lifecycle, and token metadata.
The runtime core does not interpret Spec tensor layouts.

The Metal backend may share low-level affine, K-norm, RoPE, paged-write, SDPA, and reduce components.
DSpark and DFlash2 must keep separate high-level Prefill and Decode owners.

## Main capture layout

Main writes selected residuals into one prearranged buffer:

```text
selected_main_states: [Tmain, Lselected * H]
```

`Tmain` is the number of Main rows in the invocation.
`Lselected` is the number of selected Main layers.
`H` is the Main hidden dimension.

Each selected Main layer writes directly to its assigned column range.
The implementation must not add a separate concatenate kernel or copy.
The column order must equal the checkpoint `target_layer_ids` order.

The capture owner supplies opaque capture destinations through `MainResidualCapture`.
Main must not reference a DSpark or DFlash2 concrete type.

## Spec Prefill

Spec Prefill consumes Main capture rows that can become Spec history.
It does not execute the anchor token or MASK tokens.

Spec Prefill performs this computation:

```text
selected_main_states
  -> fc
  -> hidden_norm
  -> main_feature
  -> per-layer Wk and Wv
  -> K-norm and RoPE
  -> persistent paged-KV write
```

For Main token `t`, all Spec layers consume the same `main_feature[t]`.
Each Spec layer uses its own K and V weights.

Spec Prefill has these semantic outputs:

```text
K_history[layer, token, kv_head, head_dim]
V_history[layer, token, kv_head, head_dim]
```

Spec Prefill must run after each Main invocation that produces applicable capture rows.
This rule includes prompt prefill, Main decode, and proposal verification.
A prefill-only Main batch can therefore run Spec Prefill without Spec Decode.

Spec Prefill logically publishes only committed history.
An implementation may compute unpublished candidate rows when this avoids an expensive gather.
The visible history range must exclude rejected or uncommitted rows.

## Spec Decode

Let `N` be the fixed proposal width.
Spec Decode consumes this input block for each request:

```text
row 0       anchor token
row 1..N-1  MASK tokens
```

Spec Decode performs this computation:

```text
anchor-and-MASK embedding
  -> DSpark or DFlash2 model body
  -> persistent-history attention
  -> proposal-block bidirectional attention
  -> partial-output reduce
  -> output projection and remaining layer work
  -> proposal logits and tokens
```

The Spec Decode owner owns this complete composition.
For DSpark, it owns the transformer body, Markov correction, confidence head, and proposal sampling.
For DFlash2, it owns the transformer body, dynamic convolution, candidate Top-K, and path selector.
These operations are not a separate high-level owner beside Spec Decode.

Each Spec layer creates proposal-local Q/K/V from that layer's current hidden input.
The layer reads persistent history K/V that Spec Prefill created for the same layer.
The executor discards proposal-local Q/K/V after the proposal invocation.

Spec Decode must not write proposal-local K/V to persistent history pages.
The newly sampled Main anchor has no Main-derived Spec history K/V yet.
It becomes eligible for Spec Prefill only after a later Main invocation processes it.

## Invocation order

The model executor uses this semantic order:

```text
Main invocation
  -> Main completion and result processing
  -> Spec Prefill when applicable Main capture rows exist
  -> Spec Decode when an anchor exists
```

Spec Prefill and Spec Decode are independent invocations.
They have separate inputs, metadata, replay topology, and call conditions.

The invocations have a producer-consumer dependency for each request.
Spec Decode must observe all persistent K/V that the preceding Spec Prefill publishes.

## Submission composition

Main submission must not contain Spec Prefill or Spec Decode work.
Model components must not submit or wait internally.

Separate semantic invocations do not require separate GPU submissions.
The model executor may use these compositions:

```text
prefill-only batch:
  Main submission
  Spec submission [Spec Prefill]

decode-ready batch:
  Main submission
  Spec submission [Spec Prefill, Spec Decode]
```

The executor may use separate Spec submissions when a real dependency requires that boundary.
The implementation must not add a submission boundary only for naming symmetry.

The Main capture resources must remain alive until Spec Prefill completes.
The executor must not overwrite or reuse them while Spec Prefill is in flight.

## Visible history

Each Spec Decode query reads one explicit half-open history range:

```text
[visible_history_start, anchor_position)
```

The metadata contract must provide both bounds.
It must not assume that `visible_history_start` is `0`.

Every anchor and MASK row in one proposal block can use the applicable persistent history.
Every row can also attend to the complete proposal-local block.
This rule applies when the model contract requires bidirectional block attention.

## API vocabulary

The model-specific execution owners use symmetric Prefill and Decode entry points:

```rust
Qwen3xDSparkExecution::record_prefill(...)
Qwen3xDSparkExecution::record_decode(...)

DFlashExecution::record_prefill(...)
DFlashExecution::record_decode(...)
```

The final DFlash2 type prefix must follow the repository checkpoint and model-role vocabulary.
The implementation must not add `*Stage` wrappers only to create visual symmetry.

The public or executor-level terms are `Spec Prefill` and `Spec Decode`.
The implementation can use `prefill` and `decode` when the enclosing owner already establishes the Spec role.

## Shared low-level components

DSpark and DFlash2 may reuse a low-level component only when their complete contracts match.
Compatibility includes tensor shape, data type, quantization layout, storage layout, RoPE, and paged-write ABI.

The implementation must not assume that K and V weights use the same data type or quantization layout.
It must group K weights and V weights independently when it combines compatible projections.

A cross-layer projection optimization applies only to Spec Prefill.
All layers consume the same `main_feature` in that invocation.
Main forward and Spec Decode use layer-dependent hidden inputs and cannot use that optimization.

## Correctness invariants

The implementation must preserve these invariants:

1. Main capture order equals `target_layer_ids` order.
2. Main writes selected residuals directly into the final capture layout.
3. Spec Prefill excludes the newly sampled anchor and all MASK rows.
4. Spec Decode reads persistent K/V only from its explicit visible range.
5. Spec Decode uses proposal-local K/V only for its current block.
6. Proposal-local K/V never becomes persistent state.
7. Rejected or uncommitted Main rows never become visible Spec history.
8. Spec Decode observes Spec Prefill writes before it reads the same pages.
9. Runtime core does not interpret DSpark or DFlash2 tensor layouts.
10. DSpark and DFlash2 keep separate high-level owners.

## DSpark implementation decision

The DSpark implementation uses `Qwen3xDSparkPrefill` in an independent Spec invocation.
The Main submission contains no DSpark work.
`Qwen3xDSparkExecution::record_decode` records the complete proposal graph.

The event loop can record Prefill alone or Prefill followed by Decode.
A decode-ready batch uses one ordered Spec submission.
The replay sequence inserts a dispatch barrier between Prefill and Decode programs.

The current path uses physical overcompute for Main verification rows.
The accepted request extent controls later visibility.
Rejected rows remain invisible and can be overwritten by a later Main invocation.

Prefill and Decode use independent replay keys.
Prefill keys use the Main capture row count.
Decode keys use the proposal topology.
They also use independent `Qwen3xDSparkPrefillRecording` and `Qwen3xDSparkDecodeRecording` values.
One ordered Metal submission can batch both replay recordings without merging their owners.

## DFlash2 target owner

DFlash2 is not implemented in the current `src`.
Its future wrapper is a peer of the DSpark and MTP wrappers.
It owns an independent `DFlashExecution` with DFlash-specific Prefill and Decode recordings.

```text
DFlashExecution
  |
  +-- Spec Prefill
  |     Main capture
  |       -> Main-feature projection and norm
  |       -> per-layer Wk and Wv
  |       -> K norm and RoPE
  |       -> persistent paged history K/V
  |
  `-- Spec Decode
        [anchor, MASK, ..., MASK]
          -> DFlash Embed
          -> DFlash layers
               history GQA + block bidirectional attention
               DFlash dynamic convolution
               MLP and residual path
          -> Gather + Unembed
          -> per-position candidate Top-K
          -> coherent-path selector
          -> proposal tokens and probabilities
```

The initial DFlash2 target stores all applicable committed history K/V in its persistent paged cache.
Its `sliding_window=2048` contract changes only the Decode attention read range:

```text
[max(visible_history_start, anchor_position - 2048), anchor_position)
```

The shared paged-KV write path must not discard older K/V because of the DFlash2 read policy.
A future DFlash-specific ring-cache optimization may change physical retention only when its owner proves equivalent
visibility and page-lifecycle behavior.

## Deferred optimization decisions

- Group compatible K projections across layers without assuming that K and V have the same data type or quantization
  layout.
- Write the grouped projection output directly in the K-norm, RoPE, and paged-write input layout.
- Evaluate DFlash2 only after its complete high-level owner and checkpoint contract are defined.
- Verify capture-buffer lifetime across Main and Spec submissions.
- Verify K and V weight grouping from exact checkpoint layouts.
- Add DFlash2-specific model work without moving it into shared GQA components.

After implementation, update the current component documents in the same change.
