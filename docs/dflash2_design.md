# Qwen3x DFlash2 Design

This document describes the current Qwen3x DFlash2 executor.
It covers the Qwen3.5-family outer composition, independent Spec Prefill and Decode replays, persistent history K/V,
sliding-history attention, dynamic grouped convolution, and candidate selection.

## Ownership boundary

Qwen3x DFlash2 is a model role.
It is a peer of Qwen3x DSpark.
It is not a DSpark flag.

The Qwen3.5-family executor owns the DFlash2 outer lifecycle.
The DFlash2 owner owns its checkpoint, persistent page table, replay caches, proposal workspaces, and selector state.
Main owns token embedding, the transformer, selected residual capture calls, unembedding, target sampling, and rejection
sampling.
Runtime core owns scheduling, request lifecycle, physical page allocation, and page IDs.

The DFlash2 implementation reuses lower-level components only when their contracts match.
It reuses `MainResidualCapture`, Main `Embed`, Main `Unembed`, affine projections, RMSNorm/RoPE, paged K/V write,
`BlockSpecGQA`, dense MLP, raw Top-K merge, and sparse rejection.
It owns DFlash2-specific layer composition, sliding-range construction, dynamic convolution, candidate lattice, and
proposal selection.

## Outer data flow

```text
previous Spec proposal
{draft tokens, draft probabilities}
                   |
                   v
request microbatch
{committed tokens + previous speculative suffix}
                   |
                   v
+---------------------------- Main module -----------------------------+
| token IDs -> Main Embed -> Main transformer                           |
|                              |                                        |
|                              +-> selected residual capture            |
|                              |                                        |
|                              v                                        |
|                    Gather + Main Unembed                              |
|                              |                                        |
|                              v                                        |
|              normal sampling or rejection sampling                   |
|                              |                                        |
|                              v                                        |
|        {validated draft prefix, newly sampled anchor token}           |
+------------------------------+----------------------------------------+
                               |
                               v
+----------------------- Qwen3x DFlash2 owner -------------------------+
| Spec Prefill: selected Main capture -> persistent history K/V        |
| Spec Decode:  [anchor, MASK, ..., MASK] -> proposal distributions    |
+-----------------------------------------------------------------------+
```

Main and DFlash2 are separate invocations.
Main does not record DFlash2 work.
The service waits for Main and reads the Main decision before it builds the DFlash2 Decode request.

## Independent Prefill and Decode replays

Spec Prefill and Spec Decode are independent replay recordings.
The outer owner can submit Prefill without Decode.
It can submit Decode without new Prefill work.
It can also submit Prefill followed by Decode in one model-specific Spec sequence.

```text
Spec Prefill
  selected Main residual capture
    -> affine Main-feature projection
    -> hidden normalization
    -> each DFlash2 layer Wk and Wv
    -> K normalization and RoPE
    -> persistent paged K/V write

Spec Decode
  [anchor, MASK, ..., MASK]
    -> Main Embed view
    -> DFlash2 body
    -> final normalization
    -> gather MASK rows
    -> Main Unembed view
    -> raw Top-K
    -> DFlash2 candidate lattice and path walk
    -> proposal tokens and sparse draft distributions
```

Main writes each selected residual directly into its assigned column range in one capture buffer.
DFlash2 Prefill does not run a concatenate or copy kernel.
The Main-feature projection runs once for the captured token rows.
All DFlash2 layers read that projected feature.
Each layer independently applies its Wk and Wv projection and writes its persistent K/V.

The loader does not assume that Wk and Wv have the same dtype or quantization layout.
It resolves every Q, K, V, output, gate, up, and down affine layout from its exact binding subtree.
The current implementation keeps per-layer weight owners.
It does not aggregate layer weights or add an output scatter.

## Decode attention

`--num-spec-tokens K` creates K proposal rows.
The complete query block has `K + 1` rows:

```text
[anchor, MASK_0, ..., MASK_(K-1)]
```

The anchor and MASK rows form one Q tile or part of one Q tile.
Each layer computes two attention contributions:

```text
sliding persistent history
  -> SplitKV TiledQ partials

full bidirectional local block
  -> dense block-SDPA partial

history partials + block partial
  -> shared partial-output reducer
```

The persistent cache stores every history token.
The sliding window limits history reads only.
For each query row, the DFlash2 owner supplies this explicit half-open range:

```text
[max(0, query_position + 1 - sliding_window), anchor_position)
```

The upper bound excludes the anchor because anchor and MASK K/V belong to the local bidirectional block.
Each later MASK row can have a later history lower bound.
The model owner derives these ranges.
The shared GQA backend does not infer a zero lower bound or own DFlash2 window policy.

## Dynamic grouped convolution

Each layer applies DFlash2 dynamic grouped convolution around both the attention branch and the MLP branch.
The checkpoint supplies one affine kernel projection and one F32 base kernel for each branch.

```text
normalized hidden
  -> kernel projection
  -> grouped dynamic coefficients
  -> convolution Prepare
  -> attention or MLP branch
  -> convolution Finish
  -> residual add
```

The Metal grid covers request-local query blocks and hidden values.
The replay argument limits work to active query blocks.
The kernel does not combine requests or read state from another request.

## Candidate selection

The output owner gathers only MASK rows.
It uses the immutable Main Unembed owner to produce unary vocabulary logits.
Raw Top-K merge selects exactly `selector_top_k` candidates for each proposal position.

```text
projected hidden + predecessor codebook + successor codebook + unary logits
  -> edge scores [request, step, predecessor, successor]
  -> sequential request-local probabilistic path walk
  -> proposal token IDs and probabilities
  -> exact sparse draft-distribution rows
```

The first step uses the anchor token as the predecessor.
Each later step uses the candidate selected at the prior step.
The selector applies request temperature to the fixed candidate set.
It does not apply a second top-p truncation after candidate construction.
The sparse draft distribution stores the exact distribution that generated each proposal token.
Main rejection sampling can therefore use the normal sparse rejection contract.

DFlash2 has no confidence head.
The response adapter supplies `1.0` confidence for each proposal token.

## Cache and state lifecycle

Main K/V and DFlash2 history K/V share runtime cache lane 0.
For each logical cache block, the executor interprets the supplied page IDs as:

```text
[Main page IDs | DFlash2 page IDs]
```

The executor validates and splits this list once.
Main and DFlash2 then receive complete role-local page-ID lists.
`GQARequestPageTable` does not parse model roles or runtime cache lanes.

DFlash2 persistent state contains its request page table and the shared physical page arena.
State snapshots use the separate `dflash2-gqa-request-page-table` file.
Proposal-local Q/K/V, block partials, convolution coefficients, candidate tensors, and selector outputs are ephemeral.
They do not enter the snapshot.

An executor reset fans out to the active DFlash2 request slots.
State unload and reload use the same full-state and selected-state boundaries as the peer DSpark owner.

## Checkpoint contract

`Qwen3xDFlash2Config` adapts the published `DFlash2DraftModel` schema to the repository's flat canonical schema.
Unknown architectures and unknown nested fields fail at the checkpoint boundary.
The normalized config validates Main compatibility and these DFlash2 dimensions:

- Target residual layer IDs
- Query-block size
- Sliding window
- Attention heads and head width
- Dynamic-convolution group and kernel sizes
- Selector rank and candidate count

`Qwen3xDFlash2WeightBindings` accepts only the exact published source manifest or the exact affine manifest.
DFlash2 reuses Main embedding and unembedding, so the DFlash2 manifest must not contain substitute embedding or
unembedding tensors.

The converter writes affine matrix payloads as `U32` and affine parameters as F32.
It keeps norms and dynamic-convolution base kernels as F32.
The produced checkpoint must contain no BF16 tensor.
The default conversion uses group size 64 and 4-bit affine matrices.
It uses 6-bit affine matrices for layer 2 and layer 4 `v_proj` and `down_proj` weights.

## Source layout

```text
crates/inference-executor-core/src/model/qwen/v3_x/dflash2/
  config.rs
  weight_layout.rs

crates/inference-executor-metal/src/model/qwen/v3_x/dflash2/
  mod.rs
  load.rs
  execution.rs
  execution/file_io.rs
  main_feature.rs
  embed.rs
  attention.rs
  conv.rs
  layer.rs
  model.rs
  output.rs

crates/inference-executor-metal/src/model/qwen/v3_5/executor/
  dflash2.rs

crates/inference-backend-metal/src/components/
  dynamic_grouped_conv.rs
  metal/dynamic_grouped_conv.metal
  sampling/dflash2_selector.rs
  metal/dflash2_selector.metal
```

## Verification

Focused tests cover configuration adaptation, exact tensor manifests, independent affine layouts, per-query sliding
ranges, dynamic grouped-convolution parity, candidate-selector parity, service mode normalization, and speculator
mutual exclusion.
The DFlash2 implementation does not require a real-model integration test for these structural contracts.

Use [`service.md`](service.md) for conversion and startup commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before you make a performance claim.
