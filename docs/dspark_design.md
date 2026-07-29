# Qwen3x DSpark Components

This document describes the current backend-neutral and Metal-backend DSpark components.
The Qwen3 executor does not yet load or execute these components at this commit.
The Qwen3.5 executor continues to use MTP.

## Current scope

The repository contains these foundations:

- The official flat Qwen3 DSpark configuration contract.
- Exact source and affine checkpoint binding trees.
- The `qwen3_dspark_quantize` BF16-to-affine converter.
- Backend-neutral fixed-block attention geometry.
- A reusable Metal dense bidirectional block-SDPA component.
- Metal DSpark context, block-attention, layer, embedding, output, and Markov-sampling components.

The repository no longer contains the unwired Qwen3.5-era DSpark model implementation.

## Source layout

```text
crates/inference-executor-core/src/
  attn/gqa/
    dspark_core.rs               fixed-block attention geometry and metadata
  model/qwen/v3_x/dspark/
    config.rs                    official flat configuration contract
    weight_layout.rs             exact source and affine binding trees
  bin/
    qwen3_dspark_quantize.rs     official BF16-to-affine converter

crates/inference-backend-metal/src/components/
  gqa_block_attention.rs         dense bidirectional block-SDPA component
  gqa_block_attention_test.rs    component correctness tests
  metal/gqa_block_sdpa.metal     dense bidirectional block-SDPA kernel

crates/inference-executor-metal/src/
  attn/dspark/
    backend.rs                   history and block-attention replay composition
    context.rs                   persistent DSpark context owner
    metadata.rs                  proposal-block attention metadata
    scratch.rs                   proposal-local K/V and SDPA scratch
    state.rs                     DSpark GQA state and page-table owner
  model/qwen/v3_x/dspark/
    attention.rs                 Qwen3x DSpark attention weights and record path
    embed.rs                     anchor and MASK embedding component
    layer.rs                     independent Qwen3xDSparkLayer role
    main_feature.rs              selected Main-output projection
    model.rs                     context and proposal-body components
    output.rs                    proposal gather and unembedding component
    plan.rs                      validated Metal execution plan
  sampling/
    dspark_markov.rs             sequential Markov correction and sampling
```

`inference-executor-core` owns model semantics, configuration, weight names, and replay-independent geometry.
`inference-backend-metal` owns kernel resources, buffer bindings, and dispatch.
`inference-executor-metal` owns DSpark tensor objects, page interpretation, replay composition, scratch, and sampling.

The Qwen3 executor integration is a separate later commit.

## Configuration and weights

`Qwen3xDSparkConfig` parses the official checkpoint fields.
It validates fixed-block geometry, selected Main layers, RoPE, ungated GQA, and the `vanilla` Markov head.

`Qwen3xDSparkWeightBindings` defines the exact checkpoint tree.
It supports official BF16 source weights and the affine runtime checkpoint.

The converter command is:

```sh
cargo run -p inference-executor-core --bin qwen3_dspark_quantize -- \
  --input-dir /path/to/source \
  --output-dir /path/to/output \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not already exist.

## Block attention

`UngatedDSparkGQACore` defines proposal-block geometry.
The block contains one anchor row and the configured MASK rows.
Every row can attend to the complete local block.

`GQABlockSDPAKernel` computes one dense bidirectional local-block partial.
It writes the existing `SDPAPartialOutput` ABI.
The existing GQA reduce component can combine this partial with paged-history partials.

`UngatedDSparkGQA` combines paged-history partials and the dense bidirectional block partial.
`UngatedDSparkGQAState` owns the DSpark request page table and persistent context.
`DSparkBlockScratch` owns proposal-local Q/K/V and SDPA scratch.
Proposal-local K/V does not enter persistent context.

## Model and sampling components

`Qwen3xDSparkLayer` is an independent role type.
It composes ungated DSpark GQA, normalization, residual, and dense MLP components.
It does not extend `Qwen3MainLayer` or `Qwen35MTPLayer`.

`Qwen3xDSparkContext` records selected Main features into persistent DSpark context pages.
`Qwen3xDSparkBody` records one fixed proposal block.
`Qwen3xDSparkEmbed` produces the anchor and MASK rows.
`Qwen3xDSparkGatherUnembed` produces one logit row for each proposal position.

`DSparkMarkovSampling` applies the trained Markov correction and samples each fixed-block position in sequence.
Each sampled token is the Markov input for the next position.
The component writes sparse proposal distributions to `SpecProbsStore`.

These components do not own the Qwen3 batch transaction or executor submission lifecycle.

## Verification

Core tests cover configuration, bindings, geometry, and converter round trips.
Metal tests cover DSpark planning, selected Main-output projection, sequential Markov behavior, and block attention.
Executor integration and production-path performance evidence belong to later commits.
