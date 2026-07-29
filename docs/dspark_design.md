# Qwen3x DSpark Foundations

This document describes the current backend-neutral and Metal-backend DSpark foundations.
The Qwen3 executor does not yet load or execute these components at this commit.
The Qwen3.5 executor continues to use MTP.

## Current scope

The repository contains these foundations:

- The official flat Qwen3 DSpark configuration contract.
- Exact source and affine checkpoint binding trees.
- The `qwen3_dspark_quantize` BF16-to-affine converter.
- Backend-neutral fixed-block attention geometry.
- A reusable Metal dense bidirectional block-SDPA component.

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
```

`inference-executor-core` owns model semantics, configuration, weight names, and replay-independent geometry.
`inference-backend-metal` owns kernel resources, buffer bindings, and dispatch.

The Qwen3x model composition and Qwen3 executor integration are separate later commits.

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

This component does not own model layers, persistent context pages, proposal lifecycle, or sampling.

## Verification

Core tests cover configuration, bindings, geometry, and converter round trips.
Metal tests compare block attention with CPU reference results.
Executor integration and production-path performance evidence belong to later commits.
