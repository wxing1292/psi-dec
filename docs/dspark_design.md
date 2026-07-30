# Qwen3 DSpark

This document describes the current fixed-block Qwen3 DSpark implementation.
[`executor_qwen.md`](executor_qwen.md) owns the Qwen executor lifecycle.
[`executor_sampling.md`](executor_sampling.md) owns sparse sampling and rejection.
[`executor_gqa.md`](executor_gqa.md) owns GQA page and attention-kernel details.

## Role terminology

Model roles use `Main`, `MTP`, and `DSpark` or `Spec`.
They do not use `Target` or `Draft`.

The official checkpoint fields keep their upstream names.
These fields include `target_layer_ids` and `num_target_layers`.
The generic rejection-sampling API also keeps `target_*` and `draft_*`.
In that API, the terms identify two probability distributions.
The backend-neutral `ResidualCaptureTarget` uses `Target` to mean a write destination.
The tracing `target:` field identifies a log category.
None of these names identify model roles.

## Current scope

The Qwen3 executor can load one optional DSpark checkpoint.
The implementation follows the official flat `Qwen3xDSparkConfig` schema.
It supports an ungated GQA backbone and a `vanilla` Markov head.

The first milestone has these limits:

- It produces exactly `block_size` proposals for each active decode request.
- It recognizes confidence-head fields and weights.
- It does not materialize or execute the confidence head.
- It does not schedule variable proposal lengths.
- It does not support a gated DSpark GQA checkpoint.
- It permits one in-flight batch for each executor.

The Qwen3.5 executor continues to use MTP.
It does not own a DSpark implementation.

## Source layout

```text
crates/inference-executor-core/src/
  attn/gqa/
    dspark_core.rs               block geometry and metadata
  model/qwen/v3_x/dspark/
    config.rs                    official configuration contract
    reference.rs                 CPU Markov and sampling reference
    weight_layout.rs             exact source and affine binding trees
  bin/
    qwen3_dspark_quantize.rs     official BF16 -> affine converter

crates/inference-backend-metal/src/components/
  dspark_markov_sampling.rs      fused Markov and tile-Top-K map component
  gqa_block_attention.rs         dense block-SDPA component
  metal/dspark_markov_sampling.metal
  metal/gqa_block_sdpa.metal     dense bidirectional block kernel

crates/inference-executor-metal/src/
  attn/dspark/
    backend.rs                   ungated DSpark GQA replay graph
    context.rs                   persistent Main-context append
    metadata.rs                  history and block attention metadata
    scratch.rs                   fixed-capacity proposal scratch
    state.rs                     page-table and metadata lifecycle
  model/qwen/v3_x/dspark/
    attention.rs                 DSpark attention composition
    embed.rs                     DSpark token embedding replay
    layer.rs                     independent Qwen3xDSparkLayer
    model.rs                     context and body replay owners
    output.rs                    gather, unembed, and Markov sampling
    plan.rs                      configuration-to-backend conversion
    main_feature.rs              selected Main residual projection
  model/qwen/v3/executor/
    dspark.rs                    Qwen3 proposal orchestration
  sampling/
    dspark_markov.rs             sequential Markov correction and sampling
    rejection_replay.rs          shared sparse rejection replay and microbatch contract
```

The source does not retain the earlier `qwen/v3_5/dspark/` implementation.
The current Qwen3 path does not wire that implementation through a compatibility layer.

## Model semantics

DSpark uses one fixed transformer forward.
It is not an iterative diffusion process.
The input block for proposal length `N` is:

```text
input rows:
  anchor
  MASK
  ...
  MASK                 N rows total

proposal rows:
  token[0]
  token[1]
  ...
  token[N - 1]         N proposal distributions
```

Each block row can attend to all `N` local block rows.
Each block row can also attend to persistent history before the anchor position.
The block therefore uses bidirectional local attention and causal history attention.

The local block produces temporary Q/K/V.
The executor discards these values after proposal generation.
The next Main batch verifies the proposed token sequence.

The Markov head corrects proposal logits in position order.
Position `i` depends on the token sampled at position `i - 1`.
`DSparkMarkovSampling` owns this sequential correction and sampling loop.
It stores one sparse draft distribution for each proposed token.

Each position records this pair:

```text
DSparkMarkovTopKMap
  previous token
  -> quantized W1 row
  -> quantized W2
  -> add base logits
  -> tile-local Top-K

generic TopKSampleAndSparseDistribution
  -> global Top-K and top-p
  -> sample
  -> sparse draft distribution
```

The implementation records all pairs before one Spec submission.
The sampled token creates a GPU dependency between adjacent pairs.
A seven-position block uses 14 commands.
The implementation does not allocate full Markov latent, bias-logit, or corrected-logit buffers.

The official Qwen3 checkpoint uses `markov_rank = 256`.
The current fused map uses 128 threads per threadblock and a 64-token vocabulary tile.
Each thread holds eight F32 latent values.
It uses one sequential W2 dot accumulator.
It does not retain one accumulator for each output token.
Each threadblock uses 1,024 bytes of shared memory:

```text
latent       256 * sizeof(bf16) = 512 bytes
tile logits   64 * sizeof(f32)  = 256 bytes
tile IDs      64 * sizeof(i32)  = 256 bytes
```

Kernel construction checks the pipeline thread limit, the pipeline SIMD execution width, and the device shared-memory
limit.
The static geometry requires complete SIMDgroups.
It also requires enough threads for the 64-entry bitonic network.
The number of results per wave must divide the vocabulary tile.
Metal does not expose register allocation through `MTLComputePipelineState`.
The tile decision therefore also uses the source-level live-value count and a real-weight tile sweep.

## Main context

`Qwen3Main` exposes the narrow `Qwen3MainResidualCapture` contract.
It does not depend on a DSpark model or a Metal recorder.

`Qwen3xDSparkMainFeatureProjector` selects the configured Main residual outputs.
It projects those outputs into one Main feature for each Main token.
Each DSpark layer projects that feature to its persistent context K/V.

`Qwen3xDSparkContext` records this work after `Qwen3Main` in the same Main submission:

```text
MainEmbed
  -> Main
  -> DSparkContext
```

Persistent DSpark context follows accepted Main history.
Proposal-local K/V never enters this context.

## Attention composition

The official Qwen3 checkpoint uses ungated GQA.
`UngatedDSparkGQA` owns a model-neutral QKV attention graph.
It does not add a mode to `UngatedGQA`.

One DSpark attention call records this composition:

```text
QKV projection
  -> Q/K norm and RoPE
  -> history paged-SDPA map
  -> block bidirectional-SDPA map
  -> existing GQA partial-output reduce
  -> output projection
```

The history path reads persistent paged K/V.
The block path reads dense local K/V from `DSparkBlockScratch`.
Both map paths write `SDPAPartialOutput` records with the existing ABI.
The existing `GQASDPAReduceKernel` combines both sets.

The history metadata supplies a half-open visible range.
For an anchor at position `p`, that range is `[0, p)`.
The block kernel supplies the complete local block.

One block-bidirectional map Task owns one Q token and one Q head.
The backend fixes this Task to one 32-thread SIMDgroup.
For the official `head_dim = 128`, each thread keeps four F32 Q values.
The logical Q register payload is 16 bytes per thread.
The threadblock keeps seven F32 logits in 28 bytes of shared memory.
It does not keep a shared reduction array.

This geometry preserves `block_size * num_q_heads` independent threadblocks for one request.
The 32 lanes cooperate on each 128-value Q/K dot product.
Each lane then computes four output dimensions.
Larger threadblocks add SIMDgroups without independent work at this Task boundary.
The backend does not expose this thread choice to the executor.

Kernel construction checks the pipeline SIMD width, the pipeline thread limit, and the device shared-memory limit.
Metal does not expose the compiler register allocation.
The implementation therefore validates the logical live-value count and measures the production shape.

## Page ownership

The runtime core allocates one flat page-ID span for each logical cache block.
The executor splits that span:

```text
[0 .. main_page_count)             Main K/V
[main_page_count .. total_count)   persistent DSpark context K/V
```

The executor updates separate Main and DSpark page tables.
The runtime core allocates and releases both spans as one cache block.
The runtime core does not parse the model-specific split.

The service derives:

```text
num_pages_per_kv_block =
    main_pages_per_block
  + dspark_pages_per_block
```

A DSpark-disabled Qwen3 executor uses only the Main span.

## Scratch capacity

`DSparkBlockScratch` is an executor-owned resource.
It contains local Q/K/V, normalized Q/K, attention partials, and reduced output.

Define:

```text
T_capacity = max_requests * block_size
P_capacity = next_power_of_two(2 * T_capacity)
```

Each local query needs at least one history partial and one block partial.
The metadata builder can divide long history across available history tasks.
It cannot exceed `P_capacity`.

The partial buffers use this capacity:

```text
partial_max_logits[P_capacity, num_q_heads]       f32
partial_exp_sums[P_capacity, num_q_heads]         f32
partial_output[P_capacity, num_q_heads, head_dim] model dtype
```

Context length does not set this capacity.
One history task can process many K/V tiles with online softmax.

## Sparse distribution identity

Draft distributions persist until the next Main batch.
Their identity is request-slot based:

```text
draft_distribution_index =
    req_slot * block_size
  + proposal_position
```

Main verification distributions exist only in one submission.
They use compact active-row indices:

```text
0, 1, ..., num_active_target_distributions - 1
```

This difference follows the two data lifetimes.
It is not an optimization exception.

## Execution lifecycle

The service owns all `submit` and `wait` boundaries.
Components only prepare, record, or read completed output.

The Main submission is:

```text
MainEmbed
  -> Main
  -> DSparkContext                         when DSpark is enabled
  -> GatherUnembed                         when sample rows exist
  -> Sampling or RejectionSampling         when sample rows exist
```

The Spec submission is:

```text
DSparkEmbed
  -> DSpark
  -> DSparkGatherUnembed
  -> DSparkSampling
```

The CPU reads Main results before it creates the anchor block.
This dependency requires two submissions.
No component submits or waits internally.

Prefill, decode, and mixed batches use the same Main hook order.
An empty unembed or sampling input records no component.
The service runs Spec only when Main returns at least one decode result.

The lifecycle does not use `main_stage_submitted`.
It does not use `read_sampling_output`.
It does not create a dummy completed submission.

## Configuration and conversion

The loader validates the DSpark configuration against the Qwen3 Main configuration.
It rejects unsupported attention, dtype, RoPE, `target_layer_ids`, and Markov variants.
The affine loader requires exact semantic bindings.

Convert an official checkpoint with this command:

```sh
cargo run -p inference-executor-core --bin qwen3_dspark_quantize -- \
  --input-dir /path/to/Qwen3-DSpark \
  --output-dir /path/to/Qwen3-DSpark-affine \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not exist.
The converter writes `model.safetensors` and `model.safetensors.index.json`.
It preserves DSpark-owned embedding and unembedding when the source provides them.
It omits confidence weights and reports that limit.

## Verification

Unit and Metal parity tests cover:

- Official `target_layer_ids` validation
- Exact source and affine bindings
- Converter round-trip and safetensors index generation
- Anchor plus `N - 1` MASK construction
- Flat Main and DSpark page splitting
- Bidirectional block attention
- Combined history and block reduction
- Static partial-scratch capacity
- Sequential Markov sampling
- Padded Markov replay buckets and non-contiguous request slots
- Ragged sparse rejection

The source, test, and benchmark boundaries are:

```text
src/
  production semantics, components, replay wiring, and model execution

src/*_test.rs
  unit and Metal parity tests for one production module

benches/gqa/block_attn.rs
  model-independent block-bidirectional SDPA map timing

benches/rejection_sampling.rs
  model-independent fused DSpark Markov map timing

benches/qwen3/dspark.rs
benches/qwen3/dspark/fixture.rs
  real-checkpoint Main and DSpark executor lifecycle timing
```

The benchmark fixture uses only the public executor contract.
Production `src` has no benchmark-only control path or state.

Run the component benchmark with:

```sh
cargo bench -p inference-backend-metal --bench gqa_block_attn -- \
  --block-sizes 7 --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode dspark-markov-top-k-map --rows 1 --top-k 20 --vocab 151936 \
  --markov-rank 256 --markov-w1-group-size 64 --markov-w1-bits 4 \
  --markov-w2-group-size 64 --markov-w2-bits 8 \
  --iters 1 --warmup-iters 0 --runs 1
```

Run the executor benchmark with:

```sh
cargo bench -p inference-executor-metal --bench qwen3_dspark -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --cases dspark --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1
```

The release Qwen3 service passed deterministic and probabilistic DSpark decode.
The deterministic 64-token decoded text matched the Main-only decoded text.

The first end-to-end performance comparison found a regression.
The steady deterministic median was `41.180 tok/s` for Main-only and `33.611 tok/s` for DSpark.
The DSpark result was `18.4%` slower.
The steady executor breakdown attributes approximately `85.9%` of one DSpark cycle to the eight-token Main
verification submission.
It attributes approximately `14.3%` to the DSpark proposal submission.
Steady record, read, and commit work is negligible.
The sparse-rejection kernel measures `0.351 ms` for the matching one-request, seven-proposal geometry.
Two final fused 14-command Markov replay runs measure `1.461522 ms` and `1.389819 ms`.
The earlier five-command-per-step replay measured `1.729050 ms`.
The fused path reduces this isolated stage by `15.5%` to `19.6%`.
The measured `27.7%` proposal-token acceptance rate is below the approximately `38.6%` break-even rate.
Proposal-only optimization cannot recover the regression at this acceptance rate.
The performance verification is complete.
The fixed-block setup is not suitable for this Qwen3-14B workload on the tested Apple M3 Max.
This result does not invalidate the functional fixed-block milestone.
It prevents a performance-benefit claim for this setup.
The full evidence is in [`qwen3_dspark_design_draft.md`](qwen3_dspark_design_draft.md).

## Deferred work

[`future_work.md`](future_work.md) owns these items:

- Confidence-head execution and global proposal scheduling
- Gated DSpark GQA
- Backend-neutral replay-boundary review
- Additional checkpoint variants
- Overlapping batches
