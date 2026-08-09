# Qwen3x DSpark

This document describes the current fixed-block Qwen3 and Qwen3.5 DSpark implementation.
DSpark support is experimental.
Its checkpoint contract, CLI, cache sizing, and proposal policy may change.
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
The Metal replay `ResidualAddCaptureTarget` uses `Target` to mean a write destination.
The tracing `target:` field identifies a log category.
None of these names identify model roles.

## Current scope

Qwen3 and Qwen3.5 provide separate Vanilla and DSpark initializer paths.
The implementation follows the official flat `Qwen3xDSparkConfig` schema.
It supports an ungated GQA backbone and a `vanilla` Markov head.
It requires the official Markov-conditioned confidence head.
Qwen3.5 MTP and DSpark are mutually exclusive.

At startup, `--num-spec-tokens N` selects one fixed proposal length.
`N` must not exceed the checkpoint `block_size`.
When the option is absent, the executor uses the checkpoint `block_size`.

The current implementation has these limits:

- It produces exactly `N` proposals for each active decode request.
- It executes the required official confidence head.
- It returns one confidence value for each proposal token.
- It does not schedule variable proposal lengths.
- It does not support a gated DSpark GQA checkpoint.
- It permits one in-flight batch for each executor.

## Source layout

```text
crates/inference-executor-core/src/
  attn/gqa/
    dspark_core.rs               block geometry and metadata
  sampling/
    dspark.rs                    CPU Markov, sampling, and confidence reference
  model/qwen/v3_x/dspark/
    config.rs                    official configuration contract
    weight_layout.rs             exact source and affine binding trees
  bin/
    qwen3_dspark_quantize.rs     official BF16 -> affine converter

crates/inference-backend-metal/src/components/
  sampling/dspark_markov.rs      fused Markov, confidence, and tile-Top-K map component
  gqa_block_attention.rs         dense block-SDPA component
  metal/dspark_markov_sampling.metal
  metal/gqa_block_sdpa.metal     dense bidirectional block kernel

crates/inference-executor-metal/src/
  attn/dspark/
    backend.rs                   ungated DSpark GQA replay graph
    capacity.rs                  Metal partial-output resource capacity
    context.rs                   persistent Main-context append
    metadata.rs                  history and block attention metadata
    scratch.rs                   fixed-capacity proposal scratch
    state.rs                     page-table and metadata lifecycle
  model/
    main_residual_capture.rs     shared Main residual-capture contract
    qwen/v3_x/dspark/
      attention.rs               DSpark attention composition
      embed.rs                   DSpark token embedding replay
      execution.rs               shared execution resources and per-batch recording
      layer.rs                   independent Qwen3xDSparkLayer
      load.rs                    shared checkpoint and resource load
      model.rs                   context and body replay owners
      output.rs                  gather, unembed, and Markov sampling
      main_feature.rs            selected Main residual projection
      sampling.rs                Qwen3x Markov checkpoint weights and generic backend adapter
    qwen/v3/executor/
      dspark.rs                  Qwen3 proposal orchestration
    qwen/v3_5/executor/
      load.rs                    separate Qwen3.5 Vanilla/MTP/DSpark load
      dspark.rs                  Qwen3.5 DSpark proposal orchestration
  sampling/
    dspark_markov.rs             sequential Markov, confidence, and sampling composition
    rejection_replay.rs          shared sparse rejection replay and microbatch contract
```

Qwen3 and Qwen3.5 compose the same `qwen/v3_x/dspark/` owners.
The shared loader resolves DSpark bindings and constructs the checkpoint-owned resources once.
The model-specific executors keep Main validation, batches, transactions, and proposal-result adaptation in their
executor directories.

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
At index `0`, it uses the Main `sampled_token`, which is the anchor token.
At an index greater than `0`, it uses `spec_tokens[i - 1]`.
`DSparkMarkovSampling` owns this sequential correction and sampling loop.
It owns the generic Metal operators, runtime parameters, scratch, and output buffers.
It receives borrowed Markov and confidence weights for each replay recording.
`Qwen3xDSparkMarkov` owns the Qwen checkpoint buffers and supplies those borrowed weights.
It stores one sparse draft distribution for each proposed token.

```text
Qwen3xDSparkConfig + exact Qwen3x checkpoint bindings
  -> Qwen3xDSparkMarkov
       owns Qwen3xDSparkMarkovWeights
       owns Qwen3xDSparkConfidenceWeights
       owns DSparkMarkovSampling
            |
            | record(DSparkMarkovInput + borrowed weights)
            v
       DSparkMarkovTopKMapKernel -> TopKMergeKernels
```

This owner pattern matches Qwen3x GQA, GDN, and MLP.
The Qwen wrapper owns checkpoint materialization.
The generic executor owner owns backend execution resources.
Each DSpark semantic owner loads one bounded `TensorMap` for its exact binding subtree.
It removes every tensor before it creates Metal buffers, performs any required fusion, and requires the map to be empty.

`Qwen3xDSparkExecution` is the single DSpark execution owner in each DSpark enum variant.
It owns the DSpark replay caches, request state, page layout, and reusable workspaces.
`Replay<Qwen3xDSparkSampling>` is the single Markov owner and access path.
The executor uses `Replay::component()` for Markov prepare, replay arguments, and proposal reads.
`Qwen3xDSparkRecording` owns only keys, replay arguments, replay shape, and request slots for one batch.
The executor speculator enums prevent partially initialized DSpark field combinations.
It also keeps replay programs and workspaces reusable across submissions.

Each position records this pair:

```text
input_token_ids[i]
  i = 0: sampled_token / anchor_token
  i > 0: spec_tokens[i - 1]
         |
         v
quantized Markov W1 embedding
         |
         +-----------------------------------------+
         |                                         |
         v                                         v
quantized Markov W2                     [hidden[i], Markov W1 embedding]
         |                                         |
         v                                         v
base_logits[i] + correction             confidence projection + sigmoid
         |                                         |
         v                                         v
tile-local Top-K                         spec_confidence[i]
         |
         v

TopKSampleAndWriteDistribution
  -> global Top-K and top-p
  -> spec_token[i]
  -> spec_prob[i]
  -> sparse draft distribution
```

The loader requires `enable_confidence_head = true` and `confidence_head_with_markov = true`.
The confidence contract is:

```text
confidence_input[i] = concat(hidden[i], MarkovW1(input_token_ids[i]))
confidence_raw[i] = confidence_bias + dot(confidence_weight, confidence_input[i])
spec_confidence[i] = sigmoid(confidence_raw[i])
```

The current executor uses the official default sigmoid temperature of `1.0`.
`DSparkMarkovTopKMapConfig` states the BF16 I/O and scale/bias workload facts.
The confidence weight, bias, hidden state, and Markov latent use BF16 storage.
The dot product and sigmoid use F32.
`DSparkMarkovTopKMap` computes the confidence branch from the existing W1 latent.
It does not add a replay command or a wide temporary feature buffer.

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

`Qwen3Main` and `Qwen35Main` accept the shared `MainResidualCapture` contract.
The contract does not depend on a DSpark model or a Metal recorder.

`Qwen3xDSparkMainFeatureProjector` selects the configured Main residual outputs.
It projects those outputs into one Main feature for each Main token.
Each DSpark layer projects that feature to its persistent context K/V.

`Qwen3xDSparkContext` records this work after the model-specific Main graph in the same Main submission:

```text
MainEmbed
  -> Main
  -> DSparkContext
```

Persistent DSpark context follows accepted Main history.
Proposal-local K/V never enters this context.

The reusable Qwen3x DSpark model, checkpoint-weight owners, and Main capture contract do not depend on one Main model
version.
Both executors compose these owners with their model-specific lifecycle.

## Attention composition

The supported official Qwen3x DSpark checkpoints use ungated GQA.
`UngatedDSparkGQA` owns a model-neutral QKV attention graph.
It does not add a mode to `UngatedGQA`.

Each `Qwen3xDSparkLayer` resolves its exact Q/K/V/output affine layout from its weight bindings.
It owns one `UngatedDSparkGQA` and one `UngatedDSparkGQAContextAppender`.
Different layers can use different affine layouts.
The Q/K/V/output tensors within one layer must use one layout because the current fused QKV ABI uses one
`GQAMetalConfig`.

`UngatedDSparkGQAState` owns only shared execution state.
It owns the page table, metadata, shared block/context scratch, and backend-selected single-query history compute contract.
The shared scratch contract requires equal attention geometry and I/O dtype across layers.
It does not require equal quantization layout across layers.

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

`UngatedDSparkGQAState::prepare_block` gives fixed workload facts to `GQACompute`.
The backend selects the constrained single-query-token partial ABI once.
`DSparkGQAMetadataBuffers` stores that exact `GQAComputePath` with the replay shape.
Recording uses the stored path and does not run a second selector.

One block-bidirectional map Task owns one Q token and one Q head.
The backend fixes this Task to one 32-thread SIMDgroup.
For the official `head_dim = 128`, each thread keeps four F32 Q values.
The logical Q register payload is 16 bytes per thread.
The threadblock keeps `N` F32 logits in `4 * N` bytes of shared memory.
It does not keep a shared reduction array.

This geometry preserves `N * num_q_heads` independent threadblocks for one request.
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

Main and DSpark use the same request-slot lifecycle.
The model-specific executor sends each runtime reset notification to both GQA state owners.
Both owners call `GQARequestPageTable::reset_req_slots` for the released request slots.
The reset clears executor page-table bindings.
It does not clear physical pages that the runtime core owns and releases.

The service derives:

```text
num_pages_per_kv_block =
    main_pages_per_block
  + dspark_pages_per_block
```

A DSpark-disabled Qwen3 or Qwen3.5 executor uses only the Main span.

## Scratch capacity

`DSparkBlockScratch` is an executor-owned resource.
It contains local Q/K/V, normalized Q/K, attention partials, and reduced output.

Define:

```text
T_capacity = max_requests * num_spec_tokens
P_capacity = next_power_of_two(2 * T_capacity)
```

`DSparkBlockCapacity` is backend-neutral.
It contains `max_requests`, the selected proposal length as `block_size`, and `max_tokens`.
`DSparkGQACapacity` is Metal-specific.
It derives `P_capacity` for metadata and partial scratch.
The runtime core and executor core do not contain Metal `TaskTemplate` capacity.

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

Qwen3.5 GDN also allocates candidate recurrent states for every possible accepted proposal prefix.
`Qwen35Config` resolves the request-slot capacity from `--max-requests` for Main, MTP, and DSpark.
The service passes the same capacity to the executor, runtime, and scheduler.
This rule bounds the persistent GDN arena without changing buffer, scratch, replay, or residency reuse.

The shared DSpark loader validates these limits:

```text
1 <= num_spec_tokens <= checkpoint block_size
```

`num_spec_tokens` controls the DSpark proposal width. It does not set the Main
per-request verification budget. The scheduler may send only a proposal prefix
to Main. DSpark derives its own `max_requests * num_spec_tokens` row capacity.
When DSpark reuses Main embedding or unembedding weights, it creates a
DSpark-capacity view over the same immutable kernel and weights. It does not use
the Main scheduler row capacity.

## Sparse distribution identity

Draft distributions persist until the next Main batch.
Their identity is request-slot based:

```text
draft_distribution_index =
    req_slot * num_spec_tokens
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

The loader validates the DSpark configuration against the Qwen3 or Qwen3.5 Main configuration.
It rejects unsupported attention, dtype, RoPE, `target_layer_ids`, and Markov variants.
It permits the query projection width to differ from `hidden_size` when the head geometry is valid.
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
It also preserves the confidence projection and bias as BF16.

## Verification

Unit and Metal parity tests cover:

- Official `target_layer_ids` validation
- Exact source and affine bindings
- Converter round-trip and safetensors index generation
- Indexed confidence input-token semantics and CPU/Metal parity
- Equal `spec_tokens`, `spec_probs`, and `spec_confidences` lengths
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
The final one-request deterministic comparison measured a `36.565 tok/s` Main-only median and a `42.788 tok/s`
DSpark median.
DSpark was `17.0%` faster for that workload.
The full optimization history is in [`qwen3_dspark_design_draft.md`](qwen3_dspark_design_draft.md).

### Acceptance correctness

The 2026-07-29 acceptance audit used clean commit `dad01342fb8eb4fc828b4fdd00db51d8db65fe9f` on an Apple M3
Max with a 40-core GPU.
It used `Qwen3-14B-4bit` and `dspark_qwen3_14b_block7-affine`.
All GPU runs were serial.
Each request used `temperature=0`, `top_k=1`, `top_p=1`, and `seed=1`.

| Workload             | Verification rounds | Proposed tokens | Accepted tokens | Proposal acceptance | Accepted length |
| -------------------- | ------------------: | --------------: | --------------: | ------------------: | --------------: |
| Sky explanation      |                  34 |             238 |              65 |              27.31% |            2.91 |
| Algebra              |                  20 |             140 |             113 |              80.71% |            6.65 |
| Python code          |                  20 |             140 |             109 |              77.86% |            6.45 |
| Database engineering |                  40 |             280 |              93 |              33.21% |            3.33 |
| Aggregate            |                 114 |             798 |             380 |              47.62% |            4.33 |

`Accepted length` includes the one Main continuation token from each verification round.
This is the `tau` convention in the
[DSpark paper](https://arxiv.org/abs/2607.05147).
Proposal acceptance divides only accepted draft tokens by the seven fixed proposal slots.
The two metrics must not be compared as if they use the same denominator.

The aggregate conditional acceptance by proposal position was:

```text
position:                 1      2      3      4      5      6      7
conditional acceptance: 82.5%  79.8%  74.7%  87.5%  83.7%  85.4%  85.7%
```

The per-position evidence does not show a proposal-position shift or suffix collapse.
The workload spread also follows the expected DSpark behavior.
Structured algebra and code accepted longer prefixes than open-ended explanations.

The audit verified these contracts:

- One anchor plus six MASK inputs produce seven proposal distributions.
- Every proposal row sees persistent Main context before the anchor.
- Every proposal row sees the complete local block with bidirectional attention.
- Main residual capture uses the configured decoder-layer outputs.
- Markov step zero consumes the anchor.
- Each later Markov step consumes the preceding sampled draft token.
- Rejection uses one Main distribution for each proposal and one final continuation distribution.

Three of the four greedy outputs matched Main-only byte for byte.
The database-engineering output diverged at output token 51.
At the divergence, DSpark proposed token `235`.
Rejection accepted zero tokens and the eight-row Main verification sampled token `117`.
The one-row Main path sampled token `223`.
Thus, rejection did not accept an incorrect draft token.
Both paths reproduced their own output on a second run.

Verdict: The audit found no DSpark proposal, Markov, attention, or rejection correctness defect.
The low sky-prompt acceptance is workload-specific and is not representative of algebra or code.
Strict one-row versus multi-row Main numerical determinism is not established.
That investigation is separate from DSpark acceptance correctness.

## Deferred work

[`future_work.md`](future_work.md) owns these items:

- Confidence-head execution and global proposal scheduling
- Gated DSpark GQA
- Strict one-row and multi-row Main numerical parity
- Backend-neutral replay-boundary review
- Additional checkpoint variants
- Overlapping batches
