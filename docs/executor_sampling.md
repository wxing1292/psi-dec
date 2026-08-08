# Sampling Executor

This document owns the current top-k/top-p sampling and sparse rejection-sampling contracts.
[`executor_qwen.md`](executor_qwen.md) owns Qwen stage order and Spec proposal ownership.
[`executor.md`](executor.md) owns generic executor composition.

## Source layout

```text
crates/inference-executor-core/src/sampling/
  config.rs          sampler validation and optional request seed
  domain.rs          independent Main, Spec, accept, and resample RNG domains
  reference.rs       CPU top-k/top-p and rejection correctness oracle
  dspark.rs          CPU reference for sequential Markov correction, confidence, and sampling
  rejection_sampling.rs
                     backend-neutral sparse rejection shape/request contracts
  request_state.rs   executor-owned request-slot seed lifecycle
  top_k_sampling.rs  backend-neutral sampling shape and request parameters

crates/inference-backend-metal/src/components/
  sampling/top_k.rs           generic Metal top-k sampling components
  sampling/rejection.rs       sparse rejection component
  sampling/dspark_markov.rs   fused DSpark Markov, confidence, and tile-Top-K map component
  metal/sampling.metal
  metal/dspark_markov_sampling.metal

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs       TopKSampling, parameter/scratch, and TopKSamplingOutputBuffers
  top_k_replay.rs         Sampling and DraftSampling replay components
  rejection_replay.rs     sparse rejection replay owner, microbatch contract/adapters, and bindings
  dspark_markov.rs        sequential DSpark Markov, confidence, and sampling composition
  spec_probs.rs           SpecProbsStore sparse draft/target probability workspace

crates/inference-executor-metal/src/model/qwen/v3_x/dspark/
  sampling.rs             Qwen3x Markov checkpoint weights and generic backend adapter
```

Runtime core transports sampler configuration and sampled decisions.
It does not own RNG state, sparse distributions, or replay geometry.
The executor resolves one root seed for each live request slot.
It keeps that seed stable until the slot resets.

## Normal sampling

Each compact sampled row has its own sampling parameters.
They include temperature, top-k, top-p, resolved seed, logical output-token position, and `SamplingDomain`.
The current default decode policy uses temperature 0.7, top-p 0.8, and top-k 20.
Greedy decoding uses the same contract with top-k 1 and temperature 0.

```text
logits [num_rows, vocab_size]
  -> top_k_logits_tiles
       one 256-token vocabulary tile per threadblock
       write sorted/reduced tile candidates
  -> top_k_sample_tiles
       merge tile candidates
       apply temperature and top-p
       draw from (request seed, logical position, domain)
  -> sampled token IDs + probabilities
```

The backend selects the tile kernel from the logits dtype, top-k, and required output.
The small-k path uses repeated maximum reduction for sample-only top-k <= 32.
Larger top-k and write-distribution generation use the bitonic tile path.
These operations remain separate pipeline entry points.
Unused static threadblock storage can reduce occupancy.

Sampling returns only token IDs and probabilities.
Tile candidates and merged rows are private scratch, not model-level API state.
The current input contract does not include repetition, frequency, or presence penalties.
`TopKSamplingOutputBuffers` owns the concrete sampled token and probability buffers.
`OutputBuffers` denotes GPU buffers, not a model lifecycle state machine.

## Sparse distributions and rejection

Qwen MTP and DSpark proposals use the same post-temperature/top-k/top-p distribution family as Main verification.
The production path stores sparse token and probability rows.
It does not scatter them into dense full-vocabulary buffers.

```text
proposal logits
  -> sample + sparse draft distribution

target logits
  -> sparse target distributions

target distributions + draft distributions + flat draft tokens
  -> sparse rejection
       process drafts sequentially within each request
       accept with SamplingDomain::Accept
       on rejection sample max(target - draft, 0) with SamplingDomain::Resample
       if all drafts pass, sample the final target continuation
  -> accepted draft prefix + one fallback/continuation token
```

The target table contains one distribution for each draft token.
It also contains one final continuation distribution for each request.
The draft table contains one distribution for each draft token.
`cu_target_distributions` and `cu_draft_distributions` select the ragged per-request ranges.
Requests are independent.
Drafts within one request are ordered because the first rejection ends that request's acceptance walk.

`SpecProbsStore` owns `draft_token_ids`, `draft_probs`, `target_token_ids`, and `target_probs`.
`max_k` is the maximum sparse Top-K row width, not the vocabulary size.
Debug builds also retain `expected_draft_token_ids` for lifecycle validation.
Release builds do not allocate, reset, or compare this CPU-only metadata.

Draft distributions cross a batch boundary.
Their row identity is `req_slot * max_num_spec_tokens + proposal_position`.
Main verification distributions exist only in the current submission.
They use compact active-row indices from zero.
`cu_target_distributions` uses this compact row domain.

## Replay ownership

Sampling and rejection use capacity replay keys.
Executor configuration caps power-of-two row and request capacities.
Submission-scoped `ReplayArguments` contain exact active thread counts.
Every padded kernel returns inactive lanes before it reads input, changes RNG state, or writes output.
A 0/1 capacity uses an immediate constant to preserve the common single-request decode path.

Active top-k remains in the replay shape because it changes candidate and scratch geometry.
Temperature, top-p, seed, logical position, and RNG domain are dynamic request data.
They never enter replay keys.

Writing runtime sampling or rejection parameters arms exactly one replay-argument preparation.
That preparation consumes the matching active row or request count.
It then clears the armed state.
Replay without a fresh write is an invariant violation.
It does not permit reuse of stale parameter rows.

The Qwen executor owns four distinct graph and cache stages:

- `Replay<Sampling>` handles ordinary Main output.
- `Replay<DraftSampling>` handles MTP draft sampling and sparse draft-distribution storage.
- `Replay<Qwen3xDSparkSampling>` handles DSpark Markov correction, confidence, sampling, and sparse draft storage.
- `Replay<RejectionSampling>` handles target write-distribution generation and sparse rejection.

Main and MTP share one `Rc<TopKSampling>` implementation.
`DSparkMarkovSampling` owns model-neutral Markov runtime parameters, tile candidates, and per-step outputs.
It accepts borrowed weights at record time.
`Qwen3xDSparkMarkov` owns Qwen checkpoint buffers and delegates execution to this backend owner.
It loads one bounded `TensorMap` for W1, W2, and the required Markov-conditioned confidence head.
It removes all tensors and requires the map to be empty before initialization completes.
It reuses the generic sample-and-write-distribution reducer.
The stages retain separate replay keys and programs.

Qwen3.5 Main and MTP body replays use separate capacity-bucketed token domains.
Qwen3 and DSpark body token counts remain exact.
Sampling and rejection keep their own replay domains, keys, active counts, and bucket policies.
MTP draft sampling is a distinct replay after MTP GatherUnembed.
For K MTP steps, each pass writes one request-local draft-distribution row at `step_index`.
The step index is runtime metadata and does not enter the sampling replay key.
DSpark Markov sampling is a distinct replay after `Qwen3xDSparkGatherUnembed`.
Sparse Main distributions and rejection form one Main-stage replay.

One DSpark Markov step records two commands:

```text
DSparkMarkovTopKMap
  input_token_ids[i]
    i = 0: sampled_token / anchor_token
    i > 0: spec_tokens[i - 1]
  -> affine W1 row
  -> affine W2 projection
  -> add one base-logit row
  -> 64-token tile-local Top-K
  -> confidence projection and sigmoid
  -> spec_confidence[i]

TopKSampleAndWriteDistribution
  -> global Top-K merge
  -> top-p sampling
  -> spec_token[i], spec_prob[i], and sparse draft distribution
```

The sampled token from one reducer is the Markov input for the next map.
The replay places a barrier at this dependency.
A seven-token block records 14 commands in one submission.
It does not materialize full-vocabulary Markov bias or corrected-logit buffers.

`SampledTokens::Decode` carries `spec_tokens`, `spec_probs`, and `spec_confidences`.
These vectors always have the same length.
MTP uses `1.0` for each speculative confidence because MTP has no confidence head.
Qwen3x DSpark requires and evaluates its confidence head.
The Qwen3 and Qwen3.5 response adapters preserve DSpark confidence values across the executor/runtime boundary.
The runtime does not apply a confidence threshold or proposal-length policy yet.

The current fused map preserves the earlier BF16 storage boundaries.
It dequantizes W1 to F32 and stores the latent row as BF16.
It accumulates W2 in F32.
It rounds the correction and corrected logit to BF16 before tile Top-K.
Sampling probabilities use F32.

### DSpark Markov numerical contract

Decision: Retain the current BF16 materialization boundaries.

The Qwen3 implementations in
[vLLM at `bb3b61f2fd2333ab165ebaba13f133db4210b9f2`](https://github.com/vllm-project/vllm/blob/bb3b61f2fd2333ab165ebaba13f133db4210b9f2/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)
and
[SGLang at `85618cc798ce9b5fdbfdd5c535576515d498acc2`](https://github.com/sgl-project/sglang/blob/85618cc798ce9b5fdbfdd5c535576515d498acc2/python/sglang/srt/models/dspark.py)
add the Markov correction in the model-logit dtype.
They do not explicitly promote the Qwen3 corrected-logit tensor to F32.
[SGLang sampling](https://github.com/sgl-project/sglang/blob/85618cc798ce9b5fdbfdd5c535576515d498acc2/python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py)
promotes the corrected logits when it computes sampling probabilities.

The Metal map uses F32 where reduction requires it:

```text
BF16 W1 latent in shared memory
  -> F32 W2 partials and SIMDgroup reduction in thread-local values
  -> BF16 correction
  -> BF16 corrected logit
  -> F32 tile Top-K and sampling probabilities
```

For the official rank-256 checkpoint, the map uses 1024 bytes of static shared memory.
The BF16 latent uses 512 bytes.
The 64 F32 tile logits and 64 I32 token IDs use the other 512 bytes.
An F32 corrected-logit candidate does not require more shared memory.
It keeps the correction and corrected logit in thread-local scalar values.
The Metal compiler can keep these values in registers or spill them.
Changing the W1 latent to F32 would increase the static shared-memory requirement to 1536 bytes, but that is a
different numerical contract.

The F32 corrected-logit candidate used this sequence:

```text
BF16 W1 latent
  -> F32 W2 accumulation
  -> F32 base-logit add
  -> F32 tile Top-K and sampling probabilities
```

The A/B investigation used base commit `2804bdece3c03235e9b2b18f9655a933ed7a3220` on macOS 27.0 and an Apple M3 Max
with a 40-core GPU and 48 GB of memory.
The candidate changed only the corrected-logit rounding contract and its CPU reference.
The worktree also contained only the benchmark output controls from this change.
Focused CPU/GPU parity passed for both contracts.

The real-weight Markov benchmark used
`dspark_qwen3_14b_block7-affine`, one request, `temperature=0.7`, `top_k=20`, `top_p=0.8`, and `seed=42`.
It used the production Markov weights and deterministic zero base logits to isolate Markov correction and sampling.

```text
cargo bench -p inference-executor-metal --bench qwen3_dspark_sampling -- \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 --temperature 0.7 --top-k 20 --top-p 0.8 --seed 42 \
  --warmup-iters 10 --iters 50 --runs 5
```

The current BF16 path measured 4.158 ms for the complete seven-step replay.
The F32 candidate measured 4.162 ms.
The candidate did not produce a measurable performance gain.
Both contracts sampled `[5310, 5390, 979, 14550, 448, 5091, 369]` in this isolated case.
The BF16 write-distribution fingerprint was `31642210e096a9d7`.
The F32 fingerprint was `9dacaff10a8cc01d`.
The proposal probability bits also differed.

The fused confidence check used base commit `a36024dc` with only the confidence change present in the worktree.
It used the same machine, Markov weights, sampler settings, and benchmark command.
The converted checkpoint also contained the official BF16 confidence projection and bias.
The complete seven-step replay measured 4.061 ms.
The sampled tokens and BF16 write-distribution fingerprint matched the 4.158 ms path.
The confidence output contained seven finite, nonconstant values.

The deterministic service comparison used Qwen3-14B-4bit, the same DSpark checkpoint, `temperature=0`, `top_k=1`,
`top_p=1`, and `seed=42`.

```text
target/release/qwen3 \
  --grpc-listen-addr 127.0.0.1:50151 \
  --http-listen-addr 127.0.0.1:8011 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --hf-dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-cache-pages 4096 --max-requests 4 --max-tokens 128 \
  --max-tokens-per-request 64 --logging info

target/release/decode \
  --server-url http://127.0.0.1:50151 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --prompt-str 'Explain in concise technical terms why the sky appears blue during the day.' \
  --disable-thinking --max-sampled-tokens 256 \
  --temperature 0 --top-k 1 --top-p 1 --seed 42 --show-stats
```

Three runs for each contract produced the same final 98-token text.
The current BF16 contract used 34 verification batches, generated 238 proposals, and accepted 65 proposals.
Its acceptance rate was 27.31%.
Its three decode rates were 42.095, 42.253, and 42.852 tok/s.
The F32 candidate used 33 verification batches, generated 231 proposals, and accepted 66 proposals.
Its acceptance rate was 28.57%.
Its three decode rates were 43.021, 43.320, and 43.169 tok/s.
The throughput values are not a corrected-logit kernel comparison because the contracts used different verification
trajectories.

The changed distributions and replay trajectory make F32 a model numerical-contract change.
One deterministic prompt does not justify that change.
The production path must keep the BF16 contract unless broader checkpoint evidence establishes a different contract.

## Correctness and benchmarks

CPU references define sampling, rejection, and sequential DSpark Markov math.
Focused Metal tests compare fixed and random distributions with these references.
They also compare mixed per-row parameters and deterministic seed/domain behavior.
Sparse rejection tests cover accepted and rejected MTP and DSpark paths.
The DSpark Markov parity test covers a padded replay bucket and non-contiguous request slots.
GPU tests run serially under the repository Metal reservation/lock rules.
The `qwen3_dspark_sampling` benchmark prints proposal token IDs, exact proposal probability bits, and a stable
fingerprint of the complete sparse draft distribution.
Use `--temperature`, `--top-k`, `--top-p`, and `--seed` to reproduce a sampling contract.

Synthetic backend modes:

```text
cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode top-k-sample --rows 1 --num-reqs 1 --spec-tokens 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode dspark-markov-top-k-map --rows 1 --top-k 20 --vocab 151936 \
  --markov-rank 256 --markov-w1-group-size 64 --markov-w1-bits 4 \
  --markov-w2-group-size 64 --markov-w2-bits 8 \
  --iters 1 --warmup-iters 0 --runs 1
```

Supported modes are `top-k-sample`, `top-k-write-distribution`, `top-k-sample-and-write-distribution`,
`rejection-sparse`, and `dspark-markov-top-k-map`.
The model-executor targets are `qwen35_sampling` and `qwen3_dspark_sampling`.
Qwen3.5 DSpark reuses the Qwen3x Markov component benchmark and adds end-to-end service validation.
[`executor_benchmarks.md`](executor_benchmarks.md) defines shared measurement and provenance rules.
