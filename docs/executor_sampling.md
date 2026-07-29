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
  rejection_sampling.rs
                     backend-neutral sparse rejection shape/request contracts
  request_state.rs   executor-owned request-slot seed lifecycle
  top_k_sampling.rs  backend-neutral sampling shape and request parameters

crates/inference-executor-core/src/model/qwen/v3_x/dspark/
  reference.rs       CPU reference for sequential Markov correction and sampling

crates/inference-backend-metal/src/components/
  sampling.rs                 generic Metal sampling and rejection components
  dspark_markov_sampling.rs   fused DSpark Markov and tile-Top-K map component
  metal/sampling.metal
  metal/dspark_markov_sampling.metal

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs       TopKSampling, parameter/scratch, and TopKSamplingOutputBuffers
  top_k_replay.rs         Sampling and DraftSampling replay components
  rejection_replay.rs     sparse rejection replay owner, microbatch contract/adapters, and bindings
  dspark_markov.rs        sequential DSpark Markov correction and sampling
  spec_probs.rs           SpecProbsStore sparse draft/target probability workspace
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

The small-k path uses repeated maximum reduction for normal top-k <= 32.
Larger top-k and bf16 sparse-distribution generation use the bitonic tile path.
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

The Qwen executor owns three distinct graph and cache stages:

- `Replay<Sampling>` handles ordinary Main output.
- `Replay<DraftSampling>` handles MTP draft sampling and sparse draft-distribution storage.
- `Replay<Qwen3xDSparkSampling>` handles DSpark Markov correction, sampling, and sparse draft storage.
- `Replay<RejectionSampling>` handles target sparse-distribution generation and sparse rejection.

Main and MTP share one `Rc<TopKSampling>` implementation.
DSpark owns its Markov runtime parameters, tile candidates, and per-step outputs.
It reuses the generic sample-and-sparse-distribution reducer.
The stages retain separate replay keys and programs.

Main, MTP, and DSpark body token counts remain exact.
The complete upstream model slice does not yet share an inactive-lane ABI.
MTP draft sampling is a distinct replay after MTP GatherUnembed.
DSpark Markov sampling is a distinct replay after `Qwen3xDSparkGatherUnembed`.
Sparse Main distributions and rejection form one Main-stage replay.

One DSpark Markov step records two commands:

```text
DSparkMarkovTopKMap
  previous sampled token
  -> affine W1 row
  -> affine W2 projection
  -> add one base-logit row
  -> 64-token tile-local Top-K

TopKSampleAndSparseDistribution
  -> global Top-K merge
  -> top-p sampling
  -> sampled token and sparse draft distribution
```

The sampled token from one reducer is the Markov input for the next map.
The replay places a barrier at this dependency.
A seven-token block records 14 commands in one submission.
It does not materialize full-vocabulary Markov bias or corrected-logit buffers.

The current fused map preserves the earlier BF16 storage boundaries.
It dequantizes W1 to F32 and stores the latent row as BF16.
It accumulates W2 in F32.
It rounds the correction and corrected logit to BF16 before tile Top-K.
Sampling probabilities use F32.

## Correctness and benchmarks

CPU references define sampling, rejection, and sequential DSpark Markov math.
Focused Metal tests compare fixed and random distributions with these references.
They also compare mixed per-row parameters and deterministic seed/domain behavior.
Sparse rejection tests cover accepted and rejected MTP and DSpark paths.
The DSpark Markov parity test covers a padded replay bucket and non-contiguous request slots.
GPU tests run serially under the repository Metal reservation/lock rules.

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

Supported modes are `top-k-sample`, `top-k-sparse-distribution`, `top-k-sample-and-sparse-distribution`,
`rejection-sparse`, and `dspark-markov-top-k-map`.
The model-executor targets are `qwen35_sampling` and `qwen3_dspark_sampling`.
[`executor_benchmarks.md`](executor_benchmarks.md) defines shared measurement and provenance rules.
