# Sampling Executor

This document owns the current top-k/top-p sampling and sparse rejection-sampling contracts.
[`executor_qwen.md`](executor_qwen.md) owns Qwen stage order and MTP proposal ownership.
[`executor.md`](executor.md) owns generic executor composition.

## Source layout

```text
crates/inference-executor-core/src/sampling/
  config.rs          sampler validation and optional request seed
  domain.rs          independent target, draft, accept, and resample RNG domains
  reference.rs       CPU top-k/top-p and rejection correctness oracle
  rejection_sampling.rs
                     backend-neutral sparse rejection shape/request contracts
  request_state.rs   executor-owned request-slot seed lifecycle
  top_k_sampling.rs  backend-neutral sampling shape and request parameters

crates/inference-backend-metal/src/components/
  sampling.rs        Metal component shapes, buffers, kernels, and dispatch
  metal/sampling.metal

crates/inference-executor-metal/src/sampling/
  top_k_sampling.rs       TopKSampling, parameter/scratch, and TopKSamplingOutputBuffers
  top_k_replay.rs         Sampling and DraftSampling replay components
  dspark_markov.rs        sequential DSpark Markov correction and sampling
  rejection_sampling.rs   generic sparse rejection Metal owner and bindings
  rejection_replay.rs     shared Qwen microbatch preparation and RejectionSampling composition
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

Qwen MTP proposals and target verification use the same post-temperature/top-k/top-p distribution family as ordinary
sampling.
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
- `Replay<RejectionSampling>` handles target sparse-distribution generation and sparse rejection.

These stages share one `Rc<TopKSampling>` implementation and its parameter and scratch buffers.
They retain separate replay keys and programs.

`DSparkMarkovSampling` is an independent proposal component.
It applies the Markov correction and samples the fixed block sequentially.
It stores sparse draft distributions in `SpecProbsStore`.
The Qwen3 executor does not record this component yet at this commit.

Main and MTP forward token counts remain exact.
The complete upstream model slice does not yet share an inactive-lane ABI.
MTP draft sampling is a distinct replay after MTP GatherUnembed.
Sparse target distribution and rejection form a separate target-stage replay.

## Correctness and benchmarks

CPU references define sampling and rejection math.
Focused Metal tests compare fixed and random distributions with these references.
They also compare mixed per-row parameters, deterministic seed/domain behavior, and accepted/rejected MTP paths.
GPU tests run serially under the repository Metal reservation/lock rules.

Synthetic backend modes:

```text
cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode top-k-sample --rows 1 --num-reqs 1 --spec-tokens 1 \
  --iters 1 --warmup-iters 0 --runs 1
```

Supported modes are `top-k-sample`, `top-k-sparse-distribution`, `top-k-sample-and-sparse-distribution`, and
`rejection-sparse`.
The model-executor target is `qwen35_sampling`.
[`executor_benchmarks.md`](executor_benchmarks.md) defines shared measurement and provenance rules.
