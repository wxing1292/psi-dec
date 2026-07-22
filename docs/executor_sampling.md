# Sampling Executor

This document owns the current top-k/top-p sampling and sparse rejection-sampling contracts. Qwen stage ordering and
MTP proposal ownership remain in [`executor_qwen.md`](executor_qwen.md); generic executor composition remains in
[`executor.md`](executor.md).

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
  rejection_sampling.rs   generic sparse rejection Metal owner and bindings
  spec_probs.rs           SpecProbsStore sparse draft/target probability workspace

crates/inference-executor-metal/src/model/qwen/v3_5/
  rejection_sampling.rs   Qwen microbatch preparation and RejectionSampling composition
```

Runtime core transports sampler configuration and sampled decisions. It does not own RNG state, sparse distributions,
or replay geometry. The executor resolves one root seed per live request slot and keeps it stable until that slot is
reset.

## Normal sampling

Each compact sampled row carries its own temperature, top-k, top-p, resolved seed, logical output-token position, and
`SamplingDomain`. The current default decode policy is temperature 0.7, top-p 0.8, and top-k 20. Greedy decoding is the
same contract with top-k 1 and temperature 0.

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

The small-k path uses repeated maximum reduction for normal top-k <= 32. Larger top-k and bf16 sparse-distribution
generation use the bitonic tile path. They remain separate pipeline entry points because unused static threadblock
storage can still reduce occupancy.

Sampling returns only token IDs and probabilities. Tile candidates and merged rows are private scratch, not model-level
API state. Repetition, frequency, and presence penalties are not part of the current input contract.
`TopKSamplingOutputBuffers` is the concrete sampled token/probability buffer owner; `OutputBuffers` denotes GPU buffers,
not a model lifecycle state machine.

## Sparse distributions and rejection

Qwen MTP proposals and target verification use the same post-temperature/top-k/top-p distribution family as ordinary
sampling. The production path stores sparse token/probability rows; it does not scatter them into dense full-vocabulary
buffers.

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

The target table contains one distribution per draft token plus one final continuation distribution per request. The
draft table contains one distribution per draft token. `cu_target_distributions` and `cu_draft_distributions` select the
ragged per-request ranges. Requests are independent; drafts within one request are ordered because the first rejection
ends that request's acceptance walk.

`SpecProbsStore` owns `draft_token_ids`, `draft_probs`, `target_token_ids`, and `target_probs`. `max_k` is the maximum
sparse Top-K row width, not vocabulary size. Debug builds additionally retain `expected_draft_token_ids` for lifecycle
validation; release builds do not allocate, reset, or compare that CPU-only metadata.

## Replay ownership

Sampling and rejection use capacity replay keys. Power-of-two row/request capacities are capped by executor config;
exact active thread counts are submission-scoped `ReplayArguments`. Every padded kernel returns inactive lanes before
reading input, changing RNG state, or writing output. A 0/1 capacity uses an immediate constant to preserve the common
single-request decode path.

Active top-k remains in the replay shape because it changes candidate and scratch geometry. Temperature, top-p, seed,
logical position, and RNG domain are dynamic request data and never enter replay keys.

Writing runtime sampling/rejection parameters arms exactly one replay-argument preparation. That preparation consumes
the matching active row/request count and clears the armed state; replay without a fresh write is an invariant violation,
not permission to reuse stale parameter rows.

The Qwen executor owns three distinct graph/cache stages: `Replay<Sampling>` for ordinary Main output,
`Replay<DraftSampling>` for MTP draft sampling plus sparse draft-distribution storage, and
`Replay<RejectionSampling>` for target sparse-distribution generation plus sparse rejection. They share one
`Rc<TopKSampling>` implementation and its parameter/scratch buffers, but retain separate replay keys and programs.

Main and MTP forward token counts remain exact because the complete upstream model slice does not yet share an
inactive-lane ABI. MTP draft sampling is a distinct replay composed after MTP GatherUnembed; sparse target distribution plus
rejection is a separate target-stage replay.

## Correctness and benchmarks

CPU references define sampling and rejection math. Focused Metal tests compare fixed and random distributions, mixed
per-row sampling parameters, deterministic seed/domain behavior, and accepted/rejected MTP paths against those oracles.
GPU tests run serially under the repository Metal reservation/lock rules.

Synthetic backend modes:

```text
cargo bench -p inference-backend-metal --bench sampling_rejection -- \
  --mode top-k-sample --rows 1 --num-reqs 1 --spec-tokens 1 \
  --iters 1 --warmup-iters 0 --runs 1
```

Supported modes are `top-k-sample`, `top-k-sparse-distribution`,
`top-k-sample-and-sparse-distribution`, and `rejection-sparse`. The model-executor target is
`qwen35_sampling`. See [`executor_benchmarks.md`](executor_benchmarks.md) for shared measurement and provenance rules.
