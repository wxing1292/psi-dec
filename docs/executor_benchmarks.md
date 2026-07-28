# Executor Tests, Benchmarks, and Profiling

This document owns the executor verification ladder, benchmark surfaces, metrics, profiling conventions, and performance
evidence requirements. Component math and source layouts remain in the matching `executor_*.md`.
[`service.md`](service.md) contains the service lifecycle and end-to-end commands.

## Verification ladder

Use the smallest production owner that can prove the changed contract, then compose upward:

1. Compare each optimized backend component with a slow or CPU reference.
2. Exercise the real production component API and the owner of its metadata and buffer.
3. Verify one real-weight layer path.
4. Scale Qwen layers through `layer0`, `layer4`, `first4`, and `main_all`.
5. Add embedding, final norm, unembedding, and ordinary sampling.
6. Add MTP proposal, sparse distributions, rejection, and commit last.

Component gains do not prove end-to-end performance.

Recommendation: Use the same ladder to isolate end-to-end regressions.

Establish correctness and workload identity before timing.

Run Metal and GPU commands strictly serially across tests, benchmarks, services, and worktrees. Use `--test-threads=1`
for a Metal test command.

Keep expected-panic contract tests on the host. This placement prevents one deliberate panic from obscuring later GPU
results.

## Benchmark layers

There are two benchmark levels:

```text
inference-backend-metal
  synthetic, model-independent kernel/component questions

inference-executor-metal
  real checkpoint weights and production component/layer/executor ownership
```

Backend Criterion targets:

```text
dense_mlp  sparse_mlp  moe  gqa_attn  gdn_attn  gdn_state_io
embedding  unembedding  norm
```

`sampling_rejection` is the backend custom-CLI target. Model-executor targets:

```text
qwen3_gqa         qwen35_dense_mlp  qwen35_moe  qwen35_gqa  qwen35_gdn
qwen35_embed      qwen35_layers  qwen35_output
qwen35_sampling   qwen35_executor
```

All real-weight targets except `qwen35_sampling` require `--model-dir`. These targets share `--iters`,
`--warmup-iters`, and `--runs`.

Production `src` must not gain benchmark-only state, feature paths, or environment controls.

## Target meanings

- `qwen35_gqa` selects `--gqa-model 27b|35b` and accepts `single_q_token` or `tiled_q_tokens`. It can run an explicit
  untimed `--validate-tiled-q-tokens` comparison.
- `qwen3_gqa` loads real Qwen3 ungated-GQA weights. It measures full replay and SDPA-only paths.
- `qwen3_gqa` exposes static tile geometry as CLI arguments. It can validate single-Q output against tiled output.
- `qwen35_gdn` measures the current ragged recurrent GDN path with the 35B-A3B profile.
- `qwen35_moe` compares token-major and expert-major policies for real sparse-model weights.
- `qwen35_layers` records only main transformer layers and accepts `layer0`, `layer4`, `first4`, or `main_all`.
- `qwen35_output` begins at final norm, gather, and unembedding. It can isolate sampling and readback.
- `qwen35_executor` measures the public executor contract with fixed `e2e_wo_mtp` and `e2e_w_mtp` cases.
- The `qwen35_executor` MTP case obtains proposal and draft tokens from production execution. It does not substitute a
  static draft.

Representative smoke commands:

```text
cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 128 \
  --iters 1 --warmup-iters 0 --runs 1 --validate

cargo bench -p inference-executor-metal --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-paths single_q_token \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_layers -- \
  --model-dir <27b-model-dir> --cases layer0 --tokens 1 --contexts 0 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_executor -- \
  --model-dir <35b-model-dir> --mtp-model-dir <35b-mtp-model-dir> \
  --cases e2e_w_mtp --iters 1 --warmup-iters 0 --runs 1
```

Run one performance command at a time. List the planned cases first. GPU contention and memory pressure invalidate
comparisons.

## Metrics

`setup_us` includes model loading and fixture construction. `cache_miss_wall_us` is the first complete execution.

`cache_build_estimate_us` is the CPU record and finish estimate after subtracting measured replay waits. Whole-executor
samples report wall time and main, output, and speculator replay waits.

Prepare, record and finish, feedback, and commit remain distinct host boundaries.

Force-sync and profile-summary measurements are diagnostic metrics. They are not normal wall-clock throughput. Never
compare the two measurement types as equivalent workloads.

Benchmark keys contain only comparison dimensions. Useful dimensions include:

- Model
- Storage and backend
- Operation
- Batch and tokens
- A meaningful context or state coordinate

Do not include these values:

- Default layer 0
- Generic `detail` or `sub-op`
- Metadata that output already shows elsewhere

```text
gqa/qwen36-27b/metal/full-forward/b1t64-c1024
gdn/qwen36-35b-a3b/metal/full-forward/with-state/b1t1-s1
moe/qwen36-35b-a3b/metal/token-major/b1t64
```

## Profiling and logging

The service uses static `profiling::span(...)` names for its coarse CPU tree. Put dynamic request and shape values in
structured logs.

The `--profile component|operation` modes currently produce the same tree. They do not provide GPU kernel timestamps.
[`service.md`](service.md) documents service logging fields and commands.

## Performance evidence

A performance claim records all of:

```text
commit and dirty state
machine, OS, architecture, and relevant environment
model/checkpoint and command
sampling config and deterministic seed
metric and workload/trajectory fields
baseline and current samples
verdict
```

For speculative decoding, also record these values:

- Proposals
- Sampled tokens
- Accepted tokens and chunks
- Acceptance efficiency

A throughput change with a different deterministic acceptance trajectory is not a pure executor or kernel comparison.
