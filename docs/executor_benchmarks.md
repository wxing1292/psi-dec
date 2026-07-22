# Executor Tests, Benchmarks, and Profiling

This document owns the executor verification ladder, benchmark surfaces, metrics, profiling conventions, and performance
evidence requirements. Component math and source layouts remain in the matching `executor_*.md`; service lifecycle and
end-to-end commands remain in [`service.md`](service.md).

## Verification ladder

Use the smallest production owner that can prove the changed contract, then compose upward:

1. Compare each optimized backend component with a slow/CPU reference.
2. Exercise the real production component API and its metadata/buffer owner.
3. Verify one real-weight layer path.
4. Scale Qwen layers through `layer0`, `layer4`, `first4`, and `main_all`.
5. Add embedding, final norm, unembedding, and ordinary sampling.
6. Add MTP proposal, sparse distributions, rejection, and commit last.

Component wins do not prove end-to-end performance; e2e regressions should be bisected down the same ladder. Correctness
and workload identity come before timing.

Run Metal/GPU commands strictly serially across tests, benches, services, and worktrees; use `--test-threads=1` for a
Metal test command. Keep expected-panic contract tests host-only so one deliberate panic cannot obscure later GPU
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
qwen35_dense_mlp  qwen35_moe  qwen35_gqa  qwen35_gdn
qwen35_embed      qwen35_layers  qwen35_output
qwen35_sampling   qwen35_executor
```

All real-weight targets except `qwen35_sampling` require `--model-dir`. They share `--iters`, `--warmup-iters`, and
`--runs`. Production `src` must not gain benchmark-only state, feature paths, or environment knobs.

## Target meanings

- `qwen35_gqa` selects `--gqa-model 27b|35b`, accepts context-parallel/tiled paths, and can run an explicit untimed
  `--validate-tiled` comparison.
- `qwen35_gdn` measures the current ragged recurrent GDN path with the 35B-A3B profile.
- `qwen35_moe` compares token-major and expert-major policies for real sparse-model weights.
- `qwen35_layers` records only main transformer layers and accepts `layer0`, `layer4`, `first4`, or `main_all`.
- `qwen35_output` begins at final norm/gather/unembedding and can isolate sampling/readback.
- `qwen35_executor` measures the public executor contract with fixed `e2e_wo_mtp` and `e2e_w_mtp` cases. Its MTP case
  obtains proposal/draft tokens from production execution rather than substituting a static draft.

Representative smoke commands:

```text
cargo bench -p inference-executor-metal --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-paths context_parallel \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_layers -- \
  --model-dir <27b-model-dir> --cases layer0 --tokens 1 --contexts 0 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_executor -- \
  --model-dir <35b-model-dir> --mtp-model-dir <35b-mtp-model-dir> \
  --cases e2e_w_mtp --iters 1 --warmup-iters 0 --runs 1
```

Run one perf command at a time. List planned cases first; GPU contention and memory pressure invalidate comparisons.

## Metrics

`setup_us` includes model loading and fixture construction. `cache_miss_wall_us` is the first complete execution;
`cache_build_estimate_us` is the CPU record/finish estimate after subtracting measured replay waits. Whole-executor
samples report wall time plus main/output/MTP replay waits. Prepare, record/finish, feedback, and commit remain distinct
host boundaries.

Force-sync/profile-summary measurements are diagnostic metrics, not normal wall-clock throughput. Never compare the two
as if they measured the same workload.

Benchmark keys contain only comparison dimensions. Model, storage/backend, operation, batch/tokens, and a meaningful
context/state coordinate are useful; default layer 0, generic `detail`/`sub-op`, and metadata already printed elsewhere
are not.

```text
gqa/qwen36-27b/metal/full-forward/b1t64-c1024
gdn/qwen36-35b-a3b/metal/full-forward/with-state/b1t1-s1
moe/qwen36-35b-a3b/metal/token-major/b1t64
```

## Profiling and logging

Profile paths are stable, hyphenated, static tree segments owned by local `tree_span_keys.rs` files. Dynamic request,
shape, page, layer, or state values belong in structured logs, not profile keys.

Production executor callsites remain module-qualified:

```text
profile::span(...)        tracing::debug!(...)
profile::eval_component  tracing::info!(...)
profile::eval_operation  tracing::warn!(...)
```

The service's `--profile component|operation` modes currently produce the same coarse CPU tree. They do not provide GPU
kernel timestamps. Use replay-stage debug timing for submission/wait boundaries and Metal capture/counters for kernel
attribution. Service logging fields and commands are documented in [`service.md`](service.md).

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

For speculative decoding also record proposals, sampled tokens, accepted tokens/chunks, and acceptance efficiency. A
throughput change with a different deterministic acceptance trajectory is not a pure executor/kernel comparison.
