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
dense_mlp  sparse_mlp  moe  gqa_attn  gqa_block_attn  gdn_attn  gdn_state_io
embedding  unembedding  norm
```

`rejection_sampling` is the backend custom-CLI target. Model-executor targets:

```text
qwen3_gqa              qwen3_dspark           qwen3_dspark_forward
qwen3_dspark_unembedding                       qwen3_dspark_sampling
qwen35_dense_mlp       qwen35_moe             qwen35_gqa
qwen35_gdn             qwen35_embed           qwen35_layers
qwen35_output          qwen35_sampling        qwen35_executor
```

Real-weight targets accept the model paths that their production owner needs.
`qwen3_dspark_sampling` needs only `--dspark-model-dir`.
`qwen35_sampling` does not load model weights.
These targets share `--iters`, `--warmup-iters`, and `--runs`.

Production `src` must not gain benchmark-only state, feature paths, or environment controls.

## Target meanings

- `norm` measures standalone RMSNorm and residual-add RMSNorm variants.
  The `rms-only/replay64` cases record 64 standalone BF16 RMSNorm commands in one replay.
  This method amortizes command submission and wait overhead for kernel-level comparisons.
- `qwen35_gqa` selects `--gqa-model 27b|35b` and accepts `single_q_token` or `tiled_q_tokens`. It can run an explicit
  untimed `--validate-tiled-q-tokens` comparison.
- `qwen3_gqa` loads real Qwen3 ungated-GQA weights. It measures full replay, SDPA-only paths, and exact QKV/output
  projection kernels.
- `qwen3_gqa` exposes static tile geometry as CLI arguments. It can validate single-Q output against tiled output.
  Its projection probes compare QMV, QMM BM8/BN32, and QMM BM16/BN32. These forced paths are benchmark-only.
- `gqa_block_attn` measures the model-independent dense block-bidirectional SDPA map component.
  It accepts block size, request count, head geometry, and dtype as CLI arguments.
  The backend owns its one-SIMDgroup threadblock geometry.
- `qwen3_dspark` loads real Main and DSpark checkpoints.
  It runs the public executor lifecycle for `main` or `dspark`.
  Use `--num-spec-tokens N` to select the DSpark proposal length.
  Omit the option to use the checkpoint `block_size`.
  It reports each record, submit/wait, read, and commit boundary.
  It reports `num_spec_tokens`, proposed tokens, accepted tokens, generated proposals, and acceptance.
- `qwen3_dspark_forward` loads real Main and DSpark weights.
  It compares the complete `MainEmbed` and Main forward with `DSparkEmbed` plus the complete DSpark backbone and final
  norm.
  The Main and DSpark depth-comparison cases use the same request count, seven rows per request, and history length.
  The result reports total and per-layer time.
  Use the per-layer value to compare the 40-layer Main stack with the five-layer DSpark stack.
  The additional `main-verification` result uses eight rows per request and a DSpark-enabled Main executor.
  It includes `MainEmbed`, all Main layers, Main residual capture, and DSpark context projection.
  It excludes Main `GatherUnembed` and rejection sampling.
- `qwen3_dspark_unembedding` loads the production `Qwen3xDSparkGatherUnembed` component.
  It uses DSpark-owned unembed weights when they exist.
  Otherwise, it uses the Main unembed weights, as the production executor does.
  It measures gather and unembed together for `block_size` rows per request.
- `qwen3_dspark_sampling` loads real DSpark Markov weights.
  It measures the complete sequential fused Markov-map, sampling, and write-distribution replay.
  Each proposal step uses one fused W1, W2, base-logit-add, and tile-Top-K map.
  It then uses the generic sample-and-write-distribution reducer.
  It accepts `--temperature`, `--top-k`, `--top-p`, and `--seed`.
  It prints proposal token IDs, exact proposal probability bits, and a stable fingerprint of the complete sparse draft
  distribution.
  `qwen3_dspark_forward`, `qwen3_dspark_unembedding`, and `qwen3_dspark_sampling` isolate the three ordered proposal
  segments:

  ```text
  DSparkEmbed + DSpark forward + final norm
    -> GatherUnembed
    -> Markov correction + sampling
  ```

- `qwen35_gdn` measures the current ragged recurrent GDN path with the 35B-A3B profile.
- `qwen35_moe` compares token-major and expert-major policies for real sparse-model weights.
- `qwen35_layers` records only main transformer layers and accepts `layer0`, `layer4`, `first4`, or `main_all`.
- `qwen35_output` begins at final norm, gather, and unembedding. It can isolate sampling and readback.
- `qwen35_executor` measures the public executor contract with `e2e_wo_mtp` and `e2e_w_mtp` cases.
- The `e2e_w_mtp` case accepts `--num-spec-tokens N` and defaults to one speculative token.
- The `qwen35_executor` MTP case obtains proposal and draft tokens from production execution. It does not substitute a
  static draft.
- The `qwen35_executor` fixture obtains Main and MTP page-table widths from the loaded executor.
- The fixture advances compute sequence, token index, and the next input token after each committed decode batch.
  It does not reuse `token_index=0` after model state advances.

Representative smoke commands:

```text
cargo bench -p inference-backend-metal --bench gqa_block_attn -- \
  --block-sizes 7 --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-backend-metal --bench rejection_sampling -- \
  --mode dspark-markov-top-k-map --rows 1 --top-k 20 --vocab 151936 \
  --markov-rank 256 --markov-w1-group-size 64 --markov-w1-bits 4 \
  --markov-w2-group-size 64 --markov-w2-bits 8 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 128 \
  --iters 1 --warmup-iters 0 --runs 1 --validate

cargo bench -p inference-executor-metal --bench qwen3_dspark -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --cases dspark --num-requests 1 --num-spec-tokens 2 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen3_dspark_forward -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --num-requests 1 --context 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen3_dspark_unembedding -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen3_dspark_sampling -- \
  --dspark-model-dir <dspark-model-dir> --num-requests 1 --top-k 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-paths single_q_token \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_layers -- \
  --model-dir <27b-model-dir> --cases layer0 --tokens 1 --contexts 0 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_executor -- \
  --model-dir <35b-model-dir> --mtp-model-dir <35b-mtp-model-dir> \
  --cases e2e_w_mtp --num-spec-tokens 2 \
  --iters 1 --warmup-iters 0 --runs 1
```

Run one performance command at a time. List the planned cases first. GPU contention and memory pressure invalidate
comparisons.

## Metrics

`setup_us` includes model loading and fixture construction. `cache_miss_wall_us` is the first complete execution.

`cache_build_estimate_us` is the CPU record and execute-phase estimate after subtracting measured replay waits.
Whole-executor samples report wall time and Main-only, Main-with-sampling, and speculator replay waits.

Prepare, Main record, execute/read, feedback, and commit remain distinct host boundaries.
The `finish_*` benchmark fields retain their existing output names.
They now measure Main submit/read plus optional speculator record/submit/read.

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
