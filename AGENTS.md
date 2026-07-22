# Repository Instructions

Start with `docs/high_level.md`. It defines shared repo rules and the runtime core vs model executor boundary.

Focused docs:

- `docs/engineering_conventions.md`: repository-wide naming, layouts, replay/resource safety, optimization, API, and test rules.
- `docs/core.md`: runtime scheduling, lifecycle, page/cache ownership, executor notifications.
- `docs/executor.md`: model executor architecture, symmetry, weights, and replay composition.
- `docs/executor_sampling.md`: current sampling and sparse rejection source/contracts.
- `docs/executor_benchmarks.md`: verification, benchmarks, profiling, and performance evidence.
- `docs/service.md`: model download, server/client operation, logging, and end-to-end checks.
- `docs/executor_gqa.md`: current GQA source layout and paths.
- `docs/executor_gdn.md`: current Gated DeltaNet / GDN source layout and paths.
- `docs/executor_dense_mlp.md`: current dense MLP source layout and paths.
- `docs/executor_moe.md`: current MoE source layout and paths.
- `docs/future_work.md`: active TODOs and future investigations.

When changing GQA, Gated DeltaNet, dense MLP, MoE, sampling, or MTP source layout/default paths, update the matching `docs/executor_*.md` in the same change. When changing service commands or logging, update `docs/service.md`. Current-component docs should describe current `src`, not planned cleanup.

## Shared hard constraints

- Use `panic!`, `assert!`, and `debug_assert!` for internal invariant or contract violations. Use release `assert!` only at init-time, one-time structural or ownership boundaries, or contracts whose enforcement is absolutely necessary in release. Repeated internal bug checks that would add release hot-path noise belong in `debug_assert!`; cover them with tests and debug builds. Classify by lifecycle and cost instead of converting checks mechanically.
- Use recoverable `Err(Exception::custom(...))` only for user input, model loading, runtime failures, MLX failures, IO failures, or other genuinely recoverable failures.
- Do not use `pub(crate)` or `pub(super)`. Keep items private unless intentionally exported with `pub`.
- When working with Rust, use rust-analyzer semantic operations whenever applicable: definition/reference lookup, type and diagnostic inspection, rename, and refactoring. Prefer them over textual heuristics for symbol identity and bindings; use `rg` for textual discovery, not as a substitute for semantic analysis. Rename each binding or item independently when the same spelling appears in multiple scopes. Rust-analyzer does not cover Metal, generated source strings, docs, inactive configurations, or host↔shader ABI correspondence, so audit those boundaries separately and still check for shadowing, stale references, and semantic-equivalence regressions.
- Run Rust formatting as `cargo +nightly fmt`.
- Do not reshape production `src` only to make benchmarks easier.

## Runtime core vs model executor

Runtime core owns scheduling, request lifecycle, token/block metadata, KV/state page allocation/free, page ownership, and cache/state lifecycle notifications.

The model executor owns model execution, model layout parsing, backend-side tensor/state objects, GQA, Gated DeltaNet, dense MLP, MoE, component-local page interpretation, profiling, and benchmarking.

The executor consumes runtime-provided metadata and page IDs. It should not become the scheduler. Runtime core should not parse model-specific tensor layout.

## Performance work

For Metal executor/backend performance work:

- Do not claim a gain or degradation without recording commit, dirty state, model, command, environment, metric, baseline, current result, and verdict.
- Keep force-sync/profile-summary data separate from normal wall-clock throughput.
- Run perf/bench commands one at a time. Do not parallelize them; memory pressure and GPU contention can invalidate results.
- Follow `docs/executor_benchmarks.md`: per executor component, include one-layer production forward perf and configurable setup/kernel/sub-op perf.
- Bench keys should include only dimensions with clear comparison value. Do not include default `layer0`, generic words like `detail`, or values already available as metadata.
