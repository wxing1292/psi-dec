# Documentation Guide

The top-level [README](../README.md) is the human entry point: what the project is, how one request flows, how to run it,
and where each subsystem lives. This guide answers the next question: which document owns the detail you need?

## Document roles

| Kind | Answers | Documents |
| --- | --- | --- |
| Project orientation | What is psi-dec and how do I run/read it? | [README](../README.md) |
| Architecture | Who owns lifecycle, data, and execution order? | [`core.md`](core.md), [`executor.md`](executor.md), Metal backend [README](../crates/inference-backend-metal/README.md) |
| Current components | What source implements this model component today? | [`executor_qwen.md`](executor_qwen.md), [`executor_gqa.md`](executor_gqa.md), [`executor_gdn.md`](executor_gdn.md), [`executor_dense_mlp.md`](executor_dense_mlp.md), [`executor_moe.md`](executor_moe.md), [`executor_sampling.md`](executor_sampling.md) |
| Workflows | How do I run, verify, benchmark, or profile it? | [`service.md`](service.md), [`executor_benchmarks.md`](executor_benchmarks.md) |
| Engineering rules | How should code and APIs be designed? | [`high_level.md`](high_level.md), [`engineering_conventions.md`](engineering_conventions.md) |
| Follow-up work | What remains unresolved or under investigation? | [`future_work.md`](future_work.md) |

Each document has one primary job. Link to an owning document instead of copying its whole contract. Current component
docs describe current `src`; future designs belong in `future_work.md`. Promote durable repository-wide findings into
`engineering_conventions.md` and component-specific findings into the owning current document.

## Reading paths

Choose the shortest path that reaches the owner of your question.

### Understand one request

```text
../README.md
core.md
executor.md
executor_qwen.md
```

### Change runtime scheduling or cache lifecycle

```text
high_level.md
core.md
../crates/inference-runtime-core/src/
```

### Change a model component

```text
high_level.md
executor.md
the matching executor_<component>.md
../crates/inference-executor-core/src/<component>/
../crates/inference-backend-metal/src/components/
../crates/inference-executor-metal/src/<component>/
```

GQA and GDN use `attn/gqa` and `attn/gdn`; dense MLP and MoE use `mlp/dense` and `mlp/moe`; sampling uses `sampling`.

### Change Metal recording or a kernel

```text
high_level.md
executor.md
../crates/inference-backend-metal/README.md
the matching component doc
```

### Run or validate the service

```text
service.md
executor_benchmarks.md        # when a measurement or release claim is involved
```

### Investigate performance

```text
executor_benchmarks.md
the matching current component doc
future_work.md                 # active known investigations
```

Performance notes record commit, dirty state, model, command, environment, workload/trajectory, metric, baseline,
current result, and verdict. Run GPU/perf work serially.

## Maintenance rules

- Prefer a link and a one-sentence boundary over duplicated multi-section prose.
- Keep headings navigable and source paths current.
- Put shared test/benchmark commands with the workflow that owns them; retain a component command only when it explains
  that component's production path or flags.
- Do not add broad historical note directories; consolidate or delete stale prose when a stable rule is promoted.
- Do not describe desired future state as current API or source.
