# Documentation Guide

The top-level [README](../README.md) is the main entry point.
It explains the project, request flow, startup process, and subsystem locations.
This guide identifies the document that owns each type of information.

## Document roles

| Kind | Answers | Documents |
| --- | --- | --- |
| Project orientation | What is `psi-dec`, and how do I run or read it? | [README](../README.md) |
| Architecture | Who owns lifecycle, data, and execution order? | [`core.md`](core.md), [`executor.md`](executor.md), [`gpu_execution.md`](gpu_execution.md), Metal backend [README](../crates/inference-backend-metal/README.md) |
| Current components | What source implements this model component today? | [`executor_qwen.md`](executor_qwen.md), [`executor_gqa.md`](executor_gqa.md), [`executor_gdn.md`](executor_gdn.md), [`executor_dense_mlp.md`](executor_dense_mlp.md), [`executor_moe.md`](executor_moe.md), [`executor_sampling.md`](executor_sampling.md) |
| Qwen3x DSpark | How does the current fixed-block Qwen3x DSpark path work? | [`dspark_design.md`](dspark_design.md) |
| Qwen3x DFlash2 | How do current DFlash2 Prefill, Decode, sliding attention, convolution, and selection work? | [`dflash2_design.md`](dflash2_design.md) |
| Workflows | How do I run, verify, benchmark, or profile it? | [`service.md`](service.md), [`executor_benchmarks.md`](executor_benchmarks.md) |
| Engineering rules | Which rules apply to code, APIs, and technical English? | [`high_level.md`](high_level.md), [`engineering_conventions.md`](engineering_conventions.md), [`technical_english.md`](technical_english.md) |
| Follow-up work | What remains unresolved or under investigation? | [`future_work.md`](future_work.md) |

Each document has one primary job.
Link to the owning document instead of copying its full contract.
Current component documents describe current `src`.
Put future designs in `future_work.md`.
Put durable repository rules in `engineering_conventions.md`.
Put component-specific findings in the current document that owns the component.

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

GQA and GDN use `attn/gqa` and `attn/gdn`.
Dense MLP and MoE use `mlp/dense` and `mlp/moe`.
Sampling uses `sampling`.

### Change Metal recording or a kernel

```text
high_level.md
executor.md
gpu_execution.md
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

Performance notes must record the commit, dirty state, model, command, environment, workload/trajectory, metric,
baseline, current result, and verdict.
Run GPU and performance commands one at a time.

## Maintenance rules

- Recommendation: Use a link and a one-sentence boundary instead of duplicated prose.
- Keep headings navigable and source paths current.
- Put shared test and benchmark commands in the workflow that owns them.
- Keep a component command only when it explains the production path or its flags.
- Do not add broad historical note directories.
- Consolidate or delete stale prose when a stable rule moves to an owning document.
- Do not describe desired future state as current API or source.
- Apply [`technical_english.md`](technical_english.md) to new or revised English documentation.
