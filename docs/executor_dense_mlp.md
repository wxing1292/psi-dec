# Dense MLP Executor

This document describes the current dense gated-MLP implementation.
It covers semantic shapes, scratch ownership, Metal replay, tests, and production benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency.
It owns backend-neutral dense MLP layer metadata.
`crates/inference-executor-metal` owns the current Metal replay backend:

```text
crates/inference-executor-core/src/mlp/dense/
  mod.rs
  core.rs      DenseMLPCore + DenseMLPReplayShape

crates/inference-executor-metal/src/mlp/dense/
  mod.rs
  backend.rs   DenseMLPMetalConfig + DenseMLP
  scratch.rs   reusable dense MLP scratch allocation owner and borrowed replay bindings

crates/inference-executor-metal/src/model/qwen/
  v3_x/layer/dense_mlp.rs  Qwen3xDenseMLP, private checkpoint weights, load, and record
  v3/main/layer.rs         fixed Qwen3 Main GQA + dense-MLP layer composition
  v3/main/component_config.rs
                           Qwen3 Main dense-MLP geometry/config builder
  v3_5/main/layer.rs       Qwen3.5 Main dense-MLP/MoE layer variants
  v3_5/mtp/layer.rs        Qwen3.5 MTP dense-MLP/MoE layer variants
  v3_5/component_config.rs Qwen3.5 dense-MLP geometry/config builder

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape

crates/inference-executor-metal/src/def/
  ReplayLayer              typed semantic replay input/output and record contract
crates/inference-executor-core/src/backend/
  Recorder
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal dense MLP kernels live in:

```text
crates/inference-backend-metal/src/components/dense_mlp.rs
crates/inference-backend-metal/src/components/dense_mlp_test.rs
crates/inference-backend-metal/src/components/metal/quantized_dense_mlp_swiglu.metal
```

## Shape model

`DenseMLPCore` owns immutable layer metadata:

```text
model_layer_index
hidden_dim
intermediate_dim
```

It derives dense MLP projection shapes:

```text
linear_shape
gate_up_shape
down_shape
```

`DenseMLP` connects model-level dense MLP metadata to `inference-backend-metal` kernels.
It owns the full `gate_up -> swiglu -> down` backend path.
It does not own tensor storage, runtime scheduling, or page allocation.

The backend implements `ReplayLayer`.
Qwen model and layer code use `Recorder` to append dense MLP work to a larger whole-layer or whole-model replay.
Focused tests and benches build replay programs from the same recorder path.
The internal order is `gate_up -> swiglu [barrier before] -> down [barrier before]`.
Model and layer wiring own barriers on the first consumer command and downstream residual consumers.

## Replay contract

`DenseMLP` records one dense gated MLP forward into a caller-owned `Recorder`.
It does not submit commands.
It does not own tensor storage or request lifecycle.
The semantic layer input is
`DenseMLPInput { num_total_tokens, num_active_tokens, hidden_state, next_hidden_state, scratch, weights }`.
Replay returns the caller-owned `next_hidden_state` buffer directly.

The replay order is:

```text
hidden_state
  -> fused gate/up quantized projection
  -> SwiGLU: SiLU(gate) * up
  -> down quantized projection
  -> next_hidden_state
```

`DenseMLPInput.num_total_tokens` is the recorded row capacity. `num_active_tokens` accepts
`ReplayU32::Fixed(value)` or `ReplayU32::Parameter(key)`.
Production callers allocate scratch for model capacity.
A fixed invocation requires the active and total token counts to match. Every invocation validates all buffers against
`num_total_tokens`.
All buffers and weights must match the configured dimensions, group size, bit width, and dtype.
This requirement covers hidden buffers, gate/up scratch, swiglu scratch, and immutable weights.

Qwen model replay keeps dense MLP scratch in one model-owned `DenseMLPScratch`.
Its `bindings()` method exposes borrowed `DenseMLPScratchBindings` during replay recording.
Scratch allocation geometry consists of `max_tokens`, `intermediate_dim`, and `io_dtype`.
It does not accept quantization group size or bit width because those weight facts do not affect scratch layout.
The model stream serializes Main and MTP execution.
Thus, layers can reuse `gate_up` and swiglu scratch.

The shared `Qwen3xDenseMLP` leaf directly owns immutable weights and per-layer output buffers.
It retains the core and Metal configuration that created its backend.
Weight reload uses these retained values.
`Qwen3MainLayer` and the dense variants of `Qwen35MainLayer` and `Qwen35MTPLayer` compose that leaf.
Each composition uses a separate role-specific layer and scratch type.
Their model-specific binding trees contain `Qwen3xDenseMLPWeightBindings` at the leaf boundary.
The weight owner loads one bounded `TensorMap` from that exact gate/up/down binding subtree.
It removes every tensor and materializes fused gate-up buffers while it keeps the down projection separate.
The map must be empty after construction.
When Main and MTP both use dense MLP, Qwen3.5 initialization requires their `intermediate_size` values to match.
Both users share the executor `max_tokens` value and the current BF16 model boundary.
If only one graph uses dense MLP, the loader derives the scratch geometry from that graph.

### Bucketed replay

`dense_mlp::Compute` exposes one composite topology identity:

```text
dense_mlp::ReplayTopology
  gate_up_affine
  down_affine
```

It also exposes the sorted union of the `gate_up` and `down` affine topology boundaries.
The owner of a larger replay stage must union these boundaries with the boundaries from all other token-domain components.
The larger replay stage then selects one shared `num_total_tokens` capacity.
Dense MLP does not own that final policy.

The replay key must contain `num_total_tokens` and the composite dense MLP topology.
The key must not contain `num_active_tokens`.
The submission supplies `num_active_tokens` through the caller-owned replay parameter key.

The three dense MLP stages bind the same `u32` key with the same `1..=num_total_tokens` domain:

```text
gate_up affine
SwiGLU
down affine
```

Thus, a parameterized dense MLP replay declares one parameter. A fixed active count declares no parameters.

Each stage records work for `num_total_tokens` rows.
Affine QMV returns for each inactive row before it reads input or writes output.
Affine QMM skips fully inactive tiles and masks inactive rows in a partially active tile.
SwiGLU dispatches `num_total_tokens * intermediate_dim` threads.
It returns when `gid >= num_active_tokens * intermediate_dim`.
This return occurs before row calculation, gate/up scratch reads, and SwiGLU scratch writes.

Inactive scratch rows can contain poison, output from an earlier full submission, or other stale values.
The implementation does not clear these rows.
The guards ensure that later stages do not consume or overwrite them.

`Qwen3xDenseMLP` exposes one `record(...)` API and topology accessors.
Qwen3.5 Main calls this API with the Main stage-owned total token count and active-token value.
Qwen3.5 MTP calls this API with its separate body stage-owned token capacity and active-token key when the physical MTP
layer uses dense MLP.
The DSpark path supplies equal active and total counts.

## Data flow and backend stages

Dense MLP is a pure hidden-state transform with no request page/state side effects:

```text
hidden_state[num_tokens, hidden_dim]
  -> fused gate/up quantized affine
  -> swiglu[row, intermediate] = SiLU(gate[row, col]) * up[row, col]
  -> down quantized affine
  -> next_hidden_state[num_tokens, hidden_dim]
```

The fused gate/up projection writes a stacked intermediate buffer:

```text
gate_up[row, 0..intermediate_dim)                  gate projection
gate_up[row, intermediate_dim..2*intermediate_dim) up projection
```

The `SwiGLUKernel` reads both halves.
It writes one `swiglu[num_tokens, intermediate_dim]` scratch buffer.
The down projection reads that scratch and immutable down weights.
It then writes the component output.

The hidden input and output are model-boundary bf16 buffers.
Quantized affine kernels apply the stored per-group scale/bias during accumulation.
Each kernel accumulates into its internal accumulator type.

Resource flow is:

```text
gate_up affine
  reads hidden_state + gate/up weights/scales/biases
  writes gate_up scratch

swiglu
  reads gate_up scratch
  writes swiglu scratch

down affine
  reads swiglu scratch + down weights/scales/biases
  writes next_hidden_state
```

The component records barriers between these stages.
Each stage consumes scratch from the previous stage.
Model replay records additional layer-level barriers around residual and norm consumers.
It does not put these barriers inside the dense MLP component.

Dense MLP has no expert-major policy.
Every active token row runs the same dense expert.
The backend exposes one `invoke(...)` API.
It accepts `Shape { num_total_tokens }` and one `ReplayU32` active-token value.
`ReplayU32::Fixed(value)` requires `value == num_total_tokens`.
`ReplayU32::Parameter(key)` supplies the active prefix at submission.
Capacity buffers can be larger than the active prefix.

## Execution hierarchy

Dense MLP is a semantic command graph. It is not one kernel launch:

```text
DenseMLPExecution
├── gate_up affine_quantized::Matmul
├── SwiGLU KernelLaunch
└── down affine_quantized::Matmul
```

Each affine owner defines its QMV or QMM thread-block task, tile geometry, and layout. The dense MLP owner supplies the
projection geometry and runtime row count. It does not select the affine kernel again.

The current SwiGLU kernel uses this compile-time constant hierarchy:

```text
SwiGLUKernelConstants
├── io_dtype
└── thread_block
    └── required_threads = 256
```

One non-persistent SwiGLU thread block processes a bounded flat range of `(token, intermediate)` output coordinates.
One thread processes one coordinate at a time. It reads the matching gate and up values and writes one SwiGLU value.
The tensor is row-major. The flat dispatch does not make a thread block the owner of one complete token row.

Dense MLP does not use a component-level registry, selector, or planner. The command graph does not change with the
runtime row count. The
two affine owners make independent row-dependent kernel choices. `dense_mlp::ReplayTopology` records both affine
choices so that a replay bucket cannot cross either topology boundary.

## Backend selection

`dense_mlp::Compute` owns one adaptive `affine_quantized::Matmul` for gate/up and one for down.
Each `affine_quantized::Matmul` owns the QMV/QMM candidates and selects its kernel.
The model and executor provide the complete dense-MLP dimensions and active row count.
They do not select a kernel or tile.

Large dense MLPs use this policy when `hidden_dim > 4096` or `intermediate_dim > 4096`:

| Active rows | Backend path |
| ---: | --- |
| 1–5 | QMV |
| 6–8 | QMM BM8/BN32 |
| 9–16 | QMM BM16/BN32 |
| 17 or more | QMM BM32/BN32 |

Smaller dense MLPs keep QMV for a longer range.
The QMV limit is 18 rows when both dimensions are at most 2048.
The QMV limit is 12 rows for the remaining smaller shapes.
The backend uses BM16/BN32 through 16 rows after that limit.
It uses BM32/BN32 for larger row counts.

Gate/up and down apply the same backend selector independently.
They can share a family when their dimensions select the same candidate.
The 8-row BF16 BM8/BN32 kernel uses 64 threads and 3200 bytes of static threadblock memory.
The memory contains the `8 × 40` input tile and the `32 × 40` weight tile.
The `40` stride is `BK=32` plus eight BF16 padding values.
Kernel initialization checks the SIMD width, pipeline thread limit, calculated threadblock memory, reported static
threadblock memory, and device threadblock-memory limit.

Benchmark-only QMV/QMM probes select an affine kernel policy for measurement.
The semantic data flow stays the same.

## Tests and benchmarks

The focused backend test records each dense MLP topology into an isolated test cache.
It replays non-monotonic active-token sequences and compares each active output prefix with the CPU quantized
dense-MLP reference.
The reference covers the gate/up projection, `SiLU(gate) * up`, and the down projection as one numerical contract.
It accepts separate group sizes and bit widths for the gate/up and down projections.
The test covers the production Q4/Q4 BF16-affine layout and a mixed Q4/Q6 F32-affine layout.
It records capacities on each side of the affine topology boundaries.
It does not use inactive output or scratch rows as an oracle.
Affine operator tests own kernel selection and topology-boundary contracts.

Current Metal component bench:

```text
cargo bench -p inference-backend-metal --bench dense_mlp -- --profile-time 1 --noplot
```

Current Metal real-weight comparison bench:

```text
cargo bench -p inference-executor-metal --bench qwen35_dense_mlp -- \
  --model-dir <27b-model-dir> --tokens 1 --cases full_auto \
  --iters 1 --warmup-iters 0 --runs 1
```

The bench covers the 27B dense profile.
CLI arguments select the model path, token list, case list, iteration count, warmup count, and run count.
The bench can run the automatic full dense MLP path or focused shape-policy probes:

```text
full_auto
full_qmv_bn8_bk32
full_qmm_bm8_bn32
full_qmm_bm16_bn32
full_qmm_bm32_bn32
gate_up_auto
gate_up_qmv_bn8_bk32
gate_up_qmm_bm8_bn32
gate_up_qmm_bm16_bn32
gate_up_qmm_bm32_bn32
swiglu
down_auto
down_qmv_bn8_bk32
down_qmm_bm8_bn32
down_qmm_bm16_bn32
down_qmm_bm32_bn32
```

The default forward path is the real-weight replay path:

```text
gate_up -> swiglu -> down
```

The `swiglu` stage computes `SiLU(gate) * up` from the stacked gate/up projection.
Public replay APIs call this stage `swiglu`.
It is the dense MLP SwiGLU contract, not a standalone SiLU transform.

The real-weight `*_auto` cases use `DenseMLP` and its normal shape-dependent policy.
`qmv_bn8_bk32` means the forced QMV BN8/BK32 kernel.
Each `qmm` case includes its complete BM/BN tile.
Forced qmv/qmm cases are benchmark-only operator-policy probes.

They help select the correct production threshold.
They are not separate production paths.
Dense MLP no longer keeps direct-submit or fused gate/up swiglu forward probes as production paths.

The real-weight bench prints replay metadata with each perf row:

```text
backend
command_count
retained_buffers
retained_pipelines
constant_bytes
```

The Metal stream backend name supplies `backend`.
The expected value is `backend=metal`.

Recommendation: Compare the backend component bench first.
Then compare the real-weight dense MLP wrapper and the layer/layer-ladder bench.
Dense MLP scratch is reusable at model scope.
The caller must preserve the layer-boundary hidden buffer until downstream residual consumers finish.

[`executor_benchmarks.md`](executor_benchmarks.md) defines shared GPU serialization, benchmark metrics, and
performance-evidence rules.
