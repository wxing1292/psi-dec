# Dense MLP Executor

This document maps the current dense gated-MLP implementation from semantic shapes and scratch ownership through Metal
replay and production benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency. It keeps backend-neutral dense MLP layer
metadata; `crates/inference-executor-metal` owns the current Metal replay backend:

```text
crates/inference-executor-core/src/mlp/dense/
  mod.rs
  core.rs      DenseMLPCore + DenseMLPReplayShape

crates/inference-executor-metal/src/mlp/dense/
  mod.rs
  backend.rs   DenseMLPMetalConfig + DenseMLP
  scratch.rs   reusable dense MLP scratch allocation owner and borrowed replay bindings

crates/inference-executor-metal/src/model/qwen/v3_5/layer/
  dense_mlp.rs Qwen35DenseMLP, private checkpoint weights, load, and record

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape
  Layer

crates/inference-executor-metal/src/def/
  ReplayLayer
crates/inference-executor-core/src/backend/
  Recorder
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal dense MLP kernels live in:

```text
crates/inference-backend-metal/src/components/quantized_dense_mlp.rs
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

`DenseMLP` wires model-level dense MLP metadata to `inference-backend-metal` dense MLP kernels. It owns
the full dense MLP backend path `gate_up -> activation -> down`; it does not own tensor storage, runtime scheduling,
or page allocation.

The backend implements `Layer + ReplayLayer` so Qwen model/layer code can append dense MLP work into a larger
whole-layer or whole-model replay through `Recorder`. Focused tests and benches build replay programs from that
same recorder path. Dense MLP internal replay order is
`gate_up -> activation [barrier before] -> down [barrier before]`; model/layer wiring owns barriers on the component's
first consumer command and on downstream residual consumers.

## Replay contract

`DenseMLP` records one dense gated MLP forward into a caller-owned `Recorder`. It does not submit
commands and it does not own tensor storage or request lifecycle. The semantic layer input is
`DenseMLPReplayInput { shape, hidden_state, next_hidden_state, scratch, weights }`. Replay returns the caller-owned
`next_hidden_state` buffer directly.

The replay order is:

```text
hidden_state
  -> fused gate/up quantized projection
  -> SiLU(gate) * up activation
  -> down quantized projection
  -> next_hidden_state
```

`DenseMLPReplayShape.num_tokens` is the backend-neutral current microbatch row count. The Metal lowering path maps it
to `QuantizedDenseMLPShape` only inside `crates/inference-executor-metal`. Production callers allocate scratch for model
capacity, but each replay invocation validates and uses only the current token count. The hidden input, hidden output,
gate/up scratch, activation scratch, and immutable weights must all match the configured hidden/intermediate dimensions,
group size, bit width, and dtype.

Qwen model replay keeps dense MLP scratch in one model-owned `DenseMLPScratch`. Its `bindings()` method exposes borrowed
`DenseMLPScratchBindings` to replay recording. Main and MTP execution is serialized on the model stream, so `gate_up` and
activation scratch are reusable across layers. Per-layer output buffers and immutable weights remain owned directly by
the dense variant inside `Qwen35Layer`; there is no separate production layer-weight bundle. Qwen validates scratch
layout compatibility across every Main and optional MTP dense layer at init.

## Data flow and backend stages

Dense MLP is a pure hidden-state transform with no request page/state side effects:

```text
hidden_state[num_tokens, hidden_dim]
  -> fused gate/up quantized affine
  -> activation[row, intermediate] = SiLU(gate[row, col]) * up[row, col]
  -> down quantized affine
  -> next_hidden_state[num_tokens, hidden_dim]
```

The fused gate/up projection writes a stacked intermediate buffer:

```text
gate_up[row, 0..intermediate_dim)                  gate projection
gate_up[row, intermediate_dim..2*intermediate_dim) up projection
```

The activation kernel reads both halves and writes one `activation[num_tokens, intermediate_dim]` scratch buffer. The
down projection reads that activation scratch and immutable down weights, then writes the component output. The hidden
input and output are model-boundary bf16 buffers; quantized affine kernels apply the stored per-group scale/bias while
accumulating into the kernel's internal accumulator type.

Resource flow is:

```text
gate_up affine
  reads hidden_state + gate/up weights/scales/biases
  writes gate_up scratch

activation
  reads gate_up scratch
  writes activation scratch

down affine
  reads activation scratch + down weights/scales/biases
  writes next_hidden_state
```

The component records barriers between these stages because each stage consumes the previous stage's scratch. Model
replay records any additional layer-level barriers around residual/norm consumers, not inside the dense MLP component.

Dense MLP has no token-major versus expert-major policy: every active token row runs the same dense expert. The only
shape input is `num_tokens`; capacity buffers may be larger, but each replay invocation uses the current active prefix.
Benchmark-only qmv/qmm probes choose affine kernel policy for measurement, but the semantic dataflow stays the same.

## Tests and benchmarks

Focused backend tests compare the current quantized bf16 replay path against the CPU quantized dense-MLP reference for
both fixed and random inputs. They cover gate/up projection, `SiLU(gate) * up`, and down projection as one numerical
contract.

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

The bench covers the 27B dense profile and uses CLI args for model path, token list, case list, iteration count, warmup
count, and run count. It can run the automatic full dense MLP path or focused shape-policy probes:

```text
full_auto
full_qmv
full_qmm
gate_up_auto
gate_up_qmv
gate_up_qmm
activation
down_auto
down_qmv
down_qmm
```

The default forward path is the real-weight replay path:

```text
gate_up -> activation -> down
```

The activation stage computes `SiLU(gate) * up` from the stacked gate/up projection. Public replay APIs use
`activation` for this stage because it is the dense MLP activation contract, not a standalone SiLU transform.

The real-weight `*_auto` cases use `DenseMLP` and its normal shape-dependent policy. `qmv` means the quantized
matrix-vector kernel; `qmm` means the quantized matrix-matrix kernel. Forced qmv/qmm cases are benchmark-only
operator-policy probes used to choose the correct production threshold; they are not separate production paths. Dense
MLP no longer keeps direct-submit or fused gate/up activation forward probes as production paths.

The real-weight bench prints replay metadata with each perf row:

```text
backend
command_count
retained_buffers
retained_pipelines
constant_bytes
```

`backend` is reported by the Metal stream backend name and should print `backend=metal`.

Perf and correctness checks should first compare the backend component bench, then the real-weight dense MLP wrapper,
then the layer/layer-ladder bench. Dense MLP scratch is reusable at model scope, but the caller must preserve the
layer-boundary hidden buffer until downstream residual consumers have finished.

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).
