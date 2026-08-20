# Model Primitive Executor Components

This document describes the current embedding, unembedding, normalization, residual, and fused RoPE components. It
defines their component-specific GPU tasks and selection owners.

## Source layout

Model owners live in:

```text
crates/inference-executor-metal/src/model/
  embedding.rs
  unembedding.rs
  rms_norm.rs
  residual_add.rs
```

Reusable backend components live in:

```text
crates/inference-backend-metal/src/components/
  quantized_embedding.rs
  rms_norm.rs
  residual_add.rs
  residual_add_rms_norm.rs
  rms_norm_rope.rs
```

These components use the execution vocabulary in [GPU Execution Vocabulary](gpu_execution.md).

## Embedding

`Embed` owns the model-level quantized embedding weights and one `QuantizedEmbeddingKernel`. It records an exact token
count or a fixed replay capacity with a submission-time active-token count.

The kernel computes this mapping:

```text
token_ids[token]
    -> quantized vocabulary row
    -> per-group scale and bias
    -> output_hidden[token, hidden]
```

The current kernel specialization is:

```text
QuantizedEmbeddingKernelSpecialization
├── scale_bias_dtype
├── output_dtype
└── thread_block
    └── required_threads = 256
```

One non-persistent thread block processes a bounded flat range of `(token, hidden)` output coordinates. One thread
processes one coordinate at a time. The kernel derives the token and hidden indices from the flat coordinate.

Embedding has one current kernel implementation for each supported scale/bias dtype. Dtype selection occurs at
initialization. The runtime token count changes only the grid. Embedding does not need a registry or runtime planner.

## Unembedding

`Unembed` owns one adaptive `AffineQuantizedMatmul`. It maps hidden rows to vocabulary-logit rows:

```text
hidden[num_rows, hidden_dim]
    -> affine quantized matrix multiplication
    -> logits[num_rows, vocab_size]
```

`AffineQuantizedMatmul` owns QMV/QMM registration, row-dependent selection, kernel tile geometry, and topology
boundaries. `Unembed` supplies model geometry, weights, buffers, and row counts. It must not select a second kernel.

Embedding and unembedding share weight lifecycle and replay-capacity conventions. They do not share one GPU planner.
Embedding is a row lookup. Unembedding is a matrix multiplication.

## RMSNorm

The standalone RMSNorm kernel maps one thread block to one token row:

```text
one ThreadBlock
    -> one token row
    -> reduce sum of squares
    -> apply reciprocal RMS and weight
    -> one output row
```

The dtype selects the F32 or BF16-vectorized kernel at initialization. The current kernel requires 1024 threads. The
runtime token count determines the grid dimensions. RMSNorm does not use a runtime planner.

## Residual add

The plain residual-add kernel processes a bounded flat range of output values. Its row-prefix replay form preserves
complete row boundaries for active-count guards. The capture form also copies selected output columns to a separate
buffer.

Dtypes select one static kernel at initialization. The runtime shape changes only the grid and active-prefix guard.
Residual add does not use a runtime planner.

## Residual-add RMSNorm

The fused component maps one thread block to one token row:

```text
lhs row + rhs row
    -> residual_output row
    -> RMSNorm
    -> norm_output row
```

The optional capture form also writes selected residual columns. The backend selects the scalar or BF16-vectorized
kernel from the validated dtype and hidden geometry at initialization. This is a static kernel-kind choice. It is not
a dynamic execution plan.

## RMSNorm/RoPE

The fused RMSNorm/RoPE component maps one thread block to one `(flat Q token, Q head)` row. It normalizes one head row,
reads the request position from `flat_token_indices`, applies RoPE to `rope_dim`, and writes the complete head row.

The compiled specialization contains the attention-head geometry, RoPE geometry, epsilon, scaling constants, dtype,
and fixed thread-block requirement. The runtime token count changes only the grid and the active-token guard. The
component has one current specialization for one model configuration and does not need a runtime planner.

## Replay and ownership

Model owners retain immutable weights. Replay programs retain backend pipelines and buffers according to the common
replay contract. Exact invocations record exact token or row counts. Bucketed invocations record capacity and bind one
active-count parameter.

The runtime core does not select primitive kernels. The Metal backend owns dtype specialization, affine selection, and
execution configuration. Model code owns semantic composition and buffer roles.
