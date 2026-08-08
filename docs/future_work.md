# Future Work

This file contains only unresolved implementation work and bounded investigations. Current contracts belong in the
current documents.

Move resolved repository-wide rules into `engineering_conventions.md`. Move resolved component findings into the
document that owns the component.

## Tool APIs

- Implement the Pi custom-provider session wire and persistent conversation store.
  Define the session ID, event envelope, append ordering, concurrent append behavior, and storage owner.
  Map appended events into the existing `ToolEvent` and `ToolState` domain.

  Recommendation: Register common tools once.
  Later, append only additions, removals, calls, responses, and cancellations.
  Do not retransmit complete tool definitions on every turn.
  Do not model conversation history as an evictable cache.
  Do not add a registry resynchronization protocol.

## Runtime Lifecycle

- Enforce each model's context window in the shared request lifecycle.
  Thread the model limit through admission and decode commit.
  Reject prompts that leave no output capacity.
  Produce the reserved `CompletionReason::ContextLimit`.
  Map this reason to each RPC protocol.
  Do not infer the reason from emitted token counts.
- Make trie and request cache-block extent runtime-dynamic before offering arbitrary Qwen cache-block CLI values.
  The extent is currently a const generic.
  Runtime, request, trie block, cache, and RPC service types share this generic.
  Many power-of-two monomorphizations would increase code size and maintenance cost for one simple option.
- Drain the user and reservation-completion channels before flushing when both channels are ready.
  This behavior prevents queue priority from depending on crossbeam select order.
- Replace the unbounded per-request token-output channel with bounded slow-consumer handling.
  Base the handling on explicit per-request or service-level byte accounting.
  Saturation must terminate only the affected request.
  It must return a clear RPC error.
  It must not silently drop committed output or panic the service.
- Redesign host-pinned segment ownership before enabling offload.
  An allocation must have unique ownership for mutation and free operations.
  The new design permits only read-only shared views.
  It defines their cross-thread contract explicitly.
  Then implement KV and state onload and offload as a separate lifecycle.
  This lifecycle must remain distinct from the existing reservation-wait task.
- When a later change revises the service and executor boundary, move `ReplayableModelBatchExecutor` and the executor
  timing and output traits out of `inference-runtime-core`.

## Pipeline Parallelism

- Add a bounded final-response reorder buffer for the scheduler and runtime core.
  The buffer holds completed future sequences.
  It releases only the next compute-slot sequence to FIFO commit.
  Cover the case where `n + 1` arrives before `n`.
  Do not weaken the FIFO contract or requeue pending requests.
- Define a model-agnostic `PipelineStageResult` transport envelope.
  The envelope carries compute sequence, stage, request-slot identity, and model-specific stage payload.
  The Qwen payload must preserve ragged hidden-state row association.
- Add Qwen pipeline-stage configuration for the stage index, stage count, and main-layer range.
  Derive first-stage and last-stage behavior from this configuration.
  Let non-first stages ingest transported hidden states.
  Let non-last stages materialize outgoing hidden states.
  Only the last stage runs final norm and unembedding.
  Only the last stage runs sampling, rejection, and MTP proposal.
- Send final sampling and rejection decisions back to each stage as a commit notification.
  Each stage must commit its pending GDN/cache transaction.
  It must commit the transaction before the final response returns to runtime core.
- Permit multiple batches to overlap only after each stage uses bounded, sequence-ordered pending transactions.
  Transport, commit notification, and in-flight cache-publish ownership preserve the same per-request causal order.

## Prefill

Decode correctness and latency are the current optimization priority. Revisit high-throughput long prefill as a separate
future effort. Do not complicate the decode path prematurely.

Any new path must preserve these behaviors:

- Ragged-batch correctness
- Cache and state lifecycle
- Mixed prefill and decode behavior

Use end-to-end TTFT and prompt-throughput evidence to select the implementation. Do not treat a removed or hypothetical
component path as the design.

## Replay Evolution

- Revisit the backend-neutral replay boundary before adding another backend.
  The current neutral `Runtime` and `Recorder` contracts coexist with Metal-only replay programs and fusion.
  They also coexist with executor-side adapters.
  Keep the shared execution lifecycle backend-agnostic.
  Keep Metal replay and fusion in the Metal backend.
  Recommendation: Do not mirror replay operator and recorder APIs across the backend and executor.
- Add a bounded replay catalog only if measured replay-memory growth justifies it.
  Its eviction policy must account for all stage caches.
  The policy must retain in-flight resources through GPU completion.
- Design each multi-command-buffer replay chain around the CPU readback boundary that MTP rejection requires.
  Do not bypass allocator-reset ownership or proposal dependency semantics.

## Performance Investigations

- Evaluate software-pipelined K and V loads for GQA `TiledQTokens` as a bounded experiment.
  The current `Tkv=16`, `D=256` K and V threadgroup tiles use 16.5 KiB.
  Duplicating them with the existing eight-value row padding requires 33 KiB.
  This requirement exceeds the 32 KiB threadgroup-memory limit.

  First measure the correctness, bank-conflict behavior, and performance after removing that padding.
  Then compare an exact-32-KiB ping-pong layout with the current single-workspace kernel.
  Use identical real-weight cases for the comparison.
  Do not make this design the default without component parity and end-to-end evidence.
- Isolate Qwen3.6 Metal first-request latency from steady-state decode. On
  `2b06deb0183e75c24699dedb7784b116b4987d3b` (`dirty=0`, macOS 27.0, M3 Max 40-core GPU), the observed TTFT was:

  | Model | MTP | First request | Next two requests |
  | --- | --- | ---: | ---: |
  | 27B | off | 1629 ms | 330-332 ms |
  | 35B-A3B | off | 3735 ms | 72-78 ms |
  | 27B | on | 960 ms | 337-342 ms |
  | 35B-A3B | on | 2195 ms | 76-78 ms |

  The first 35B-A3B MTP-off 1024-token run also reached 84.3 tok/s.
  Later runs reached 93.1-93.4 tok/s.
  Reproduce the baseline and current commits with identical cache, scheduler, logging, model, sampling, and cooldown
  settings.
  Report cold and steady-state results separately.

  Instrument model and device initialization.
  Instrument generated Metal pipeline compilation and cache hits.
  Instrument replay recording and the first distinct replay shapes.
  Collect this evidence before assigning the latency to one layer.

## Model and Backend Investigations

- Audit Qwen3 and Qwen3.5 tensor-level quantization overrides outside MoE routing.
  Their current Main and MTP GQA, GDN, and MLP builders use model-level affine defaults.
  Confirm the supported checkpoint contracts before changing the loaders.
  If a checkpoint permits per-layer layouts, resolve each semantic layer from its exact binding subtree.
  Keep page tables, metadata, and compatible scratch shared.
  Do not share a weight-dependent backend across incompatible affine layouts.
- Wire the reusable Qwen3x DSpark model into Qwen3.5 only when a supported checkpoint defines the Main compatibility
  and executor lifecycle. Reuse `MainResidualCapture`, `Qwen3xDSparkModel`, `Qwen3xDSparkMarkov`, and the generic
  DSpark attention/state owners. Do not create a parallel `qwen/v3_5/dspark` implementation.
- Use the Qwen3 DSpark confidence output to add dynamic proposal lengths.
  Keep global proposal ranking and verification-budget allocation in runtime scheduling.
  Keep the executor output fixed at `block_size` until the scheduler owns this cross-request policy.
- Investigate strict one-row and multi-row Qwen3 Main numerical parity.
  The 2026-07-29 greedy acceptance audit found one deterministic output divergence in four prompts.
  At the first divergence, sparse rejection accepted zero draft tokens.
  The eight-row Main verification sampled token `117`.
  The one-row Main path sampled token `223`.
  Compare the Main GQA, dense MLP, and unembed outputs before changing production kernels.
  Treat exact batch-shape reproducibility as a separate requirement from DSpark rejection correctness.
- Add a separate gated DSpark GQA implementation when a supported checkpoint requires it.
  Do not add a runtime gate flag to `UngatedDSparkGQA`.
  Keep the history map, block map, reducer, page layout, and scratch contracts gate-neutral.
- Add Qwen3 DSpark checkpoint variants only from real checkpoint contracts.
  This work can include a final-layer entry in the official `target_layer_ids` field, additional Markov heads, or a
  different proposal layout.
  Do not add speculative compatibility fields without a supported checkpoint.
- Permit overlapping Qwen3 DSpark batches only after all proposal scratch, outputs, replay arguments, and probability
  stores have bounded in-flight ownership.
- Design backend-agnostic immutable `Weight` / `Tensor` / `Storage` ownership.
  Recommendation: Assign file and mapped-storage lifetime to checkpoint readers.
  Assign tensor identity and semantic layout to model planning.
  Assign immutable buffer and view materialization to each backend.

  Preserve relayout and conversion only at initialization.
  Do not leak Metal buffers into executor-core.
  Do not add per-request weight preparation.
- Revisit native-FP8 GQA KV only with evidence against BF16.
  The evidence must cover generated instructions, traffic, occupancy, component parity, and end-to-end decode.
- Add an opt-in Metal timeline profiler.
  Use counter and timestamp samples for stable replay and component boundaries.
  Keep these measurements separate from ordinary throughput runs.
