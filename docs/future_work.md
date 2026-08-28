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

- Revisit cross-request resource sharing after the complete multimodal path is stable.
  Adapt the existing runtime pin-cache ownership model instead of adding a second ad hoc resource cache.
  The shared object must be immutable while it has readers.
  It must keep one RAII owner for arena allocation release.
  Measure repeated-media workloads before selecting admission and eviction policies.
- Add bounded retries for retryable resource materialization failures.
  Keep programming errors as panics or assertions.
  Abort the request after the retry limit.
- Replace the single-type async-task response downcast before runtime core adds a second async-task response type.
  The current request path assumes that every `AsyncTaskResp` is a `ResourceMaterializationResp`.
  Keep this temporary assumption while resource materialization is the only async task.
  The replacement must provide typed response dispatch without a request `Any` downcast and without an
  `AsyncTaskPool` dependency on `UserRequest`.
- Redesign host-pinned segment ownership before enabling offload.
  An allocation must have unique ownership for mutation and free operations.
  The new design permits only read-only shared views.
  It defines their cross-thread contract explicitly.
  Then implement KV and state onload and offload as a separate lifecycle.
  This lifecycle must remain distinct from the existing reservation-wait task.
  Follow the ownership and lifecycle design in [`model_state_io.md`](model_state_io.md).
- Implement explicit per-request model-state offload and onload tasks only for real model-state movement.
  Keep `AwaitReservation` in the scheduler-owned wait collection.
  Reserve `Swapped` for a request whose model state is not device-resident.
- Measure and optimize full and selected model-state snapshot I/O.
  Keep weight handling outside this path.
  `unload_weights()` must drop Metal weight residency without writing weights to the snapshot.
  `load_weights()` must reload weights from the original checkpoint.
  The current v3 path uses one snapshot directory and one semantic file for each resource.
  It uses the Metal backend `BufferIO` primitive without an application staging buffer.
  It opens Metal buffer files with `BufferIOFileCacheMode::Uncached`.
  It uses native-endian `wincode` for the manifest and streams direct `GDNRequestSlots` metadata.
  Selected-state I/O converts allocation bitmaps directly to canonical ranges.
  Measure allocation-bitmap scanning at production page capacity.
  Keep this work on the runtime event-loop thread unless measurements justify a worker.
  A worker must preserve the FIFO Stop boundary and must not let a later batch overtake Stop.
  Measure batched Metal read submission and vectored positional writes for workloads with many noncontiguous ranges.
  Preserve temporary-directory sync, atomic rename, parent-directory sync, and synchronous model-executor lifecycle
  APIs.
  The current writer and reader validate one topology-specific file set.
  The reader validates the local snapshot schema, manifest lengths, and directory contents before allocation.
  Each component validates its resource length before transfer.
  Evaluate optional per-file integrity hashing without a second SSD pass.
  `buffer_to_file` can update a CRC or SHA digest over each successfully written byte range from the shared Metal
  buffer pointer. The metadata path can use a hashing writer. `file_to_buffer` can verify the digest with one CPU scan
  of the loaded shared Metal buffer after Metal I/O completes. Measure the added CPU memory scan before selecting a
  digest algorithm or enabling this feature.
  Record snapshot bytes, write and read duration, effective bandwidth, and peak host-memory overhead before and after
  the change.
  Follow [`model_state_io.md`](model_state_io.md).
## Pipeline Parallelism

- Fill all free compute slots while runnable work exists. The event loop currently calls `do_flush()` at most once for
  each received event. It must call `do_flush()` until `Scheduler::can_flush()` is false. `can_flush()` must stop the
  loop when no compute slot is free. This behavior fills the pipeline when `max_compute_slots` equals the pipeline
  stage count. Implement this loop with the PP>1 request lifecycle.
- Define per-request in-flight readiness for overlapping Prefill and Decode batches. A response for an earlier batch
  must not make a request runnable when a later request-local Decode is still in flight. Cover Prefill-to-Prefill and
  Prefill-to-Decode sequences with `max_compute_slots > 1`.
- Add a `FIFOBatcher` regression test for an unused sticky token budget. Cover an ID-map-only request whose later
  Decode remains in flight. Verify that the unused map entry does not block the next runnable FIFO request.
- Audit the Trie Decode fallback `token_index` after an overlapping MTP Prefill. The current no-ready-token path uses
  `num_cached_tokens()`. It must account for the preceding scheduled range before runtime commit.
- Enforce request-local FIFO commit and LIFO cancellation before either operation changes scheduler state.
- Preserve strict compute-sequence FIFO order across compute slots, batch requests, and batch responses. Validate the
  response sequence directly. Treat an out-of-order response as a contract violation. Do not add a reorder buffer.
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

## Confidence-aware Scheduling

Complete the PP>1 request lifecycle before the following work:

- Add shadow telemetry for proposal position, confidence bin, selected prefix length, validation result, request, and
  model. Use this data to verify that confidence values are calibrated and comparable across requests.
- Add a runtime-owned `ConfidenceCalibration` component. Keep the identity transform until shadow telemetry supports a
  different calibration. Do not add an absolute confidence threshold before this gate passes.
- Add a measured cost estimator to the token-budget allocator. Include request-side fixed cost, Main rows, padding,
  replay bucket changes, validated-token transitions, and new-request admission. Use measured latency and throughput
  data. Do not guess a threshold.
- Add page-feasibility feedback to the planning pass. Keep page capacity separate from token-budget capacity.
- Add end-to-end verification for mixed Prefill and Decode requests, request-local proposal-prefix trimming, the
  proposal token/probability/confidence length invariant, visible-token limits, and PP pipeline fill and drain.
- Keep each executor proposal run fixed at its configured `num_spec_tokens` until these follow-up items are complete.
  Use [`token_budget_allocator.md`](token_budget_allocator.md) for the current policy and open decisions.

## Prefill

Decode correctness and latency are the current optimization priority. Revisit high-throughput long prefill as a separate
future effort. Do not complicate the decode path prematurely.

Any new path must preserve these behaviors:

- Ragged-batch correctness
- Cache and state lifecycle
- Mixed prefill and decode behavior

Use end-to-end TTFT and prompt-throughput evidence to select the implementation. Do not treat a removed or hypothetical
component path as the design.

## Multimodal Input

- Measure resource-arena memory before and after resource-level early unload.
  The current runtime converts one concrete resource to symbolic form after all placement ranges for that resource
  enter committed KV cache.
  Measure peak arena bytes and later rematerialization frequency for requests with repeated audio, image, or video
  placements.
  Keep resource lifetime independent from cache-block and resource-segment lifetime.
- Add long-audio chunking and streaming only after the collected Qwen3-ASR path has parity fixtures.
  Define timestamp, overlap, context carry, and output-merge contracts before implementation.
- Add image support with a model-owned image processor, Vision Tower, spatial merger, and true T/H/W M-RoPE.
  Reuse the runtime resource contracts and `ResourceEmbed` replacement semantics.
  Do not put vision position logic in runtime core.
- Add video support after image support.
  Define demux, frame sampling, timestamp placement, and multi-segment mapping from reference fixtures.
  Reuse the compatible Vision Tower while keeping image and video processors separate.
- Measure first-Prefill resource materialization latency and batch interference.
  Add separate resource-materialization scheduling only if the measurements justify it.
  Do not add modality-specific scheduler states.
- Compare `tower_block_attention` with the decoder bidirectional block GQA implementation after Audio Tower and
  Vision Tower correctness and performance baselines are stable.
  Preserve the tower contract for contiguous Q/K/V without KV-cache or page metadata.
  Preserve the decoder contract for GQA heads, paged state, request metadata, and SplitKV reduction.
  Share a lower-level SDPA operation only if the operation has the same contract and a measured benefit.

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

- Compare a complete base-2 GQA softmax family with the current natural-log family.
  The experiment must migrate SplitKV SingleQ, SplitKV TiledQ, bidirectional block SDPA, and their reducers as one compatible
  partial-state ABI. It must not mix base-2 and natural-log partial statistics or add a per-partial log-base flag.
  Compare focused Metal component performance and representative one-layer production GQA performance.
  Change the default only if the complete base-2 family preserves parity and gives a measured gain.
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

- Extend the independent GDN recurrent/conv version maps for C4 only after the owner can late-bind recurrent slots.
  The selected recurrent version is not known at `begin_txn(...)`. Commit 2 must bind one replay destination slot to
  the accepted state version and to each known publish target that replay satisfies. It must not register every
  candidate version in `recurrent_materialized_state_versions`. That behavior would recreate C0 recurrent capacity.
- Audit Qwen3 and Qwen3.5 tensor-level quantization overrides outside MoE routing.
  Their current Main and MTP GQA, GDN, and MLP builders use model-level affine defaults.
  Confirm the supported checkpoint contracts before changing the loaders.
  If a checkpoint permits per-layer layouts, resolve each semantic layer from its exact binding subtree.
  Keep page tables, metadata, and compatible scratch shared.
  Do not share a weight-dependent backend across incompatible affine layouts.
- Allow an MTP request to start with fewer than `num_spec_tokens` input tokens.
  The runtime currently returns `InvalidArgument` because Trie initialization requires one input token for each MTP
  cache lane. Add a Main-only warm-up phase that creates enough verified history before it enables MTP. The runtime
  core must own the mode transition and cache-lane initialization. Do not use placeholder tokens or partially
  initialized cache lanes.
- Investigate strict one-row and multi-row Qwen3 Main numerical parity.
  The 2026-07-29 greedy acceptance audit found one deterministic output divergence in four prompts.
  At the first divergence, sparse rejection accepted zero draft tokens.
  The eight-row Main verification sampled token `117`.
  The one-row Main path sampled token `223`.
  Compare the Main GQA, dense MLP, and unembed outputs before changing production kernels.
  Treat exact batch-shape reproducibility as a separate requirement from DSpark rejection correctness.
- Add a separate gated DSpark GQA implementation when a supported checkpoint requires it.
  Do not add a runtime gate flag to `BiDiBlockGQA`.
  Keep the history map, block map, reducer, page layout, and scratch contracts gate-neutral.
- Add Qwen3 DSpark checkpoint variants only from real checkpoint contracts.
  This work can include a final-layer entry in the official `target_layer_ids` field, additional Markov heads, or a
  different proposal layout.
  Do not add speculative compatibility fields without a supported checkpoint.
- Permit overlapping Qwen3 DSpark batches only after all proposal scratch, outputs, replay arguments, and probability
  stores have bounded in-flight ownership.
- Evaluate a DFlash2 ring cache as an owner-local alternative to the current persistent paged history cache.
  A likely candidate is a short-lived or ephemeral request that does not use prefix sharing or request forks.
  Compare this case with long-lived requests that exceed the attention window, prefix reuse, request forks, page
  eviction, and fragmentation.
  Measure retained bytes, page-table work, copy traffic, and end-to-end latency.
  Keep persistent paged history KV as the default until a ring cache improves a real workload.
  Keep the policy in the DFlash2 owner. Do not put it in the shared GQA backend.
- Evaluate grouped DSpark and DFlash2 Prefill projections only with a real model-level bottleneck measurement.
  Group Wk projections by compatible affine layout and group Wv projections separately.
  Do not assume that Wk and Wv have the same dtype or quantization layout.
  Preserve direct paged-K/V output and avoid an aggregation copy, transpose, or scatter.
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
