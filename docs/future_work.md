# Future Work

This file contains only unresolved implementation work and bounded
investigations. Current contracts belong in the current docs. Promote resolved
repository-wide rules into `engineering_conventions.md` and component-specific
findings into the owning component document.

## Runtime Lifecycle

- Simplify FIFO dispatch to submit whenever runnable work and a free compute slot are available. Keep request/token
  budgets as hard batch-capacity limits, but remove the time-bound and accumulated-size-bound flush policy
  (`max_wait_duration`, `flush_instant`, and the dual timeout/full decision); deferred batch aggregation is an
  optimization that adds scheduler state and edge cases without a demonstrated product need.
- Make trie/request cache-block extent runtime-dynamic before offering arbitrary Qwen cache-block CLI values. It is
  currently the const generic shared by runtime, request, trie block/cache, and RPC service types; enumerating many
  power-of-two monomorphizations would trade a simple option for code-size and maintenance cost.
- Complete asynchronous request swap-in/swap-out as one lifecycle: reservation
  waits, KV/state onload and offload, request-status ownership, bounded task
  queues, completion delivery, and event-loop reinsertion. Do not enable a
  partial reservation-only path.
- Redesign host-pinned segment ownership before enabling offload. An allocation
  must have unique free/mutation ownership, while any shared views are
  read-only and their cross-thread contract is explicit.
- Move `ReplayableModelBatchExecutor` and executor timing/output traits out of
  `inference-runtime-core` when the service/executor boundary is revised.

## Pipeline Parallelism

- Add a bounded scheduler/runtime-core final-response reorder buffer. It holds
  completed future sequences and releases only the next compute-slot sequence
  to FIFO commit; cover `n + 1` arriving before `n` without weakening the FIFO
  contract or requeueing pending requests.
- Define a model-agnostic `PipelineStageResult`/transport envelope carrying
  compute sequence, stage, request/slot identity, and model-specific stage
  payload. The Qwen payload must preserve ragged hidden-state row association.
- Add Qwen pipeline-stage configuration for stage index/count and main-layer
  range; derive first/last-stage behavior from it. Let non-first stages ingest
  transported hidden states and non-last stages materialize outgoing hidden
  states. Only the last stage performs final norm/unembed, sampling/rejection,
  and MTP proposal.
- Fan final sampling/rejection decisions back to every stage as a commit
  notification. Each stage must commit its own pending GDN/cache transaction
  before the final response returns to runtime core.
- Permit multiple batches to overlap only after every stage uses bounded,
  sequence-ordered pending transactions together with transport, commit
  notification, and in-flight cache-publish ownership that preserve the same
  per-request causal order.

## Prefill

Decode correctness and latency are the current optimization priority. Revisit high-throughput long prefill as a separate
future effort rather than complicating the decode path prematurely. Any new path must preserve ragged-batch correctness,
cache/state lifecycle, and mixed prefill/decode behavior; choose its implementation from end-to-end TTFT and prompt-
throughput evidence instead of treating a removed or hypothetical component path as the design.

## Replay Evolution

- Evaluate capacity-bucketed main and MTP forward replays only after every
  participating kernel has a guarded inactive-lane ABI and parity coverage.
  Main/MTP counts remain exact until then.
- Add a bounded replay catalog only if measured replay-memory growth justifies
  it. Its eviction policy must account for all stage caches and retain in-flight
  resources through GPU completion.
- Design any multi-command-buffer replay chain around the CPU readback boundary
  required by MTP rejection; do not bypass allocator-reset ownership or proposal
  dependency semantics.

## Performance Investigations

- Isolate Qwen3.6 Metal first-request latency from steady-state decode. On
  `2b06deb0183e75c24699dedb7784b116b4987d3b` (`dirty=0`, macOS 27.0, M3 Max 40-core GPU), the observed TTFT was:

  | Model | MTP | First request | Next two requests |
  | --- | --- | ---: | ---: |
  | 27B | off | 1629 ms | 330-332 ms |
  | 35B-A3B | off | 3735 ms | 72-78 ms |
  | 27B | on | 960 ms | 337-342 ms |
  | 35B-A3B | on | 2195 ms | 76-78 ms |

  The first 35B-A3B MTP-off 1024-token run also reached 84.3 tok/s versus 93.1-93.4 tok/s afterward. Reproduce baseline
  and current commits with identical cache, scheduler, logging, model, sampling, and cooldown settings; report cold and
  steady-state results separately. Instrument model/device initialization, generated Metal pipeline compilation/cache
  hits, replay recording, and the first distinct replay shapes before assigning the latency to any one layer.

## Model and Backend Investigations

- Design backend-agnostic immutable `Weight` / `Tensor` / `Storage` ownership.
  Checkpoint readers should own file and mapped-storage lifetime, model planning
  should own tensor identity and semantic layout, and each backend should own
  immutable buffer/view materialization. Preserve init-time-only relayout and
  conversion; do not leak Metal buffers into executor-core or add per-request
  weight preparation.
- Revisit native-FP8 GQA KV only with generated-instruction, traffic,
  occupancy, component-parity, and end-to-end decode evidence against BF16.
- Add an opt-in Metal timeline profiler using counter/timestamp samples for
  stable replay/component boundaries, separate from ordinary throughput runs.
