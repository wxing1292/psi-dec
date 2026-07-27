# Future Work

This file contains only unresolved implementation work and bounded
investigations. Current contracts belong in the current docs. Promote resolved
repository-wide rules into `engineering_conventions.md` and component-specific
findings into the owning component document.

## HTTP APIs

- Implement the explicit HTTP request/response preprocessing placeholders.
  Chat Completions must lower the wire request into model-ready token IDs
  before calling `Inference::decode`, then transform the token response into
  either the collected or streaming wire response after decode. Chat-template
  rendering, tokenization, detokenization, and tool-call dialect handling
  remain outside the token-level inference API and the HTTP transport adapter.
  Evaluate an HF-compatible streaming response parser for the postprocessing
  boundary.

## Tool APIs

- Implement the Pi custom-provider session wire and persistent conversation
  store. Define the session ID, event envelope, append ordering, concurrent
  append behavior, and storage owner; map appended events into the existing
  `ToolEvent`/`ToolState` domain. The provider should register common tools once
  and later append only additions, removals, calls, responses, and
  cancellations. Do not retransmit complete tool definitions on every turn,
  model conversation history as an evictable cache, or add a registry
  resynchronization protocol.

## Runtime Lifecycle

- Make trie/request cache-block extent runtime-dynamic before offering arbitrary Qwen cache-block CLI values. It is
  currently the const generic shared by runtime, request, trie block/cache, and RPC service types; enumerating many
  power-of-two monomorphizations would trade a simple option for code-size and maintenance cost.
- When user and reservation-completion channels are both ready, drain both
  before flushing so queue priority does not depend on crossbeam select order.
- Replace the unbounded per-request token-output channel with bounded
  slow-consumer handling based on explicit per-request or service-level byte
  accounting. Saturation must terminate only the affected request and return a
  clear RPC error; it must neither silently drop committed output nor panic the
  service.
- Redesign host-pinned segment ownership before enabling offload. An allocation
  must have unique free/mutation ownership, while any shared views are
  read-only and their cross-thread contract is explicit. Then implement
  KV/state onload and offload as a lifecycle distinct from the existing
  reservation-wait task.
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

- Evaluate software-pipelined K/V loads for GQA `TiledQTokens` as a bounded experiment. The current
  `Tkv=16`, `D=256` K/V threadgroup tiles use 16.5 KiB; duplicating them with the existing eight-value row padding
  requires 33 KiB and exceeds a 32 KiB threadgroup-memory limit. First measure the correctness, bank-conflict behavior,
  and performance of removing that padding, then compare an exact-32-KiB ping-pong layout against the current
  single-workspace kernel with identical real-weight cases. Do not make it a default without component parity and
  end-to-end evidence.
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
