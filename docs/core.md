# Runtime Core Guidance

Runtime core owns scheduling, request lifecycle, and page/cache ownership. It should stay model-agnostic and separate
from the model executor.

## Scope

Core may know requests, batches, token/block indices, page IDs, lifecycle state, and allocation ownership.

Core should not parse model-specific tensor layout, checkpoint/Metal buffer shapes, or GQA/gated-delta-net/MLP
internals.

`RuntimeConfig::num_tokens_per_cache_block` is model-neutral cache metadata:
the token extent of one trie block and its attached opaque KV/state-page
vectors. It is distinct from the number of tokens that a single physical KV
page holds. The executor interprets each vector according to its model layout;
the core only allocates, shares, and reports the vector as one logical block.

## Core-owned responsibilities

```text
request lifecycle
request-slot allocation/reset/drop
batch scheduling
prefill/decode/spec scheduling decisions
token_index and block_index metadata
accepted/rejected token metadata
KV cache page allocation/free
GDN state page allocation/free
page ownership and reuse
cache/state lifecycle notifications to executor
executor input metadata construction
```

Prefer explicit contracts:

```text
request slot -> active request metadata
block_index -> kv_page_ids
block_index -> linear_state_page_ids
batch row -> token_index / block_index / page IDs
```

## Scheduler and trie cache object model

The scheduler owns request placement, budgets and compute-slot ordering. A
request owns its decoder-block view; preparing that view may reserve or reuse
trie-cache blocks and physical pages before producing executor metadata:

A full block is eligible for trie lookup only when at least one request token
remains after it. This keeps a real forward suffix for logits and recurrent
state instead of introducing a model-specific full-prefix replay operation.
Earlier blocks still reuse trie entries, and a completed terminal mutable block
is committed normally for use as a non-terminal prefix by later requests.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ QueuedRequest                                                               │
│ request ID + tokens + sampling/lifecycle state; no request slot            │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ event-loop admission when a slot is free
                                    │ -> allocate RequestSlot
                                    │ -> convert to InternalRequest
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ ScheduleQueue                                                              │
│ new_queue | run_queue | ID -> request map | bounded swap-out task sender   │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ InstrumentedScheduler<SimpleScheduler>                                     │
│ scheduler API latency/counts + hard request/token/per-request limits       │
│ free/used compute slots + ordered compute-slot sequence                    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ runnable work + free slot
                                    │ -> allocate ordered compute slot
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ FIFOBatcher::prepare                                                       │
│ apply scheduler-provided limits to FIFO preparation                        │
│ pop requests, estimate token cost, call InternalRequest::prepare           │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ InternalRequest::prepare                                                   │
│ initialize/reserve decoder blocks -> schedule query tokens                 │
│ -> build DecoderSyncBlocks -> DeviceRequest                                │
└──────────┬───────────────┬────────────────┬──────────────────┬─────────────┘
           │ Continue      │ Pending        │ Await            │ resource/term
           v               v                v                  v
┌──────────────────┐  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ BatchDeviceReq   │  │ ID map only   │  │ swap-out task  │  │ preempt/drop  │
│ ragged requests  │  │ await response│  │ async wait     │  │ or terminal   │
└────────┬─────────┘  └───────────────┘  └────────────────┘  └───────────────┘
         │ executor submission
         v
┌──────────────────┐     ┌───────────────────────────────────────────────────┐
│ BatchDeviceResp  │────>│ SimpleScheduler/FIFOBatcher::commit               │
│ same slot order  │     │ match response by ID and oldest compute-slot seq  │
└──────────────────┘     └────────────────────────┬──────────────────────────┘
                                                  v
                               ┌──────────────────────────────────────┐
                               │ InternalRequest::commit              │
                               │ commit decoder tokens/cache state    │
                               │ emit visible output + stop handling  │
                               └──────────────┬───────────────┬───────┘
                                              │ Continue      │ Terminal
                                              v               v
                                       ┌─────────────┐  ┌──────────────┐
                                       │ run_queue   │  │ drop request │
                                       │ push front  │  │ release slot │
                                       └─────────────┘  │ and pages    │
                                                        └──────────────┘
```

`PrepareResult::Pending` deliberately leaves the request in the ID map without
putting it on `run_queue`: previously submitted work owns its next transition,
and the executor response returns the request through `commit`.

Runtime admission and scheduler batching have separate limits.
`RuntimeConfig::max_queued_requests` bounds the user-request channel, while
`RuntimeConfig::max_running_requests` bounds the request-slot domain and both
reservation-task channels. `SchedulerConfig::max_requests`,
`max_tokens`, and `max_tokens_per_request` remain per-batch limits, and
`SchedulerConfig::max_compute_slots` bounds outstanding batches and therefore
the batch-request and batch-response channels. Runtime initialization requires
the per-batch request limit to be no larger than the running-request limit.

A queued request owns no request slot. The synchronous event loop registers the
user-request receiver only while the request-slot allocator reports free
capacity, allocates one slot after receiving the request, and then constructs
the `InternalRequest` consumed by the scheduler. The same slot follows that
request through run queues, device-pending work, reservation-task queues and
async runners until request drop. Request-slot allocator usage is therefore the
single admission count; no separate running-request counter exists.

`PrepareResult::Await` transfers the request into a bounded asynchronous task
queue without producing a device request for that request. Once preparation is
invoked, the scheduler still allocates a compute slot and submits the resulting
batch even when it contains zero device requests. Empty batches preserve
executor participation for backends such as distributed expert parallelism;
the empty response releases the compute slot in normal sequence order.

A task runner transitions `Running -> Swapped` immediately before awaiting the
reservation, then `Swapped -> Running` after completion. It publishes the
request through a bounded synchronous completion channel; the synchronous event
loop owns that receiver and appends the request to the run queue. A request
that becomes terminal before or during the wait stays terminal and is logged
and released by the event loop instead of being re-enqueued. Shutdown cancels
the wait by dropping task/request ownership. The task pool runs on the service's
Tokio runtime; the scheduler event loop remains a synchronous thread.

This is the complete lifecycle for asynchronous cache-reservation waits. It is
not KV/state offload: backing storage remains where it was, and future
onload/offload requires an additional explicit ownership design.

The event loop submits immediately after receiving a user request, receiving a
completed reservation wait, or committing a device response when the scheduler
has runnable token work and a free compute slot. Request and token budgets
remain hard per-batch limits in
`FIFOBatcher::prepare`; they are not mutable aggregation thresholds, and the
current path has no scheduler flush timer.

The trie cache is the storage and reuse subsystem reached through each
request's `TrieDecoderBlocks`:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ TrieDecoderBlocks for one request                                          │
│ queued/ready/spec tokens + epoch + per-lane block sequences                │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
          ┌──────────────────────────────────────────────────────┐
          │ Request block lifecycle                              │
          │ Mutable -> SemiImmutable -> Immutable                │
          │ each logical block contains one block per cache lane │
          └─────────────────────────┬────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ MultiLaneBlockCache                                                        │
│ reserve/commit/free all main + MTP cache lanes as one request operation    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Per-lane block cache                                                       │
│ block metadata + token/resource annotations + physical page ownership      │
└───────────────────┬──────────────────────────────┬─────────────────────────┘
                    │ immutable identity           │ allocate/free
                    v                              v
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│ Trie                                 │  │ Physical page allocators         │
│ roots + token/resource edges         │  │ KV pages + GDN state pages       │
│ partitioned TrieNodeStore            │  │ globally unique page IDs         │
│ external/child pin counts            │  └─────────────────┬────────────────┘
└───────────────────┬──────────────────┘                    │
                    │ unpinned leaf candidates              │
                    v                                       │
┌──────────────────────────────────────┐                    │
│ S3FIFO                               │                    │
│ S queue + M queue + ghost history    │                    │
│ select/reject/commit eviction        │                    │
└───────────────────┬──────────────────┘                    │
                    │ successful eviction                   │
                    └──────────────────────┬────────────────┘
                                           v
                         ┌────────────────────────────────┐
                         │ Page IDs may be reused         │
                         │ after ownership is released    │
                         └────────────────────────────────┘

TrieDecoderBlocks::prepare_blocks()
  -> DecoderSyncBlocks
     block_index + lane/layer page IDs
  -> DeviceRequest
  -> model executor page/state tables
```

Mutable and semi-immutable blocks are request-local lifecycle objects;
immutable identity is represented by trie nodes. Pin counts protect reusable
nodes from eviction, while S3FIFO tracks eligible unpinned leaves. The trie
stores logical identity and ownership links; physical allocators own page IDs
and backing storage.

## Page and cache model

Separate logical executor metadata from physical storage.

Executor-facing KV metadata is logical and lane-first:

```text
lane -> kv_block -> layer -> page_id
```

Physical storage is allocator-facing and flat:

```text
page_id -> page buffer
```

`page_id` is globally allocated. It is not scoped by lane, block, or layer.

Runtime core does not define a GPU KV page tensor layout. It allocates opaque
physical pages and reports their IDs in logical cache-block metadata. The
executor interprets each page according to the active model; the current Qwen
GQA layout is documented in [`executor_gqa.md`](executor_gqa.md).

CPU/onload/offload should consume the same logical metadata and physical page IDs. Do not invent a separate cache object hierarchy unless scheduling or placement policy needs it.

## Managed objects

Likely core-managed objects:

```text
request
request slot
batch
batch row
scheduler queue
KV page allocator
GDN state page allocator
page ownership table
request lifecycle table
accepted token/state metadata
```

Each object should expose a small lifecycle surface:

```text
get / get_ref / get_mut
set
allocate / free
push / pop
reset
```

Avoid near-synonyms such as `clear`, `zero`, `reset`, `load`, `insert`, and `publish` on the same object unless they operate on clearly different objects.

## Executor notification contract

Core should notify the executor when core-owned lifecycle state changes:

```text
request slot initialized
request slot completed/dropped/reset
prefix/cache hit selected
batch metadata constructed
accepted/rejected tokens finalized
KV/state pages allocated or freed
```

The executor should not infer global lifecycle state from a forward call. If a request slot must be globally reset, that should come from core lifecycle.

If only executor-local state must be replaced for one forward, use a specific set/update operation instead of resetting the whole slot.

## Stop and EOS handling

Runtime core owns token-id stop and EOS completion at the request commit boundary. The executor reports sampled tokens; core checks configured stop token sequences, commits decoder/cache state, truncates the user-visible output after the first matched stop sequence, and then marks the request completed. EOS is represented as a one-token stop sequence.

Model executors may provide model-specific default stop sequences, such as Qwen EOS token IDs. The service merges those defaults with request-provided stop sequences and de-duplicates the token sequences before submitting the request to core.

String stop conditions belong in tokenizer/detokenizer or service output handling, not in the model executor: token-id EOS/stop is handled by scheduler/request lifecycle code, while text stop strings are applied in output/detokenization paths.

Per-request token/probability delivery must not silently drop committed output.
The current runtime uses an unbounded internal channel between request commit
and the asynchronous RPC forwarding task. Dropping the external request closes
the channel and cancels the request lifecycle. Slow-consumer memory accounting
and a bounded request-local cancellation policy remain future work.

`max_sampled_tokens` is a caller-visible output limit. A speculative step may
commit more sampled tokens to decoder/cache state than the caller's remaining
budget. Core truncates only the `TokenProbs` sent to the caller, leaves the
sampled-token commit unchanged, marks the request completed, and lets request
drop release its decoder/cache ownership.

Request-slot drop adds its slot to a deduplicated reset set before returning the
slot to the allocator. A capacity-one channel only wakes the executor; it
never carries slot IDs, and a full wake channel means a wake is already pending.
The executor event loop selects over wake notifications and device batches, so
slots dropped before any device batch are reset while the executor is otherwise
idle. When a batch arrives, the executor also drains the reset set before
preparing that batch; a newly reused slot therefore cannot execute ahead of its
prior reset notification.

## Scheduler contracts and invariants

If the scheduler guarantees a condition, rely on it and assert at the boundary where violations become visible.

Examples:

```text
page IDs are allocated before executor sees them
batch rows have valid token_index/block_index metadata
accepted/rejected token metadata is internally consistent
request slot lifecycle notifications are ordered
```

The scheduler forms FIFO ragged batches from runtime readiness and token-budget
constraints. It does not group requests by model-executor sampling parameters
such as top-k, top-p, temperature, seed, or speculative stage. A model executor
must accept the resulting mixed batch and compact, partition, or select replay
geometry internally without changing scheduler policy.

Request status tracks lifecycle ownership, not scheduler placement:

```text
Initialized -> queued request exists but has not entered runtime admission
Running     -> request owns a request slot in the normal runtime path
Swapped     -> request ownership is held by an async reservation task and may later be re-enqueued
terminal    -> Cancelled, TimedOut, Aborted, or Completed
```

Scheduler-internal locations such as new queue, run queue, pending device work,
and response commit do not create additional request statuses.

Runtime-critical background threads hold a `ShutdownGuard` for their full
lifetime. Normal return and panic unwind both drop the guard and notify the
other runtime loops to stop; a failed worker must not leave a partially live
service waiting forever.

`PrepareResult::Pending` means a request has no additional runnable query while
previous scheduled work is still in flight. The batcher keeps it in the
request-ID map but does not put it back on the run queue; the model executor's
response returns it through `commit`. Do not requeue `Pending` requests. Device
batch responses currently commit in submission order because decoder scheduled
token ranges are FIFO-owned. Pipeline stages may overlap, but their final
responses must preserve that order until core owns an explicit epoch/reorder
buffer. An out-of-order response is an internal contract violation and should
fail fast until a real transport reorder contract is introduced.

Reservation waits use bounded swap-out and swap-in channels. Their capacities
equal `RuntimeConfig::max_running_requests`, which is also the hard
request-slot limit and the executor request-state domain. Every admitted
request owns exactly one slot and can occupy either channel once, so a full
channel is an ownership/accounting bug and fails fast.

Queue priority is run queue before new queue: a continued request goes to the
front, a completed reservation wait goes to the back, and a new request enters
the new queue. Crossbeam selection can still receive a new request just before
an already-ready swap-in completion; draining both ready input channels before
flushing remains future work. Actual KV/state onload and offload also remains
future work and must not reuse the reservation-wait name as if data movement
already existed.

Do not add executor-side recovery for impossible scheduler states unless there is a real runtime recovery path.

## Core should avoid

```text
model-specific layer parsing
checkpoint/Metal buffer shape interpretation
GQA KV kernel semantics
gated-delta-net recurrent-state layout
MLP routing/fusion policy
per-component benchmark/profiling details
```
