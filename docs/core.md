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
│ Incoming InternalRequest                                                   │
│ request ID + request slot + tokens + sampling/lifecycle state              │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ enqueue
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ ScheduleQueue                                                              │
│ new_queue | run_queue | ID -> request map | placeholder swap task queues   │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ FIFOScheduler                                                              │
│ request/token budgets + max tokens/request + free/used compute slots       │
│ decision: Noop | Timeout | Flush                                           │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ Flush -> allocate ordered compute slot
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ FIFOBatcher::prepare                                                       │
│ pop FIFO requests, estimate token cost, call InternalRequest::prepare      │
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
│ ragged requests  │  │ await response│  │ placeholder    │  │ or terminal   │
└────────┬─────────┘  └───────────────┘  └────────────────┘  └───────────────┘
         │ executor submission
         v
┌──────────────────┐     ┌───────────────────────────────────────────────────┐
│ BatchDeviceResp  │────>│ FIFOScheduler/FIFOBatcher::commit                 │
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
and the executor response returns the request through `commit`. Swap queues are
shown as placeholders because the async onload/offload lifecycle is not yet a
complete production path.

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

Per-request token/probability delivery must not silently drop committed output
under client backpressure. The runtime uses a bounded per-request channel;
filling it is a contract violation and fails fast because the client must keep
consuming committed output. Dropping the external request closes the channel
and cancels the request lifecycle.

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
Initialized -> request objects exist but have not been submitted
Running     -> request is owned by the normal runtime path
Swapped     -> request ownership is held by an async swap/offload path and may later be re-enqueued
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

Swap-in/swap-out task types and bounded queues are placeholders for a complete
asynchronous request-task design. Cache reservation waits, KV/state onload and
offload, lifecycle transitions, and event-loop reinsertion must be implemented
together; the current event loop does not claim that path is complete.

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
