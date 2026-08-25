# Runtime Core Guidance

Runtime core owns scheduling, request lifecycle, and page and cache ownership.

Recommendation: Keep runtime core model-agnostic and separate from the model executor.

## Selection

Core may know requests, batches, token and block indices, page IDs, lifecycle state, and allocation ownership.

Recommendation: Keep these details out of runtime core:

- Model-specific tensor layouts
- Checkpoint and Metal buffer shapes
- GQA, Gated DeltaNet, and MLP internals

`RuntimeConfig::num_tokens_per_cache_block` is model-neutral cache metadata. It gives the token extent of one trie
block and its attached opaque KV and state-page vectors.

The service compiles this extent as const `N` for each production artifact. Production selects the artifact and its
configuration at build and deployment time. Runtime initialization requires
`RuntimeConfig::num_tokens_per_cache_block == N`. One live runtime does not change this extent.

This extent differs from the token capacity of one physical KV page. The executor interprets each vector according to
its model layout. Core only allocates, shares, and reports the vector as one logical block.

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

Recommendation: Use explicit contracts:

```text
request slot -> active request metadata
block_index -> kv_page_ids
block_index -> linear_state_page_ids
batch row -> token_index / block_index / page IDs
```

## Scheduler and trie cache object model

The scheduler owns request placement, budgets, and compute-slot ordering. A request owns its decoder-block view.
Preparation can reserve or reuse trie-cache blocks and physical pages before preparation produces executor metadata.

A full block is eligible for trie lookup only when at least one request token remains after the block. This rule keeps
a real forward suffix for logits and recurrent state. This design does not need a model-specific full-prefix replay
operation.

Earlier blocks still reuse trie entries. Core commits a completed terminal mutable block normally. Later requests can
then use that block as a non-terminal prefix.

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
│ new_queue | unique run_queue | ID -> request map                           │
│ unordered AwaitReservation futures | swap-out task sender                  │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ InstrumentedScheduler<SimpleScheduler>                                     │
│ periodical/lifetime API and Spec stats + hard scheduler limits             │
│ free/used compute slots + ordered compute-slot sequence                    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ runnable work + free slot
                                    │ -> allocate ordered compute slot
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ FIFOBatcher::prepare                                                       │
│ plan slot-local sticky requests without changing request state             │
│ allocate minimum validated, maximum validated, then speculative budgets    │
│ use remaining request and token budgets for FIFO preparation               │
│ call InternalRequest::prepare with one absolute token budget per request   │
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
│ BatchDeviceReq   │  │ ID map only   │  │ waiting_reqs   │  │ preempt/drop  │
│ Prefill: requeue │  │ await response│  │ wait future    │  │ or terminal   │
│ Decode: map only │  └───────────────┘  └────────────────┘  └───────────────┘
└────────┬─────────┘
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

`PrepareResult::Pending` deliberately leaves the request in the ID map. It does not put the request on `run_queue`.
Previously submitted work owns the next transition. The executor response returns the request through `commit`.

`PrepareResult::Continue` carries a `PreparePhase`. A Prefill query returns to `run_queue` after preparation so that
another Prefill query can enter the pipeline. A Decode query stays only in the ID map until a response commits or a
cancellation restores it.

`run_queue` stores each request ID at most once. `push_front` moves an existing ID to the front. `push_back` keeps an
existing ID at its current position. Commit and cancellation can therefore restore a request without creating a
second runnable occurrence.

Each compute slot keeps the request ID order from its most recent device batch. The next use of that slot treats these
IDs as its sticky working set. Different compute slots can contain the same request ID during pipelined Prefill. One
device batch must contain each request ID at most once.

`SimpleScheduler` resolves only runnable sticky IDs. It skips IDs that are pending, terminal, swapped, or absent. It
first creates immutable `ReqTokenInventory` values. It then allocates minimum validated, maximum validated, and
speculative token budgets. The speculative phase uses a request-local causal candidate heap. The allocator returns a
request-ID-to-token-budget map. `FIFOBatcher` consumes this map before it uses the remaining hard request and token
budgets for the normal FIFO queue.

Runtime admission and scheduler batching have separate limits. `RuntimeConfig::max_queued_requests` bounds the
user-request channel. `RuntimeConfig::max_running_requests` bounds the request-slot domain and the async-task request
and response channels.

`SchedulerConfig::max_requests`, `max_tokens`, and `max_tokens_per_request` remain per-batch limits.
`max_tokens_per_request` must not exceed `max_tokens`.
`SchedulerConfig::max_compute_slots` bounds outstanding batches. It therefore also bounds the batch-request and
batch-response channels.

Runtime initialization requires the per-batch request limit to be no larger than the running-request limit.
Request admission requires at least `max(1, L - 1)` initial input tokens. This rule gives each configured cache lane
its required initial token. The runtime returns `InvalidArgument` before it constructs Trie blocks when the request is
too short.

`RuntimeConfig::context_window` is the effective model-input token limit. Request initialization validates
`history.len() + prompt.len() + sampled.len() < context_window` before it constructs Trie blocks. It also requires the
initial sampled-token count to be less than `max_sampled_tokens`.

A queued request owns no request slot. The synchronous event loop registers the user-request receiver only when the
request-slot allocator reports free capacity.

After the event loop receives a request, it allocates one slot. It then constructs the `InternalRequest` that the
scheduler consumes.

The same slot follows the request until request drop. It passes through run queues, device-pending work, and the
scheduler-owned reservation-wait collection.

Request-slot allocator usage is therefore the single admission count. No separate running-request counter exists.

`PrepareResult::Await` transfers the request into the `ScheduleQueue` reservation-wait collection. The collection is a
`FuturesUnordered<AwaitReservation<...>>`. It does not produce a device request for that request.

`AwaitReservation` is not a general asynchronous task. Its wait future must retain completion until the scheduler
polls it. It must not require an independent timer or I/O wake to make the synchronous event loop poll the collection.

After preparation starts, the scheduler still allocates a compute slot and submits the resulting batch. It does this
even when the batch contains zero device requests.

Empty batches preserve executor participation for backends such as distributed expert parallelism. The empty response
releases the compute slot in normal sequence order.

The request remains `Running` while it waits for the reservation. A reservation wait does not change model-state
residency.

`SimpleScheduler::commit` commits the device response first. After `commit` returns, the event loop calls
`Scheduler::pop_ready_reqs` until the method returns `None`. `SimpleScheduler` delegates each call to
`ScheduleQueue::pop_ready_reqs`. This method synchronously polls the unordered wait collection and returns at most one
ready request.

A request can become terminal before or during the wait. The request then stays terminal. The event loop releases a
terminal reservation-wait request. Otherwise, it calls `Scheduler::resume`, which appends the request to the back of
`run_queue`.

Shutdown cancels reservation waits when it drops the scheduler and its wait collection. The scheduler event loop
remains a synchronous thread.

This is the complete lifecycle for cache-reservation waits. This lifecycle is not KV and state offload.
Backing storage remains unchanged. Future onload and offload require an additional explicit ownership design.

The event loop can submit after two request-work events:

- It receives a user request.
- It commits a device response.

A device-response commit also discovers completed reservation waits before the event loop prepares the next batch.

The event loop submits immediately when the scheduler has runnable token work and a free compute slot.

Request and token budgets remain hard per-batch limits in `FIFOBatcher::prepare`. They are not mutable aggregation
thresholds. The current path has no scheduler flush timer.

The scheduler stats timer fires every 30 seconds. It does not flush scheduler work. `InstrumentedScheduler` prints and
resets non-empty periodical stats. It retains separate lifetime stats and always prints them during shutdown.

Both scopes contain scheduler API counts/latencies and speculative-token counts. The speculative table counts every
proposal position that the Spec forward produced. It counts an accepted position when `validated_tokens` contains that
position. The rate is `accepted / proposed` at the same index.

The trie decoder keeps proposal tokens, probabilities, and confidence values in one request-local proposal state.
These vectors must have the same length. A verification-prefix trim changes all three vectors. Proposal confidence
does not change rejection sampling.

The initial scheduler policy applies an identity transform to proposal confidence. It uses cumulative confidence as a
ranking score. The runtime does not yet have evidence that these values are calibrated or comparable across requests.
The policy does not use an absolute confidence threshold.

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
│ reserve/commit/free all configured cache lanes as one request operation    │
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

The runtime core carries the compile-time cache-lane count as const `L`.
`MultiLaneTrieBlockCache<P, L, ...>`, runtime requests, and service runtime types preserve this const through scheduling.
`RuntimeConfig::cache_lanes` must contain exactly `L` lane configurations.
Each multi-lane allocation, reserve, commit, and free operation requires one entry for every lane.
Qwen3.5 MTP configures one Main lane and one lane for each logical MTP step.
Thus, Qwen3.5 uses `L = num_spec_tokens + 1`.

Mutable and semi-immutable blocks are request-local lifecycle objects. Trie nodes represent immutable identity. Pin
counts protect reusable nodes from eviction. S3FIFO tracks eligible unpinned leaves.

The trie stores logical identity and ownership links. Physical allocators own page IDs and backing storage.

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

Runtime core does not define a GPU KV page tensor layout. It allocates opaque physical pages. It reports their IDs in
logical cache-block metadata.

The executor interprets each page according to the active model. [`executor_gqa.md`](executor_gqa.md) documents the
current Qwen GQA layout.

Recommendation: Use the same logical metadata and physical page IDs for CPU, onload, and offload paths.

Do not create a separate cache object hierarchy unless scheduling or placement policy needs it.

## Managed objects

Core-owned concepts include:

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

Recommendation: Give each object a small lifecycle surface:

```text
get / get_ref / get_mut
set
allocate / free
push / pop
reset
```

Recommendation: Do not use near-synonyms such as `clear`, `zero`, `reset`, `load`, `insert`, and `publish` on the same
object. Use them together only when they operate on clearly different objects.

## Executor notification contract

Runtime core and the model executor use one ordered request-response protocol:

```text
Batch(request) -> Batch(response)
Start(plan)   -> Started
Stop(plan)    -> Stopped
```

The runtime event loop tracks the commanded `Started` or `Stopped(plan)` state.
It appends `Stop` to the same ordered channel after all batches that it has already sent.
The executor completes those batches before it handles `Stop`.
It sends `Start` with the stored Stop plan when a stopped executor has work to flush.
It can append a batch after `Start` without a separate transition state.
The ordered channel guarantees that the executor handles `Start` before that batch.
The runtime consumes `Started` and `Stopped` as acknowledgements and then attempts the next flush.

Each protocol channel reserves one entry in addition to the compute-slot capacity.
This entry lets runtime core append `Stop` after every compute slot has submitted a batch.

The idle condition does not require zero live requests.
A request can remain in runtime core while it waits for resources.
The model state snapshot preserves executor state for that request across stop and start.

`RuntimeConfig::executor_hibernation_mode` fixes the Stop/Start plan policy at runtime construction.
For `Selected`, runtime core scans the allocation bitmaps for request slots and page IDs.
It converts set bits directly to sorted, disjoint, and nonadjacent ID ranges.
The page ranges include active requests, reusable trie cache blocks, and resources held by runtime tasks.
For `All`, runtime core sends `ExecutorHibernationPlan::All`.

The bitmap scan is not linearizable.
An allocation that occurs after its bitmap word is read cannot publish executor-visible state before the queued `Stop`.
A free that occurs after its bitmap word is read can add unused state to the snapshot without changing correctness.
Runtime core must quiesce future independent per-request I/O tasks before it scans the bitmaps.

Recommendation: Send core-owned lifecycle changes from core to the executor:

```text
request slot initialized
request slot completed/dropped/reset
prefix/cache hit selected
batch metadata constructed
accepted/rejected tokens finalized
KV/state pages allocated or freed
```

Recommendation: Do not infer global lifecycle state from a forward call. Initiate a required global request-slot reset
from core lifecycle.

If one forward must replace only executor-local state, use a specific set or update operation. Do not reset the whole
slot.

## Stop and EOS handling

Runtime core owns token-ID stop and EOS completion at the request commit boundary. The executor reports sampled tokens.

Core checks the configured stop token sequences. It commits decoder and cache state. It truncates user-visible output
after the first matched stop sequence. Core then marks the request completed.

A one-token stop sequence represents EOS.

The successful status records one of these completion reasons:

- `CompletionReason::StopSequence` when the stop matcher observed a match
- `LengthLimit` when caller-visible output reaches its limit
- `ContextLimit` when caller-visible output reaches the effective context window

A stop match wins when multiple completion conditions occur on the same commit. `LengthLimit` wins a tie with
`ContextLimit`. Service layers map the recorded reason. They must not infer it from the emitted token count.

Model executors may provide model-specific default stop sequences, such as Qwen
EOS token IDs. The service merges caller-provided token sequences with model
defaults and de-duplicates them before submitting the request to core.

Per-request token and probability delivery must not silently drop committed output. The current runtime uses an
unbounded internal channel between request commit and the transport-neutral `DecodeResponse`. This channel keeps
synchronous request commit non-blocking and preserves every committed output. `max_running_requests` bounds the number
of admitted request channels. `max_sampled_tokens` and `context_window` bound the caller-visible output of each request.
The runtime does not terminate a request only because its transport consumer is slow.

Dropping that response causes three actions:

- It drops the external request.
- It closes the receiver.
- It cancels the request lifecycle.

`max_sampled_tokens` and `context_window` are caller-visible output limits. A speculative step can commit more sampled
tokens to decoder state than the caller's remaining budget.

Core truncates only the `TokenProbs` sent to the caller. It leaves the sampled-token commit unchanged and marks the
request completed. Tokens that entered cache were valid model inputs inside `context_window`. The final sampled token
can remain in a terminal request's queued-token state beyond the caller-visible context boundary. It does not enter
cache or another model input. Request drop then releases its decoder and cache ownership.

If the request continues, commit truncates the next speculative token, probability, and confidence vectors to the
remaining model-input extent. `InternalRequest::prepare` asserts that the resulting model-input extent does not exceed
`context_window`.

Request-slot drop adds its slot to a deduplicated reset set. It does this before it returns the slot to the allocator.

A capacity-one channel only wakes the executor. It never carries slot IDs. A full wake channel means that a wake is
already pending.

The executor event loop selects over wake notifications and device batches. It can therefore reset dropped slots while
the executor is otherwise idle.

When a batch arrives, the executor drains the reset set before it prepares the batch. A reused slot cannot execute
before its prior reset notification.

## Scheduler contracts and invariants

If the scheduler guarantees a condition, rely on it and assert at the boundary where violations become visible.

Examples:

```text
page IDs are allocated before executor sees them
batch rows have valid token_index/block_index metadata
accepted/rejected token metadata is internally consistent
request slot lifecycle notifications are ordered
```

The scheduler forms FIFO ragged batches from runtime readiness and token-budget constraints. It does not group requests
by these model-executor sampling parameters:

- Top-k
- Top-p
- Temperature
- Seed
- Speculative stage

A model executor must accept the resulting mixed batch. It must compact or partition the batch, or select replay
geometry internally. It must not change scheduler policy.

Request status tracks lifecycle ownership, not scheduler placement:

```text
Initialized -> queued request exists but has not entered runtime admission
Running     -> request owns a request slot in the normal runtime path
Swapped     -> reserved for a request whose model state is not device-resident
terminal    -> Cancelled, TimedOut, Aborted, or Completed(CompletionReason)
```

Scheduler-internal locations such as new queue, run queue, pending device work,
and response commit do not create additional request statuses.

Runtime-critical background threads hold a `ShutdownGuard` for their full lifetime. Normal return and panic unwind both
drop the guard. Both paths notify the other runtime loops to stop.

A failed worker must not leave a partially live service waiting indefinitely.

`PrepareResult::Pending` means a request has no additional runnable query while
previous scheduled work is still in flight. The batcher keeps it in the
request-ID map. It does not put the request back on the run queue. The model executor response returns it through
`commit`.

Do not requeue `Pending` requests. Device batch responses currently commit in submission order because decoder
scheduled token ranges are FIFO-owned.

Commit and cancellation put a continuing request at the front of the run queue. The queue deduplicates an ID that is
already runnable.

Pipeline stages can overlap. Their final responses must preserve submission order until core owns an explicit epoch or
reorder buffer.

An out-of-order response is an internal contract violation.

Recommendation: Fail fast on this response until a real transport reorder contract exists.

Reservation waits do not use the swap-out and swap-in channels. `ScheduleQueue` owns them in `waiting_reqs`.
`RuntimeConfig::max_running_requests` bounds this collection through the request-slot limit. Every admitted request
owns exactly one slot while it waits.

The existing swap-task channels remain separate from reservation waits. Their capacities equal
`RuntimeConfig::max_running_requests`.

The run queue has priority over the new queue:

- A continued request goes to the front.
- A completed reservation wait goes to the back.
- A new request enters the new queue.

Actual KV and state onload and offload also remain future work. This work must not use the reservation-wait name as if
data movement already existed.

Do not add executor-side recovery for impossible scheduler states unless there is a real runtime recovery path.

## Core exclusions

Recommendation: Keep these details out of core:

```text
model-specific layer parsing
checkpoint/Metal buffer shape interpretation
GQA KV kernel semantics
gated-delta-net recurrent-state layout
MLP routing/fusion policy
per-component benchmark/profiling details
```
