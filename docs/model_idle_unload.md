# Model Idle Unload Design

This document records the implemented model resource operations, executor protocol, and service event-loop wiring.
Runtime idle detection and model residency tracking are implemented.

[`model_state_io.md`](model_state_io.md) defines the planned selected-state and request-mobility design.

## Objective

The final service must release model weights and backend state after an idle period.
The service process and its RPC listeners must remain active.
The first new request must load the model before model execution resumes.

The implementation must preserve runtime cache identity across a successful stop and start.
It must discard runtime cache metadata after any snapshot or restore failure.

## Current implementation boundary

`inference-executor-core` owns `ReplayableModel`.
The trait defines batch execution and these synchronous resource operations:

```rust
trait ReplayableModel {
    fn clear_replay_cache(&mut self);
    fn unload_state(
        &mut self,
        snapshot_path: &Path,
    ) -> Result<(), ModelExecutorError>;
    fn unload_weights(&mut self);
    fn load_weights(&mut self) -> Result<(), ModelExecutorError>;
    fn load_state(
        &mut self,
        snapshot_path: &Path,
    ) -> Result<(), ModelExecutorError>;
}
```

Qwen3 and Qwen3.5 implement these operations.
GQA, GDN, MTP, DSpark, MLP, embed, unembed, and sampling owners participate in symmetric resource traversal.
The operations run synchronously on the model executor thread.

`ReplayableModelEventLoop` owns the loaded model, its stable `Started` or `Stopped` state, and one state snapshot path.
It handles `Batch`, `Start`, and `Stop` requests synchronously on the executor thread.
`Start` and `Stop` are idempotent.
`Batch` starts a stopped model before it executes the batch.

The event loop defers request-slot resets while the model is stopped.
It applies all deferred resets after state loading and before it acknowledges `Start` or executes a batch.

Runtime core tracks the commanded `Started` or `Stopped` state.
It appends `Stop` after all batches that it has already sent.
The model event loop completes those batches before it handles `Stop`.
It sends `Start` when a stopped executor has work to flush.
It can append a batch after `Start` without a separate transition state.
The ordered request channel guarantees that the model event loop handles `Start` before that batch.

The idle timeout defaults to 300 seconds.
`--model-idle-timeout-secs` accepts a positive integer.
The service does not have a lifecycle status API or status route.

## Current resource order

The implemented resource sequence is:

```text
clear_replay_cache
    |
    v
unload_state(snapshot path)
    |
    v
unload_weights

load_weights
    |
    v
load_state(snapshot path)
```

`clear_replay_cache` removes recorded programs that retain Metal resources.
`unload_state` writes a complete state snapshot before it releases state buffers.
`unload_weights` removes all shared weight owners before it drops the final owner.

Load and unload traversal must remain logically symmetric.
Names and ownership levels must also remain symmetric.

## State snapshot

The current implementation stores complete resources.
It does not select page IDs or request slots.

The snapshot contains these resources:

- The full `PageArena` payload.
- The full Main GQA request page table.
- The full MTP or DSpark GQA request page table when configured.
- The full GDN recurrent arena.
- The full GDN convolution arena.
- The durable GDN request state table.

The durable GDN request state table contains these values:

- The current state slot for each request slot.
- The current state version for each request slot.
- Free state slots in allocator order.
- Future state versions that must publish to runtime-owned page IDs.
- The page IDs for each future publish.

The snapshot does not store submitted restore jobs, submitted publish jobs, or current batch transactions.
The model must finish or clear this transient work before it writes the snapshot.

The snapshot header contains a magic value, schema version, process-local model fingerprint, section count, and checksum.
Each section contains a resource ID, byte length, and payload checksum.
The writer syncs a temporary file before it publishes the snapshot with an atomic rename.

The first implementation uses bounded CPU staging buffers.
Future state I/O may use mapped storage or aligned direct I/O.
It must preserve complete validation, atomic publication, and synchronous model APIs.

Future selective I/O may add symmetric `write_state` and `read_state` operations.
Its metadata must identify each resource, page, request, layer, and state slot.

## Ownership boundary

Runtime core owns these resources:

- Request lifecycle and request admission.
- Request-slot identity.
- Trie cache metadata.
- Physical page allocation and ownership.
- Runtime page IDs and their logical cache identity.

The model executor owns these resources:

- Model weights.
- `PageArena` payload data.
- GQA request page tables.
- GDN recurrent and convolution state.
- GDN request state versions and future publish mappings.
- Component-local state interpretation.
- Replay programs and retained Metal resources.

Runtime core owns idle detection and executor protocol state.
The service owns the configured idle timeout.
The model event loop owns the resource operation order.
Runtime core must not parse model-specific state.
The model executor must not allocate or free runtime-owned page IDs.

## Runtime-to-device protocol

The protocol is symmetric:

```rust
pub enum ReplayableModelExecutorRequest {
    Batch(BatchDeviceRequest),
    Start,
    Stop,
}

pub enum ReplayableModelExecutorResponse {
    Batch(BatchDeviceResponse),
    Started,
    Stopped,
}
```

The request-response pairs are:

```text
Batch(request) -> Batch(response)
Start          -> Started
Stop           -> Stopped
```

`Start` and `Stop` are idempotent.
One request channel orders `Batch`, `Start`, and `Stop`.
One response channel preserves the matching response order.
The model event loop processes one request at a time.
Each channel reserves one entry in addition to the compute-slot capacity.
This entry lets `Stop` follow a full set of submitted batches.

Runtime core sends and receives all protocol variants.
It consumes `Started` and `Stopped` as acknowledgements and then attempts the next flush.
It does not maintain separate transition states for these acknowledgements.
Future wiring may expose the tracked residency through a status API.

## Lifecycle flow

```text
runtime core                        ReplayableModelEventLoop
    |                                         |
    | Stop                                    |
    |---------------------------------------->|
    |                                         | clear replay cache
    |                                         | unload state to SSD
    |                                         | unload weights
    | Stopped                                 |
    |<----------------------------------------|
    |                                         |
    | Start                                   |
    |---------------------------------------->|
    |                                         | load weights
    |                                         | load state from SSD
    | Started                                 |
    |<----------------------------------------|
```

The stop boundary must preserve any completed request state that runtime core still owns.
The model must not serialize an in-flight GPU job.
Runtime core and the model executor must agree on a completed batch boundary before `Stop` runs.
Runtime core can retain live requests across this boundary.
If a live request becomes runnable after `Stop` is queued, runtime core appends `Start` and then the next batch.
The request channel preserves the required `Stop`, `Start`, and `Batch` execution order.

The event loop uses this operation order:

```text
Stop:
  drain request-slot resets
  clear_replay_cache
  unload_state(snapshot path)
  unload_weights

Start or Batch while stopped:
  load_weights
  load_state(snapshot path)
  remove snapshot
  drain deferred request-slot resets
```

Each event-loop instance uses a unique process-local path in the system temporary directory.
The event loop removes the snapshot after a successful start and when the event loop exits.

## Failure contract

Snapshot and checkpoint I/O failures are recoverable operation errors.
Lifecycle misuse is an internal invariant violation.

The event-loop wiring fails closed after these errors:

- Snapshot write failure.
- Snapshot corruption.
- Snapshot schema mismatch.
- Model fingerprint mismatch.
- Weight load failure.
- State load failure.

The current failure path invokes global shutdown and does not send a success response.
Process shutdown discards the full runtime cache metadata.
The service must not continue with runtime page IDs that refer to invalid executor state.

## Integration verification

The ignored model residency integration tests cover this sequence:

```text
hash(state + weights)
  -> clear_replay_cache
  -> unload_state
  -> unload_weights
  -> load_weights
  -> load_state
  -> hash(state + weights)
```

The digests must be equal.
The test matrix contains Qwen3.6 27B and 35B with MTP and DSpark.
Each test requires the matching Main and speculative checkpoint environment variables.

Run the matrix with this command:

```sh
cargo test --release -p inference-executor-metal \
  --test model_residency_round_trip -- --ignored --nocapture
```
