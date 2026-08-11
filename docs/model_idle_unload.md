# Model Idle Unload Design Draft

This document records the implemented model resource operations and the planned service wiring.
The service wiring is not implemented.
The runtime-to-device protocol is not final.

## Objective

The final service must release model weights and backend state after an idle period.
The service process and its RPC listeners must remain active.
The first new request must load the model before model execution resumes.

The implementation must preserve runtime cache identity across a successful stop and start.
It must discard runtime cache metadata after any snapshot or restore failure.

## Current implementation boundary

`ReplayableModel` defines batch execution and these synchronous resource operations:

```rust
trait ReplayableModel {
    type LifecycleError;

    fn clear_replay_cache(&mut self);
    fn unload_state(
        &mut self,
        snapshot_path: &Path,
    ) -> Result<(), Self::LifecycleError>;
    fn unload_weights(&mut self);
    fn load_weights(&mut self) -> Result<(), Self::LifecycleError>;
    fn load_state(
        &mut self,
        snapshot_path: &Path,
    ) -> Result<(), Self::LifecycleError>;
}
```

Qwen3 and Qwen3.5 implement these operations.
GQA, GDN, MTP, DSpark, MLP, embed, unembed, and sampling owners participate in symmetric resource traversal.
The operations run synchronously on the model executor thread.

`ReplayableModelEventLoop` currently accepts only batch requests and request-slot reset notifications.
It does not invoke model resource operations.
The service has no start command, stop command, idle timer, lifecycle status, or status route.

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

The future service wiring must own the idle policy and operation order.
Runtime core must not parse model-specific state.
The model executor must not allocate or free runtime-owned page IDs.

## Planned runtime-to-device protocol

The preferred protocol shape is symmetric, but it is not final and is not implemented:

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

The intended request-response pairs are:

```text
Batch(request) -> Batch(response)
Start          -> Started
Stop           -> Stopped
```

`Start` and `Stop` must be idempotent.
The final protocol must define ordering with batch execution and request-slot reset notifications.
It must not require runtime core to interpret service policy.

The future wiring belongs in a separate commit.
That commit must define admission control, idle detection, runtime cache coordination, failure handling, and status exposure.

## Planned flow

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

## Failure contract

Snapshot and checkpoint I/O failures are recoverable operation errors.
Lifecycle misuse is an internal invariant violation.

The final wiring must fail closed after these errors:

- Snapshot write failure.
- Snapshot corruption.
- Snapshot schema mismatch.
- Model fingerprint mismatch.
- Weight load failure.
- State load failure.

The failure path must invoke global shutdown or rebuild the full runtime cache generation.
It must not continue with runtime page IDs that refer to invalid executor state.

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
