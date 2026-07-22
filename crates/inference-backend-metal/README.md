# inference-backend-metal

`inference-backend-metal` owns reusable Metal execution primitives, operators,
and buffer-first components consumed by model executors. It does not own
request scheduling, request lifecycle, or model-specific orchestration.

The crate has one execution path: operators record reusable indirect command
buffers (ICBs), and a `Stream` submits those replays through Metal 4.

## System Overview

```text
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Runtime Core                                                                               │
│ scheduler + request lifecycle + token/block metadata + cache page ownership                │
└──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                           │ ragged batch metadata + page IDs
                                           v
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Model Executor                                                                             │
│                                                                                            │
│ Embedding / GQA / GDN / Dense MLP / MoE / Final Norm / Sampling / MTP                      │
│                                      │ semantic component order                            │
└──────────────────────────────────────┼─────────────────────────────────────────────────────┘
                                       │
             ┌─────────────────────────┼────────────────────────────┐
             │                         │                            │
             v                         v                            v
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ Weights                │  │ Workspace / State      │  │ Runtime Input          │
│                        │  │                        │  │                        │
│ quantized buffers      │  │ layer/component input  │  │ token IDs              │
│ scales / biases        │  │ and output buffers     │  │ cu_tokens              │
│ norm weights           │  │ GQA KV pages           │  │ page IDs               │
│ embedding / unembed    │  │ GDN state              │  │ sampling parameters    │
│                        │  │ scratch buffers        │  │ num_active_threads     │
│                        │  │                        │  │ ────────────────────── │
│ long-lived             │  │ reused across replay   │  │ per submission         │
└───────────┬────────────┘  └───────────┬────────────┘  └───────────┬────────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        v
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Metal Resource Layer                                                                       │
│                                                                                            │
│ Device                                                                                     │
│   ├─ Buffer                                      // owns one MTLBuffer                     │
│   │    └─ BufferView                             // borrow + dtype + shape + offset        │
│   ├─ Kernel                                      // compiled MTLComputePipelineState       │
│   └─ Stream                                                                                │
│        ├─ command queue                          // MTL4CommandQueue                       │
│        ├─ command allocator                      // MTL4CommandAllocator                   │
│        ├─ commit completion                      // MTL4 commit feedback                    │
│        └─ ResidencySet                           // wraps queue-attached MTLResidencySet   │
└──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                           │ buffers + kernels + semantic order
                                           v
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Recording Layer                                                                            │
│                                                                                            │
│ Component::record                                                                          │
│        v                                                                                   │
│ ReplayOp                                                                                   │
│        v                                                                                   │
│ ReplayRecorder                              // ordering + operator fusion                  │
│        v                                                                                   │
│ Operator / Invocation                                                                      │
│   pipeline + buffers + offsets + constants                                                 │
│   num_total_threads + num_threads_per_threadblock   // replay-static dispatch capacity     │
│        │ Operator::record                                                                  │
│        v                                                                                   │
│ CommandRecorder                                                                            │
│   set_kernel + set_buffer_read/write + set_* + bind_* + dispatch                           │
│        v                                                                                   │
│ CommandMetadata[]                                                                          │
│   pipeline + resource bindings + dispatch + barrier-before                                 │
└──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                           │ ReplayProgramBuilder::build
                                           v
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Replay Build / Cache                                                                       │
│                                                                                            │
│ CommandMetadata[] + initial parameter bytes + ReplayParameterTable + stable resources      │
│        v                                                                                   │
│ indirect command buffer                     // MTLIndirectCommandBuffer                    │
│   command slots [C0, C1, ... CN]                                                           │
│        v                                                                                   │
│ ReplayResources                                                                            │
│   ├─ ICB                                                                                   │
│   ├─ retained buffers                         // MTLBuffer resources                       │
│   ├─ retained pipelines                       // MTLComputePipelineState resources         │
│   ├─ replay parameter buffer                                                               │
│   └─ Residency                                // lease in ResidencySet                     │
│        v                                                                                   │
│ ReplayProgram                                  // cached by topology/algorithm/capacity key  │
└──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                           │ ReplayArguments + runtime input
                                           │ per submission
                                           v
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Submission / GPU Execution                                                                 │
│                                                                                            │
│ validate ReplayArguments ──> write num_active_threads and other replay parameters          │
│                                      │                                                     │
│ command allocator ──> command buffer ──> compute command encoder                           │
│                                              │                                             │
│                                              └─ executeCommandsInBuffer(ICB)               │
│                                                               │                            │
│                                                               v                            │
│ command queue ──> GPU executes command slots ──> invoke commit feedback                    │
│                                                               │                            │
│                                                               v                            │
│ allocator.reset() + release in-flight ReplayResources retention                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

The top three resource groups differ by lifecycle, not by Metal allocation
type. Weights are immutable after initialization. Workspace and state buffers
keep stable identities so cached commands can retain their bindings, while
their contents may change. Runtime input is written once per submission.
`num_total_threads` is part of the recorded dispatch capacity;
`num_active_threads` is the submission value that masks unused capacity.

## Add One

This complete example creates every persistent object from `Device`, records a
fixed-capacity ICB, submits it twice with different active work, and reads the
result back:

```rust
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::CommandRecorder;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Kernel;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::Stream;

const ADD_ONE_SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void add_one(
        device float* values [[buffer(0)]],
        constant uint& num_active_threads [[buffer(1)]],
        uint global_thread_id [[thread_position_in_grid]]
    ) {
        if (global_thread_id >= num_active_threads) {
            return;
        }

        values[global_thread_id] += 1.0f;
    }
"#;

const NUM_ACTIVE_THREADS: ReplayParameterKey =
    ReplayParameterKey::new("add_one.num_active_threads");

struct AddOne<'a> {
    kernel: &'a Kernel,
    values: &'a Buffer,
    num_total_threads: u32,
    num_threads_per_threadblock: u32,
}

impl Operator for AddOne<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read_write(0, self.values, 0);

        // MSL [[buffer(1)]] reads this per-submission replay parameter.
        recorder.bind_u32(1, NUM_ACTIVE_THREADS, 0, self.num_total_threads);

        // The ICB records this fixed bucket-capacity grid.
        recorder.dispatch_1d(
            self.num_total_threads as usize,
            self.num_threads_per_threadblock as usize,
        );
    }
}

fn main() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernel = Kernel::new(&device, ADD_ONE_SOURCE, "add_one");
    let values = Buffer::from_slice(&device, &vec![1.0_f32; 128]);

    let mut builder = stream.create_replay_program();
    builder.record(AddOne {
        kernel: &kernel,
        values: &values,
        num_total_threads: 128,
        num_threads_per_threadblock: 64,
    });
    let replay = builder.build();

    let first = ReplayArguments::new().with_u32(NUM_ACTIVE_THREADS, 96);
    stream
        .submit_replay_with_arguments(&replay, &first)
        .wait();

    let second = ReplayArguments::new().with_u32(NUM_ACTIVE_THREADS, 117);
    stream
        .submit_replay_with_arguments(&replay, &second)
        .wait();

    let output = values.read_typed::<f32>(0, 128);
    assert!(output[..96].iter().all(|&value| value == 3.0));
    assert!(output[96..117].iter().all(|&value| value == 2.0));
    assert!(output[117..].iter().all(|&value| value == 1.0));
}
```

The Metal bindings are:

```text
binding index 0   values
binding index 1   num_active_threads
dispatch grid     num_total_threads recorded in the ICB
```

Both submissions execute the same recorded grid:

```text
num_total_threads            = 128
num_threads_per_threadblock  = 64
num_threadblocks             = 2
```

Only `num_active_threads` changes. On the second submission, threads `0..116`
perform the operation and threads `117..127` return before touching memory.
Keep these four identities visibly separate:

```text
binding index         1                    MSL [[buffer(1)]]
replay parameter key  NUM_ACTIVE_THREADS   parameter-layout identity
submission value      96 or 117            current active workload
fixed ICB grid         128 total threads    recorded bucket capacity
```

The inactive-lane guard must run before every read, write, state update, or RNG
effect. A kernel with threadblock barriers must either submit whole active
threadblocks or keep every lane participating in each barrier and guard only
its memory and state effects.

Standalone RMSNorm and residual-RMSNorm expose both forms explicitly. Their
shape stores the fixed `num_total_tokens` dispatch capacity. Exact invocation
records that value directly; bucketed invocation binds a replay parameter key
whose per-submission `num_active_tokens` value is asserted to be within the
capacity. Residual-add plus RMSNorm fusion preserves the same dynamic binding.

## Recording Layers

The minimal add-one replay uses these layers:

```text
AddOne: Operator
        |
        | ReplayProgramBuilder::record
        v
CommandRecorder
  records pipeline + bindings + parameters + dispatch
        |
        v
CommandMetadata
        |
        v
ReplayProgramBuilder::build -> ReplayProgram / ICB
```

`CommandRecorder` is the low-level command-recording surface.
`ReplayProgramBuilder` collects concrete commands, consumer-side barrier
attributes, initial parameter bytes, and the replay parameter table for one ICB.
`ReplayProgramBuilder::new(&stream)` binds the builder to that stream's device,
queue, and residency set; `Stream::create_replay_program()` is the equivalent
convenience entry point. The builder constructs each `CommandRecorder`;
operators receive a recorder borrow but cannot create or finish a recorder
independently.

Model components may add the optional `ReplayRecorder` above those layers:

```text
ReplayOp
  residual_add + rms_norm, or an opaque Operator
        |
        v
ReplayRecorder
  orders pending ops and performs operator fusion
        |
        | emits resulting Operator values
        v
ReplayProgramBuilder -> CommandRecorder -> CommandMetadata -> ICB
```

`ReplayRecorder` belongs to model composition and is deliberately absent from
the basic add-one example.

## Component, Operator, and Command Order

These layers describe different units and are not one-to-one:

| Level | Examples | Owns |
| --- | --- | --- |
| Component | GQA, GDN, dense MLP, MoE, sampling | Model-semantic input/output shape, typed weights/state/scratch bindings, and the algorithm's operator composition |
| Operator | residual add, RMSNorm, fused residual + RMSNorm, quantized matmul | One backend tensor operation: kernel selection, backend shape, resource usage, parameters, and lowering into commands |
| Backend command | on Metal, one compute dispatch in one ICB slot | Exactly one backend pipeline, its resource/parameter bindings, execution geometry, and consumer-side barrier attribute |

A component emits one or more operators. An operator records one or more
backend commands. The command representation is backend-specific; Metal lowers
it to an ICB compute command, while another backend may use a different launch
or graph-node representation. Components and their semantic contracts remain
above that backend boundary. `ReplayRecorder` is an operator-stream rewrite
layer: it may fuse adjacent compatible operators, but it is not another model
computation level.

This architectural classification is not inferred from the Rust directory
name. The backend `components` module contains reusable Metal building blocks;
for example, `ResidualInvocation` is architecturally an operator, while the
executor's GQA implementation is a component that composes operators.

```text
Model / Layer
  | chooses semantic component order
  v
Component::record
  | emits ReplayOp values for algorithm phases
  v
ReplayRecorder
  | preserves order, but may fuse adjacent compatible ReplayOp values
  v
Operator::record
  | binds and dispatches one or more kernels
  v
CommandRecorder
  | one set_kernel ... dispatch sequence becomes one command
  v
ReplayProgramBuilder
  | concatenates commands in recording order
  v
ICB slots [C0, C1, C2, ...]
```

The model or layer owns semantic order. A component lowers that algorithm into
backend operators. `ReplayRecorder` may replace adjacent compatible operators
with one fused operator, but preserves the same dependency. An `Operator` is a
recording unit, not necessarily one kernel or one ICB slot; each completed
kernel dispatch becomes one concrete backend command. On Metal that command has
one compute pipeline, resource/parameter bindings, dispatch geometry, and one
ICB slot.

ICB slot order identifies commands but does not by itself serialize their
resource accesses. Replays use concurrent compute dispatches, so commands
without a dependency may overlap.

## Barrier Ownership

A barrier belongs to the command that consumes earlier results:

```text
phase 0:  C0 producer A     C1 producer B       independent commands may overlap
                   |
                   | C2 has barrier-before
                   v
phase 1:  C2 consumer C     C3 independent D
                   |
                   | C4 has barrier-before
                   v
phase 2:  C4 consumer E
```

Apple's `MTLIndirectComputeCommand::setBarrier()` is attached to the consumer
command: commands before it complete before it executes. The project therefore
records `barrier_before` on `C2`, not a barrier-after property on a producer.

At a component boundary, the consumer records its first command with
`record_with_barrier_before(...)`. At the low-level command API, an operator
that has already selected the consumer kernel calls `set_barrier_before()`.
The enclosing program builder turns a barrier request on the first program command
into a no-op because no producer precedes it.

The replay builder also infers RAW, WAR, and WAW hazards when commands bind the
same `MTLBuffer` handle with declared read/write usage. Explicit component
barriers remain necessary for dependencies that buffer identity cannot express,
such as aliased views or semantic phase boundaries.

There is no pending or trailing barrier state. `build()` freezes the recorded
commands and parameter table; it does not append a final ICB barrier. The
commit feedback proves that the entire submitted workload finished before
allocator reset and in-flight resource release.

## Execution Resources

An ICB command records how to execute work; it does not copy tensor data into
the ICB:

```text
one indirect compute command
  |-- Kernel pipeline binding               // MTLComputePipelineState
  |-- Buffer bindings and byte offsets      // MTLBuffer references
  |-- replay parameter-buffer binding
  |-- threadblock-memory lengths
  |-- fixed dispatch geometry
  `-- barrier-before state
```

These GPU objects must remain alive and resident while a replay may execute:

```text
Buffer data                       // MTLBuffer
replay parameter buffer           // MTLBuffer
Kernel pipeline                   // MTLComputePipelineState
indirect command buffer           // MTLIndirectCommandBuffer
```

These host-side objects are not resident GPU resources:

```text
CommandRecorder
ReplayRecorder
ReplayProgramBuilder
ReplayArguments
CommandParameterLayoutBuilder
```

Retention and residency solve different problems:

```text
Retained / Rc ownership
  keeps Metal objects alive and prevents use-after-free

Residency
  keeps allocations registered in the Stream's MTLResidencySet
```

The Rust ownership direction expresses both lifetimes directly:

```text
Stream
  `-- Rc<ResidencySet> -> one MTLResidencySet attached to its MTL4 queue
            |
            `-- Rc<Residency> lease
                    |
                    v
ReplayProgram
  `-- Rc<ReplayResources>
        |-- ICB
        |-- retained buffers / pipelines
        |-- parameter/ICB allocations
        `-- Rc<Residency>
                    ^
                    |
ReplaySubmission --+-- Vec<Rc<ReplayResources>>
  |-- command allocator + in-flight flag
  |-- command queue + command buffer
  |-- Rc<CommitCompletion>
  `-- wait/drop: receive commit feedback, then reset allocator
```

Dropping a cached `ReplayProgram` cannot release its resources while an in-flight
`ReplaySubmission` retains the same `Rc<ReplayResources>`. There is no second
internal submission owner or parallel resource list. A replay is tied to the
`Stream` whose residency set registered its allocations and must be submitted through
that Stream's queue.

One `Residency` covers many allocations. Leases may overlap on weights and
pipelines; `ResidencySet` deduplicates them with per-allocation reference
counts, so the queue-attached residency set contains the union of all live
replay allocations. When new allocations first enter the set, the wrapper
commits the membership update and requests residency before their first replay
submission.

## End-To-End Lifecycle

Persistent objects are created first:

```text
Device::system_default()                    // MTLDevice
  |
  |-- Kernel::new(add_one)
  |     `-- compile -> MTLComputePipelineState
  |
  |-- Buffer::from_slice(values)
  |     `-- allocate -> MTLBuffer
  |
  `-- Stream::new(&device)
        |-- create MTL4CommandQueue
        |-- create MTL4CommandAllocator
        |-- create CommitCompletion
        |     `-- commit options + feedback block + bounded channel
        `-- create ResidencySet
              `-- create and attach its MTLResidencySet
```

Build records reusable work once:

```text
Stream::create_replay_program
        |
        v
ReplayProgramBuilder::record(AddOne)
        |
        v
CommandRecorder
  |-- record Kernel pipeline
  |-- record Buffer bindings
  |-- bind NUM_ACTIVE_THREADS
  `-- record fixed num_total_threads
        |
        v
ReplayProgramBuilder::build
  |-- build initial parameter bytes + ReplayParameterTable
  |-- allocate the GPU parameter MTLBuffer
  |-- create and populate MTLIndirectCommandBuffer
  |-- register allocations -> Residency
  |-- construct Rc<ReplayResources>
  `-- return ReplayProgram
```

Each submission creates transient command state around the persistent ICB:

```text
ReplayArguments { NUM_ACTIVE_THREADS: 96 }
        |
        | validate and write replay parameter buffer
        v
MTL4CommandAllocator
        |
        | beginCommandBufferWithAllocator
        v
MTL4CommandBuffer
        |
        v
MTL4ComputeCommandEncoder
        |
        | executeCommandsInBuffer(ICB)
        v
endEncoding -> endCommandBuffer
        |
        | register the retained feedback block for this commit
        v
MTL4 queue commit
        |
        v
GPU executes the ICB
        |
        v
Metal invokes the commit feedback block
        |
        | proves the whole command buffer completed
        v
allocator.reset()
        |
        v
wait returns; dropping ReplaySubmission releases command state and ReplayResources
```

The allocator backs transient submission commands. It does not own the ICB,
model buffers, pipelines, parameter buffer, or residency lease. `wait()` proves
completion and resets the allocator but leaves the submission's retained fields
intact; those fields are released when the `ReplaySubmission` itself is dropped.

## Current Stream Contract

One `Stream` currently owns:

```text
1 MTL4CommandQueue
1 MTL4CommandAllocator         transient command storage
1 CommitCompletion             reusable options/block/channel; registered per commit
1 ResidencySet                 wraps one queue-attached MTLResidencySet
```

Because one allocator is reused, a Stream currently allows one in-flight
submission. Completion resets the allocator before another submission can be
encoded. Supporting multiple in-flight submissions requires an allocator
pool or ring; it does not require duplicating persistent ICBs.

A submission may execute a sequence of distinct `ReplayProgram` values in one
Metal 4 command buffer. The stream inserts an execution barrier between ICBs
and retains every program's resources through completion. If one program
appears multiple times in a sequence, every occurrence must use identical arguments
because its parameter buffer is replay-owned.

## Why Replay

Conceptually, a direct path would bind pipelines, buffers, constants, and the
current dispatch grid into a transient compute encoder on every submission:

```text
conceptual direct submission
  Operator metadata
    -> bind resources again
    -> encode current dispatch again
    -> submit once
```

Replay moves the stable work to build time:

```text
implemented replay path
  build once: pipeline + resources + capacity grid -> ICB
  submit many: validate/write ReplayArguments -> execute ICB
```

The crate intentionally does not expose `DirectBatch` or direct operator
variants. Production, benches, and Metal correctness tests all exercise the
same replay/ICB path; CPU implementations remain the correctness oracles.
Keeping this conceptual comparison is useful for understanding what replay
removes, but it is not a second execution API.

Executable add-one coverage lives in
[`src/metal/stream/mod.rs`](src/metal/stream/mod.rs).
