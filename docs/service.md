# Qwen Service and Decode RPC

The service runs one transport-neutral `Inference` API over the runtime. The gRPC adapter submits model-ready token IDs
through its `decode` operation.

The HTTP adapter provides collected and streaming OpenAI-compatible Chat Completions. It uses the same `decode`
operation.

[`core.md`](core.md) defines runtime scheduling and page ownership. The executor documents define model execution.

## Source ownership

`crates/inference-runtime-service/src/api/` owns these functions:

- Token-level validation
- Server request IDs
- Stop-sequence merging
- Submission and output streaming
- Cancellation when a response is dropped

`rpc/grpc/` owns protobuf conversion and tonic status mapping. `rpc/http/` owns HTTP listener setup, JSON parsing, and
HTTP error mapping.

The Qwen server loads one immutable `QwenCodec` from the model directory. It shares this codec with HTTP requests.

RPC converts wire objects to the codec domain. It also maps codec events back to wire objects. RPC does not implement
these functions:

- Prompt syntax
- Tokenization algorithms
- The Qwen tool dialect

`crates/inference-runtime-service/src/tool/` owns the transport-neutral and model-neutral tool domain. It does not own
these functions:

- Conversation storage
- Tool handlers
- Execution scheduling
- Prompt rendering
- Model-specific tool syntax

`inference-runtime-core/src/chat_template/` loads the checkpoint-authoritative Hugging Face chat template. It compiles
the template with `hf-chat-template`.

A standalone `chat_template.jinja` takes priority over an inline tokenizer-config template. `codec/qwen.rs` owns the
model-specific composition.

`QwenCodec` contains the compiled template and shared Hugging Face tokenizer. Its `encode` method renders messages and
tools before tokenization.

The `decode` method transforms a token stream into these outputs:

- Reasoning
- Answer text
- Tool calls
- A terminal event

When thinking is enabled, the codec consumes the Qwen `</think>` boundary. It never exposes model control tokens as
content.

Each response stream keeps its incremental detokenization and Qwen response-grammar state private.

The runtime token and probability channel remains `async_channel::unbounded`. Dropping a `DecodeResponse` drops its
`ExternalRequest`. This action cancels only that request.

[`future_work.md`](future_work.md) contains the bounded slow-consumer work. The service keeps one output channel at its
current capacity.

## Tool state

`ToolState` is the only lifecycle owner for state derived from ordered tool events in one conversation:

```text
ToolState
├── tools: currently callable ToolDefinition values
└── executions: HashMap<ToolCallID, ToolCallRequest>
```

`ToolState` does not retain completed calls. It also does not retain registration, unregistration, response, or
cancellation events. The persistent conversation history owns these events.

`ToolState::fold` applies history in order. This operation reconstructs the two derived collections.

`ToolRegistration` and `ToolUnregistration` are non-empty batches without duplicates. `register_tool` and
`unregister_tool` are atomic.

Each operation updates all definitions or leaves the state unchanged. Definition order follows conversation-history
order. Thus, a later prompt adapter can render tools deterministically.

Removing a tool prevents new calls. A later event can register the same `ToolID` again.

A `ToolCallRequest` has the following transport-neutral shape:

```text
ToolCallRequest
├── tool_id: ToolID
├── tool_call_id: ToolCallID
└── arguments: ToolArguments
```

`ToolState` is a correlation ledger. It is not a tool executor.

`request_execution` requires a currently registered `tool_id`. It also requires a `tool_call_id` that is not in flight.
Multiple requests can remain in flight.

`respond_execution` and `cancel_execution` address an execution only by `ToolCallID`. They do not examine the currently
callable definitions.

Thus, unregistration does not cancel an accepted call. The response for that call remains valid. Completed and canceled
entries leave the in-flight ledger immediately.

Execution iteration has no ordering contract.

A `ToolCallResponse` carries:

```text
ToolCallResponse
├── tool_call_id: ToolCallID
├── raw_content: Vec<ToolRawContent>
├── structured_content: Option<ToolStructuredContent>
└── is_error: bool
```

`ToolRawContent` is model-facing content and currently supports text. `ToolStructuredContent` is an arbitrary JSON value
for programmatic consumers.

The two content forms can coexist. A tool-level failure sets `is_error`. It also includes a safe, non-empty error
message for the model.

Service and protocol failures remain ordinary `Err(Error)` values. They are not false tool responses.

The event projection is:

```text
Registration     tools + definitions    executions unchanged
Unregistration   tools - definitions    executions unchanged
CallRequest      require active tool    executions + request
CallResponse     tools ignored          executions - request
CallCancellation tools ignored          executions - request
```

The persistent conversation history is the authority. Its position is the natural version.

`ToolState` is only a derived in-memory projection. It is not a history store or recoverable cache.

Calls from one model response are independently in flight. The agent or tool environment can run them concurrently.
The inference server does not own these controls:

- Approval and permissions
- Side-effect serialization
- Concurrency limits
- Handler cancellation
- Argument-schema validation

## Binaries and checkpoints

The service provides Qwen3 and Qwen3.5 binaries with optional DSpark.
DSpark support is experimental.
Its checkpoint contract, CLI, cache sizing, and proposal policy may change.
It also provides Qwen3.5 binaries that retain their names for current Qwen3.6 MLX checkpoints:

| Model                  | Binary           | Main checkpoint                      | Optional Spec checkpoint                  |
| ---------------------- | ---------------- | ------------------------------------ | ----------------------------------------- |
| Qwen3 dense 14B        | `qwen3`          | `mlx-community/Qwen3-14B-4bit`       | optional official Qwen3 DSpark checkpoint |
| Qwen3.6 dense 27B      | `qwen3_5_dense`  | `mlx-community/Qwen3.6-27B-4bit`     | matching MTP or official Qwen3x DSpark    |
| Qwen3.6 sparse 35B-A3B | `qwen3_5_sparse` | `mlx-community/Qwen3.6-35B-A3B-4bit` | matching MTP or official Qwen3x DSpark    |

Download with the Hugging Face CLI:

```sh
hf auth login
hf download mlx-community/Qwen3-14B-4bit --local-dir models/Qwen3-14B-4bit
hf download mlx-community/Qwen3.6-27B-4bit --local-dir models/Qwen3.6-27B-4bit
hf download mlx-community/Qwen3.6-27B-MTP-4bit --local-dir models/Qwen3.6-27B-MTP-4bit
```

Use the corresponding 35B-A3B names for the sparse model. MTP checkpoints contain Spec weights. They must match the
Main model family.

### Qwen3x DSpark conversion

Convert an official BF16 Qwen3x DSpark checkpoint to the affine executor format:

```sh
cargo run -p inference-executor-core --bin qwen3_dspark_quantize -- \
  --input-dir /path/to/Qwen3-DSpark \
  --output-dir /path/to/Qwen3-DSpark-affine \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not exist before you run the converter.
The converter writes `model.safetensors` and `model.safetensors.index.json`.
It preserves the confidence projection and bias as BF16.
The input must be the official BF16 DSpark checkpoint.
The flat DSpark query projection width can differ from `hidden_size`.
The converter and loader derive the query width from `num_attention_heads * head_dim`.
An affine checkpoint that was generated before confidence support does not contain these tensors.
Regenerate that checkpoint into a new output directory.

Qwen3 Main-only startup:

```sh
cargo run --release --bin qwen3 -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit"
```

Qwen3 startup with DSpark:

```sh
cargo run --release --bin qwen3 -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit" \
  --hf-dspark-model-dir "$PWD/models/Qwen3-DSpark-affine" \
  --num-spec-tokens 4
```

The Qwen3 executor gets stop tokens from the checkpoint configuration when `generation_config.json` is absent.

Qwen3.5/Qwen3.6 dense startup with DSpark:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-dspark-model-dir "$PWD/models/Qwen3.6-27B-DSpark-affine" \
  --num-spec-tokens 4
```

The Qwen3.5 services reject a configuration that specifies both `--hf-mtp-model-dir` and
`--hf-dspark-model-dir`.

Qwen3.5 startup with MTP enabled:

Dense:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-27B-MTP-4bit" \
  --num-spec-tokens 4
```

Sparse:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit"
```

The normal MTP and DSpark commands use the same service and scheduler arguments.
Only the Spec checkpoint argument changes:

```text
--hf-mtp-model-dir DIR
--hf-dspark-model-dir DIR
```

An MTP checkpoint enables one speculative MTP step by default.
`--num-spec-tokens K` takes a positive `usize` value for MTP or DSpark.
The executor reuses the checkpoint's one physical MTP layer for K dependent logical steps.
For MTP, omit `--num-spec-tokens` to use one step.
`--max-tokens-per-request` must not exceed `--max-tokens`.
For MTP with K speculative tokens, `--max-tokens-per-request` must be at least K.
An MTP decode request must contain at least K initial input tokens.
For DSpark, omit `--num-spec-tokens` to use the checkpoint `block_size`.
An explicit DSpark value must not exceed the checkpoint `block_size`.
The DSpark value controls proposal generation. It is independent of
`--max-tokens-per-request`, which limits the Main verification batch.
The scheduler may verify only a proposal prefix.

The service specialization module provides the model-independent worker build and process lifecycle.
`SpecializedWorker` uses `escargot` to build an executable for the active profile and target.
It owns the dedicated target directory, build environment, artifact path, and process replacement.
A model launcher supplies its worker manifest, binary name, specialization target directory, build environment, and worker arguments.

`qwen3_5_dense` and `qwen3_5_sparse` are thin model-specific launchers.
Each launcher validates the normal CLI. For MTP with K speculative tokens, it calculates `L = K + 1`.
For Vanilla and DSpark, it uses `L = 1`.
It configures `SpecializedWorker` to build a const-specialized copy of the same `qwen3_5_dense` or `qwen3_5_sparse` binary.
The `inference-runtime-service` `build.rs` generates compile-time const `L`.
An internal environment marker makes the specialized binary run the model instead of starting another build.
The launcher then replaces itself with that specialized binary.

Each cache-lane count uses `target/qwen3_5_specialized/cache_lanes_L` as its Cargo target directory.
Cargo fingerprints the source, features, target, and active debug or release profile in that directory.
A warm launch checks and reuses the existing artifact.
A cold launch requires the repository source, Cargo, the pinned Rust toolchain, and access to all required build inputs.

The gRPC address defaults to `127.0.0.1:50051`. The HTTP address defaults to `127.0.0.1:8000`.

The model idle timeout defaults to 300 seconds.
After this period without executable model work, the service writes model state to SSD and unloads model resources.
The listeners and runtime requests remain active.
The next executable batch loads weights and state before execution.
`--model-idle-timeout-secs` accepts a positive integer.

One lifecycle owner stops both listeners in these conditions:

- The runtime stops.
- A listener fails.
- The process receives SIGINT or SIGTERM.

`--num-spec-tokens` requires `--hf-mtp-model-dir` or `--hf-dspark-model-dir`.
The service rejects zero, a missing Spec directory, or simultaneous MTP and DSpark directories.
For a Main-only run, omit both Spec checkpoint arguments and `--num-spec-tokens`.

Qwen uses 32 KiB physical cache pages. Qwen3 and Qwen3.5 default to 256K pages. The Qwen3-14B geometry stores eight
tokens in one physical page.

Its 16-token logical cache block uses 80 pages across 40 layers. Thus, the default holds 3,276 complete blocks. These
blocks contain 52,416 resident tokens in aggregate.

When DSpark is enabled, the executor adds persistent DSpark context pages to the same logical block.
The page count depends on the DSpark layer and KV geometry.

Qwen3.5 keeps 2,048-token logical blocks to amortize its GDN snapshots. It defaults to 256K shared pages.
MTP step K adds K logical KV cache lanes to the Main lane.
All lanes allocate from the same shared physical-page arena.

At startup, each service derives the page count for one block from the initialized executor. The service rejects
`--num-cache-pages` when one complete block cannot fit.
The service classifies a model-executor initialization failure as an internal startup error.

The rejection reports this dynamic minimum.

Recommendation: For performance comparisons, pass `--num-cache-pages` explicitly. This setting controls memory
pressure.

`Qwen3Config` and `Qwen35Config` resolve the queued-request, running-request, and per-batch capacities.
CLI checkpoint arguments remain optional parser inputs.
Configuration validation converts them to one `Vanilla`, `MTP`, or `DSpark` model mode.
The validated configuration does not store independent MTP and DSpark options.
`--max-requests` defines both the running request-slot capacity and the per-batch request capacity.
The model service passes this value to the executor, `RuntimeConfig`, and `SchedulerConfig`.
The services default to 32 queued requests and 4 running request slots. Queued requests do not consume executor
request-slot state.

Admission assigns a slot before a request enters the scheduler.
GQA page tables, GDN request state, sampling state, and request-indexed workspaces use the same slot domain.
For Qwen3.5 DSpark, GDN retains one candidate state for every possible accepted proposal prefix in each slot.
Thus, `--max-requests` also bounds the persistent GDN candidate-state arena.
Buffers, scratch allocations, replay resources, and resident model resources remain reusable.

Qwen3.5 wiring derives this request-local GDN slot count:

```text
decision_candidate_states = match mode {
  Vanilla => 1,
  MTP { num_spec_tokens } | DSpark { num_spec_tokens } => num_spec_tokens + 1,
}
block_boundary_candidates = ceil(max_tokens_per_request / num_tokens_per_block)
candidate_states = decision_candidate_states + block_boundary_candidates
state_slots = 1 + candidate_states
```

The leading slot stores the current state.
Candidate slots store decision prefixes and logical cache-block boundary states.
The two sets can be disjoint when a request contains more than one fixed token.
MTP shifts the complete decision-candidate version range by `num_spec_tokens - 1`.
The shift changes the physical replay frontier. It does not change the candidate count.
Qwen verification calculates the shifted range. GDN commit receives one selected physical version as its next source.
The total arena scales with `--max-requests * state_slots * full_model_state_bytes`.

For the Qwen3.6-27B checkpoint, one full-model GDN state is 149.625 MiB.
With the default `--max-requests 4`, one-step MTP uses four state slots for each request and allocates
approximately 2.34 GiB for the arena.
Two-step MTP uses five state slots for each request and allocates approximately 2.92 GiB.
Four-step MTP uses seven state slots for each request and allocates approximately 4.09 GiB.
A DSpark run with `num_spec_tokens=15` uses 18 state slots for each request and allocates approximately 10.52 GiB.
These values do not include model weights, cache pages, or other executor workspaces.

One default batch has these scheduler limits:

- 4 requests
- 128 flattened tokens
- 64 tokens for each request

## gRPC decode

`DecodeRequest` contains model-ready tokens and sampling fields. It does not contain a caller request ID or client-side
default stop tokens.

The server assigns a nonzero ID. Each `DecodeResponse` envelope contains this ID.

Each response has one of these forms:

- A `chunk` with equal, non-empty token and probability arrays
- One `completion` event

Completion reasons are `STOP_SEQUENCE`, `LENGTH_LIMIT`, and the reserved `CONTEXT_LIMIT`. EOF without a completion event
means that the stream failed.

The external diagnostic client remains a gRPC client:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --top-k 64 \
  --top-p 0.8 \
  --seed 59 \
  --chat-template auto \
  --prompt-str "Explain paged KV cache in one paragraph." \
  --max-sampled-tokens 8192 \
  --show-stats \
  --print-prompt
```

The client records the server ID from the first envelope. It requires the explicit completion event and streams
incremental text.

The client compares the incremental text with one final full decode. It reports inter-chunk metrics, not inter-token
metrics.

The client model directory supplies tokenizer and chat-template files. The server controls the Main model.

`--chat-template auto` runs the checkpoint template. It does not use a hard-coded Qwen prompt formatter.

Add `--disable-thinking` for the checkpoint-defined non-thinking generation prefix. `--chat-template raw` is an
explicit diagnostic bypass.

## HTTP Chat Completions

The HTTP listener runs with gRPC. It provides the OpenAI-compatible `POST /v1/chat/completions` route.

### Matched gRPC and HTTP tests

The following command pairs use the same prompt, output limit, sampling values, seed, and thinking mode.
The HTTP `model` field is optional.
When the request omits it, the response uses the loaded executor model name.
When the request provides it, the value labels the response.
It does not select a different loaded model.

Use this gRPC command for a Qwen3 server:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit" \
  --chat-template auto \
  --disable-thinking \
  --prompt-str "Reply with exactly: hello" \
  --max-sampled-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 1
```

Use this equivalent HTTP command:

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Use this gRPC command for a Qwen3.5/Qwen3.6 dense server:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --chat-template auto \
  --disable-thinking \
  --prompt-str "Reply with exactly: hello" \
  --max-sampled-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 1
```

Use this equivalent HTTP command:

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

For the sparse Qwen3.5/Qwen3.6 server, use `"$PWD/models/Qwen3.6-35B-A3B-4bit"`.

### Response modes

Omit `stream` or set it to `false` to return one collected response:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Set `stream:true` to return server-sent events (SSE). `stream_options.include_usage:true` adds a separate usage chunk
before `[DONE]`:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Both paths process messages and active function definitions through the shared `QwenCodec`. Each path calls
`Inference::decode` one time.

The path maps the `ResponseEvent` stream to content, tool-call, completion, and usage fields.

The service validates historical tool calls against their results. It does not validate them against currently active
definitions.

Thus, removing a tool prevents a new call without invalidating completed conversation history.

`tool_choice` supports `auto` and `none`. The service rejects `required`. It does not silently weaken the `required`
contract.

The endpoint is a documented OpenAI-compatible subset. It does not claim full API coverage.

The Pi `openai-completions` provider can use this endpoint directly. The wire adapter accepts these Pi fields:

- Non-persistent `store:false`
- Non-constrained `strict:false` tool definitions
- A leading `developer` message

A supplied `reasoning_effort` enables Qwen thinking. Streaming reasoning returns in `delta.reasoning_content`.
Later history can supply the reasoning as assistant `reasoning_content`.

The adapter rejects `reasoning_effort` with `enable_thinking:false` because the fields are contradictory.

## Operational notes

Run one GPU service at a time. The first request includes these operations:

- Model initialization
- Metal pipeline compilation
- Replay construction
- Cache warmup

Measure the first request separately from steady-state throughput.

Normal sampling submits these replay programs in one ordered Metal command buffer:

- MainEmbed
- Main
- GatherUnembed
- Sampling

Speculative Main verification replaces Sampling with RejectionSampling.

An MTP proposal uses K ordered passes of MTPEmbed, MTP, GatherUnembed, and DraftSampling.
The public Spec lifecycle remains one `submit_spec -> wait -> read_spec` transaction.
The current implementation submits each dependent pass separately and reads its sampled token on the CPU before it
prepares the next pass.
All passes reuse the same recorded replay programs, weights, scratch, and bound workspaces.

A DSpark proposal uses a separate submission with DSparkEmbed, DSpark, DSparkGatherUnembed, and DSparkSampling.
The Main submission records DSparkContext before GatherUnembed.
Main decisions cross the CPU boundary before DSpark constructs the anchor block.

`--logging info` emits one batch event with these fields:

- Model and batch sequence
- Request and input counts
- Speculative input
- Accepted speculative tokens
- Committed output tokens
- Acceptance rate
- Total latency

`--logging debug` uses the same event model. It adds these fields:

- Request-kind counts
- Rejected and next speculative tokens
- Sampled rows
- Replay-stage submit and wait timing

It does not duplicate an INFO event.

Internal model `Start` and `Stop` commands emit INFO lifecycle events on the
`inference-runtime-service::lifecycle` target.
The events use `model.start.begin`, `model.start.complete`, `model.stop.begin`, and `model.stop.complete` phases.
Completion events include elapsed milliseconds.
Failure events include the model name and error, and then trigger global shutdown.
Runtime core issues these commands after the configured idle timeout.

The Main timing fields are:

- `model_output_main_replay_ms`: `MainEmbed -> Main` when the batch has no sampling rows
- `model_output_main_sample_replay_ms`: `MainEmbed -> Main -> GatherUnembed -> Sampling/RejectionSampling`

Executor profiling spans use the lifecycle hook names:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> read_main
embed_spec -> forward_spec -> unembed_spec -> sample_spec
submit_spec -> read_spec
```

The model-neutral speculator timing fields are:

- `model_output_spec_build_ms`
- `model_output_spec_replay_ms`
- `model_output_spec_read_ms`
- `model_output_spec_passes`

A pass is one auxiliary speculator forward that runs.
For Qwen3.5, each logical MTP step or DSpark block forward is one pass.
For Qwen3, each DSpark block forward is one pass.
Qwen3 does not run DSpark Spec for a prefill-only batch because no sampled anchor exists.

Runtime shutdown emits a scheduler table. The table contains call counts and latency percentiles for the runtime
lifetime.

Enqueue and swap-in operations report request counts. Prepare, cancel, and commit operations report counts and latency
percentiles.

Set `PSI_QWEN35_STATE_TRACE=1` to write executor lifecycle lines to standard error. These lines include:

- Replay cache hit or miss keys
- GDN restore and publish decisions
- Synchronous `prepare_sync` timing

The timing fields are `gqa_us`, `gdn_states_us`, dependent `gdn_metadata_us`, and total `wall_us`.

Set `PSI_GDN_STATE_TRACE=1` only for the detailed GDN request-state transition trace. Both settings are diagnostic.
Leave them unset for throughput measurements.

`--profile component` and `--profile operation` enable the same coarse CPU tree. It contains prepare, model
input/forward/output, and commit.

These modes do not attribute GPU time to components or kernels. Use Metal capture or counters for GPU time.

## Correctness and long decode

For a release correctness check, first use deterministic sampling:

```text
--temperature 0 --top-k 1 --top-p 1 --seed 1
```

Use a prompt with an objective oracle. Then, run a long 8K-token generation.

Validate the dense and sparse models. If MTP paths changed, validate MTP off and on.

Record these facts:

- Prompt tokens and sampled tokens
- Termination reason and output sanity
- Commit and dirty state
- Model directory

Investigate cold-start stalls separately from later decode stalls.

## End-to-end performance helper

The Qwen3 helper runs Main-only and DSpark decode measurements:

```sh
scripts/qwen3_e2e_decode_perf.sh \
  --model <qwen3-model-dir> \
  --dspark <qwen3-affine-dspark-model-dir> \
  --runs 7
```

The helper uses the model directory for tokenization by default.
Use `--tokenizer` to select a different directory.
Use `--tokens` to select the comma-separated output-token counts.
The default Qwen3 matrix runs `14b_off` and `14b_dspark`.
The `14b_dspark` group runs `--num-spec-tokens 1` and then `--num-spec-tokens 2`.
Each summary label includes the selected value, for example, `14b_dspark1` or `14b_dspark2`.
If the DSpark checkpoint is absent and no download repository is configured, the helper prints a warning and skips
that case. A missing Main checkpoint remains an error.

The Qwen3.5/3.6 helper runs controlled 27B/35B Main-only, MTP, and DSpark comparisons:

```sh
scripts/qwen35_e2e_decode_perf.sh \
  --tokenizer <tokenizer-model-dir> \
  --model-27b <27b-model-dir> \
  --mtp-27b <27b-mtp-model-dir> \
  --dspark-27b <27b-affine-dspark-model-dir> \
  --model-35b <35b-model-dir> \
  --mtp-35b <35b-mtp-model-dir> \
  --dspark-35b <35b-affine-dspark-model-dir> \
  --runs 7
```

Each `*_mtp` or `*_dspark` group runs `--num-spec-tokens 1` and then `--num-spec-tokens 2`.
The default case matrix runs `27b_off`, `27b_mtp`, `27b_dspark`, `35b_off`, `35b_mtp`, and `35b_dspark`.
If an MTP or DSpark checkpoint is absent and no download repository is configured, the helper prints a warning and
skips that case. A missing Main checkpoint remains an error.
The helper stops the server between speculative-token counts.
It applies the configured cooldown before the second count.
Each summary label includes the speculative mode and token count, for example, `27b_mtp2` or `27b_dspark2`.

Both helpers print these facts:

- Commit and dirty state
- Model directories
- Machine and operating system
- Cache and request capacity
- Scheduler capacities
- Sampling configuration and seed
- Trajectory fields

Both helpers also print cooldown and speculative-acceptance fields.

The Qwen3 helper does not contain a checked-in performance baseline.

The Qwen3.5/3.6 helper contains an M3 Max baseline for Main-only and one-step MTP cases.
Two-step MTP and all DSpark cases report that no hardware baseline exists.
The baseline was recorded on 2026-07-21 at `132c5073`.
It used these settings:

- 384K pages
- 2048-token cache blocks
- Four running requests
- The 4/128/64 scheduler configuration

The current default uses four running requests.

A summary reports `baseline_status=comparable` only when all comparison inputs match. It then reports typed decode,
TTFT, and inter-chunk delta percentages.

These comparison inputs must match:

- Machine and operating system
- Checkpoint directory names
- Prompt and sampling configuration
- Capacities and cooldown
- Sampled trajectory

Baseline throughput and trajectory use machine, case, and token count as keys. A configuration or trajectory mismatch
remains visible. It does not produce a performance delta.

Summaries report these metrics:

- Decode throughput
- TTFT and prompt throughput
- RPC inter-chunk p50 and p95
- Tokens for each chunk
- Exact accepted/proposed speculative-token rate for the matching server-log interval

For Main-only decoding, a chunk contains one token. Thus, inter-chunk time is inter-token latency.

With MTP, inter-chunk time measures burst cadence. Interpret it with tokens for each chunk and the acceptance rate.

A positive decode delta is faster. A positive TTFT or inter-chunk latency delta is slower.

Use `--case-cooldown-secs 0` only for an intentional sustained-load experiment.

Follow [`executor_benchmarks.md`](executor_benchmarks.md) before you make a performance claim.
