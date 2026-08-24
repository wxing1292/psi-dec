# Qwen Service and Decode RPC

The service runs one transport-neutral `Inference` API over the runtime. The gRPC adapter submits model-ready token IDs
through its `decode` operation.

The HTTP adapter provides collected and streaming OpenAI-compatible Chat Completions. It uses the same `decode`
operation.

[`core.md`](core.md) defines runtime scheduling and page ownership. The executor documents define model execution.

## Source ownership

`crates/inference-runtime-service/src/api/` owns these functions:

- Token-level validation
- Process-local runtime request IDs
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

The runtime token and probability channel is intentionally `async_channel::unbounded`. It preserves committed output
without blocking synchronous runtime commit. Dropping a `DecodeResponse` drops its `ExternalRequest`. This action
cancels only that request. A slow transport consumer does not terminate the request.

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

The service provides Qwen3 and Qwen3.5 binaries with optional speculative models.
DSpark and DFlash2 support is experimental.
Their checkpoint contracts, CLI, cache sizing, and proposal policies may change.
It also provides Qwen3.5 binaries that retain their names for compatible Qwen3.6 and Qwen3.8 MLX checkpoints:

| Model                  | Binary           | Main checkpoint                      | Optional Spec checkpoint                  |
| ---------------------- | ---------------- | ------------------------------------ | ----------------------------------------- |
| Qwen3 dense 14B        | `qwen3`          | `mlx-community/Qwen3-14B-4bit`       | optional official Qwen3 DSpark checkpoint |
| Qwen3.8 dense 27B      | `qwen3_5_dense`  | `mlx-community/Qwen3.8-27B-4bit`     | matching MTP, Qwen3x DSpark, or Qwen3x DFlash2 |
| Qwen3.6 sparse 35B-A3B | `qwen3_5_sparse` | `mlx-community/Qwen3.6-35B-A3B-4bit` | matching MTP, Qwen3x DSpark, or Qwen3x DFlash2 |

Download with the Hugging Face CLI:

```sh
hf auth login
hf download mlx-community/Qwen3-14B-4bit --local-dir models/Qwen3-14B-4bit
hf download mlx-community/Qwen3.8-27B-4bit --local-dir models/Qwen3.8-27B-4bit
hf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir models/Qwen3.8-27B-MTP-4bit
hf download RadixArk/Qwen3.8-27B-DSpark --local-dir models/Qwen3.8-27B-DSpark
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir models/Qwen3.8-27B-DFlash2
```

Use the corresponding 35B-A3B names for the sparse model. MTP checkpoints contain Spec weights. They must match the
Main model family.

The DSpark and DFlash2 repositories contain BF16 source checkpoints. Convert these checkpoints before service startup.
Both converter subcommands use the same affine format:

- Matrix payloads use packed `U32` storage.
- Affine scales and biases use BF16 storage.
- Quantization calculations use F32.
- Packed codes use the final stored BF16 scale and bias values.
- Unquantized tensors preserve their source BF16 dtype.

### Qwen3x DSpark conversion

Convert an official BF16 Qwen3x DSpark checkpoint to the affine executor format:

```sh
cargo run --bin qwen3x_spec_quantize -- dspark \
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

### Qwen3x DFlash2 conversion

Convert the official BF16 Qwen3.8 DFlash2 checkpoint to the affine executor format:

```sh
cargo run --bin qwen3x_spec_quantize -- dflash2 \
  --input-dir /path/to/Qwen3.8-27B-DFlash2 \
  --output-dir /path/to/Qwen3.8-27B-DFlash2-affine
```

The output directory must not exist before you run the converter.
The default policy uses group size 64 and 4-bit affine matrices.
It uses 6-bit affine matrices for layer 2 and layer 4 `v_proj` and `down_proj` weights.
This policy matches the tensor-level Q4_K_M choices in `z-lab/Qwen3.8-27B-DFlash2-GGUF`.
It preserves norms and dynamic-convolution base kernels as BF16.
The output checkpoint contains no unquantized BF16 weight matrix.

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
  --hf-spec-model-dir "$PWD/models/Qwen3-DSpark-affine" \
  --spec-type dspark
```

The Qwen3 executor gets stop tokens from the checkpoint configuration when `generation_config.json` is absent.

Qwen3.5/Qwen3.6 dense startup with DSpark:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-27B-DSpark-affine" \
  --spec-type dspark
```

Qwen3.8 dense startup with DFlash2:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine" \
  --spec-type dflash2
```

The Qwen3 service accepts only `--spec-type dspark`.
The Qwen3.5 services accept `mtp`, `dspark`, and `dflash2`.

Qwen3.5/Qwen3.6/Qwen3.8 startup with MTP enabled:

Dense:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 4
```

Sparse:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --spec-type mtp
```

The normal MTP, DSpark, and DFlash2 commands use the same service and scheduler arguments.
Select the Spec checkpoint with these paired arguments:

```text
--hf-spec-model-dir DIR
--spec-type {mtp,dspark,dflash2}
```

An MTP checkpoint enables one speculative MTP step by default.
`--num-spec-tokens K` is an MTP-only option. It takes a positive `usize` value.
The value is the number of speculative tokens in one MTP proposal.
The executor reuses the checkpoint's one physical MTP layer for K dependent logical steps.
Omit `--num-spec-tokens` to use one MTP step.
`--max-tokens-per-request` must not exceed `--max-tokens`.
For MTP with K speculative tokens, `--max-tokens-per-request` must be at least K.
An MTP decode request must contain at least K initial input tokens.
DSpark gets its proposal count from the checkpoint `block_size`.
DFlash2 gets its query-block size from the checkpoint `block_size`.
One DFlash2 query block contains one anchor and `block_size - 1` MASK proposal rows.
The service rejects `--num-spec-tokens` with DSpark or DFlash2.
The checkpoint-defined block geometry is independent of `--max-tokens-per-request`.
That option limits the Main verification batch.
The scheduler may verify only a proposal prefix.

The service specialization module provides the model-independent worker build and process lifecycle.
`SpecializedWorker` uses `escargot` to build an executable for the active profile and target.
It owns the dedicated target directory, build environment, artifact path, and process replacement.
A model launcher supplies its worker manifest, binary name, specialization target directory, build environment, and worker arguments.

`qwen3_5_dense` and `qwen3_5_sparse` are thin model-specific launchers.
Each launcher validates the normal CLI. For MTP with K speculative tokens, it calculates `L = K + 1`.
For Vanilla, DSpark, and DFlash2, it uses `L = 1`.
It configures `SpecializedWorker` to build a const-specialized copy of the same `qwen3_5_dense` or `qwen3_5_sparse` binary.
The `inference-runtime-service` `build.rs` generates compile-time const `L`.
An internal environment marker makes the specialized binary run the model instead of starting another build.
The launcher then replaces itself with that specialized binary.

Each cache-lane count uses `target/qwen3_5_specialized/cache_lanes_L` as its Cargo target directory.
Cargo fingerprints the source, features, target, and active debug or release profile in that directory.
A warm launch checks and reuses the existing artifact.
A cold launch requires the repository source, Cargo, the pinned Rust toolchain, and access to all required build inputs.

The gRPC address defaults to `127.0.0.1:50051`. The HTTP address defaults to `127.0.0.1:8000`.

The executor hibernation timeout defaults to 300 seconds.
After this period without executable model work, the service writes model state to SSD and unloads model resources.
The listeners and runtime requests remain active.
The next executable batch loads weights and state before execution.
`--executor-hibernation-timeout-secs` accepts a positive integer.
`--executor-hibernation-mode` accepts `all` or `selected`. It defaults to `selected`.
Use `all` to write every state entry.

One lifecycle owner stops both listeners in these conditions:

- The runtime stops.
- A listener fails.
- The process receives SIGINT or SIGTERM.

`--num-spec-tokens` requires `--spec-type mtp`.
The service rejects zero, use with another Spec type, or an incomplete Spec checkpoint argument pair.
For a Main-only run, omit both Spec checkpoint arguments and `--num-spec-tokens`.

Qwen uses 32 KiB physical cache pages. Qwen3 and Qwen3.5 default to 256K pages. The Qwen3-14B geometry stores eight
tokens in one physical page.

Its 16-token logical cache block uses 80 pages across 40 layers. Thus, the default holds 3,276 complete blocks. These
blocks contain 52,416 resident tokens in aggregate.

When DSpark or DFlash2 is enabled, the executor adds persistent Spec history pages to the same logical block.
The page count depends on the selected Spec layer and KV geometry.

Qwen3.5 keeps 2,048-token logical blocks to amortize its GDN snapshots. It defaults to 256K shared pages.
MTP step K adds K logical KV cache lanes to the Main lane.
All lanes allocate from the same shared physical-page arena.

At startup, each service derives the page count for one block from the initialized executor. The service rejects
`--num-cache-pages` when one complete block cannot fit.
The service classifies a model-executor initialization failure as an internal startup error.

The service also derives the runtime `context_window` from the Main model's `max_position_embeddings`. Vanilla and MTP
use the Main value. DSpark and DFlash2 subtract the checkpoint-derived proposal count because the block-Spec model
applies RoPE to the Main sampled anchor and the complete proposal block. Startup configuration logs include the
effective `context_window`.

The rejection reports this dynamic minimum.

Recommendation: For performance comparisons, pass `--num-cache-pages` explicitly. This setting controls memory
pressure.

`Qwen3Config` and `Qwen35Config` resolve the queued-request, running-request, and per-batch capacities.
CLI checkpoint arguments remain optional parser inputs.
Configuration validation converts them to one `Vanilla`, `MTP`, `DSpark`, or `DFlash2` model mode.
The validated configuration does not store independent speculative-model options.
`--max-requests` defines both the running request-slot capacity and the per-batch request capacity.
The model service passes this value to the executor, `RuntimeConfig`, and `SchedulerConfig`.
The services default to 32 queued requests and 4 running request slots. Queued requests do not consume executor
request-slot state.

Admission assigns a slot before a request enters the scheduler.
GQA page tables, GDN request state, sampling state, and request-indexed workspaces use the same slot domain.
For Qwen3.5 DSpark or DFlash2, GDN retains one candidate state for every possible accepted proposal prefix in each
slot.
Thus, `--max-requests` also bounds the persistent GDN candidate-state arena.
Buffers, scratch allocations, replay resources, and resident model resources remain reusable.

Qwen3.5 wiring derives this request-local GDN slot count:

```text
decision_candidate_states = match mode {
  Vanilla => 1,
  MTP { num_spec_tokens } => num_spec_tokens + 1,
  DSpark { block_size } => block_size + 1,
  DFlash2 { block_size } => block_size,
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
A DSpark checkpoint with `block_size=15` uses 18 state slots for each request and allocates approximately 10.52 GiB.
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

Completion reasons are `STOP_SEQUENCE`, `LENGTH_LIMIT`, and `CONTEXT_LIMIT`. EOF without a completion event
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

### Request and response identity

Each Chat Completions request may supply an `x-request-id` header with a UUID.
The UUID is a request-correlation value.
The caller may reuse it when the caller replays a request.
The HTTP adapter generates a UUIDv4 when the header is absent.
In this case, a later replay gets a different request UUID.
The adapter returns the selected UUID in each successful response `x-request-id` header.

The request UUID remains in the HTTP protocol layer.
The runtime core and model executor continue to use the process-local numeric `RawRequestID`.
The response body uses an independent `chatcmpl-{response UUID}` ID.
Each new tool call uses an independent UUID as its tool-call ID.
The streaming `tool_calls[].index` value identifies the array position only.
It is not part of the tool-call identity.

Axum invokes the Chat Completions handler for each HTTP request.
The handler invocation owns its tool-call validation state.
The shared `HTTPServer` does not own a cross-request tool-call registry.
The handler reconstructs a tool-call ID set and pending-call map from the submitted conversation history.

The caller stores each tool-call UUID in the assistant message.
The caller uses the same UUID in the applicable tool-result message.
On resume, the caller sends the complete history again.
The new handler invocation scans this history and validates the correlation.
It rejects duplicate assistant tool-call IDs in one submitted history.
It does not compare tool-call IDs from independent HTTP requests.
It does not regenerate historical tool-call IDs.

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

Use this gRPC command for a Qwen3.5/Qwen3.6/Qwen3.8 dense server:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
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
    "model": "qwen3.8-27b",
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
The public MTP lifecycle remains one `submit_spec -> wait -> read_spec` transaction.
The current implementation submits each dependent pass separately and reads its sampled token on the CPU before it
prepares the next pass.
All passes reuse the same recorded replay programs, weights, scratch, and bound workspaces.

A DSpark-enabled batch records DSpark Prefill after the Main CPU read.
DSpark Prefill creates persistent DSpark history K/V from the completed Main capture.
A decode-ready batch then records DSpark Decode in the same Spec submission.
DSpark Decode contains DSparkEmbed, DSpark, DSparkGatherUnembed, and DSparkSampling.

A DFlash2-enabled batch uses the same outer two-stage lifecycle through its separate owner.
DFlash2 Prefill creates persistent DFlash2 history K/V from the completed Main capture.
A decode-ready batch records DFlash2 Decode after Prefill.
DFlash2 Decode contains DFlash2Embed, the DFlash2 body, and DFlash2Output.
The body combines per-row sliding-history attention with bidirectional local-block attention and applies dynamic
grouped convolution.
DFlash2Output builds and samples the candidate lattice and writes sparse draft distributions.

`--logging debug` emits one DEBUG `phase="executor.batch.perf"` event after each non-empty executor batch. The event
uses the same schema for Vanilla, MTP, DSpark, and DFlash2. It also uses the same schema for prefill and decode batches:

```text
component="executor"
phase="executor.batch.perf"
model="qwen3.5"
model_mode="vanilla|mtp|dspark|dflash2"
batch_seq=42
num_reqs=4
num_input_tokens=12
num_spec_tokens=9
num_verified_tokens=6
acceptance_rate=0.6667
num_spec_token_by_index=[4, 3, 2]
num_verified_token_by_index=[3, 2, 1]
acceptance_rate_by_index=[0.7500, 0.6667, 0.5000]
main_ms=14.2500
spec_ms=6.5000
spec_passes=2
```

`num_verified_tokens` is the number of returned `validated_tokens`. It does not include the final `sampled_token`.
`acceptance_rate` is `num_verified_tokens / num_spec_tokens`.

The per-index rate is conditional. Index `i` is eligible only when the request contains speculative token `i` and
all earlier speculative tokens passed verification. `num_spec_token_by_index[i]` is this eligible count.
`num_verified_token_by_index[i]` is the verified count at the same index. The helper scripts sum these counts across
batches before they calculate `acceptance_rate_by_index`.

`main_ms` covers `MainEmbed -> Main -> GatherUnembed -> Sampling/RejectionSampling -> submit/wait -> read`.
`spec_ms` covers the complete MTP, DSpark, or DFlash2 Spec lifecycle.
It includes all dependent MTP passes and any recorded block-Spec Prefill and Decode work.
DSpark or DFlash2 prefill-only batches can have nonzero `spec_ms` and zero `spec_passes`.
`spec_passes` counts Spec Decode forwards. These values are host elapsed latencies. They are not GPU kernel timings.

`--logging info` does not emit the executor batch performance event. `--logging debug` also emits request and response
diagnostics. The end-to-end performance helpers enable only the `inference-runtime-service::perf` DEBUG target when
the selected server logging level is INFO.

Internal model `Start` and `Stop` commands emit INFO lifecycle events on the
`inference-runtime-service::lifecycle` target.
The events use `component="model"` with `start.begin`, `start.complete`, `stop.begin`, and `stop.complete` phases.
Completion events include elapsed milliseconds.
Failure events include the model name and error, and then trigger global shutdown.
Runtime core issues these commands after the configured executor hibernation timeout.

Executor profiling spans use the lifecycle hook names:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> read_main

embed_spec -> forward_spec -> unembed_spec -> sample_spec    MTP

prefill_spec -> decode_spec                                DSpark or DFlash2
submit_spec -> read_spec
```

A pass is one proposal forward that runs.
For Qwen3.5, each logical MTP step, DSpark Decode block, or DFlash2 Decode block is one pass.
For Qwen3, each DSpark block forward is one pass.
Qwen3 runs DSpark Prefill for a prefill-only batch.
It does not run DSpark Decode because no sampled anchor exists.

The runtime emits non-empty periodical scheduler stats every 30 seconds. It resets these stats after each output.
Runtime shutdown always emits separate lifetime scheduler stats. It does not reset the lifetime stats.

Each scheduler stats output contains two tables. The scheduler API table contains enqueue and swap-in counts. It also
contains prepare, cancel, and commit counts and latency percentiles. The Speculative acceptance table uses proposal
indexes as columns:

```text
spec stat | overall | index@0 | index@1 | ...
proposed  | ...
accepted  | ...
rate      | ...
```

The `overall` column sums the proposed and accepted counts across all indexes. Each remaining column reports one
proposal index. `proposed` counts each proposal position that the Spec forward produced. `accepted` counts the
responses whose `validated_tokens` include that position. `rate` is `accepted / proposed`. A column with no proposals
has rate `N/A`.

This rate differs from the conditional `acceptance_rate_by_index` in the executor batch performance event. The
scheduler denominator includes every produced proposal at the index. The executor event includes an index only when
all earlier proposal tokens passed verification.

Long-running service and runtime components use these spans:

```text
runtime
replayable-executor
async-task-pool
s3-fifo
grpc-server
http-server
```

Each component emits one INFO `started` event and one INFO `stopped` event in its span. Normal shutdown receipt does not
emit a separate INFO event. The `runtime` span belongs to the runtime-core event loop in
`runtime/scheduler/event_loop.rs`; it is not a second service-level wrapper.

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
The `14b_dspark` case uses the checkpoint `block_size` as its proposal count.
If the DSpark checkpoint is absent and no download repository is configured, the helper prints a warning and skips
that case. A missing Main checkpoint remains an error.

The Qwen3.5/3.6/3.8 helper runs controlled 27B/35B Main-only, MTP, DSpark, and DFlash2 comparisons:

```sh
scripts/qwen35_e2e_decode_perf.sh \
  --tokenizer <tokenizer-model-dir> \
  --model-27b <27b-model-dir> \
  --mtp-27b <27b-mtp-model-dir> \
  --dspark-27b <27b-affine-dspark-model-dir> \
  --dflash2-27b <27b-affine-dflash2-model-dir> \
  --model-35b <35b-model-dir> \
  --mtp-35b <35b-mtp-model-dir> \
  --dspark-35b <35b-affine-dspark-model-dir> \
  --dflash2-35b <35b-affine-dflash2-model-dir> \
  --runs 7
```

Use `27b_mtp1`, `27b_mtp2`, `35b_mtp1`, or `35b_mtp2` to select an MTP proposal count.
The `*_mtp` aliases run MTP proposal counts 1 and 2.
The DSpark and DFlash2 cases use checkpoint-defined block geometry.
The default case matrix uses this order:

1. `27b_off`
2. `35b_off`
3. `27b_mtp1`
4. `35b_mtp1`
5. `27b_dspark`
6. `35b_dspark`
7. `27b_dflash2`
8. `35b_dflash2`
9. `27b_mtp2`
10. `35b_mtp2`

The default 27B Main and MTP checkpoints use Qwen3.8. The default 35B Main and MTP checkpoints use Qwen3.6.
Each case uses its Main checkpoint for tokenization by default. Use `--tokenizer` to override all cases.
The default `representative2` workload contains one fixed GSM8K prompt and the original Beijing travel prompt.
The helper repeats and summarizes each prompt independently. The default output contains a configuration table,
measurement progress, and a compact result table. The result table contains case, prompt, output limit, sampled tokens,
decode throughput, tokens per chunk, and verified/proposed speculative tokens. Use `--show-runs` to also print
machine-readable `CONFIG`, `RUN`, and `SUMMARY` rows. Each `RUN` and `SUMMARY` row contains the stable prompt ID. The
configuration output contains the prompt-set name, prompt IDs, prompt count, and prompt-set SHA-256.
Use `--prompt <text>` to replace the set with one custom prompt.
If an MTP, DSpark, or DFlash2 checkpoint is absent and no download repository is configured, the helper prints a
warning and skips that case. A missing Main checkpoint remains an error.
Before it starts a DFlash2 case, the helper validates the affine config and safetensors headers.
It rejects a checkpoint that contains a BF16 matrix.
The helper stops the server after each explicit case.
It applies the configured cooldown between runnable cases. The default cooldown is 8 seconds.
Each MTP summary label includes its proposal count, for example, `27b_mtp2`.
DSpark and DFlash2 summary labels identify only the model and mode because their block geometry comes from the
checkpoint.

Both helpers record these facts:

- Commit and dirty state
- Model directories
- Machine and operating system
- Cache and request capacity
- Scheduler capacities
- Sampling configuration and seed
- Prompt identity
- Trajectory fields

Both helpers also record cooldown and speculative-acceptance fields.

The Qwen3.5/3.6/3.8 helper contains one observed reference run for its exact `representative2` workload. This reference
is not a pass/fail threshold. The helper reports a throughput delta only when the machine, OS, architecture, clean
state, model names, prompt hash, sampling configuration, scheduler capacities, and cooldown match. The run must contain
at least as many samples as the reference. The input-token, sampled-token, chunk, proposal, verified-token, and
conditional-acceptance trajectory must also match. A `SUMMARY` row reports the mismatch instead of a delta when one of
these conditions does not match. The final output reports the reference-status counts. Use `--show-runs` for each
reference delta and mismatch. Use `--no-reference` to disable this comparison.

The Qwen3 helper does not contain a checked-in reference run. Record its comparison results outside the script with the
complete provenance required by [`executor_benchmarks.md`](executor_benchmarks.md).

The current default uses four running requests.

Summaries report these metrics:

- Decode throughput
- TTFT and prompt throughput
- RPC inter-chunk p50 and p95
- Tokens for each chunk
- Exact verified/proposed speculative-token rate for the matching server-log interval
- Conditional speculative-token acceptance rate at each index

For Main-only decoding, a chunk contains one token. Thus, inter-chunk time is inter-token latency.

With MTP, inter-chunk time measures burst cadence. Interpret it with tokens for each chunk and the acceptance rate.

Use `--case-cooldown-secs 0` only for an intentional sustained-load experiment.

Follow [`executor_benchmarks.md`](executor_benchmarks.md) before you make a performance claim.
