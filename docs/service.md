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

The service provides a target-only Qwen3 binary. It also provides Qwen3.5 binaries that retain their names for current
Qwen3.6 MLX checkpoints:

| Model                  | Binary           | Target checkpoint                    | Optional speculator checkpoint           |
| ---------------------- | ---------------- | ------------------------------------ | ---------------------------------------- |
| Qwen3 dense 14B        | `qwen3`          | `mlx-community/Qwen3-14B-4bit`       | None                                     |
| Qwen3.6 dense 27B      | `qwen3_5_dense`  | `mlx-community/Qwen3.6-27B-4bit`     | `mlx-community/Qwen3.6-27B-MTP-4bit`     |
| Qwen3.6 sparse 35B-A3B | `qwen3_5_sparse` | `mlx-community/Qwen3.6-35B-A3B-4bit` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` |

Download with the Hugging Face CLI:

```sh
hf auth login
hf download mlx-community/Qwen3-14B-4bit --local-dir models/Qwen3-14B-4bit
hf download mlx-community/Qwen3.6-27B-4bit --local-dir models/Qwen3.6-27B-4bit
hf download mlx-community/Qwen3.6-27B-MTP-4bit --local-dir models/Qwen3.6-27B-MTP-4bit
```

Use the corresponding 35B-A3B names for the sparse model. MTP checkpoints contain drafter weights. They must match the
target family.

### DSpark conversion tool

The repository retains the low-level DSpark checkpoint converter and component contracts:

```sh
cargo run -p inference-executor-core --bin qwen3_dspark_quantize -- \
  --input-dir /path/to/DSpark-Qwen3.6-27B-AEON-draft \
  --output-dir /path/to/DSpark-Qwen3.6-27B-AEON-draft-psi-dec \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not exist before you run the converter.

The service does not connect DSpark at this commit. It has no `--hf-dspark-model-dir` option. The current server cannot
select converted weights.

The repository provides the converter and foundation tests for later executor integration.

Qwen3 target-only startup:

```sh
cargo run --release --bin qwen3 -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit"
```

Qwen3 does not provide MTP or DSpark options. Its executor gets stop tokens from the checkpoint configuration when
`generation_config.json` is absent.

Qwen3.5 startup with MTP enabled:

Dense:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-27B-MTP-4bit" \
  --mtp-module 1
```

Sparse:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --mtp-module 1
```

The gRPC address defaults to `127.0.0.1:50051`. The HTTP address defaults to `127.0.0.1:8000`.

One lifecycle owner stops both listeners in these conditions:

- The runtime stops.
- A listener fails.
- The process receives SIGINT or SIGTERM.

`--mtp-module` accepts `0` or `1`. It defaults to `1` when `--hf-mtp-model-dir` is present. Otherwise, it defaults to
`0`.

Value `1` requires that directory. Explicit `--mtp-module 0` ignores an optional MTP directory. Use this value for
controlled target-only tests.

Qwen uses 32 KiB physical cache pages. Qwen3 and Qwen3.5 default to 384K pages. The Qwen3-14B geometry stores eight
tokens in one physical page.

Its 16-token logical cache block uses 80 pages across 40 layers. Thus, the default holds 4,915 complete blocks. These
blocks contain 78,640 resident tokens in aggregate.

Qwen3.5 keeps 2,048-token logical blocks to amortize its GDN snapshots. It defaults to 384K shared pages.

At startup, each service derives the page count for one block from the initialized executor. The service rejects
`--num-cache-pages` when one complete block cannot fit.

The rejection reports this dynamic minimum.

Recommendation: For performance comparisons, pass `--num-cache-pages` explicitly. This setting controls memory
pressure.

The services default to 32 queued requests and 8 running request slots. Queued requests do not consume executor
request-slot state.

Admission assigns a slot before a request enters the scheduler. `--max-requests` cannot exceed the eight executor slots.

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

The client model directory supplies tokenizer and chat-template files. The server controls the target model.

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
It does not select a different target.

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

Speculative target verification replaces Sampling with RejectionSampling.

An MTP proposal uses a separate submission with MTPEmbed, MTP, GatherUnembed, and DraftSampling. Rejection decisions
cross the CPU boundary before the next proposal input.

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

A pass is one auxiliary speculator forward that runs. For Qwen3.5, each MTP module forward is one pass. This includes
prefill forwards for cache maintenance.

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

Validate the dense and sparse targets. If MTP paths changed, validate MTP off and on.

Record these facts:

- Prompt tokens and sampled tokens
- Termination reason and output sanity
- Commit and dirty state
- Model directory

Investigate cold-start stalls separately from later decode stalls.

## End-to-end performance helper

The Qwen3 helper runs target-only decode measurements:

```sh
PSI_DEC_QWEN3_MODEL_DIR=<qwen3-model-dir> \
scripts/qwen3_e2e_decode_perf.sh --runs 7
```

The helper uses the model directory for tokenization by default.
Set `PSI_DEC_QWEN3_TOKENIZER_DIR` or use `--tokenizer` to select a different directory.
Use `--tokens` to select the comma-separated output-token counts.

The Qwen3.5/3.6 helper runs controlled 27B/35B, MTP-off/on comparisons:

```sh
PSI_DEC_QWEN_TOKENIZER_DIR=<tokenizer-model-dir> \
PSI_DEC_QWEN_27B_MODEL_DIR=<27b-model-dir> \
PSI_DEC_QWEN_27B_MTP_DIR=<27b-mtp-model-dir> \
PSI_DEC_QWEN_35B_MODEL_DIR=<35b-model-dir> \
PSI_DEC_QWEN_35B_MTP_DIR=<35b-mtp-model-dir> \
scripts/qwen35_e2e_decode_perf.sh --runs 7
```

Both helpers print these facts:

- Commit and dirty state
- Model directories
- Machine and operating system
- Cache and runtime request capacity
- Scheduler capacities
- Sampling configuration and seed
- Trajectory fields

The Qwen3.5/3.6 helper also prints cooldown and speculative-acceptance fields.

The Qwen3 helper does not contain a checked-in performance baseline.

The Qwen3.5/3.6 helper contains an M3 Max baseline.
The baseline was recorded on 2026-07-21 at `132c5073`.
It used these settings:

- 384K pages
- 2048-token cache blocks
- Four running requests
- The 4/128/64 scheduler configuration

The current default uses eight running requests. Thus, this baseline is not comparable until it is refreshed.

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

For target-only decoding, a chunk contains one token. Thus, inter-chunk time is inter-token latency.

With MTP, inter-chunk time measures burst cadence. Interpret it with tokens for each chunk and the acceptance rate.

A positive decode delta is faster. A positive TTFT or inter-chunk latency delta is slower.

Use `--case-cooldown-secs 0` only for an intentional sustained-load experiment.

Follow [`executor_benchmarks.md`](executor_benchmarks.md) before you make a performance claim.
