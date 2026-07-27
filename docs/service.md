# Qwen Service and Decode Client

This document owns model download, server/client commands, operational logging, and the checked-in end-to-end perf
workflow. Runtime internals are in [`core.md`](core.md); Qwen executor stages are in
[`executor_qwen.md`](executor_qwen.md); benchmark methodology is in
[`executor_benchmarks.md`](executor_benchmarks.md).

## Tool state

`crates/inference-runtime-service/src/tool/` owns this transport- and
model-neutral domain. It does not own conversation storage, tool handlers,
execution scheduling, prompt rendering, or model-specific tool syntax.

`ToolState` is the single lifecycle owner for the state derived from one
conversation's ordered tool events:

```text
ToolState
├── tools: currently callable ToolDefinition values
└── executions: HashMap<ToolCallID, ToolCallRequest>
```

It does not retain completed calls or registration, unregistration, response,
or cancellation events; the persistent conversation history owns those.
`ToolState::fold` reconstructs the two derived collections by applying history
in order.

`ToolRegistration` and `ToolUnregistration` are non-empty, duplicate-free
batches. `register_tool` and `unregister_tool` are atomic: an operation either
updates all definitions or leaves the state unchanged. Definition order
follows conversation-history order so a later prompt adapter can render tools
deterministically. Removing a tool prevents new calls and permits a later
registration of the same `ToolID`.

A `ToolCallRequest` has the following transport-neutral shape:

```text
ToolCallRequest
├── tool_id: ToolID
├── tool_call_id: ToolCallID
└── arguments: ToolArguments
```

`ToolState` is a correlation ledger, not a tool executor. `request_execution`
requires that `tool_id` is currently registered and that `tool_call_id` is not
already in flight. Multiple requests may remain in flight. `respond_execution`
and `cancel_execution` address the execution only by `ToolCallID`; neither
consults the currently callable definitions. Consequently, unregistration does
not cancel an accepted call, and its response remains valid. Completed and
cancelled entries leave the in-flight ledger immediately. Execution iteration
has no ordering contract.

A `ToolCallResponse` carries:

```text
ToolCallResponse
├── tool_call_id: ToolCallID
├── raw_content: Vec<ToolRawContent>
├── structured_content: Option<ToolStructuredContent>
└── is_error: bool
```

`ToolRawContent` is model-facing content and currently supports text.
`ToolStructuredContent` is an arbitrary JSON value for programmatic consumers;
both forms may coexist. A tool-level failure sets `is_error` and includes a
safe, non-empty model-facing error message. Service/protocol failures remain
ordinary `Err(Error)` values rather than fake tool responses.

The event projection is:

```text
Registration     tools + definitions    executions unchanged
Unregistration   tools - definitions    executions unchanged
CallRequest      require active tool    executions + request
CallResponse     tools ignored          executions - request
CallCancellation tools ignored          executions - request
```

The persistent per-conversation history is the authority and its position is
the natural version. `ToolState` is only a derived in-memory projection; it is
not a history store or recoverable cache. Calls emitted in one model response
are independently in flight and may be executed concurrently. Approval,
permissions, side-effect serialization, concurrency limits, handler
cancellation, and argument-schema validation belong to the agent/tool
environment, not the inference server.

## Binaries and checkpoints

The implementation module retains Qwen3.5 names while consuming the current Qwen3.6 MLX checkpoint layout.

| Model | Server binary | Target checkpoint | Optional MTP checkpoint |
| --- | --- | --- | --- |
| 27B dense | `qwen3_5_dense` | `mlx-community/Qwen3.6-27B-4bit` | `mlx-community/Qwen3.6-27B-MTP-4bit` |
| 35B-A3B sparse | `qwen3_5_sparse` | `mlx-community/Qwen3.6-35B-A3B-4bit` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` |

Download with the Hugging Face CLI:

```sh
hf auth login
hf download mlx-community/Qwen3.6-27B-4bit --local-dir models/Qwen3.6-27B-4bit
hf download mlx-community/Qwen3.6-27B-MTP-4bit --local-dir models/Qwen3.6-27B-MTP-4bit
```

Use the corresponding 35B-A3B names for the sparse model. MTP checkpoints are drafter weights and must match the target
family.

### Retained DSpark conversion tool

The repository retains the low-level DSpark checkpoint converter and component contracts:

```sh
cargo run -p inference-executor-core --bin qwen35_dspark_quantize -- \
  --input-dir /path/to/DSpark-Qwen3.6-27B-AEON-draft \
  --output-dir /path/to/DSpark-Qwen3.6-27B-AEON-draft-psi-dec \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not already exist. Qwen3.5 service wiring for DSpark is intentionally absent: there is no
`--hf-dspark-model-dir` option and converted weights cannot be selected by the current Qwen3.5 server. The retained
converter and component tests are for future integration work.

## Start a server

Use the MTP-enabled service path for general decoding:

Dense:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-27B-MTP-4bit" \
  --mtp-module 1
```

Sparse:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --mtp-module 1
```

`--mtp-module 0` is useful for controlled target-only tests. Run one GPU service at a time. Qwen uses
32 KiB physical cache pages and defaults to 384K shared pages; perf comparisons should pass `--num-cache-pages`
explicitly so memory pressure is controlled.

The Qwen service defaults to 32 queued requests and 8 running request slots.
Queued requests do not consume executor request-slot state. Admission assigns a
slot before entering the scheduler; the default scheduler remains limited to 4
requests, 128 flattened tokens, and 64 tokens per request in one batch.

## Decode client

```sh
cargo run --release -p inference-runtime-service --bin decode -- \
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

The client model directory supplies tokenizer and chat-template files; the server controls the target model. Use
`--chat-template qwen-fixed --disable-thinking` when a deterministic non-thinking Qwen prompt is required. That mode
emits the checkpoint-canonical closed empty reasoning block before generation.

## HTTP checkpoint

The HTTP listener binds alongside gRPC and exposes OpenAI-compatible
`POST /v1/chat/completions` parsing scaffolding. Omitting `stream` or setting it
to `false` selects a collected Chat Completion response; setting it to `true`
selects Chat Completion chunks over SSE. The handler stops at explicit
unavailable preprocessing/postprocessing boundaries and is not operational yet.
It does not run Qwen chat-template or tool-call logic inside RPC. The remaining
work is tracked in [`future_work.md`](future_work.md).

## Execution and logging

Normal sampling submits MainEmbed, Main, GatherUnembed, and Sampling replay programs in one ordered Metal command
buffer. Speculative target verification submits MainEmbed, Main, GatherUnembed, and RejectionSampling in one ordered
command buffer. MTP proposal composes MTPEmbed, MTP, GatherUnembed, and DraftSampling in a separate submission because
rejection decisions cross the CPU boundary before the next proposal input is formed.

`--logging info` emits one concise batch event: model, batch sequence, request/input counts, speculative input, accepted
speculative tokens, committed output tokens, acceptance rate, and total latency. `--logging debug` uses the same event
model and adds request-kind counts, rejected/next speculative tokens, sampled rows, and replay-stage submit/wait timing.
It does not duplicate an INFO event. Runtime shutdown also emits a scheduler table containing call counts and latency
percentiles over the runtime lifetime. Enqueue and swap-in report request counts; prepare, cancel, and commit report
both counts and latency percentiles.

Set `PSI_QWEN35_STATE_TRACE=1` for opt-in executor lifecycle lines on stderr. These include replay cache hit/miss keys,
GDN restore/publish decisions, and synchronous `prepare_sync` timing (`gqa_us`, `gdn_states_us`, dependent
`gdn_metadata_us`, and total `wall_us`). Set `PSI_GDN_STATE_TRACE=1` only for the more detailed
GDN request-state transition trace. Both are diagnostic modes; leave them unset for throughput measurements.

`--profile component` and `--profile operation` currently enable the same coarse CPU tree over prepare, model
input/forward/output, and commit. They do not attribute GPU time to components or kernels; use Metal capture/counters for
that.

## Correctness and long decode

For a release correctness check, use deterministic sampling (`--temperature 0 --top-k 1 --top-p 1 --seed 1`) and a
prompt with an objective oracle before running a long 8K-token generation. Validate both dense and sparse targets, MTP
off and on when those paths changed. Record prompt tokens, sampled tokens, termination reason, output sanity, commit,
dirty state, and model directory.

First decode includes model initialization, Metal pipeline compilation, replay construction, and cache warmup. Treat it
as cold-start evidence, not steady-state token latency. Investigate cold-start separately from second/subsequent decode
stalls.

## End-to-end performance helper

The checked-in helper runs controlled 27B/35B, MTP-off/on comparisons:

```sh
PSI_DEC_QWEN_TOKENIZER_DIR=<tokenizer-model-dir> \
PSI_DEC_QWEN_27B_MODEL_DIR=<27b-model-dir> \
PSI_DEC_QWEN_27B_MTP_DIR=<27b-mtp-model-dir> \
PSI_DEC_QWEN_35B_MODEL_DIR=<35b-model-dir> \
PSI_DEC_QWEN_35B_MTP_DIR=<35b-mtp-model-dir> \
scripts/qwen35_e2e_decode_perf.sh --runs 7
```

It prints commit, dirty state, model directories, machine/OS, cache, runtime running-request capacity, scheduler
capacities, cooldown, seed, and trajectory fields. The checked-in M3 Max baseline was recorded on 2026-07-21 at
`132c5073` with the 384K-page, 2048-token cache-block, four-running-request, and 4/128/64 scheduler configuration.
The current eight-running-request default intentionally makes that baseline non-comparable until it is refreshed.
A summary reports `baseline_status=comparable` and typed
decode/TTFT/inter-chunk delta percentages only when machine, OS, checkpoint directory names, prompt/sampling config,
capacities, cooldown, and sampled trajectory match. Baseline throughput and trajectory are keyed by machine, case, and
token count; another hardware
baseline can therefore be added without replacing the M3 Max values. Config or trajectory mismatches remain visible
but produce no performance delta. Summaries report decode throughput, TTFT/prompt throughput, RPC inter-chunk p50/p95,
tokens per chunk, and exact accepted/proposed speculative-token rate from the matching server-log interval. For
target-only decoding a chunk contains one token, so inter-chunk is inter-token latency; under MTP it measures burst
cadence and must be interpreted together with tokens per chunk and acceptance rate. Positive decode delta is faster,
while positive TTFT or inter-chunk latency delta is slower. Use
`--case-cooldown-secs 0` only for an intentional sustained-load experiment.

Follow [`executor_benchmarks.md`](executor_benchmarks.md) before making a performance claim.
