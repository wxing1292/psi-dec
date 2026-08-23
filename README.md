# psi-dec

`psi-dec` is a production-quality Qwen inference engine for Apple Silicon.
It combines a model-agnostic Rust runtime, a Qwen executor, and a Metal replay backend.

## Architecture

```text
HTTP Chat Completions ──► Qwen codec ──► token IDs ──┐
                                                     ├─► Inference::decode
gRPC Decode ────────────────────────► token IDs ─────┘          │
                                                                ▼
                   Runtime core ──► Qwen executor ──► Metal backend
                        ▲                                    │
                        └──────── tokens + lifecycle ────────┘
```

Each layer has a separate owner:

- **Runtime core:** Owns scheduling, request lifecycle, and cache ownership.
- **Qwen executor:** Owns model layout, components, sampling, speculative model roles, and replay order.
- **Metal backend:** Owns devices, buffers, kernels, recording, and ICB submission.

## Quick start

You need these items:

- Apple Silicon Mac
- Rust toolchain
- Xcode command-line tools
- Hugging Face CLI with access to the model

Download the Qwen3.8 dense and Qwen3.6 sparse Main and MTP checkpoints:

```sh
hf auth login

hf download mlx-community/Qwen3.8-27B-4bit \
  --local-dir models/Qwen3.8-27B-4bit

hf download mlx-community/Qwen3.8-27B-MTP-4bit \
  --local-dir models/Qwen3.8-27B-MTP-4bit

hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir models/Qwen3.6-35B-A3B-4bit

hf download mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --local-dir models/Qwen3.6-35B-A3B-MTP-4bit
```

The `qwen3_5_*` binaries support compatible Qwen3.5, Qwen3.6, and Qwen3.8 checkpoints.

Start the dense 27B service with MTP:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --num-spec-tokens 1
```

Start the sparse 35B-A3B service with MTP:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --num-spec-tokens 1
```

The service unloads model state and weights after 300 seconds without executable model work.
The next request reloads them automatically.

Stream an HTTP Chat Completions response:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
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

The [service guide](docs/service.md) also covers Main-only startup, gRPC, tool calls, and the HTTP API.

## Workspace map

```text
inference-runtime-core      scheduling, lifecycle, and cache ownership
inference-runtime-service   inference API, RPC, codecs, and server binaries
inference-executor-core     backend-neutral model/component contracts
inference-executor-metal    Qwen execution, replay, sampling, MTP, DSpark, and DFlash2
inference-backend-metal     Metal resources, kernels, and ICB runtime
```

All paths above live under `crates/`.

## Documentation

- [Service](docs/service.md): Setup, APIs, operations, and end-to-end checks.
- [Runtime core](docs/core.md): Scheduling, request lifecycle, and cache ownership.
- [Executor](docs/executor.md): Qwen execution and component composition.
- [Qwen executor](docs/executor_qwen.md): Vanilla, MTP, DSpark, and DFlash2 ownership and replay architecture.
- [DSpark](docs/dspark_design.md): Fixed-block attention, Markov sampling, state, and lifecycle.
- [DFlash2](docs/dflash2_design.md): Persistent history, sliding attention, convolution, selection, and lifecycle.
- [Metal backend](crates/inference-backend-metal/README.md): Resources, kernels, and replay.
- [Verification](docs/executor_benchmarks.md): Correctness, benchmarks, profiling, and performance evidence.
- [Documentation index](docs/README.md): Component guides, engineering rules, and current work.

## Acknowledgements

`psi-dec` is an independent Rust and Metal implementation inspired by these projects:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

The project uses [MLX](https://github.com/ml-explore/mlx) by Apple.
The build embeds downloaded MLX Metal kernel headers and retains their MIT notice in [`NOTICE`](NOTICE).
The [Qwen team](https://qwen.ai/) develops the supported models.
Model weights are separate artifacts with their own terms.

## License

`psi-dec` is distributed under the [MIT License](LICENSE).
See [NOTICE](NOTICE) for the retained MLX header attribution.
