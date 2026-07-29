# psi-dec

`psi-dec` runs Qwen decoder models on Apple Silicon.
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
- **Qwen executor:** Owns model layout, components, sampling, MTP, and replay order.
- **Metal backend:** Owns devices, buffers, kernels, recording, and ICB submission.

HTTP accepts messages and tools.
The checkpoint chat template and tokenizer convert them to token IDs.
The HTTP and gRPC paths use the same token-level decode API.
The runtime owns scheduling and cache page lifecycles.
The executor owns Qwen model computation.
The Metal backend owns GPU execution.

## Quick start

You need these items:

- Apple Silicon Mac
- Rust toolchain
- Xcode command-line tools
- Hugging Face CLI with access to the model

Download the matching Main and MTP checkpoints:

```sh
hf auth login
hf download mlx-community/Qwen3.6-27B-4bit \
  --local-dir models/Qwen3.6-27B-4bit

hf download mlx-community/Qwen3.6-27B-MTP-4bit \
  --local-dir models/Qwen3.6-27B-MTP-4bit
```

Start gRPC and HTTP listeners with MTP enabled:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-27B-MTP-4bit" \
  --mtp-module 1
```

Stream an HTTP Chat Completions response:

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

The [service guide](docs/service.md) includes the sparse 35B-A3B command, Main-only startup, gRPC decode, tool calls, and the supported OpenAI-compatible subset.

## Workspace map

```text
inference-runtime-core      scheduling, lifecycle, and cache ownership
inference-runtime-service   inference API, RPC, codecs, and server binaries
inference-executor-core     backend-neutral model/component contracts
inference-executor-metal    Qwen execution, replay, sampling, and MTP
inference-backend-metal     Metal resources, kernels, and ICB runtime
```

All paths above live under `crates/`.

## Documentation

- Use the [service guide](docs/service.md) to run or test the server.
- Use the [runtime core guide](docs/core.md) to understand request and cache lifecycles.
- Use the [executor architecture](docs/executor.md) to follow Qwen execution.
- Use the [Metal backend guide](crates/inference-backend-metal/README.md) to inspect replay and kernels.
- Use [executor verification](docs/executor_benchmarks.md) to benchmark or profile changes.
- Apply the rules in [high-level guidance](docs/high_level.md).
- Apply the [technical English guide](docs/technical_english.md) when you write documentation.
- Review active work in [future work](docs/future_work.md).

The [documentation index](docs/README.md) links all current component guides.

## Acknowledgements

`psi-dec` is an independent Rust and Metal implementation.
These open-source projects inspired its design:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

Credit goes to their authors and contributor communities.
The project uses [MLX](https://github.com/ml-explore/mlx) by Apple.
The build process downloads and embeds the MLX Metal kernel headers.
The project retains the MIT notice in [`NOTICE`](NOTICE).
The [Qwen team](https://qwen.ai/) develops the supported Qwen models.
Model weights are separate artifacts, and their own terms apply.

## License

`psi-dec` is distributed under the [MIT License](LICENSE).
See [NOTICE](NOTICE) for the retained MLX header attribution.
