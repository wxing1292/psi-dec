# psi-dec

`psi-dec` runs Qwen decoder models on Apple Silicon. It combines a
model-agnostic Rust runtime, a Qwen executor, and a Metal replay backend.

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

The layers have separate owners:

- **Runtime core:** scheduling, request lifecycle, and cache ownership.
- **Qwen executor:** model layout, components, sampling, MTP, and replay order.
- **Metal backend:** devices, buffers, kernels, recording, and ICB submission.

HTTP accepts messages and tools. The checkpoint chat template and tokenizer
lower them into the same token-level decode API used by gRPC. The runtime owns
when work runs and how cache pages live; the executor owns what Qwen computes;
Metal owns how that computation runs on the GPU.

## Quick start

Requirements:

- Apple Silicon Mac
- Rust toolchain
- Xcode command-line tools
- Hugging Face CLI with access to the model

Download the dense checkpoint:

```sh
hf auth login
hf download mlx-community/Qwen3.6-27B-4bit \
  --local-dir models/Qwen3.6-27B-4bit
```

Start gRPC and HTTP listeners:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --mtp-module 0
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

This minimal path disables MTP. See the [service guide](docs/service.md) for
MTP, sparse-model commands, gRPC decode, tool calls, and the supported
OpenAI-compatible subset.

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

- Run or test the server: [service guide](docs/service.md).
- Understand request and cache lifecycle: [runtime core](docs/core.md).
- Follow Qwen execution: [executor architecture](docs/executor.md).
- Inspect replay and kernels:
  [Metal backend](crates/inference-backend-metal/README.md).
- Benchmark or profile changes:
  [executor verification](docs/executor_benchmarks.md).
- Apply engineering rules: [high-level guidance](docs/high_level.md).
- Review active follow-up work: [future work](docs/future_work.md).

The [documentation index](docs/README.md) links every component guide.

## Acknowledgements

`psi-dec` is an independent Rust and Metal implementation inspired by the
open-source work of [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[mistral.rs](https://github.com/EricLBuehler/mistral.rs); credit is due to their
authors and contributor communities. It uses [MLX](https://github.com/ml-explore/mlx)
by Apple; its Metal kernel headers are downloaded and embedded at build time.
The MIT notice is retained in [`NOTICE`](NOTICE). Supported Qwen models are
developed by the [Qwen team](https://qwen.ai/); model weights are separate
artifacts governed by their own terms.

## License

`psi-dec` is distributed under the [MIT License](LICENSE). See
[NOTICE](NOTICE) for the MLX header attribution retained with this project.
