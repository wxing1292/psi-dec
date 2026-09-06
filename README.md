# psi-dec

`psi-dec` is a production-quality Qwen inference engine for Apple Silicon.
It combines a model-agnostic Rust runtime, a Qwen executor, and a Metal replay backend.
It supports text generation and Qwen3-ASR audio transcription.

## Architecture

```text
HTTP Chat Completions ──► Qwen codec ─────────► token IDs ───┐
HTTP Transcriptions ────► Qwen3-ASR processor ─► resources ──┼─► Inference::decode
gRPC Decode ───────────────────────────────────► token IDs ──┘          │
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

### DSpark and DFlash2 checkpoints

Download the BF16 speculative checkpoints from
[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2):

```sh
hf download RadixArk/Qwen3.8-27B-DSpark \
  --local-dir models/Qwen3.8-27B-DSpark

hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir models/Qwen3.8-27B-DFlash2
```

Convert the source checkpoints to the psi-dec affine format:

```sh
cargo run --release -p inference-executor-core --bin qwen3x_spec_quantize -- dspark \
  --input-dir "$PWD/models/Qwen3.8-27B-DSpark" \
  --output-dir "$PWD/models/Qwen3.8-27B-DSpark-affine"

cargo run --release -p inference-executor-core --bin qwen3x_spec_quantize -- dflash2 \
  --input-dir "$PWD/models/Qwen3.8-27B-DFlash2" \
  --output-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine"
```

Each output directory must not exist before conversion.
Both commands use group size 64 and 4-bit matrix payloads by default.
DSpark uses 8 bits for `markov_head.markov_w2`.
DFlash2 uses 6 bits for the selected layer 2 and layer 4 projections.
Both formats store affine scales and biases as BF16.
See the [service guide](docs/service.md) for the full conversion and startup contracts.

Start the dense 27B service with MTP:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 1
```

Start the dense 27B service with DSpark:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-DSpark-affine" \
  --spec-type dspark
```

Start the dense 27B service with DFlash2:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine" \
  --spec-type dflash2
```

Start the sparse 35B-A3B service with MTP:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --spec-type mtp \
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

The [service guide](docs/service.md) also covers Main-only startup, Qwen3-ASR, gRPC, tool calls, and the HTTP APIs.

## Reference performance

Decode throughput in tokens/s at an output limit of 1024 tokens, from the reference supplied on 2026-09-05.
The reference profile is Apple M3 Max (40 GPU cores), macOS 27.0, arm64.
`27B` is Qwen3.8-27B. `35B` is Qwen3.6-35B-A3B.

| Mode      | 27B GSM8K | 27B Chat | 35B GSM8K | 35B Chat |
| --------- | --------: | -------: | --------: | -------: |
| Vanilla   |    22.945 |   22.843 |    95.438 |   95.845 |
| MTP=1     |    39.712 |   35.782 |   148.661 |  135.356 |
| MTP=2     |    44.159 |   30.340 |   154.331 |  122.635 |
| MTP=3     |    38.845 |   24.564 |   162.583 |  119.719 |
| MTP=4     |    34.226 |   21.043 |   149.558 |  105.510 |
| DSpark=7  |    47.824 |   17.848 |         — |        — |
| DFlash2=7 |    58.082 |   19.250 |         — |        — |

Output lengths and acceptance trajectories vary by case. These results are not performance thresholds.
The producing commit, dirty state, run count, and variance were not supplied. A dash means no reference result.
See the [perf script](scripts/qwen35_e2e_decode_perf.sh) for all 48 rows, output counts, and acceptance metrics.
See the [performance helper guide](docs/service.md#end-to-end-performance-helper) for commands, configuration, and comparison limits.

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
- [Pi provider](agent-plugins/pi/README.md): Install and configure the resident-session Pi extension.
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
