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

The [service guide](docs/service.md) also covers Main-only startup, gRPC, tool calls, and the HTTP API.

## Reference performance

The following results are observed medians from the checked-in Qwen decode reference.
They are not performance thresholds.
The run used clean commit `43d55f4e5df2edf9b29d1bba15dca31007f76c72` on an Apple M3 Max with 40 GPU
cores, macOS 27.0, and arm64.
It used three runs, an 8-second case cooldown, seed 42, temperature 0.7, top-k 20, top-p 0.8, and thinking mode.
The tables select the `max_new=384` results for Qwen3.8-27B.

GSM8K typing-average prompt:

| Mode             | Output | Decode tok/s | vs Vanilla | Tokens/chunk | Spec acceptance |
| ---------------- | -----: | -----------: | ---------: | -----------: | --------------: |
| Vanilla          |    290 |       22.804 |     1.000x |        1.000 |               — |
| MTP, 1 proposal  |    302 |       38.180 |     1.674x |        1.899 | 143/158 (90.5%) |
| MTP, 2 proposals |    293 |       33.912 |     1.487x |        2.688 | 184/216 (85.2%) |
| DSpark           |    331 |       44.508 |     1.952x |        4.597 | 261/497 (52.5%) |
| DFlash2          |    350 |       56.047 |     2.458x |        5.738 | 290/420 (69.0%) |

Chat prompt:

| Mode             | Output | Decode tok/s | vs Vanilla | Tokens/chunk |  Spec acceptance |
| ---------------- | -----: | -----------: | ---------: | -----------: | ---------------: |
| Vanilla          |    384 |       22.800 |     1.000x |        1.000 |                — |
| MTP, 1 proposal  |    384 |       33.332 |     1.462x |        1.634 |  149/234 (63.7%) |
| MTP, 2 proposals |    384 |       24.641 |     1.081x |        1.959 |  190/390 (48.7%) |
| DSpark           |    384 |       15.942 |     0.699x |        1.655 |  153/1617 (9.5%) |
| DFlash2          |    384 |       18.364 |     0.805x |        1.892 | 181/1414 (12.8%) |

`vs Vanilla` is the decode-throughput ratio for the same prompt.
It is not a pure executor speedup when the output lengths differ.
`Tokens/chunk` includes the target token.
`Spec acceptance` is verified proposal tokens divided by proposed tokens.
It is not the published average-acceptance-length metric.
The prompt-dependent difference between the two tables is part of the result.

Reproduce this 27B subset with:

```sh
./scripts/qwen35_e2e_decode_perf.sh \
  --cases 27b_off,27b_mtp1,27b_mtp2,27b_dspark,27b_dflash2 \
  --runs 3
```

The DFlash2 reference row used the F32 affine-parameter checkpoint format from that commit.
The current converter stores DFlash2 affine parameters as BF16.
Treat the row as historical workload evidence until a matched current-format run replaces it.
See [Executor Tests, Benchmarks, and Profiling](docs/executor_benchmarks.md) for metric and provenance requirements.

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
