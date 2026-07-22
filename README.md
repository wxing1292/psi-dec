# psi-dec

`psi-dec` is a Rust inference runtime and Metal executor for Qwen decoder
models on Apple Silicon. It separates a model-agnostic runtime from a Qwen
model executor and a Metal backend, so request lifecycle and model execution
can evolve independently.

## Mental model

```text
client
  -> service: tokenize and validate the request
  -> runtime core: schedule work, own request lifecycle and cache pages
  -> Qwen executor: turn runtime metadata into model execution
  -> Metal backend: replay recorded GPU commands
  -> runtime core: commit tokens, update cache ownership, finish requests
```

The runtime owns *when* and *which* tokens execute. The executor owns *what*
the Qwen model computes. Metal owns *how* those computations run on the GPU.

| Layer | Owns | Does not own |
| --- | --- | --- |
| Runtime core | scheduling, request lifecycle, logical cache blocks, physical KV/state pages | model tensor layout or Metal details |
| Qwen executor | model layout, weights, GQA, GDN, dense MLP, MoE, sampling, MTP, replay stage ordering | scheduler policy or global page ownership |
| Metal backend | devices, buffers, kernels, command recording, ICB replay submission | request lifecycle or model-specific scheduling |

Runtime cache blocks are logical units of reusable decoder work. Their physical
KV and GDN-state pages have globally allocated page IDs. Runtime decides their
lifetime; the executor interprets those IDs for the current model layer.

GQA (grouped-query attention) and GDN (Gated DeltaNet) are alternative
attention families in a Qwen layer; dense MLP and MoE are alternative
feed-forward families. They are not four consecutive stages in every layer.
MTP is a speculative proposal path around target-model execution, while replay
is the Metal command-execution mechanism beneath both.

## Current scope

The checked-in service path is local and synchronous: one executor processes a
batch, returns its result, and the runtime commits it before the next batch.
Asynchronous request tasks, KV/state SSD onload/offload, and pipeline-parallel
hidden-state transport remain future work.

## Requirements and first run

You need an Apple Silicon Mac, the Rust toolchain, Xcode command-line tools,
and a Hugging Face CLI login that can download model weights. The current
service runners use MLX Community Qwen3.6 checkpoints.

If you are setting up a machine for the first time, install Xcode command-line
tools with `xcode-select --install`, install a current stable Rust toolchain and
the Hugging Face CLI with `brew install hf`, then log in once:

```sh
hf auth login
```

Download the dense 27B target and matching MTP checkpoints. MTP is part of the supported first-run path; target-only
full-prefix resampling still has a [known recurrent-state limitation](docs/future_work.md#runtime-lifecycle).

```sh
hf download mlx-community/Qwen3.6-27B-4bit \
  --local-dir models/Qwen3.6-27B-4bit

hf download mlx-community/Qwen3.6-27B-MTP-4bit \
  --local-dir models/Qwen3.6-27B-MTP-4bit
```

Start the service:

```sh
cargo run --release -p inference-runtime-service --bin qwen3_5_dense -- \
  --listen-addr 127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-mtp-model-dir "$PWD/models/Qwen3.6-27B-MTP-4bit" \
  --mtp-module 1
```

Then send a request from another terminal:

```sh
cargo run --release -p inference-runtime-service --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --prompt-str "Explain paged KV cache in one paragraph." \
  --max-sampled-tokens 128
```

For sparse 35B-A3B, use `qwen3_5_sparse` with the matching
`Qwen3.6-35B-A3B-4bit` and `Qwen3.6-35B-A3B-MTP-4bit` checkpoints. The runner
names retain their Qwen3.5 implementation name but consume the current Qwen3.6
MLX checkpoint layout. Full service options are in the
[service guide](docs/service.md).

## Where to go next

Start with the document matching your question:

- Requests, blocks, pages, and cache lifecycle: [runtime core](docs/core.md).

- Qwen layers and replay composition: [executor architecture](docs/executor.md), then [Qwen wiring](docs/executor_qwen.md).

- Sampling and rejection: [sampling](docs/executor_sampling.md).

- Metal recording, ICB replay, and a minimal kernel: [executor architecture](docs/executor.md), then the Metal backend's [Add One example](crates/inference-backend-metal/README.md#add-one).

- Attention and feed-forward internals: [GQA](docs/executor_gqa.md), [GDN](docs/executor_gdn.md), [dense MLP](docs/executor_dense_mlp.md), or [MoE](docs/executor_moe.md).

- Running the service or validating a release: [service and decode client](docs/service.md).

- Tests, benchmarks, profiling, and performance evidence: [executor verification](docs/executor_benchmarks.md).

- Development rules: [engineering boundaries](docs/high_level.md) and
  [engineering conventions](docs/engineering_conventions.md).

- Active follow-up work: [future work](docs/future_work.md).

The [documentation guide](docs/README.md) gives the complete reading map.
Detailed operational and measurement guidance belongs in the workflow
documents; this overview keeps only the shortest working first run.

## Workspace map

```text
crates/inference-runtime-core    model-agnostic scheduling, lifecycle, and cache ownership
crates/inference-runtime-service server binaries, tokenizer/client handling, and service configuration
crates/inference-executor-core   backend-neutral Qwen/component contracts and CPU references
crates/inference-executor-metal  Qwen-on-Metal weights, components, replay, sampling, and MTP
crates/inference-backend-metal   Metal resources, kernels, operators, and ICB replay runtime
```

## Acknowledgements

`psi-dec` is an independent Rust and Metal implementation inspired by the
open-source work of [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[mistral.rs](https://github.com/EricLBuehler/mistral.rs); credit is due to their
authors and contributor communities. It uses [MLX](https://github.com/ml-explore/mlx)
by Apple; its Metal kernel headers are downloaded and embedded at build time
under the MIT License, with the required notice in [`NOTICE`](NOTICE). Supported
Qwen models are developed by the [Qwen team](https://qwen.ai/); model weights
are separate artifacts governed by their own terms.

## License

`psi-dec` is distributed under the [MIT License](LICENSE). See
[NOTICE](NOTICE) for the MLX header attribution retained with this project.
