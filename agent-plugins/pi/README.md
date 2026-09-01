# psi-dec Pi provider

This Pi extension connects Pi to the psi-dec `GenerateMessagesStream` gRPC API. One HTTP/2 gRPC stream owns one
resident runtime session. Pi owns conversation history and compaction.

## Requirements

Install these tools before you install the extension:

- Pi `0.84.4` or later
- Node.js and npm
- A built psi-dec text-generation service

The extension supports text input and text tool results. It does not support image content.

## Install

Install a detached global copy for normal use. This copy does not depend on the repository path.

Run these commands from the psi-dec repository root:

```sh
psi_dec_extension="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/extensions/psi-dec"

mkdir -p "$psi_dec_extension/src"
cp agent-plugins/pi/package.json "$psi_dec_extension/package.json"
cp agent-plugins/pi/package-lock.json "$psi_dec_extension/package-lock.json"
cp crates/inference-runtime-proto/proto/inference_runtime.proto \
  "$psi_dec_extension/inference_runtime.proto"
sed 's#../../../crates/inference-runtime-proto/proto/inference_runtime.proto#../inference_runtime.proto#' \
  agent-plugins/pi/src/index.ts > "$psi_dec_extension/src/index.ts"
(
  cd "$psi_dec_extension"
  npm ci --omit=dev --omit=peer
)
```

Pi discovers the extension under `~/.pi/agent/extensions/`. The install does not add a repository path to
`~/.pi/agent/settings.json`.

Use a local-path package only for extension development:

```sh
cd agent-plugins/pi
npm ci
npm run check
pi install "$PWD"
```

Pi stores a local package as a path reference. It does not copy the package. Run `pi remove "$PWD"` before you use
the detached global copy.

Repeat the detached install commands after an extension or protobuf change. Restart Pi or run `/reload` after the
copy completes.

## Start psi-dec

Start one text-generation service with gRPC enabled. This example starts Qwen3.8 27B with MTP:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 1
```

The resident provider uses the gRPC address. The OpenAI-compatible provider uses the HTTP address.

## Configure Pi

Add both providers to `~/.pi/agent/models.json` when you want resident and stateless access:

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "unused",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_completion_tokens",
        "thinkingFormat": "qwen",
        "supportsStrictMode": false,
        "supportsLongCacheRetention": false
      },
      "models": [
        {
          "id": "qwen3_5",
          "name": "Qwen 3.8 27B Local HTTP",
          "reasoning": true,
          "contextWindow": 81920,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": null,
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": null
          },
          "samplingParams": {
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      ]
    },
    "psi-dec": {
      "baseUrl": "http://127.0.0.1:50061",
      "api": "psi-dec-messages",
      "apiKey": "unused",
      "models": [
        {
          "id": "qwen3_5",
          "name": "Qwen 3.8 27B Local Session",
          "reasoning": true,
          "contextWindow": 81920,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": null,
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": null
          },
          "samplingParams": {
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      ]
    }
  }
}
```

The resident provider uses these fields:

| Field | Owner and effect |
| --- | --- |
| `baseUrl` | The extension connects to this psi-dec gRPC endpoint. |
| `api` | `psi-dec-messages` selects the extension transport. |
| `id` | Pi uses this value for model selection. The running psi-dec service owns the loaded checkpoint. |
| `contextWindow` | Pi uses this value for context accounting and compaction. It must not exceed the service limit. |
| `maxTokens` | The extension sends this value as the default per-turn sampled-token limit. |
| `temperature` | The extension sends this value for each turn. |
| `top_k` | The extension sends this value for each turn. |
| `top_p` | The extension sends this value for each turn. |
| `thinkingLevelMap` | Pi maps its thinking level before the extension maps the level to psi-dec reasoning effort. |

Pi turn options override the model sampling defaults. The plugin does not read model settings from environment
variables.

## Use

Start an interactive Pi session with the resident provider:

```sh
pi --model psi-dec/qwen3_5 --thinking high
```

Use the HTTP provider when you want a stateless request path:

```sh
pi --model local/qwen3_5 --thinking high
```

The resident provider sends the complete Pi context on the first turn. It records the submitted message cursor. Each
later request sends only new user or tool-result messages. The same Pi session ID selects the same resident stream.

The system prompt, tool definitions, and thinking configuration are fixed for one resident stream. The provider opens
a new stream and sends the complete context when one of these values changes.

An `AbortSignal` closes the full resident stream. Compaction, history rewrite, and branch do not reuse the resident
session. Start a new Pi session with the complete context after one of these operations.

## Verify

Confirm that Pi can see the configured model:

```sh
pi --list-models psi-dec
```

Run one request through the resident provider:

```sh
pi --model psi-dec/qwen3_5 --thinking high --print "Reply with exactly: hello"
```

The first request creates a runtime request and materializes the full prompt. A later turn in the same interactive Pi
session appends only its new prompt suffix.
