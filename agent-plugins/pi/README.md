# psi-dec Pi provider

This Pi extension uses the `GenerateMessagesStream` gRPC API. One HTTP/2 gRPC stream owns one resident psi-dec runtime
session. Pi remains the owner of conversation history and compaction.

Install the package from the repository checkout:

```sh
cd agent-plugins/pi
npm install
pi install .
```

Start a text-generation server on `127.0.0.1:50061`. Add the `psi-dec` provider and its models to Pi's `models.json`.
The plugin owns the `psi-dec-messages` transport. `models.json` owns the endpoint, model identity, context limit, and
sampling defaults.

The HTTP and resident-session providers can exist in the same configuration:

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "thinkingFormat": "openai"
      },
      "models": [
        {
          "id": "qwen3_5",
          "name": "Qwen 3.5 27B Local HTTP",
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
          "name": "Qwen 3.5 27B Local Session",
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

Select `local/qwen3_5` for the OpenAI-compatible HTTP path. Select `psi-dec/qwen3_5` for the resident-session path.

The provider sends the complete Pi context on the first turn. It records the submitted message cursor. Each later
request sends only the new user or tool-result messages. The provider does not retry an append request as a complete
context request. The system prompt, tool definitions, and thinking configuration are fixed for one resident stream.
The provider opens a new stream and sends the complete context when one of these values changes.

The first version accepts text input and text tool results. It rejects image content. An `AbortSignal` closes the full
resident stream. Compaction, history rewrite, and branch are not supported in the first version. A later protocol
revision can distinguish turn interruption from session shutdown.
