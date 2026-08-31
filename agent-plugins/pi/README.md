# psi-dec Pi provider

This Pi extension uses the `GenerateMessagesStream` gRPC API. One HTTP/2 gRPC stream owns one resident psi-dec runtime
session. Pi remains the owner of conversation history and compaction.

Install the package from the repository checkout:

```sh
cd agent-plugins/pi
npm install
pi install .
```

Start a text-generation server on `127.0.0.1:50061`. Then select the `psi-dec` provider in Pi. These environment
variables configure the default model:

| Variable | Default |
| --- | --- |
| `PSI_DEC_GRPC_URL` | `http://127.0.0.1:50061` |
| `PSI_DEC_MODEL` | `local-model` |
| `PSI_DEC_CONTEXT_WINDOW` | `262144` |
| `PSI_DEC_MAX_TOKENS` | `32768` |

The provider sends the complete Pi context on the first turn. It records the submitted message cursor. Each later
request sends only the new user or tool-result messages. The provider does not retry an append request as a complete
context request. The system prompt, tool definitions, and thinking configuration are fixed for one resident stream.
The provider opens a new stream and sends the complete context when one of these values changes.

The first version accepts text input and text tool results. It rejects image content. An `AbortSignal` closes the full
resident stream. Compaction, history rewrite, and branch are not supported in the first version. A later protocol
revision can distinguish turn interruption from session shutdown.
