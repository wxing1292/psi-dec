//! OpenAI-compatible Chat Completions wire contract.
//!
//! Request object graph:
//!
//! ```text
//! Request
//! ├── model
//! ├── messages: Vec<Message>
//! │   ├── System
//! │   │   └── content
//! │   ├── Developer
//! │   │   └── content
//! │   ├── User
//! │   │   └── Content
//! │   │       ├── Text
//! │   │       └── Parts: Vec<ContentPart>
//! │   ├── Assistant
//! │   │   ├── content?
//! │   │   ├── reasoning_content?
//! │   │   └── tool_calls: Vec<ToolCall>
//! │   │       └── Function
//! │   │           ├── id
//! │   │           └── FunctionCall { name, arguments }
//! │   └── Tool
//! │       ├── tool_call_id ───────────────► ToolCall.id
//! │       ├── content
//! │       └── name?
//! ├── tools: Vec<Tool>
//! │   └── Function
//! │       └── FunctionDefinition { name, description?, parameters }
//! │                              ▲
//! │                              └──── FunctionCall.name selects one
//! ├── tool_choice
//! ├── stream / stream_options
//! ├── store / reasoning_effort
//! └── generation options
//! ```
//!
//! `Tool::Function` defines one callable function. `ToolCall::Function`
//! records one selected invocation. A later `Message::Tool` returns its result
//! by referencing `ToolCall.id`.
//!
//! Standard OpenAI stateless tool-call lifecycle:
//!
//! ```text
//! ┌──────────┐                                  ┌──────────────────┐
//! │ Pi agent │                                  │ Inference server │
//! └────┬─────┘                                  └────────┬─────────┘
//!      │ 1. Request { messages, tools }                  │
//!      │────────────────────────────────────────────────►│
//!      │                                                 │
//!      │ 2. ResponseChunk { tool_calls }                 │
//!      │◄────────────────────────────────────────────────│
//!      │                                                 │
//!      │ 3. Execute tool                                 │
//!      │                                                 │
//!      │ 4. New Request { full history + Message::Tool } │
//!      │────────────────────────────────────────────────►│
//!      │                                                 │
//!      │ 5. ResponseChunk { text / next tool_calls }     │
//!      │◄────────────────────────────────────────────────│
//!      │                                                 │
//! ```
//!
//! A future Pi custom provider may instead append typed events to persistent
//! per-conversation history; that stateful wire contract is separate from this
//! OpenAI-compatible request schema.
//!
//! Streaming response object graph:
//!
//! ```text
//! ResponseChunk
//! ├── id / object / created / model
//! ├── choices: Vec<Choice>
//! │   └── Choice
//! │       ├── index
//! │       ├── delta: Delta
//! │       │   ├── role?
//! │       │   ├── content?
//! │       │   ├── reasoning_content?
//! │       │   └── tool_calls: Vec<ToolCallDelta>?
//! │       │       └── FunctionCallDelta { name, arguments }
//! │       └── finish_reason?
//! └── usage?
//!     └── prompt_tokens / completion_tokens / total_tokens
//!
//! role chunk → content/tool-call chunks → finish chunk → usage chunk? → [DONE]
//! ```

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::Response as AxumResponse;
use request::Request;
use request::preprocess;
use response::ResponseMetadata;
use response::postprocess_response;
use response::postprocess_stream;

use crate::rpc::http::HTTPServer;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::invalid_request;
use crate::rpc::http::error::map_error;

mod request;
mod response;

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(server): State<HTTPServer<N, L, P>>,
    payload: Result<Json<Request>, JsonRejection>,
) -> Result<AxumResponse, HTTPError> {
    let Json(request) = payload.map_err(|error| invalid_request(format!("invalid JSON request: {error}")))?;
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let model = request.model.clone();
    let (request, prompt_tokens, tool_ids, enable_thinking) = preprocess(request, &server.qwen_codec)?;
    let response = server.inference.decode(request).map_err(map_error)?;
    let metadata = ResponseMetadata::new(response.request_id(), model, prompt_tokens);
    let response = server.qwen_codec.decode(response, tool_ids, enable_thinking);
    if stream {
        Ok(postprocess_stream(response, metadata, include_usage))
    } else {
        postprocess_response(response, metadata).await
    }
}
