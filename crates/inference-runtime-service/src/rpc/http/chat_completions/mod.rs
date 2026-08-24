//! OpenAI-compatible Chat Completions wire contract.
//!
//! The optional `x-request-id` header contains the caller's request-correlation UUID.
//! The service generates a UUID when the header is absent and returns the UUID in a successful response header.
//! Each handler invocation reconstructs tool-call correlation from its submitted conversation history.
//!
//! Request object graph:
//! `Option` marks an optional field. `Vec` marks a repeated field.
//! The graph omits scalar and object types.
//!
//! ```text
//! Request
//! ├── model: Option
//! ├── messages: Vec
//! │   ├── System
//! │   │   └── content
//! │   ├── Developer
//! │   │   └── content
//! │   ├── User
//! │   │   └── content
//! │   │       ├── Text
//! │   │       └── Parts: Vec
//! │   ├── Assistant
//! │   │   ├── content: Option
//! │   │   ├── reasoning_content: Option
//! │   │   └── tool_calls: Vec
//! │   │       └── Function
//! │   │           ├── id
//! │   │           └── function
//! │   │               ├── name
//! │   │               └── arguments
//! │   └── Tool
//! │       ├── tool_call_id ───────────────► ToolCall.id
//! │       ├── content
//! │       └── name: Option
//! ├── store: Option
//! ├── tools: Vec
//! │   └── Function
//! │       └── function
//! │           ├── name ◄──────────── FunctionCall.name selects one
//! │           ├── description: Option
//! │           ├── parameters
//! │           └── strict: Option
//! ├── tool_choice: Option
//! ├── stream
//! ├── stream_options: Option
//! ├── max_completion_tokens: Option
//! ├── temperature: Option
//! ├── top_k: Option
//! ├── top_p: Option
//! ├── seed: Option
//! ├── reasoning_effort: Option
//! └── enable_thinking: Option
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
//! ├── id
//! ├── object
//! ├── created
//! ├── model
//! ├── choices: Vec
//! │   └── ResponseChunkChoice
//! │       ├── index
//! │       ├── delta
//! │       │   ├── role: Option
//! │       │   ├── content: Option
//! │       │   ├── reasoning_content: Option
//! │       │   └── tool_calls: Option<Vec>
//! │       │       └── FunctionCallDelta { name, arguments }
//! │       └── finish_reason: Option
//! └── usage: Option
//!     ├── prompt_tokens
//!     ├── completion_tokens
//!     └── total_tokens
//!
//! role chunk → content/tool-call chunks → finish chunk → optional usage chunk → [DONE]
//! ```

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::response::Response as AxumResponse;
use request::Request;
use request::preprocess;
use response::ResponseMetadata;
use response::postprocess_response;
use response::postprocess_stream;
use uuid::Uuid;

use crate::rpc::http::HTTPServer;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::invalid_request;
use crate::rpc::http::error::map_error;

mod request;
mod response;

const REQUEST_ID_HEADER: &str = "x-request-id";

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(server): State<HTTPServer<N, L, P>>,
    headers: HeaderMap,
    payload: Result<Json<Request>, JsonRejection>,
) -> Result<AxumResponse, HTTPError> {
    let Json(request) = payload.map_err(|error| invalid_request(format!("invalid JSON request: {error}")))?;
    let request_id = resolve_request_id(&headers)?;
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let model = request.model.clone().unwrap_or_else(|| server.model_name.to_string());
    let (request, prompt_tokens, tool_ids, enable_thinking) = preprocess(request, &server.qwen_codec)?;
    let response = server.inference.decode(request).map_err(map_error)?;
    let metadata = ResponseMetadata::new(model, prompt_tokens);
    let response = server.qwen_codec.decode(response, tool_ids, enable_thinking);
    let mut response = if stream {
        postprocess_stream(response, metadata, include_usage)
    } else {
        postprocess_response(response, metadata).await?
    };
    response.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::try_from(request_id.to_string()).expect("UUID must be a valid HTTP header value"),
    );
    Ok(response)
}

fn resolve_request_id(headers: &HeaderMap) -> Result<Uuid, HTTPError> {
    let Some(value) = headers.get(REQUEST_ID_HEADER) else {
        return Ok(Uuid::new_v4());
    };
    let value = value
        .to_str()
        .map_err(|_| invalid_request("x-request-id must be a UUID"))?;
    Uuid::parse_str(value).map_err(|_| invalid_request("x-request-id must be a UUID"))
}

fn new_tool_call_id() -> String {
    Uuid::new_v4().to_string()
}
