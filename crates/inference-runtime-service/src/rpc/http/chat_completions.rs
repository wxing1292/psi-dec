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
//! │   ├── User
//! │   │   └── Content
//! │   │       ├── Text
//! │   │       └── Parts: Vec<ContentPart>
//! │   ├── Assistant
//! │   │   ├── content?
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
use inference_runtime_core::Error;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use super::HTTPServer;
use super::error::HTTPError;
use super::error::invalid_request;
use super::error::map_error;
use crate::api::decode::DecodeRequest;
use crate::api::decode::DecodeResponse;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    tools: Vec<Tool>,
    tool_choice: Option<ToolChoice>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[serde(alias = "max_tokens")]
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum Message {
    System {
        content: String,
    },
    User {
        content: Content,
    },
    Assistant {
        content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
        name: Option<String>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum ContentPart {
    Text { text: String },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum Tool {
    Function { function: FunctionDefinition },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionDefinition {
    name: String,
    description: Option<String>,
    parameters: Value,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum ToolCall {
    Function { id: String, function: FunctionCall },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ToolChoice {
    None,
    Auto,
    Required,
}

#[derive(Serialize)]
struct Response {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ResponseChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ResponseChoice {
    index: usize,
    message: ResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct ResponseChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ResponseChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ResponseChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Serialize)]
struct ToolCallDelta {
    index: usize,
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionCallDelta,
}

#[derive(Serialize)]
struct FunctionCallDelta {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(server): State<HTTPServer<N, L, P>>,
    payload: Result<Json<Request>, JsonRejection>,
) -> Result<AxumResponse, HTTPError> {
    let Json(request) = payload.map_err(|error| invalid_request(format!("invalid JSON request: {error}")))?;
    let stream = request.stream;
    let request = preprocess(request)?;
    let response = server.inference.decode(request).map_err(map_error)?;
    if stream {
        postprocess_stream(response).await
    } else {
        postprocess_response(response).await
    }
}

fn preprocess(_request: Request) -> Result<DecodeRequest, HTTPError> {
    Err(map_error(Error::unavailable(
        "OpenAI chat request preprocessing is not implemented",
    )))
}

async fn postprocess_response(_response: DecodeResponse) -> Result<AxumResponse, HTTPError> {
    Err(map_error(Error::unavailable(
        "OpenAI chat response postprocessing is not implemented",
    )))
}

async fn postprocess_stream(_response: DecodeResponse) -> Result<AxumResponse, HTTPError> {
    // TODO: Evaluate an HF-compatible streaming response parser here.
    Err(map_error(Error::unavailable(
        "OpenAI chat stream postprocessing is not implemented",
    )))
}

#[cfg(test)]
#[path = "chat_completions_test.rs"]
mod tests;
