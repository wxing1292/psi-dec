use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use axum::Json;
use axum::response::IntoResponse;
use axum::response::Response as AxumResponse;
use axum::response::sse::Event;
use axum::response::sse::KeepAlive;
use axum::response::sse::Sse;
use futures_util::Stream;
use futures_util::StreamExt;
use inference_runtime_core::Error;
use inference_runtime_core::runtime::CompletionReason;
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::codec::qwen::ResponseEvent;
use crate::rpc::http::chat_completions::new_tool_call_id;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::map_error;
use crate::rpc::http::error::openai_error_body;

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
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ToolCall {
    Function { id: String, function: FunctionCall },
}

#[derive(Serialize)]
struct FunctionCall {
    name: String,
    arguments: String,
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
    reasoning_content: Option<String>,
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

pub async fn postprocess_response<S>(response: S, metadata: ResponseMetadata) -> Result<AxumResponse, HTTPError>
where
    S: Stream<Item = Result<ResponseEvent, Error>>,
{
    collect_response(response, metadata)
        .await
        .map(|response| Json(response).into_response())
        .map_err(map_error)
}

async fn collect_response<S>(response: S, metadata: ResponseMetadata) -> Result<Response, Error>
where
    S: Stream<Item = Result<ResponseEvent, Error>>,
{
    let mut response = Box::pin(response);
    let mut thinking = String::new();
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    while let Some(event) = response.next().await {
        match event? {
            ResponseEvent::Thinking(chunk) => thinking.push_str(&chunk),
            ResponseEvent::Text(chunk) => text.push_str(&chunk),
            ResponseEvent::ToolCall(call) => {
                tool_calls.push(ToolCall::Function {
                    id: new_tool_call_id(),
                    function: FunctionCall {
                        name: call.tool_id().as_str().to_string(),
                        arguments: call.arguments().as_value().to_string(),
                    },
                });
            },
            ResponseEvent::Completed {
                reason,
                num_output_tokens,
            } => {
                let usage = metadata.usage(num_output_tokens);
                let has_tool_calls = !tool_calls.is_empty();
                let content = if text.is_empty() && has_tool_calls {
                    None
                } else {
                    Some(text)
                };
                return Ok(Response {
                    id: metadata.id,
                    object: "chat.completion",
                    created: metadata.created,
                    model: metadata.model,
                    choices: vec![ResponseChoice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant",
                            content,
                            reasoning_content: (!thinking.is_empty()).then_some(thinking),
                            tool_calls: has_tool_calls.then_some(tool_calls),
                        },
                        finish_reason: finish_reason(reason, has_tool_calls),
                    }],
                    usage,
                });
            },
        }
    }
    Err(Error::internal("Qwen response ended without a completion event"))
}

pub fn postprocess_stream<S>(response: S, metadata: ResponseMetadata, include_usage: bool) -> AxumResponse
where
    S: Stream<Item = Result<ResponseEvent, Error>> + Send + 'static,
{
    let response = ResponseStream::new(response, metadata, include_usage)
        .map(|event| event.map(|data| Event::default().data(data)));
    Sse::new(response)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
        .into_response()
}

pub struct ResponseMetadata {
    id: String,
    model: String,
    created: u64,
    prompt_tokens: usize,
}

impl ResponseMetadata {
    pub fn new(model: String, prompt_tokens: usize) -> Self {
        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            model,
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            prompt_tokens,
        }
    }

    fn usage(&self, completion_tokens: usize) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            total_tokens: self.prompt_tokens + completion_tokens,
        }
    }
}

struct ResponseStream<S> {
    response: Pin<Box<S>>,
    metadata: ResponseMetadata,
    include_usage: bool,
    num_tool_calls: usize,
    pending: VecDeque<String>,
    closed: bool,
}

impl<S> ResponseStream<S>
where
    S: Stream<Item = Result<ResponseEvent, Error>>,
{
    fn new(response: S, metadata: ResponseMetadata, include_usage: bool) -> Self {
        let role = serialize(&ResponseChunk {
            id: metadata.id.clone(),
            object: "chat.completion.chunk",
            created: metadata.created,
            model: metadata.model.clone(),
            choices: vec![ResponseChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant"),
                    content: Some(String::new()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        });
        Self {
            response: Box::pin(response),
            metadata,
            include_usage,
            num_tool_calls: 0,
            pending: VecDeque::from([role]),
            closed: false,
        }
    }

    fn push_event(&mut self, event: ResponseEvent) {
        match event {
            ResponseEvent::Thinking(reasoning_content) => {
                self.pending.push_back(serialize(&self.chunk(Delta {
                    role: None,
                    content: None,
                    reasoning_content: Some(reasoning_content),
                    tool_calls: None,
                })));
            },
            ResponseEvent::Text(content) => {
                self.pending.push_back(serialize(&self.chunk(Delta {
                    role: None,
                    content: Some(content),
                    reasoning_content: None,
                    tool_calls: None,
                })));
            },
            ResponseEvent::ToolCall(call) => {
                let index = self.num_tool_calls;
                self.num_tool_calls += 1;
                self.pending.push_back(serialize(&self.chunk(Delta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index,
                        id: new_tool_call_id(),
                        kind: "function",
                        function: FunctionCallDelta {
                            name: call.tool_id().as_str().to_string(),
                            arguments: call.arguments().as_value().to_string(),
                        },
                    }]),
                })));
            },
            ResponseEvent::Completed {
                reason,
                num_output_tokens,
            } => {
                self.pending.push_back(serialize(&ResponseChunk {
                    id: self.metadata.id.clone(),
                    object: "chat.completion.chunk",
                    created: self.metadata.created,
                    model: self.metadata.model.clone(),
                    choices: vec![ResponseChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(finish_reason(reason, self.num_tool_calls > 0)),
                    }],
                    usage: None,
                }));
                if self.include_usage {
                    self.pending.push_back(serialize(&ResponseChunk {
                        id: self.metadata.id.clone(),
                        object: "chat.completion.chunk",
                        created: self.metadata.created,
                        model: self.metadata.model.clone(),
                        choices: Vec::new(),
                        usage: Some(self.metadata.usage(num_output_tokens)),
                    }));
                }
                self.pending.push_back("[DONE]".to_string());
                self.closed = true;
            },
        }
    }

    fn chunk(&self, delta: Delta) -> ResponseChunk {
        ResponseChunk {
            id: self.metadata.id.clone(),
            object: "chat.completion.chunk",
            created: self.metadata.created,
            model: self.metadata.model.clone(),
            choices: vec![ResponseChunkChoice {
                index: 0,
                delta,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn push_error(&mut self, error: Error) {
        let mut body = openai_error_body(&error);
        body.as_object_mut()
            .expect("OpenAI error body must be an object")
            .insert("id".to_string(), Value::String(self.metadata.id.clone()));
        self.pending.push_back(body.to_string());
        self.closed = true;
    }
}

impl<S> Stream for ResponseStream<S>
where
    S: Stream<Item = Result<ResponseEvent, Error>>,
{
    type Item = Result<String, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let this = self.as_mut().get_mut();
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if this.closed {
                return Poll::Ready(None);
            }
            match this.response.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(event))) => this.push_event(event),
                Poll::Ready(Some(Err(error))) => this.push_error(error),
                Poll::Ready(None) => {
                    this.push_error(Error::internal("Qwen response ended without a completion event"));
                },
            }
        }
    }
}

fn finish_reason(reason: CompletionReason, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        "tool_calls"
    } else {
        match reason {
            CompletionReason::StopSequence => "stop",
            CompletionReason::LengthLimit | CompletionReason::ContextLimit => "length",
        }
    }
}

fn serialize<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("Chat Completions response must serialize")
}

#[cfg(test)]
#[path = "response_test.rs"]
mod tests;
