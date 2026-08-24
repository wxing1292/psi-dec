use futures_util::StreamExt;
use futures_util::stream;
use inference_runtime_core::Error;
use inference_runtime_core::runtime::CompletionReason;
use uuid::Uuid;

use crate::codec::qwen::ResponseEvent;
use crate::rpc::http::chat_completions::response::Delta;
use crate::rpc::http::chat_completions::response::Response;
use crate::rpc::http::chat_completions::response::ResponseChoice;
use crate::rpc::http::chat_completions::response::ResponseChunk;
use crate::rpc::http::chat_completions::response::ResponseChunkChoice;
use crate::rpc::http::chat_completions::response::ResponseMessage;
use crate::rpc::http::chat_completions::response::ResponseMetadata;
use crate::rpc::http::chat_completions::response::ResponseStream;
use crate::rpc::http::chat_completions::response::Usage;
use crate::rpc::http::chat_completions::response::collect_response;

#[test]
fn test_response_schema() {
    let response = serde_json::to_value(Response {
        id: "chatcmpl-1".to_string(),
        object: "chat.completion",
        created: 42,
        model: "qwen".to_string(),
        choices: vec![ResponseChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: Some("hello".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
        },
    })
    .unwrap();

    assert_eq!(response["id"], "chatcmpl-1");
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    assert_eq!(response["choices"][0]["message"]["content"], "hello");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["total_tokens"], 4);
}

#[test]
fn test_response_chunk_schema() {
    let chunk = serde_json::to_value(ResponseChunk {
        id: "chatcmpl-1".to_string(),
        object: "chat.completion.chunk",
        created: 42,
        model: "qwen".to_string(),
        choices: vec![ResponseChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: Some("hello".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: Some(Usage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
        }),
    })
    .unwrap();

    assert_eq!(chunk["id"], "chatcmpl-1");
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunk["choices"][0]["delta"]["content"], "hello");
    assert_eq!(chunk["usage"]["total_tokens"], 4);
}

#[tokio::test]
async fn test_response() {
    let response = collect_response(
        stream::iter([
            Ok(ResponseEvent::Thinking("reasoning".to_string())),
            Ok(ResponseEvent::Text("hello".to_string())),
            Ok(ResponseEvent::Completed {
                reason: CompletionReason::StopSequence,
                num_output_tokens: 2,
            }),
        ]),
        fixture_metadata(),
    )
    .await
    .unwrap();

    assert!(
        Uuid::parse_str(
            response
                .id
                .strip_prefix("chatcmpl-")
                .expect("response ID must have the Chat Completions prefix"),
        )
        .is_ok()
    );
    assert_eq!(
        response.choices[0].message.reasoning_content.as_deref(),
        Some("reasoning")
    );
    assert_eq!(response.choices[0].message.content.as_deref(), Some("hello"));
    assert_eq!(response.choices[0].finish_reason, "stop");
    assert_eq!(response.usage.completion_tokens, 2);
}

#[tokio::test]
async fn test_stream() {
    let events = ResponseStream::new(
        stream::iter([
            Ok(ResponseEvent::Thinking("reasoning".to_string())),
            Ok(ResponseEvent::Text("hello".to_string())),
            Ok(ResponseEvent::Completed {
                reason: CompletionReason::LengthLimit,
                num_output_tokens: 2,
            }),
        ]),
        fixture_metadata(),
        true,
    )
    .map(|event| event.unwrap())
    .collect::<Vec<_>>()
    .await;

    assert_eq!(events.len(), 6);
    let role = serde_json::from_str::<serde_json::Value>(&events[0]).unwrap();
    let reasoning = serde_json::from_str::<serde_json::Value>(&events[1]).unwrap();
    let content = serde_json::from_str::<serde_json::Value>(&events[2]).unwrap();
    let finish = serde_json::from_str::<serde_json::Value>(&events[3]).unwrap();
    let usage = serde_json::from_str::<serde_json::Value>(&events[4]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(reasoning["choices"][0]["delta"]["reasoning_content"], "reasoning");
    assert_eq!(content["choices"][0]["delta"]["content"], "hello");
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(events[5], "[DONE]");
}

#[tokio::test]
async fn test_stream_error() {
    let events = ResponseStream::new(
        stream::iter([Err(Error::aborted("request aborted"))]),
        fixture_metadata(),
        false,
    )
    .map(|event| event.unwrap())
    .collect::<Vec<_>>()
    .await;

    assert_eq!(events.len(), 2);
    let role = serde_json::from_str::<serde_json::Value>(&events[0]).unwrap();
    let error = serde_json::from_str::<serde_json::Value>(&events[1]).unwrap();
    assert_eq!(error["id"], role["id"]);
    let response_id = error["id"].as_str().unwrap();
    assert!(
        Uuid::parse_str(
            response_id
                .strip_prefix("chatcmpl-")
                .expect("response ID must have the Chat Completions prefix"),
        )
        .is_ok()
    );
    assert_eq!(error["error"]["code"], "aborted");
    assert!(!events.iter().any(|event| event == "[DONE]"));
}

fn fixture_metadata() -> ResponseMetadata {
    ResponseMetadata::new("qwen".to_string(), 3)
}
