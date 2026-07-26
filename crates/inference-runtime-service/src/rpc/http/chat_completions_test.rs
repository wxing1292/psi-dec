use serde_json::json;

use super::Content;
use super::Delta;
use super::Message;
use super::Request;
use super::Response;
use super::ResponseChoice;
use super::ResponseChunk;
use super::ResponseChunkChoice;
use super::ResponseMessage;
use super::Tool;
use super::ToolCall;
use super::ToolChoice;
use super::Usage;

#[test]
fn test_request_schema() {
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": [{"type": "text", "text": "read"}]},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "contents",
                "name": "read_file"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"}
            }
        }],
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": 32,
        "temperature": 0.2,
        "top_k": 8,
        "top_p": 0.9,
        "seed": 42,
        "enable_thinking": true
    }))
    .unwrap();

    assert_eq!(request.model, "qwen");
    assert_eq!(request.messages.len(), 4);
    assert!(matches!(
        &request.messages[1],
        Message::User {
            content: Content::Parts(parts)
        } if parts.len() == 1
    ));
    assert!(matches!(
        &request.messages[2],
        Message::Assistant { tool_calls, .. }
        if matches!(tool_calls.as_slice(), [ToolCall::Function { .. }])
    ));
    assert!(matches!(request.tools.as_slice(), [Tool::Function { .. }]));
    assert!(matches!(request.tool_choice, Some(ToolChoice::Auto)));
    assert!(request.stream);
    assert!(request.stream_options.unwrap().include_usage);
    assert_eq!(request.max_completion_tokens, Some(32));
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_k, Some(8));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.seed, Some(42));
    assert_eq!(request.enable_thinking, Some(true));
}

#[test]
fn test_stream_defaults_to_false() {
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    assert!(!request.stream);
}

#[test]
fn test_role_fields_are_structural() {
    for message in [
        json!({"role": "system", "content": "hello", "tool_call_id": "call_1"}),
        json!({"role": "user", "content": "hello", "tool_calls": []}),
        json!({"role": "assistant", "content": "hello", "name": "assistant"}),
        json!({"role": "tool", "content": "result"}),
    ] {
        assert!(serde_json::from_value::<Message>(message).is_err());
    }
}

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
