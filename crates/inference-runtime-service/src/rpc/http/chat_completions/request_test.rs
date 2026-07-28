use std::sync::Arc;

use hf_chat_template::ChatTemplate;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use serde_json::json;
use tokenizers::models::wordlevel::WordLevel;

use crate::codec::qwen::QwenCodec;
use crate::rpc::http::chat_completions::request::Content;
use crate::rpc::http::chat_completions::request::Message;
use crate::rpc::http::chat_completions::request::Request;
use crate::rpc::http::chat_completions::request::Tool;
use crate::rpc::http::chat_completions::request::ToolCall;
use crate::rpc::http::chat_completions::request::ToolChoice;
use crate::rpc::http::chat_completions::request::preprocess;

#[test]
fn test_schema() {
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

    assert_eq!(request.model.as_deref(), Some("qwen"));
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
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    assert_eq!(request.model, None);
    assert!(!request.stream);
}

#[test]
fn test_role_fields_are_structural() {
    for message in [
        json!({"role": "system", "content": "hello", "tool_call_id": "call_1"}),
        json!({"role": "developer", "content": "hello", "tool_call_id": "call_1"}),
        json!({"role": "user", "content": "hello", "tool_calls": []}),
        json!({"role": "assistant", "content": "hello", "name": "assistant"}),
        json!({"role": "tool", "content": "result"}),
    ] {
        assert!(serde_json::from_value::<Message>(message).is_err());
    }
}

#[test]
fn test_preprocess() {
    let codec = fixture_codec();
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "developer", "content": "be concise"},
            {"role": "user", "content": "read"}
        ],
        "store": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
                "strict": false
            }
        }],
        "stream": true,
        "stream_options": {"include_usage": true},
        "reasoning_effort": "medium",
        "max_completion_tokens": 4
    }))
    .unwrap();

    let Ok((_, prompt_tokens, tool_ids, enable_thinking)) = preprocess(request, &codec) else {
        panic!("Pi Chat Completions request must preprocess");
    };
    assert!(prompt_tokens > 0);
    assert_eq!(tool_ids.len(), 1);
    assert_eq!(tool_ids[0].as_str(), "read_file");
    assert!(enable_thinking);
}

#[test]
fn test_tool_history() {
    let codec = fixture_codec();
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "use the retired tool"},
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "I used the retired tool.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "retired_tool",
                        "arguments": "{}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "retired_tool",
                "content": "contents"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "active_tool",
                "parameters": {"type": "object"}
            }
        }],
        "max_completion_tokens": 4
    }))
    .unwrap();

    let Ok((_, prompt_tokens, tool_ids, enable_thinking)) = preprocess(request, &codec) else {
        panic!("valid Chat Completions request must preprocess");
    };
    assert!(prompt_tokens > 0);
    assert_eq!(tool_ids.len(), 1);
    assert_eq!(tool_ids[0].as_str(), "active_tool");
    assert!(!enable_thinking);

    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "missing", "arguments": "{}"}
                }]
            }
        ]
    }))
    .unwrap();
    assert!(preprocess(request, &codec).is_err());
}

#[test]
fn test_tool_definitions() {
    let codec = fixture_codec();
    for request in [
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "function", "function": {"name": "same", "parameters": {}}},
                {"type": "function", "function": {"name": "same", "parameters": {}}}
            ]
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                    "strict": true
                }
            }]
        }),
    ] {
        let request = serde_json::from_value::<Request>(request).unwrap();
        assert!(preprocess(request, &codec).is_err());
    }
}

#[test]
fn test_unsupported_options() {
    let codec = fixture_codec();
    for request in [
        json!({
            "model": "",
            "messages": [{"role": "user", "content": "hello"}]
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "required"
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "store": true
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium",
            "enable_thinking": false
        }),
    ] {
        let request = serde_json::from_value::<Request>(request).unwrap();
        assert!(preprocess(request, &codec).is_err());
    }
}

fn fixture_codec() -> QwenCodec {
    let template = ChatTemplate::from_str(
        "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}{% for tool in tools %}{{ \
         tool.function.name }}{% endfor %}",
    )
    .unwrap();
    QwenCodec::new(template, Arc::new(fixture_tokenizer())).unwrap()
}

fn fixture_tokenizer() -> HFTokenizer {
    let model = WordLevel::builder()
        .vocab(
            [("[UNK]".to_string(), 0), ("</think>".to_string(), 1)]
                .into_iter()
                .collect(),
        )
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer
        .add_special_tokens([tokenizers::AddedToken::from("</think>", true)])
        .unwrap();
    HFTokenizer::new(tokenizer)
}
