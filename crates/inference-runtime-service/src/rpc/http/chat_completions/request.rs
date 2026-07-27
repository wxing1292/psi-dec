use std::collections::HashMap;
use std::collections::HashSet;

use hf_chat_template::Content as HFContent;
use hf_chat_template::Message as HFMessage;
use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::config::SamplingConfig;
use serde::Deserialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

use crate::api::decode::DecodeRequest;
use crate::codec::qwen::QwenCodec;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::invalid_request;
use crate::rpc::http::error::map_error;
use crate::tool::ToolArguments;
use crate::tool::ToolCallID;
use crate::tool::ToolDefinition;
use crate::tool::ToolID;
use crate::tool::ToolInputSchema;

const DEFAULT_MAX_COMPLETION_TOKENS: usize = 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model: String,
    messages: Vec<Message>,
    store: Option<bool>,
    #[serde(default)]
    tools: Vec<Tool>,
    tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    #[serde(alias = "max_tokens")]
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    reasoning_effort: Option<ReasoningEffort>,
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum Message {
    System {
        content: String,
    },
    Developer {
        content: String,
    },
    User {
        content: Content,
    },
    Assistant {
        content: Option<String>,
        reasoning_content: Option<String>,
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
    strict: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum ToolCall {
    Function { id: String, function: FunctionCall },
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

pub fn preprocess(
    request: Request,
    qwen_codec: &QwenCodec,
) -> Result<(DecodeRequest, usize, Vec<ToolID>, bool), HTTPError> {
    if request.model.is_empty() {
        return Err(invalid_request("model must not be empty"));
    }
    if request.store == Some(true) {
        return Err(invalid_request("store true is not supported"));
    }
    let enable_thinking = match (request.enable_thinking, request.reasoning_effort) {
        (Some(false), Some(_)) => {
            return Err(invalid_request("reasoning_effort conflicts with enable_thinking false"));
        },
        (Some(enable_thinking), _) => enable_thinking,
        (None, Some(_)) => true,
        (None, None) => false,
    };
    let tools = convert_tools(request.tools)?;
    let messages = convert_messages(request.messages)?;
    let enable_tools = match request.tool_choice.unwrap_or(ToolChoice::Auto) {
        ToolChoice::None => false,
        ToolChoice::Auto => true,
        ToolChoice::Required => {
            return Err(invalid_request("tool_choice required is not supported"));
        },
    };
    let rendered_tools = if enable_tools { tools.as_slice() } else { &[] };
    let tokens = qwen_codec
        .encode(messages, rendered_tools, enable_thinking, false)
        .map_err(map_error)?;
    let prompt_tokens = tokens.len();
    let max_sampled_tokens = request
        .max_completion_tokens
        .map(|tokens| tokens as usize)
        .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);
    let seed = request
        .seed
        .map(|seed| {
            u32::try_from(seed).map_err(|_| invalid_request(format!("seed must be in [0, {}], got {seed}", u32::MAX)))
        })
        .transpose()?;
    let sampling = SamplingConfig {
        max_sampled_tokens,
        temperature: request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
        top_k: request
            .top_k
            .map(|top_k| top_k as usize)
            .unwrap_or(DEFAULT_SAMPLING_TOP_K),
        top_p: request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
        seed,
        stop_sequences: Vec::new(),
    };
    let request = DecodeRequest::new(tokens, sampling).map_err(map_error)?;
    let tool_ids = if enable_tools {
        tools.into_iter().map(|tool| tool.tool_id().clone()).collect()
    } else {
        Vec::new()
    };
    Ok((request, prompt_tokens, tool_ids, enable_thinking))
}

fn convert_tools(tools: Vec<Tool>) -> Result<Vec<ToolDefinition>, HTTPError> {
    let mut tool_ids = HashSet::with_capacity(tools.len());
    tools
        .into_iter()
        .map(|tool| {
            let Tool::Function { function } = tool;
            if function.strict == Some(true) {
                return Err(invalid_request("strict tool definitions are not supported"));
            }
            let tool_id = ToolID::new(function.name).map_err(map_error)?;
            if !tool_ids.insert(tool_id.clone()) {
                return Err(invalid_request(format!("duplicate tool ID {:?}", tool_id.as_str())));
            }
            let input_schema = ToolInputSchema::new(function.parameters).map_err(map_error)?;
            Ok(ToolDefinition::new(tool_id, function.description, input_schema))
        })
        .collect()
}

fn convert_messages(messages: Vec<Message>) -> Result<Vec<HFMessage>, HTTPError> {
    if messages.is_empty() {
        return Err(invalid_request("messages must not be empty"));
    }
    let mut seen_call_ids = HashSet::new();
    let mut pending_calls = HashMap::new();
    let num_messages = messages.len();
    let mut converted = Vec::with_capacity(num_messages);
    for (index, message) in messages.into_iter().enumerate() {
        if !matches!(message, Message::Tool { .. }) && !pending_calls.is_empty() {
            return Err(invalid_request(
                "every assistant tool call must have a result before the next turn",
            ));
        }
        let message = match message {
            Message::System { content } | Message::Developer { content } => {
                if index != 0 {
                    return Err(invalid_request("system message must be first"));
                }
                HFMessage::system(content)
            },
            Message::User { content } => {
                let content = convert_content(content);
                if content_is_empty(&content) {
                    return Err(invalid_request("user content must not be empty"));
                }
                HFMessage {
                    role: "user".to_string(),
                    content: Some(content),
                    tool_calls: Vec::new(),
                    extra: Map::new(),
                }
            },
            Message::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                if content.as_ref().is_none_or(String::is_empty)
                    && reasoning_content.as_ref().is_none_or(String::is_empty)
                    && tool_calls.is_empty()
                {
                    return Err(invalid_request(
                        "assistant message must include content, reasoning content, or tool calls",
                    ));
                }
                let mut converted_calls = Vec::with_capacity(tool_calls.len());
                for call in tool_calls {
                    let ToolCall::Function { id, function } = call;
                    let tool_call_id = ToolCallID::new(id).map_err(map_error)?;
                    if !seen_call_ids.insert(tool_call_id.clone()) {
                        return Err(invalid_request("tool call IDs must be unique"));
                    }
                    let tool_id = ToolID::new(function.name).map_err(map_error)?;
                    let arguments = serde_json::from_str::<Value>(&function.arguments)
                        .map_err(|_| invalid_request("tool call arguments must be valid JSON"))?;
                    let arguments = ToolArguments::new(arguments).map_err(map_error)?.into_value();
                    converted_calls.push(json!({
                        "id": tool_call_id.as_str(),
                        "type": "function",
                        "function": {
                            "name": tool_id.as_str(),
                            "arguments": arguments,
                        }
                    }));
                    pending_calls.insert(tool_call_id, tool_id);
                }
                let mut extra = Map::new();
                if let Some(reasoning_content) = reasoning_content.filter(|content| !content.is_empty()) {
                    extra.insert("reasoning_content".to_string(), Value::String(reasoning_content));
                }
                HFMessage {
                    role: "assistant".to_string(),
                    content: content.map(HFContent::Text),
                    tool_calls: converted_calls,
                    extra,
                }
            },
            Message::Tool {
                tool_call_id,
                content,
                name,
            } => {
                let tool_call_id = ToolCallID::new(tool_call_id).map_err(map_error)?;
                let Some(expected_tool_id) = pending_calls.remove(&tool_call_id) else {
                    return Err(invalid_request("tool result must match a preceding tool call"));
                };
                let name = name.map(ToolID::new).transpose().map_err(map_error)?;
                if name.as_ref().is_some_and(|name| name != &expected_tool_id) {
                    return Err(invalid_request("tool result name must match its tool call"));
                }
                let mut extra = Map::new();
                extra.insert("tool_call_id".to_string(), Value::String(tool_call_id.into_string()));
                if let Some(name) = name {
                    extra.insert("name".to_string(), Value::String(name.into_string()));
                }
                HFMessage {
                    role: "tool".to_string(),
                    content: Some(HFContent::Text(content)),
                    tool_calls: Vec::new(),
                    extra,
                }
            },
        };
        converted.push(message);
    }
    if !pending_calls.is_empty() {
        return Err(invalid_request(
            "every assistant tool call must have a result before generation",
        ));
    }
    if matches!(
        converted.last(),
        Some(HFMessage { role, .. }) if role == "system" || role == "assistant"
    ) {
        return Err(invalid_request("conversation must end with a user or tool message"));
    }
    Ok(converted)
}

fn convert_content(content: Content) -> HFContent {
    match content {
        Content::Text(text) => HFContent::Text(text),
        Content::Parts(parts) => {
            HFContent::Parts(
                parts
                    .into_iter()
                    .map(|part| {
                        let ContentPart::Text { text } = part;
                        json!({"type": "text", "text": text})
                    })
                    .collect(),
            )
        },
    }
}

fn content_is_empty(content: &HFContent) -> bool {
    match content {
        HFContent::Text(text) => text.is_empty(),
        HFContent::Parts(parts) => {
            parts
                .iter()
                .all(|part| part.get("text").and_then(Value::as_str).is_none_or(str::is_empty))
        },
    }
}

#[cfg(test)]
#[path = "request_test.rs"]
mod tests;
