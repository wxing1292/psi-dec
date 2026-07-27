use std::collections::HashSet;

use inference_runtime_core::Error;
use inference_runtime_core::Result;
use serde_json::Value;

pub mod state;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ToolID(String);

impl ToolID {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        Ok(Self(validate_id(value.into(), "tool ID")?))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ToolCallID(String);

impl ToolCallID {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        Ok(Self(validate_id(value.into(), "tool call ID")?))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolInputSchema(Value);

impl ToolInputSchema {
    pub fn new(value: Value) -> Result<Self> {
        if !value.is_object() {
            return Err(Error::invalid_argument("tool input schema must be a JSON object"));
        }
        Ok(Self(value))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    tool_id: ToolID,
    description: Option<String>,
    input_schema: ToolInputSchema,
}

impl ToolDefinition {
    pub fn new(tool_id: ToolID, description: Option<String>, input_schema: ToolInputSchema) -> Self {
        Self {
            tool_id,
            description,
            input_schema,
        }
    }

    pub fn tool_id(&self) -> &ToolID {
        &self.tool_id
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn input_schema(&self) -> &ToolInputSchema {
        &self.input_schema
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolRegistration {
    definitions: Vec<ToolDefinition>,
}

impl ToolRegistration {
    pub fn new(definitions: Vec<ToolDefinition>) -> Result<Self> {
        if definitions.is_empty() {
            return Err(Error::invalid_argument(
                "tool registration must include at least one definition",
            ));
        }
        let mut tool_ids = HashSet::with_capacity(definitions.len());
        for definition in &definitions {
            if !tool_ids.insert(definition.tool_id().clone()) {
                return Err(Error::invalid_argument(format!(
                    "tool registration contains duplicate tool ID {:?}",
                    definition.tool_id().as_str()
                )));
            }
        }
        Ok(Self { definitions })
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolUnregistration {
    tool_ids: Vec<ToolID>,
}

impl ToolUnregistration {
    pub fn new(tool_ids: Vec<ToolID>) -> Result<Self> {
        if tool_ids.is_empty() {
            return Err(Error::invalid_argument(
                "tool unregistration must include at least one tool ID",
            ));
        }
        let mut unique_tool_ids = HashSet::with_capacity(tool_ids.len());
        for tool_id in &tool_ids {
            if !unique_tool_ids.insert(tool_id.clone()) {
                return Err(Error::invalid_argument(format!(
                    "tool unregistration contains duplicate tool ID {:?}",
                    tool_id.as_str()
                )));
            }
        }
        Ok(Self { tool_ids })
    }

    pub fn tool_ids(&self) -> &[ToolID] {
        &self.tool_ids
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolArguments(Value);

impl ToolArguments {
    pub fn new(value: Value) -> Result<Self> {
        if !value.is_object() {
            return Err(Error::invalid_argument("tool arguments must be a JSON object"));
        }
        Ok(Self(value))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolRawContent {
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolStructuredContent(Value);

impl ToolStructuredContent {
    pub fn new(value: Value) -> Self {
        Self(value)
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallRequest {
    tool_id: ToolID,
    tool_call_id: ToolCallID,
    arguments: ToolArguments,
}

impl ToolCallRequest {
    pub fn new(tool_id: ToolID, tool_call_id: ToolCallID, arguments: ToolArguments) -> Self {
        Self {
            tool_id,
            tool_call_id,
            arguments,
        }
    }

    pub fn tool_id(&self) -> &ToolID {
        &self.tool_id
    }

    pub fn tool_call_id(&self) -> &ToolCallID {
        &self.tool_call_id
    }

    pub fn arguments(&self) -> &ToolArguments {
        &self.arguments
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallResponse {
    tool_call_id: ToolCallID,
    raw_content: Vec<ToolRawContent>,
    structured_content: Option<ToolStructuredContent>,
    is_error: bool,
}

impl ToolCallResponse {
    pub fn new(
        tool_call_id: ToolCallID,
        raw_content: Vec<ToolRawContent>,
        structured_content: Option<ToolStructuredContent>,
        is_error: bool,
    ) -> Result<Self> {
        if is_error
            && !raw_content
                .iter()
                .any(|content| matches!(content, ToolRawContent::Text(text) if !text.is_empty()))
        {
            return Err(Error::invalid_argument(
                "tool error response must include non-empty model-facing text",
            ));
        }
        Ok(Self {
            tool_call_id,
            raw_content,
            structured_content,
            is_error,
        })
    }

    pub fn tool_call_id(&self) -> &ToolCallID {
        &self.tool_call_id
    }

    pub fn raw_content(&self) -> &[ToolRawContent] {
        &self.raw_content
    }

    pub fn structured_content(&self) -> Option<&ToolStructuredContent> {
        self.structured_content.as_ref()
    }

    pub fn is_error(&self) -> bool {
        self.is_error
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallCancellation {
    tool_call_id: ToolCallID,
}

impl ToolCallCancellation {
    pub fn new(tool_call_id: ToolCallID) -> Self {
        Self { tool_call_id }
    }

    pub fn tool_call_id(&self) -> &ToolCallID {
        &self.tool_call_id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolEvent {
    Registration(ToolRegistration),
    Unregistration(ToolUnregistration),
    CallRequest(ToolCallRequest),
    CallResponse(ToolCallResponse),
    CallCancellation(ToolCallCancellation),
}

fn validate_id(value: String, kind: &str) -> Result<String> {
    if value.is_empty() {
        return Err(Error::invalid_argument(format!("{kind} must not be empty")));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::ToolCallID;
    use super::ToolID;

    #[test]
    fn test_tool_id() {
        let id = ToolID::new("read_file").unwrap();
        assert_eq!(id.as_str(), "read_file");
        assert_eq!(id.into_string(), "read_file");
        assert!(matches!(
            ToolID::new(""),
            Err(Error::InvalidArgument(message)) if message == "tool ID must not be empty"
        ));
    }

    #[test]
    fn test_tool_call_id() {
        let id = ToolCallID::new("call-1").unwrap();
        assert_eq!(id.as_str(), "call-1");
        assert_eq!(id.into_string(), "call-1");
        assert!(matches!(
            ToolCallID::new(""),
            Err(Error::InvalidArgument(message)) if message == "tool call ID must not be empty"
        ));
    }
}
