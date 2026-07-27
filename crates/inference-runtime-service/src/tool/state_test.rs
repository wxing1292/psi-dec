use inference_runtime_core::Error;
use serde_json::json;

use super::ToolState;
use crate::tool::ToolArguments;
use crate::tool::ToolCallCancellation;
use crate::tool::ToolCallID;
use crate::tool::ToolCallRequest;
use crate::tool::ToolCallResponse;
use crate::tool::ToolDefinition;
use crate::tool::ToolEvent;
use crate::tool::ToolID;
use crate::tool::ToolInputSchema;
use crate::tool::ToolRawContent;
use crate::tool::ToolRegistration;
use crate::tool::ToolStructuredContent;
use crate::tool::ToolUnregistration;

#[test]
fn test_register_unregister_tool() {
    assert!(matches!(
        ToolInputSchema::new(json!("object")),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ToolRegistration::new(Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ToolRegistration::new(vec![new_definition("read_file"), new_definition("read_file")]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ToolUnregistration::new(Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ToolUnregistration::new(vec![new_tool_id("read_file"), new_tool_id("read_file")]),
        Err(Error::InvalidArgument(_))
    ));

    let mut state = ToolState::new();
    state
        .register_tool(&new_registration(&["read_file", "write_file"]))
        .unwrap();
    assert_eq!(
        state
            .list_tools()
            .iter()
            .map(|definition| definition.tool_id().as_str())
            .collect::<Vec<_>>(),
        ["read_file", "write_file"]
    );

    let original = state.clone();
    assert!(matches!(
        state.register_tool(&new_registration(&["list_files", "read_file"])),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(state, original);
    assert!(matches!(
        state.unregister_tool(&new_unregistration(&["read_file", "missing"])),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(state, original);

    state.unregister_tool(&new_unregistration(&["read_file"])).unwrap();
    state.register_tool(&new_registration(&["read_file"])).unwrap();
    assert_eq!(
        state
            .list_tools()
            .iter()
            .map(|definition| definition.tool_id().as_str())
            .collect::<Vec<_>>(),
        ["write_file", "read_file"]
    );
}

#[test]
fn test_request_respond_execution() {
    assert!(matches!(
        ToolArguments::new(json!(["README.md"])),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ToolCallResponse::new(new_tool_call_id("call-1"), Vec::new(), None, true),
        Err(Error::InvalidArgument(_))
    ));

    let mut state = ToolState::new();
    assert!(matches!(
        state.request_execution(&new_request("read_file", "call-1")),
        Err(Error::InvalidArgument(_))
    ));
    state
        .register_tool(&new_registration(&["read_file", "write_file"]))
        .unwrap();
    state.request_execution(&new_request("read_file", "call-1")).unwrap();
    assert!(matches!(
        state.request_execution(&new_request("write_file", "call-1")),
        Err(Error::InvalidArgument(_))
    ));
    state.request_execution(&new_request("write_file", "call-2")).unwrap();
    assert_eq!(state.list_executions().len(), 2);
    assert!(state.get_execution(&new_tool_call_id("call-1")).is_some());
    assert!(state.get_execution(&new_tool_call_id("call-2")).is_some());

    state.unregister_tool(&new_unregistration(&["write_file"])).unwrap();
    assert!(state.get_tool(&new_tool_id("write_file")).is_none());
    assert!(state.get_execution(&new_tool_call_id("call-2")).is_some());

    let response = ToolCallResponse::new(
        new_tool_call_id("call-2"),
        vec![ToolRawContent::Text("permission denied".to_string())],
        Some(ToolStructuredContent::new(json!({"code": "permission_denied"}))),
        true,
    )
    .unwrap();
    assert!(response.is_error());
    assert_eq!(response.raw_content().len(), 1);
    assert_eq!(
        response.structured_content().unwrap().as_value(),
        &json!({"code": "permission_denied"})
    );
    state.respond_execution(&response).unwrap();
    state.respond_execution(&new_response("call-1")).unwrap();
    assert_eq!(state.list_executions().len(), 0);

    assert!(matches!(
        state.respond_execution(&new_response("call-1")),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn test_cancel_execution() {
    let mut state = ToolState::new();
    state.register_tool(&new_registration(&["read_file"])).unwrap();
    state.request_execution(&new_request("read_file", "call-1")).unwrap();
    state
        .cancel_execution(&ToolCallCancellation::new(new_tool_call_id("call-1")))
        .unwrap();

    assert_eq!(state.list_executions().len(), 0);
    assert!(matches!(
        state.cancel_execution(&ToolCallCancellation::new(new_tool_call_id("call-1"))),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        state.respond_execution(&new_response("call-1")),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn test_fold() {
    let events = vec![
        ToolEvent::Registration(new_registration(&["read_file"])),
        ToolEvent::CallRequest(new_request("read_file", "old-call")),
        ToolEvent::Unregistration(new_unregistration(&["read_file"])),
        ToolEvent::Registration(new_registration(&["read_file"])),
        ToolEvent::CallRequest(new_request("read_file", "new-call")),
        ToolEvent::CallResponse(new_response("old-call")),
        ToolEvent::CallCancellation(ToolCallCancellation::new(new_tool_call_id("new-call"))),
    ];

    let state = ToolState::fold(&events).unwrap();

    assert!(state.get_tool(&new_tool_id("read_file")).is_some());
    assert_eq!(state.list_tools().len(), 1);
    assert_eq!(state.list_executions().len(), 0);
}

fn new_registration(tool_ids: &[&str]) -> ToolRegistration {
    ToolRegistration::new(tool_ids.iter().map(|tool_id| new_definition(tool_id)).collect()).unwrap()
}

fn new_unregistration(tool_ids: &[&str]) -> ToolUnregistration {
    ToolUnregistration::new(tool_ids.iter().map(|tool_id| new_tool_id(tool_id)).collect()).unwrap()
}

fn new_definition(tool_id: &str) -> ToolDefinition {
    ToolDefinition::new(
        new_tool_id(tool_id),
        None,
        ToolInputSchema::new(json!({"type": "object"})).unwrap(),
    )
}

fn new_request(tool_id: &str, tool_call_id: &str) -> ToolCallRequest {
    ToolCallRequest::new(
        new_tool_id(tool_id),
        new_tool_call_id(tool_call_id),
        ToolArguments::new(json!({})).unwrap(),
    )
}

fn new_response(tool_call_id: &str) -> ToolCallResponse {
    ToolCallResponse::new(
        new_tool_call_id(tool_call_id),
        vec![ToolRawContent::Text("ok".to_string())],
        None,
        false,
    )
    .unwrap()
}

fn new_tool_id(value: &str) -> ToolID {
    ToolID::new(value).unwrap()
}

fn new_tool_call_id(value: &str) -> ToolCallID {
    ToolCallID::new(value).unwrap()
}
