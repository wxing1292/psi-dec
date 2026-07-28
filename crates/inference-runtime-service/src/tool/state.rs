use std::collections::HashMap;
use std::collections::hash_map::Entry;

use inference_runtime_core::Error;
use inference_runtime_core::Result;

use crate::tool::ToolCallCancellation;
use crate::tool::ToolCallID;
use crate::tool::ToolCallRequest;
use crate::tool::ToolCallResponse;
use crate::tool::ToolDefinition;
use crate::tool::ToolEvent;
use crate::tool::ToolID;
use crate::tool::ToolRegistration;
use crate::tool::ToolUnregistration;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolState {
    tools: Vec<ToolDefinition>,
    executions: HashMap<ToolCallID, ToolCallRequest>,
}

impl ToolState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fold<'a>(events: impl IntoIterator<Item = &'a ToolEvent>) -> Result<Self> {
        let mut state = Self::new();
        for event in events {
            state.apply_event(event)?;
        }
        Ok(state)
    }

    pub fn apply_event(&mut self, event: &ToolEvent) -> Result<()> {
        match event {
            ToolEvent::Registration(registration) => self.register_tool(registration),
            ToolEvent::Unregistration(unregistration) => self.unregister_tool(unregistration),
            ToolEvent::CallRequest(request) => self.request_execution(request),
            ToolEvent::CallResponse(response) => self.respond_execution(response),
            ToolEvent::CallCancellation(cancellation) => self.cancel_execution(cancellation),
        }
    }

    pub fn register_tool(&mut self, registration: &ToolRegistration) -> Result<()> {
        for definition in registration.definitions() {
            if self.get_tool(definition.tool_id()).is_some() {
                return Err(Error::invalid_argument(format!(
                    "tool ID {:?} is already registered",
                    definition.tool_id().as_str()
                )));
            }
        }
        self.tools.extend(registration.definitions().iter().cloned());
        Ok(())
    }

    pub fn unregister_tool(&mut self, unregistration: &ToolUnregistration) -> Result<()> {
        for tool_id in unregistration.tool_ids() {
            if self.get_tool(tool_id).is_none() {
                return Err(Error::invalid_argument(format!(
                    "tool ID {:?} is not registered",
                    tool_id.as_str()
                )));
            }
        }
        self.tools
            .retain(|definition| !unregistration.tool_ids().contains(definition.tool_id()));
        Ok(())
    }

    pub fn request_execution(&mut self, request: &ToolCallRequest) -> Result<()> {
        if self.get_tool(request.tool_id()).is_none() {
            return Err(Error::invalid_argument(format!(
                "tool ID {:?} is not registered",
                request.tool_id().as_str()
            )));
        }
        match self.executions.entry(request.tool_call_id().clone()) {
            Entry::Occupied(_) => {
                Err(Error::invalid_argument(format!(
                    "tool call ID {:?} is already in flight",
                    request.tool_call_id().as_str()
                )))
            },
            Entry::Vacant(entry) => {
                entry.insert(request.clone());
                Ok(())
            },
        }
    }

    pub fn respond_execution(&mut self, response: &ToolCallResponse) -> Result<()> {
        self.complete_execution(response.tool_call_id(), "response")
    }

    pub fn cancel_execution(&mut self, cancellation: &ToolCallCancellation) -> Result<()> {
        self.complete_execution(cancellation.tool_call_id(), "cancellation")
    }

    pub fn get_tool(&self, tool_id: &ToolID) -> Option<&ToolDefinition> {
        self.tools.iter().find(|definition| definition.tool_id() == tool_id)
    }

    pub fn list_tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    pub fn get_execution(&self, tool_call_id: &ToolCallID) -> Option<&ToolCallRequest> {
        self.executions.get(tool_call_id)
    }

    pub fn list_executions(&self) -> impl ExactSizeIterator<Item = &ToolCallRequest> {
        self.executions.values()
    }

    fn complete_execution(&mut self, tool_call_id: &ToolCallID, event: &str) -> Result<()> {
        if self.executions.remove(tool_call_id).is_none() {
            return Err(Error::invalid_argument(format!(
                "tool call ID {:?} has no in-flight request for {event}",
                tool_call_id.as_str()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "state_test.rs"]
mod tests;
