use std::sync::Arc;

use futures_lite::future::Boxed;

use crate::Result;
use crate::runtime::RawRequestID;
use crate::runtime::resource::ConcreteResource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::processor::ResourceProcessors;
use crate::runtime::tasks::AsyncTaskReq;
use crate::runtime::tasks::AsyncTaskResp;

pub struct ResourceMaterializationReq {
    request_id: RawRequestID,
    resource_ids: Vec<ResourceID>,
    resource_processors: Arc<ResourceProcessors>,
}

impl ResourceMaterializationReq {
    pub fn new(
        request_id: RawRequestID,
        resource_ids: Vec<ResourceID>,
        resource_processors: Arc<ResourceProcessors>,
    ) -> Self {
        debug_assert!(
            !resource_ids.is_empty(),
            "resource materialization request requires at least one resource ID"
        );
        Self {
            request_id,
            resource_ids,
            resource_processors,
        }
    }

    async fn materialize(self) -> Result<Vec<ConcreteResource>> {
        let mut concrete_resources = Vec::with_capacity(self.resource_ids.len());
        for resource_id in self.resource_ids {
            let processor = self.resource_processors.get(resource_id);
            let resource = processor.process(resource_id).await?;
            assert_eq!(
                resource.id(),
                resource_id,
                "resource processor must preserve resource identity"
            );
            concrete_resources.push(resource);
        }
        Ok(concrete_resources)
    }
}

pub struct ResourceMaterializationResp {
    request_id: RawRequestID,
    result: Result<Vec<ConcreteResource>>,
}

impl ResourceMaterializationResp {
    pub fn into_result(self) -> Result<Vec<ConcreteResource>> {
        self.result
    }
}

impl AsyncTaskReq for ResourceMaterializationReq {
    type Resp = dyn AsyncTaskResp;

    fn request_id(&self) -> RawRequestID {
        self.request_id
    }

    fn run(self: Box<Self>) -> Boxed<Box<Self::Resp>> {
        Box::pin(async move {
            let request_id = self.request_id;
            // TODO: Retry retryable materialization failures. Abort after the retry limit.
            let result = self.materialize().await;
            Box::new(ResourceMaterializationResp { request_id, result }) as Box<dyn AsyncTaskResp>
        })
    }
}

impl AsyncTaskResp for ResourceMaterializationResp {
    fn request_id(&self) -> RawRequestID {
        self.request_id
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
