use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_lite::future::Boxed;

use crate::Result;
use crate::runtime::RawRequestID;
use crate::runtime::resource::ConcreteResource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourceTypeID;
use crate::runtime::tasks::AsyncTaskReq;
use crate::runtime::tasks::AsyncTaskResp;

pub type ResourceFuture<'a> = Pin<Box<dyn Future<Output = Result<ConcreteResource>> + Send + 'a>>;

pub trait ResourceTypeProcessor: Send + Sync + 'static {
    fn materialize<'a>(&'a self, resource_id: ResourceID) -> ResourceFuture<'a>;
}

/// Maps runtime-neutral resource type IDs to model-specific resource processors.
#[derive(Default)]
pub struct ResourceProcessor {
    processors: HashMap<ResourceTypeID, Arc<dyn ResourceTypeProcessor>>,
}

impl ResourceProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a processor before the registry is shared with runtime requests.
    pub fn register(&mut self, resource_type: ResourceTypeID, processor: Arc<dyn ResourceTypeProcessor>) {
        assert!(
            self.processors.insert(resource_type, processor).is_none(),
            "resource processor type ID {} must be registered exactly once",
            resource_type.value()
        );
    }

    pub fn unregister(&mut self, resource_type: ResourceTypeID) -> Arc<dyn ResourceTypeProcessor> {
        self.processors.remove(&resource_type).unwrap_or_else(|| {
            panic!(
                "resource processor type ID {} must be registered before it is unregistered",
                resource_type.value()
            )
        })
    }

    pub fn processor(&self, resource_id: ResourceID) -> Arc<dyn ResourceTypeProcessor> {
        self.processors
            .get(&resource_id.resource_type())
            .unwrap_or_else(|| {
                panic!(
                    "no resource processor is registered for resource type ID {}",
                    resource_id.resource_type().value()
                )
            })
            .clone()
    }
}

pub struct ResourceMaterializationReq {
    request_id: RawRequestID,
    resource_ids: Vec<ResourceID>,
    resource_processor: Arc<ResourceProcessor>,
}

impl ResourceMaterializationReq {
    pub fn new(
        request_id: RawRequestID,
        resource_ids: Vec<ResourceID>,
        resource_processor: Arc<ResourceProcessor>,
    ) -> Self {
        debug_assert!(
            !resource_ids.is_empty(),
            "resource materialization request requires at least one resource ID"
        );
        Self {
            request_id,
            resource_ids,
            resource_processor,
        }
    }

    async fn materialize(self) -> Result<Vec<ConcreteResource>> {
        let mut concrete_resources = Vec::with_capacity(self.resource_ids.len());
        for resource_id in self.resource_ids {
            let processor = self.resource_processor.processor(resource_id);
            let resource = processor.materialize(resource_id).await?;
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
