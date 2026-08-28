use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::ConcreteResource;
use super::ResourceID;
use super::ResourceTypeID;
use crate::Result;

pub type ResourceProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<ConcreteResource>> + Send + 'a>>;

pub trait ResourceProcessor: Send + Sync + 'static {
    fn process<'a>(&'a self, resource_id: ResourceID) -> ResourceProcessFuture<'a>;
}

/// Maps runtime-neutral resource type IDs to model-specific resource processors.
#[derive(Default)]
pub struct ResourceProcessors {
    processors: HashMap<ResourceTypeID, Arc<dyn ResourceProcessor>>,
}

impl ResourceProcessors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a processor before this `ResourceProcessors` collection is shared with runtime requests.
    pub fn register(&mut self, resource_type: ResourceTypeID, processor: Arc<dyn ResourceProcessor>) {
        assert!(
            self.processors.insert(resource_type, processor).is_none(),
            "resource processor type ID {} must be registered exactly once",
            resource_type.value()
        );
    }

    pub fn unregister(&mut self, resource_type: ResourceTypeID) -> Arc<dyn ResourceProcessor> {
        self.processors.remove(&resource_type).unwrap_or_else(|| {
            panic!(
                "resource processor type ID {} must be registered before it is unregistered",
                resource_type.value()
            )
        })
    }

    pub fn get(&self, resource_id: ResourceID) -> &dyn ResourceProcessor {
        self.processors
            .get(&resource_id.resource_type())
            .unwrap_or_else(|| {
                panic!(
                    "no resource processor is registered for resource type ID {}",
                    resource_id.resource_type().value()
                )
            })
            .as_ref()
    }
}
