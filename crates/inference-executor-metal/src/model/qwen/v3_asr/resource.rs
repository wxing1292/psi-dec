use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_asr::QWEN3_ASR_AUDIO_RESOURCE_TYPE;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRModelConfig;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRWeightBindings;
use inference_runtime_core::Error;
use inference_runtime_core::memory::BlockAllocator;
use inference_runtime_core::memory::OffsetAllocation;
use inference_runtime_core::runtime::ConcreteResource;
use inference_runtime_core::runtime::ResourceID;
use inference_runtime_core::runtime::ResourceURI;
use inference_runtime_core::runtime::resource::processor::ResourceProcessFuture;
use inference_runtime_core::runtime::resource::processor::ResourceProcessor;

use super::AudioEncoderExecutor;
use crate::model::resource_arena::MetalResourceArena;

pub struct Qwen3ASRAudioProcessor {
    executor: Arc<AudioEncoderExecutor>,
    arena: Arc<MetalResourceArena>,
    sources: Arc<AudioSourceStore>,
}

pub struct AudioSourceRegistration {
    resource_id: ResourceID,
    sources: Arc<AudioSourceStore>,
}

#[derive(Default)]
struct AudioSourceStore {
    sources: Mutex<HashMap<ResourceID, (ResourceURI, Arc<Qwen3ASRAudioSource>)>>,
}

impl Qwen3ASRAudioProcessor {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &Qwen3ASRModelConfig,
        bindings: &Qwen3ASRWeightBindings,
        arena: Arc<MetalResourceArena>,
    ) -> std::result::Result<Arc<Self>, ModelExecutorError> {
        let sources = Arc::new(AudioSourceStore::default());
        let executor = Arc::new(AudioEncoderExecutor::load(
            model_dir,
            config.audio.clone(),
            bindings.audio.clone(),
            Arc::clone(&arena),
        )?);
        Ok(Arc::new(Self {
            executor,
            arena,
            sources,
        }))
    }

    pub fn encoder_executor(&self) -> Arc<AudioEncoderExecutor> {
        Arc::clone(&self.executor)
    }

    pub fn register_source(
        &self,
        resource_id: ResourceID,
        source: Qwen3ASRAudioSource,
    ) -> (ResourceURI, AudioSourceRegistration) {
        assert_eq!(
            resource_id.resource_type(),
            QWEN3_ASR_AUDIO_RESOURCE_TYPE,
            "Qwen3-ASR audio source must use the Qwen3-ASR audio resource type"
        );
        let uri = ResourceURI::new(format!("qwen3-asr://prepared/{}", resource_id.uuid()));
        let registration = self.sources.register(resource_id, uri.clone(), source);
        (uri, registration)
    }
}

impl AudioSourceStore {
    fn register(
        self: &Arc<Self>,
        resource_id: ResourceID,
        uri: ResourceURI,
        source: Qwen3ASRAudioSource,
    ) -> AudioSourceRegistration {
        let previous = self
            .sources
            .lock()
            .unwrap()
            .insert(resource_id, (uri, Arc::new(source)));
        assert!(previous.is_none(), "Qwen3-ASR audio resource ID must be unique");
        AudioSourceRegistration {
            resource_id,
            sources: Arc::clone(self),
        }
    }

    fn resolve(&self, resource_id: ResourceID) -> Option<(ResourceURI, Arc<Qwen3ASRAudioSource>)> {
        self.sources.lock().unwrap().get(&resource_id).cloned()
    }
}

impl Drop for AudioSourceRegistration {
    fn drop(&mut self) {
        let source = self.sources.sources.lock().unwrap().remove(&self.resource_id);
        assert!(source.is_some(), "registered Qwen3-ASR audio source must exist");
    }
}

impl ResourceProcessor for Qwen3ASRAudioProcessor {
    fn process<'a>(&'a self, resource_id: ResourceID) -> ResourceProcessFuture<'a> {
        let source = self.sources.resolve(resource_id);
        Box::pin(async move {
            let (uri, source) = source.ok_or_else(|| {
                Error::cancelled(format!(
                    "Qwen3-ASR audio source {} is no longer registered",
                    resource_id.uuid()
                ))
            })?;
            let num_resource_tokens = source.num_resource_tokens();
            let allocation = self.executor.encode(source).await?;
            let allocator: Arc<dyn BlockAllocator<BlockSegment = OffsetAllocation>> = self.arena.clone();
            Ok(ConcreteResource::new(
                resource_id,
                uri,
                allocator,
                allocation,
                num_resource_tokens as u32,
            ))
        })
    }
}

#[cfg(test)]
#[path = "resource_test.rs"]
mod tests;
