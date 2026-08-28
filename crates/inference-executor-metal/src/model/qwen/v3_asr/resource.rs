use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use inference_backend_metal::metal::Device;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_asr::QWEN3_ASR_AUDIO_RESOURCE_TYPE;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRModelConfig;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRWeightBindings;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::memory::BlockAllocator;
use inference_runtime_core::memory::OffsetAllocation;
use inference_runtime_core::runtime::ConcreteResource;
use inference_runtime_core::runtime::ResourceID;
use inference_runtime_core::runtime::ResourceURI;
use inference_runtime_core::runtime::resource::processor::ResourceProcessFuture;
use inference_runtime_core::runtime::resource::processor::ResourceProcessor;

use super::AudioTower;
use crate::model::resource_arena::MetalResourceArena;

pub struct Qwen3ASRAudioProcessor {
    jobs: async_channel::Sender<AudioMaterializationJob>,
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

struct AudioMaterializationJob {
    resource_id: ResourceID,
    uri: ResourceURI,
    source: Arc<Qwen3ASRAudioSource>,
    response: async_channel::Sender<Result<ConcreteResource>>,
}

impl Qwen3ASRAudioProcessor {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &Qwen3ASRModelConfig,
        bindings: &Qwen3ASRWeightBindings,
        arena: Arc<MetalResourceArena>,
    ) -> std::result::Result<Arc<Self>, ModelExecutorError> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let audio_config = config.audio.clone();
        let audio_bindings = bindings.audio.clone();
        let hidden_dim_bytes = config.audio.output_dim * size_of::<half::bf16>();
        let sources = Arc::new(AudioSourceStore::default());
        let (jobs, worker_jobs) = async_channel::unbounded();
        let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("qwen3-asr-audio".to_string())
            .spawn(move || {
                let device = Device::system_default();
                let result = SafeTensorStore::from_model_dir(&model_dir)
                    .and_then(|mut store| AudioTower::load(&device, &mut store, audio_config, audio_bindings));
                match result {
                    Ok(tower) => {
                        if ready_sender.send(Ok(())).is_err() {
                            return;
                        }
                        run_worker(&tower, &arena, hidden_dim_bytes, worker_jobs);
                    },
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                    },
                }
            })
            .map_err(|error| ModelExecutorError::custom(format!("unable to start Qwen3-ASR audio worker: {error}")))?;
        ready_receiver.recv().map_err(|error| {
            ModelExecutorError::custom(format!("Qwen3-ASR audio worker stopped during init: {error}"))
        })??;
        Ok(Arc::new(Self { jobs, sources }))
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
            let (response, result) = async_channel::bounded(1);
            self.jobs
                .send(AudioMaterializationJob {
                    resource_id,
                    uri,
                    source,
                    response,
                })
                .await
                .map_err(|_| Error::unavailable("Qwen3-ASR audio worker is not available"))?;
            result
                .recv()
                .await
                .map_err(|_| Error::unavailable("Qwen3-ASR audio worker stopped before it returned a resource"))?
        })
    }
}

fn run_worker(
    tower: &AudioTower,
    arena: &Arc<MetalResourceArena>,
    hidden_dim_bytes: usize,
    jobs: async_channel::Receiver<AudioMaterializationJob>,
) {
    while let Ok(job) = jobs.recv_blocking() {
        let result = materialize(tower, arena, hidden_dim_bytes, job.resource_id, job.uri, &job.source);
        let _ = job.response.send_blocking(result);
    }
}

fn materialize(
    tower: &AudioTower,
    arena: &Arc<MetalResourceArena>,
    hidden_dim_bytes: usize,
    resource_id: ResourceID,
    uri: ResourceURI,
    source: &Qwen3ASRAudioSource,
) -> Result<ConcreteResource> {
    let num_resource_tokens = source.num_resource_tokens();
    let len_bytes = num_resource_tokens * hidden_dim_bytes;
    let allocation = arena.alloc_segment(len_bytes)?;
    tower.encode(source, arena, &allocation);
    let allocator: Arc<dyn BlockAllocator<BlockSegment = OffsetAllocation>> = arena.clone();
    Ok(ConcreteResource::new(
        resource_id,
        uri,
        allocator,
        allocation,
        num_resource_tokens as u32,
    ))
}

#[cfg(test)]
#[path = "resource_test.rs"]
mod tests;
