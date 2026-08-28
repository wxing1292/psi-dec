use std::sync::Arc;

use inference_backend_metal::components::resource_embed;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_runtime_core::compute::DeviceRequest;

use crate::def::replay_op::MetalReplayRuntime;
use crate::model::resource_arena::MetalResourceArena;
use crate::model::resource_embed::ResourceEmbed;
use crate::model::resource_embed::ResourceEmbedInput;
use crate::model::resource_embed::build_mapping_table;
use crate::replay::Replay;

pub enum Qwen3InputEmbedding {
    Text,
    Resource(Box<Qwen3ResourceEmbedding>),
}

pub struct Qwen3ResourceEmbedding {
    config: resource_embed::Config,
    arena: Arc<MetalResourceArena>,
    mappings: Buffer,
    replay: Replay<ResourceEmbed>,
    prepared_table: Option<resource_embed::MappingTable>,
}

impl Qwen3InputEmbedding {
    pub fn resource(
        device: &Device,
        config: resource_embed::Config,
        arena: Arc<MetalResourceArena>,
        max_tokens: usize,
    ) -> Self {
        assert!(max_tokens > 0, "Qwen3 ResourceEmbed requires token capacity");
        assert!(
            u32::try_from(max_tokens).is_ok(),
            "Qwen3 ResourceEmbed token capacity must fit u32"
        );
        let num_mapping_values = max_tokens
            .checked_mul(3)
            .expect("Qwen3 ResourceEmbed mapping capacity must fit usize");
        Self::Resource(Box::new(Qwen3ResourceEmbedding {
            config,
            arena,
            mappings: Buffer::new_zeroed_elements(
                device,
                num_mapping_values,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            replay: Replay::new("qwen3 ResourceEmbed", ResourceEmbed::new(device, config)),
            prepared_table: None,
        }))
    }

    pub fn prepare(&mut self, requests: &[DeviceRequest]) {
        match self {
            Self::Text => {
                debug_assert!(
                    requests.iter().all(|request| request.resource_placements.is_empty()),
                    "text-only Qwen3 does not accept resource placements"
                );
            },
            Self::Resource(resource) => resource.prepare(requests),
        }
    }

    pub fn prepare_replay(&self) -> Option<(resource_embed::Shape, ReplayArguments)> {
        match self {
            Self::Text => None,
            Self::Resource(resource) => {
                resource
                    .prepared_table
                    .as_ref()
                    .map(|table| resource.replay.component().prepare_replay(table))
            },
        }
    }

    pub fn record(&mut self, runtime: &MetalReplayRuntime<'_>, hidden: &Buffer) -> Option<resource_embed::Shape> {
        let Self::Resource(resource) = self else {
            return None;
        };
        let table = resource.prepared_table.as_ref()?;
        let input = ResourceEmbedInput {
            table,
            arena: resource.arena.storage().buffer(),
            mappings: &resource.mappings,
            hidden,
        };
        Some(resource.replay.record(runtime, &input).0)
    }

    pub fn replay(&self, key: &resource_embed::Shape) -> &inference_backend_metal::metal::ReplayProgram {
        match self {
            Self::Text => panic!("text-only Qwen3 has no ResourceEmbed replay"),
            Self::Resource(resource) => resource.replay.replay(key),
        }
    }

    pub fn clear(&mut self) {
        if let Self::Resource(resource) = self {
            resource.replay.clear();
        }
    }
}

impl Qwen3ResourceEmbedding {
    fn prepare(&mut self, requests: &[DeviceRequest]) {
        self.prepared_table = build_mapping_table(self.config, requests);
        if let Some(table) = &self.prepared_table {
            self.mappings.write_typed(0, table.encoded_u32s());
        }
    }
}
