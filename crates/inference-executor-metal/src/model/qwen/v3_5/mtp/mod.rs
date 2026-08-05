use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_runtime_core::compute::BatchDeviceRequest;

use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::Qwen35GQAReplayKey;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayer;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerInput;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod layer;

pub const QWEN35_MTP_GQA_LAYER_INDEX: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.mtp.gqa_layer_index");
const QWEN35_MTP_FIRST_CACHE_LANE: usize = 1;

pub struct Qwen35MTP {
    layer: Qwen35MTPLayer,
    output_norm: RMSNorm,
    request_page_table: Rc<GQARequestPageTable>,
    num_cache_pages: usize,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gqa_replay_topology: GQAReplayTopology,
    pub pages: &'a Buffer,
}

impl Qwen35MTP {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        main_config: &Qwen35ModelConfig,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35LayerWeightBindings,
        final_norm_weight: String,
        gqa_state: &Qwen3xGQAState,
        num_cache_pages: usize,
        layer_scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let layer = Qwen35MTPLayer::load(
            device,
            store,
            config,
            defaults,
            main_config.text_config.num_hidden_layers,
            bindings,
            gqa_state,
            layer_scratch,
            dense_scratch,
            moe_scratch,
        )?;
        let mut tensors = store.load_tensors([final_norm_weight.as_str()])?;
        let final_norm_weight = remove_qwen3x_norm_weight(device, &mut tensors, &final_norm_weight, &[hidden_dim])?;
        assert!(tensors.is_empty(), "qwen3.5 MTP must consume its final norm tensor map");
        Ok(Self {
            layer,
            output_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps, final_norm_weight),
            request_page_table: Rc::clone(gqa_state.request_page_table()),
            num_cache_pages,
        })
    }

    pub fn prepare_pages(&self, batch: &BatchDeviceRequest) {
        prepare_request_page_table(&self.request_page_table, batch, self.num_cache_pages);
    }

    pub fn validate_batch(&self, microbatch: &Qwen35Microbatch) {
        let max_context_tokens = (0..microbatch.num_reqs())
            .map(|req_index| {
                microbatch.token_indices()[req_index]
                    .checked_add(microbatch.q_len(req_index))
                    .expect("qwen3.5 MTP GQA request context length overflow")
            })
            .max()
            .expect("qwen3.5 MTP batch requires requests") as usize;
        let page_capacity = self
            .request_page_table
            .num_blocks()
            .checked_mul(self.request_page_table.num_page_ids_per_block())
            .expect("qwen3.5 MTP GQA page capacity must fit usize");
        let tokens_per_page = self.layer.gqa_tokens_per_page();
        assert!(
            max_context_tokens.div_ceil(tokens_per_page.max(1)) <= page_capacity,
            "qwen3.5 MTP GQA request context exceeds page-table capacity"
        );
    }
}

fn prepare_request_page_table(
    request_page_table: &GQARequestPageTable,
    batch: &BatchDeviceRequest,
    num_cache_pages: usize,
) {
    assert!(
        request_page_table.num_layers() > 0,
        "qwen3.5 MTP page table requires logical layers"
    );
    for request in &batch.dev_reqs {
        let page_ids_by_lane_and_block = request.decoder_sync_blocks.kv_page_ids();
        for gqa_layer_index in 0..request_page_table.num_layers() {
            let cache_lane = QWEN35_MTP_FIRST_CACHE_LANE
                .checked_add(gqa_layer_index)
                .expect("qwen3.5 MTP cache lane must fit usize");
            let page_ids_by_block = page_ids_by_lane_and_block
                .get(cache_lane)
                .unwrap_or_else(|| panic!("qwen3.5 MTP request page table missing cache lane {cache_lane}"));
            for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                assert_eq!(
                    page_ids.len(),
                    request_page_table.num_page_ids_per_block(),
                    "qwen3.5 MTP cache lane {cache_lane} must contain one physical layer's page IDs"
                );
                assert!(
                    page_ids.iter().all(|&page_id| (page_id as usize) < num_cache_pages),
                    "runtime supplied a qwen3.5 MTP page ID outside the cache-page buffer"
                );
                request_page_table.write_page_ids(
                    request.req_slot,
                    gqa_layer_index,
                    request
                        .decoder_sync_blocks
                        .block_index()
                        .checked_add(block_offset)
                        .expect("qwen3.5 MTP cache-block index must fit usize"),
                    page_ids,
                );
            }
        }
    }
}

impl Qwen35MTP {
    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MTPArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let hidden = self.layer.record(
            recorder,
            Qwen35MTPLayerInput {
                gqa: args.gqa,
                num_tokens,
                pages: args.pages,
                residual_input: args.hidden_input,
            },
        );
        self.output_norm
            .record(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPReplayKey {
    num_tokens: usize,
    gqa: Qwen35GQAReplayKey,
}

impl Qwen35MTPReplayKey {
    pub fn new(
        num_tokens: usize,
        gqa_shape: inference_executor_core::attn::GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Self {
        gqa_shape.validate();
        Self {
            num_tokens,
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
        }
    }
}

impl ReplayComponent for Qwen35MTP {
    type Key = Qwen35MTPReplayKey;
    type Input<'a> = Qwen35MTPArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::Key::new(
            input.num_tokens as usize,
            input.gqa.replay_shape(),
            input.gqa_replay_topology,
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35MTP::record(self, recorder, *input);
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GQAPageTableLayout;
    use inference_runtime_core::compute::BatchDeviceRequest;
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::Token;

    use super::GQARequestPageTable;
    use super::prepare_request_page_table;

    #[test]
    fn test_prepare_request_page_table_maps_cache_lanes_to_gqa_layers() {
        let device = Device::system_default();
        let page_table = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 3,
                num_blocks: 4,
                num_page_ids_per_block: 2,
            },
        );
        let batch = BatchDeviceRequest::new(
            1,
            [DeviceRequest::new(
                7,
                1,
                QueryTokens::Decode {
                    epoch: 0,
                    token_index: 1,
                    tokens: vec![Token::new(42)],
                    spec_tokens: Vec::new(),
                },
                DecoderSyncBlocks::new(
                    1,
                    vec![
                        vec![vec![1, 2]],
                        vec![vec![10, 11], vec![12, 13]],
                        vec![vec![20, 21], vec![22, 23]],
                        vec![vec![30, 31], vec![32, 33]],
                    ],
                    vec![Vec::new(); 4],
                ),
                SamplingConfig::default(),
            )],
        );

        prepare_request_page_table(&page_table, &batch, 64);

        assert_eq!(page_table.read_page_ids(1, 0, 1), vec![10, 11]);
        assert_eq!(page_table.read_page_ids(1, 0, 2), vec![12, 13]);
        assert_eq!(page_table.read_page_ids(1, 1, 1), vec![20, 21]);
        assert_eq!(page_table.read_page_ids(1, 1, 2), vec![22, 23]);
        assert_eq!(page_table.read_page_ids(1, 2, 1), vec![30, 31]);
        assert_eq!(page_table.read_page_ids(1, 2, 2), vec![32, 33]);
    }
}
