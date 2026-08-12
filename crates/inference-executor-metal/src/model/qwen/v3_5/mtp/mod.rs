use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;
use inference_runtime_core::compute::BatchDeviceRequest;

use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::backend::add_gqa_private_replay_arguments;
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
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPMLPReplayTopology;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod layer;

pub const QWEN35_MTP_GQA_LAYER_INDEX: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.mtp.gqa_layer_index");
const QWEN35_MTP_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.mtp.num_active_tokens");
const QWEN35_MTP_FIRST_CACHE_LANE: usize = 1;

pub struct Qwen35MTP {
    layer: Qwen35MTPLayer,
    output_norm: RMSNorm,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    num_cache_pages: usize,
    replay_bucket_policy: ReplayBucketPolicy,
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
    pub fn new(
        device: &Device,
        main_config: &Qwen35ModelConfig,
        config: &Qwen35ModelConfig,
        max_tokens: usize,
        defaults: Qwen35MetalDefaults,
        gqa_state: &Qwen3xGQAState,
        num_cache_pages: usize,
        layer_scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let layer = Qwen35MTPLayer::new(
            device,
            config,
            defaults,
            main_config.text_config.num_hidden_layers,
            gqa_state,
            layer_scratch,
            dense_scratch,
            moe_scratch,
        )?;
        let max_tokens = max_tokens
            .try_into()
            .expect("qwen3.5 MTP replay token capacity must fit u32");
        let mut topology_boundaries = gqa_state.replay_token_topology_boundaries().into_vec();
        topology_boundaries.extend(layer.mlp_replay_topology_boundaries());
        Ok(Self {
            layer,
            output_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps),
            request_page_table: Some(Rc::clone(gqa_state.request_page_table())),
            num_cache_pages,
            replay_bucket_policy: mtp_replay_bucket_policy(max_tokens, topology_boundaries),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        main_config: &Qwen35ModelConfig,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35LayerWeightBindings,
        final_norm_weight: String,
    ) -> Result<(), ModelExecutorError> {
        self.layer.load_weights(
            device,
            store,
            config,
            defaults,
            main_config.text_config.num_hidden_layers,
            bindings,
        )?;
        let hidden_dim = config.text_config.hidden_size;
        let mut tensors = store.load_tensors([final_norm_weight.as_str()])?;
        self.output_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &final_norm_weight,
            &[hidden_dim],
        )?);
        assert!(tensors.is_empty(), "qwen3.5 MTP must consume its final norm tensor map");
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.output_norm.unload_weights();
        self.layer.unload_weights();
    }

    pub fn unload_state(&mut self) {
        self.layer.unload_state();
        self.request_page_table
            .take()
            .expect("qwen3.5 MTP request page-table state must be loaded");
    }

    pub fn load_state(&mut self, state: &Qwen3xGQAState) {
        assert!(
            self.request_page_table.is_none(),
            "qwen3.5 MTP request page-table state is already loaded"
        );
        self.request_page_table = Some(Rc::clone(state.request_page_table()));
        self.layer.load_state(state);
    }

    pub fn replay_token_capacity(&self, num_active_tokens: u32) -> u32 {
        self.replay_bucket_policy.capacity(num_active_tokens)
    }

    pub fn prepare_replay(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Qwen35MTPReplayKey {
        self.bucketed_replay_key(num_active_tokens, gqa_shape, gqa_topology)
    }

    pub fn replay_arguments(
        &self,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gqa_layer_index: u32,
    ) -> ReplayArguments {
        self.validate_bucketed_capacity(gqa_shape.num_tokens, gqa_shape.total_tokens);
        mtp_replay_arguments(gqa_shape, gqa_topology, gqa_layer_index)
    }

    pub fn prepare_pages(&self, batch: &BatchDeviceRequest) {
        let request_page_table = self.request_page_table();
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
                    self.write_page_ids(
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

    pub fn write_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize, page_ids: &[u32]) {
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a qwen3.5 MTP page ID outside the cache-page buffer"
        );
        self.request_page_table()
            .write_page_ids(req_slot, layer_index, block_index, page_ids);
    }

    pub fn read_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize) -> Vec<u32> {
        self.request_page_table()
            .read_page_ids(req_slot, layer_index, block_index)
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
            .request_page_table()
            .num_blocks()
            .checked_mul(self.request_page_table().num_page_ids_per_block())
            .expect("qwen3.5 MTP GQA page capacity must fit usize");
        let tokens_per_page = self.layer.gqa_tokens_per_page();
        assert!(
            max_context_tokens.div_ceil(tokens_per_page.max(1)) <= page_capacity,
            "qwen3.5 MTP GQA request context exceeds page-table capacity"
        );
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("qwen3.5 MTP request page-table state must be loaded before execution")
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
            .record_with_barrier(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }

    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        args: Qwen35MTPArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_bucketed_capacity(args.num_tokens, num_total_tokens);
        let hidden = self.layer.record_bucketed(
            recorder,
            num_total_tokens,
            QWEN35_MTP_NUM_ACTIVE_TOKENS,
            Qwen35MTPLayerInput {
                gqa: args.gqa,
                num_tokens: args.num_tokens,
                pages: args.pages,
                residual_input: args.hidden_input,
            },
        );
        self.output_norm.record_bucketed_with_barrier(
            recorder,
            num_total_tokens,
            QWEN35_MTP_NUM_ACTIVE_TOKENS,
            hidden,
            args.hidden_output,
        );
        args.hidden_output
    }

    fn bucketed_replay_key(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Qwen35MTPReplayKey {
        gqa_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, num_active_tokens,
            "qwen3.5 MTP GQA active tokens must match the stage"
        );
        let num_total_tokens = gqa_shape.total_tokens;
        self.validate_bucketed_capacity(num_active_tokens, num_total_tokens);
        Qwen35MTPReplayKey::for_bucketed(
            num_total_tokens,
            gqa_shape,
            gqa_topology,
            self.layer.mlp_replay_topology(num_total_tokens),
        )
    }

    fn validate_bucketed_capacity(&self, num_active_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.replay_token_capacity(num_active_tokens),
            num_total_tokens,
            "qwen3.5 MTP metadata token capacity must match the stage policy"
        );
        assert_eq!(
            self.layer.mlp_replay_topology(num_active_tokens),
            self.layer.mlp_replay_topology(num_total_tokens),
            "qwen3.5 MTP replay token capacity must preserve the MLP topology"
        );
    }
}

fn mtp_replay_bucket_policy(max_tokens: u32, mut topology_boundaries: Vec<u32>) -> ReplayBucketPolicy {
    topology_boundaries.sort_unstable();
    topology_boundaries.dedup();
    ReplayBucketPolicy::with_topology_boundaries(max_tokens, &topology_boundaries)
}

fn mtp_replay_arguments(
    gqa_shape: GQAReplayShape,
    gqa_topology: GQAReplayTopology,
    gqa_layer_index: u32,
) -> ReplayArguments {
    let mut arguments = ReplayArguments::new().with_u32(QWEN35_MTP_NUM_ACTIVE_TOKENS, gqa_shape.num_tokens);
    add_gqa_private_replay_arguments(gqa_shape, gqa_topology, &mut arguments);
    arguments.set_u32(QWEN35_MTP_GQA_LAYER_INDEX, gqa_layer_index);
    arguments
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPReplayKey {
    mode: Qwen35MTPReplayMode,
    gqa: Qwen35GQAReplayKey,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Qwen35MTPReplayMode {
    Legacy {
        num_tokens: usize,
    },
    Bucketed {
        num_total_tokens: u32,
        mlp_topology: Qwen35MTPMLPReplayTopology,
    },
}

impl Qwen35MTPReplayKey {
    /// Creates a source-compatible legacy exact/manual identity.
    ///
    /// Production replay uses the MTP-owned bucket policy and records the MLP
    /// topology through [`Qwen35MTP::prepare_replay`].
    pub fn new(
        num_tokens: usize,
        gqa_shape: inference_executor_core::attn::GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Self {
        gqa_shape.validate();
        Self {
            mode: Qwen35MTPReplayMode::Legacy { num_tokens },
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
        }
    }

    fn for_bucketed(
        num_total_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        mlp_topology: Qwen35MTPMLPReplayTopology,
    ) -> Self {
        gqa_shape.validate();
        assert_eq!(
            gqa_shape.total_tokens, num_total_tokens,
            "qwen3.5 MTP GQA key capacity must match the stage"
        );
        Self {
            mode: Qwen35MTPReplayMode::Bucketed {
                num_total_tokens,
                mlp_topology,
            },
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
        }
    }

    fn bucketed_num_total_tokens(&self) -> u32 {
        match self.mode {
            Qwen35MTPReplayMode::Legacy { .. } => {
                panic!("legacy qwen3.5 MTP replay key does not select a bucketed token capacity")
            },
            Qwen35MTPReplayMode::Bucketed { num_total_tokens, .. } => num_total_tokens,
        }
    }
}

impl ReplayComponent for Qwen35MTP {
    type Key = Qwen35MTPReplayKey;
    type Input<'a> = Qwen35MTPArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.bucketed_replay_key(input.num_tokens, input.gqa.replay_shape(), input.gqa_replay_topology)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        self.record_bucketed(recorder, key.bucketed_num_total_tokens(), *input);
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use inference_backend_metal::components::GQAComputePath;
    use inference_backend_metal::components::QuantizedDenseMLPReplayTopology;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
    use inference_executor_core::attn::GQAPageTableLayout;
    use inference_executor_core::model::qwen::v3_5::Qwen35TextConfig;
    use inference_runtime_core::compute::BatchDeviceRequest;
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::Token;

    use super::*;
    use crate::attn::gqa::backend::GQA_NUM_ACTIVE_Q_TOKEN_TILES;
    use crate::attn::gqa::backend::GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES;
    use crate::mlp::moe::backend::GatedMoEComputePath;
    use crate::mlp::moe::backend::GatedMoEReplayTopology;
    use crate::model::qwen::v3_5::plan::qwen35_dense_mlp_core_and_metal;
    use crate::model::qwen::v3_5::plan::qwen35_gqa_core_and_metal;

    fn gqa_topology() -> GQAReplayTopology {
        GQAReplayTopology {
            compute_path: GQAComputePath::SingleQueryToken {
                kv_token_tile_size: 256,
                num_threads_per_threadblock: 256,
                q_head_tile_size: 6,
            },
            qgkv_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }

    fn dense_topology() -> Qwen35MTPMLPReplayTopology {
        Qwen35MTPMLPReplayTopology::Dense(QuantizedDenseMLPReplayTopology {
            gate_up_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            down_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        })
    }

    fn moe_topology() -> Qwen35MTPMLPReplayTopology {
        Qwen35MTPMLPReplayTopology::MoE(GatedMoEReplayTopology {
            compute_path: GatedMoEComputePath::TokenMajor,
            router_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            shared_expert_gate_affine: None,
            shared_experts_dense: None,
        })
    }

    fn gqa_shape(num_tokens: u32, total_tokens: u32) -> GQAReplayShape {
        GQAReplayShape::new(num_tokens, total_tokens, 1, 2, 2, 4, false)
    }

    #[test]
    fn test_mtp_policy_composes_base_buckets_and_topology_boundaries() {
        let policy = mtp_replay_bucket_policy(16, vec![10, 5, 10]);

        assert_eq!(policy.buckets(), [1, 2, 4, 6, 8, 9, 12, 16]);
        assert_eq!(policy.capacity(3), 4);
        assert_eq!(policy.capacity(4), 4);
        assert_eq!(policy.capacity(5), 6);
        assert_eq!(policy.capacity(9), 9);
        assert_eq!(policy.capacity(10), 12);
    }

    #[test]
    fn test_mtp_arguments_use_stage_token_private_gqa_values_and_layer_index() {
        let shape = gqa_shape(3, 4);
        let single_arguments = mtp_replay_arguments(shape, gqa_topology(), 7);
        assert_eq!(
            single_arguments,
            ReplayArguments::new()
                .with_u32(QWEN35_MTP_NUM_ACTIVE_TOKENS, 3)
                .with_u32(GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 2)
                .with_u32(QWEN35_MTP_GQA_LAYER_INDEX, 7)
        );

        let tiled_topology = GQAReplayTopology {
            compute_path: GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            },
            ..gqa_topology()
        };
        let tiled_arguments = mtp_replay_arguments(shape, tiled_topology, 8);
        assert_eq!(
            tiled_arguments,
            ReplayArguments::new()
                .with_u32(QWEN35_MTP_NUM_ACTIVE_TOKENS, 3)
                .with_u32(GQA_NUM_ACTIVE_Q_TOKEN_TILES, 1)
                .with_u32(GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 2)
                .with_u32(QWEN35_MTP_GQA_LAYER_INDEX, 8)
        );
    }

    #[test]
    fn test_bucketed_mtp_key_ignores_active_count_and_isolates_legacy_mode() {
        let active_three = Qwen35MTPReplayKey::for_bucketed(4, gqa_shape(3, 4), gqa_topology(), dense_topology());
        let active_four = Qwen35MTPReplayKey::for_bucketed(4, gqa_shape(4, 4), gqa_topology(), dense_topology());
        let legacy = Qwen35MTPReplayKey::new(3, gqa_shape(3, 4), gqa_topology());

        assert_eq!(active_three, active_four);
        assert_ne!(active_three, legacy);
        assert_eq!(active_three.bucketed_num_total_tokens(), 4);
    }

    #[test]
    fn test_bucketed_mtp_key_separates_capacity_gqa_and_mlp_topology() {
        let base = Qwen35MTPReplayKey::for_bucketed(4, gqa_shape(3, 4), gqa_topology(), dense_topology());
        let different_capacity = Qwen35MTPReplayKey::for_bucketed(6, gqa_shape(3, 6), gqa_topology(), dense_topology());
        let different_gqa_capacity = Qwen35MTPReplayKey::for_bucketed(
            4,
            GQAReplayShape::new(3, 4, 1, 4, 2, 4, false),
            gqa_topology(),
            dense_topology(),
        );
        let different_gqa_topology = Qwen35MTPReplayKey::for_bucketed(
            4,
            gqa_shape(3, 4),
            GQAReplayTopology {
                qgkv_affine: AffineQuantizedMatmulKernelKind::QmmBm8Bn32,
                ..gqa_topology()
            },
            dense_topology(),
        );
        let different_mlp_topology =
            Qwen35MTPReplayKey::for_bucketed(4, gqa_shape(3, 4), gqa_topology(), moe_topology());

        assert_ne!(base, different_capacity);
        assert_ne!(base, different_gqa_capacity);
        assert_ne!(base, different_gqa_topology);
        assert_ne!(base, different_mlp_topology);
    }

    #[test]
    fn test_prepare_pages_maps_cache_lanes_to_gqa_layers() {
        let device = Device::system_default();
        let mtp = test_mtp(&device);
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

        mtp.prepare_pages(&batch);

        assert_eq!(mtp.read_page_ids(1, 0, 1), vec![10, 11]);
        assert_eq!(mtp.read_page_ids(1, 0, 2), vec![12, 13]);
        assert_eq!(mtp.read_page_ids(1, 1, 1), vec![20, 21]);
        assert_eq!(mtp.read_page_ids(1, 1, 2), vec![22, 23]);
        assert_eq!(mtp.read_page_ids(1, 2, 1), vec![30, 31]);
        assert_eq!(mtp.read_page_ids(1, 2, 2), vec![32, 33]);
    }

    fn test_mtp(device: &Device) -> Qwen35MTP {
        let main_config = test_model_config();
        let mtp_config = test_model_config();
        let defaults = Qwen35MetalDefaults::default();
        let (gqa_core, gqa_metal) = qwen35_gqa_core_and_metal(
            main_config.text_config.num_hidden_layers,
            &mtp_config.text_config,
            defaults,
        )
        .unwrap();
        let gqa_state = Qwen3xGQAState::new(
            device,
            gqa_core,
            gqa_metal,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 3,
                num_blocks: 4,
                num_page_ids_per_block: 2,
            },
            2,
            64,
        );
        let (dense_core, dense_metal) = qwen35_dense_mlp_core_and_metal(0, &mtp_config.text_config, defaults).unwrap();
        let dense_scratch = Rc::new(DenseMLPScratch::new(device, &dense_core, dense_metal.io_dtype, 2));
        Qwen35MTP::new(
            device,
            &main_config,
            &mtp_config,
            2,
            defaults,
            &gqa_state,
            64,
            Rc::new(Qwen35MTPLayerScratch::new(
                device,
                2,
                mtp_config.text_config.hidden_size,
            )),
            Some(&dense_scratch),
            None,
        )
        .unwrap()
    }

    fn test_model_config() -> Qwen35ModelConfig {
        Qwen35ModelConfig {
            model_type: "qwen3_5".to_string(),
            tie_word_embeddings: false,
            text_config: Qwen35TextConfig {
                model_type: "qwen3_5_text".to_string(),
                hidden_size: 128,
                hidden_act: "silu".to_string(),
                intermediate_size: 128,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 128,
                rms_norm_eps: 1e-6,
                vocab_size: 16,
                max_position_embeddings: 32,
                attention_bias: false,
                tie_word_embeddings: false,
                layer_types: vec!["full_attention".to_string()],
                full_attention_interval: 1,
                linear_num_value_heads: 1,
                linear_num_key_heads: 1,
                linear_key_head_dim: 128,
                linear_value_head_dim: 128,
                linear_conv_kernel_dim: 4,
                decoder_sparse_step: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                shared_expert_intermediate_size: 0,
                moe_intermediate_size: 0,
                norm_topk_prob: true,
                mtp_num_hidden_layers: 1,
                mtp_use_dedicated_embeddings: false,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_parameters: None,
                use_cache: true,
                dtype: None,
                scale: 1.0 / 128.0_f32.sqrt(),
                rope_dim: 128,
            },
            quantization: None,
        }
    }
}
