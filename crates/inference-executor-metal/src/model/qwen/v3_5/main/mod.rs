use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MainWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::attn::gdn::backend::GDNReplayTopology;
use crate::attn::gqa::backend::GQAReplayTopology;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3_5::Qwen35GQAReplayKey;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayer;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerInput;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerScratch;
use crate::model::qwen::v3_5::main::layer::Qwen35MainMLPReplayTopology;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod embed;
pub mod layer;
pub mod output;

const QWEN35_MAIN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.main.num_active_tokens");

pub struct Qwen35Main {
    layers: Vec<Qwen35MainLayer>,
    final_norm: RMSNorm,
    residual_capture: Option<Rc<dyn MainResidualCapture>>,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gqa_replay_topology: GQAReplayTopology,
    pub gdn: &'a crate::attn::gdn::batch_metadata::GDNMetadataBuffers,
    pub gdn_replay_topology: GDNReplayTopology,
    pub pages: &'a Buffer,
}

impl Qwen35Main {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        max_tokens: usize,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35MainWeightBindings,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
        residual_capture: Option<Rc<dyn MainResidualCapture>>,
        layer_scratch: Rc<Qwen35MainLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let Qwen35MainWeightBindings {
            final_norm_weight,
            layers: layer_bindings,
        } = bindings;
        assert_eq!(
            layer_bindings.len(),
            config.text_config.num_hidden_layers,
            "qwen3.5 Main config and checkpoint binding layer counts must match"
        );
        let mut num_loaded_gqa_layers = 0;
        let mut num_loaded_gdn_layers = 0;
        let mut layers = Vec::with_capacity(layer_bindings.len());
        for (layer_index, bindings) in layer_bindings.into_iter().enumerate() {
            let layer_type = config.layer_type_at(layer_index)?;
            let attn_layer_index = match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers,
                LayerType::GDN => num_loaded_gdn_layers,
            };
            layers.push(Qwen35MainLayer::load(
                device,
                store,
                config,
                defaults,
                layer_index,
                attn_layer_index,
                bindings,
                gqa_state,
                gdn_state,
                Rc::clone(&layer_scratch),
                dense_scratch,
                moe_scratch,
            )?);
            match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers += 1,
                LayerType::GDN => num_loaded_gdn_layers += 1,
            }
            store.unload_all();
        }

        let final_norm_weight =
            load_qwen3x_norm_weight(device, store, &final_norm_weight, &[config.text_config.hidden_size])?;
        let max_tokens = max_tokens
            .try_into()
            .expect("qwen3.5 Main replay token capacity must fit u32");
        let mut topology_boundaries = gqa_state.replay_token_topology_boundaries().into_vec();
        topology_boundaries.extend(gdn_state.replay_token_topology_boundaries());
        for layer in &layers {
            topology_boundaries.extend(layer.mlp_replay_topology_boundaries());
        }
        Ok(Self {
            layers,
            final_norm: RMSNorm::new(
                device,
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
                final_norm_weight,
            ),
            residual_capture,
            replay_bucket_policy: main_replay_bucket_policy(max_tokens, topology_boundaries),
        })
    }

    pub fn replay_token_capacity(&self, num_active_tokens: u32) -> u32 {
        self.replay_bucket_policy.capacity(num_active_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_replay(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
    ) -> (Qwen35MainReplayKey, ReplayArguments) {
        let key = self.bucketed_replay_key(num_active_tokens, gqa_shape, gqa_topology, gdn_shape, gdn_topology);
        let arguments = main_replay_arguments(num_active_tokens);
        (key, arguments)
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = <Qwen35MainLayer as ReplayLayer>::record(
                layer,
                recorder,
                Qwen35MainLayerInput {
                    gdn: args.gdn,
                    gqa: args.gqa,
                    num_tokens,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                    residual_capture_dest: self
                        .residual_capture
                        .as_ref()
                        .and_then(|capture| capture.capture_for_model_layer(layer.layer_index())),
                },
            );
        }
        self.final_norm
            .record_with_barrier(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }

    pub fn record_bucketed<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        args: Qwen35MainArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_bucketed_capacity(args.num_tokens, num_total_tokens);
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = layer.record_bucketed(
                recorder,
                num_total_tokens,
                QWEN35_MAIN_NUM_ACTIVE_TOKENS,
                Qwen35MainLayerInput {
                    gdn: args.gdn,
                    gqa: args.gqa,
                    num_tokens: args.num_tokens,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                    residual_capture_dest: self
                        .residual_capture
                        .as_ref()
                        .and_then(|capture| capture.capture_for_model_layer(layer.layer_index())),
                },
            );
        }
        self.final_norm.record_bucketed_with_barrier(
            recorder,
            num_total_tokens,
            QWEN35_MAIN_NUM_ACTIVE_TOKENS,
            hidden,
            args.hidden_output,
        );
        args.hidden_output
    }

    #[allow(clippy::too_many_arguments)]
    fn bucketed_replay_key(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
    ) -> Qwen35MainReplayKey {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, num_active_tokens,
            "qwen3.5 Main GQA active tokens must match the stage"
        );
        assert_eq!(
            gdn_shape.num_tokens, num_active_tokens,
            "qwen3.5 Main GDN active tokens must match the stage"
        );
        assert_eq!(
            gqa_shape.total_tokens, gdn_shape.total_tokens,
            "qwen3.5 Main GQA and GDN token capacities must match"
        );
        let num_total_tokens = gqa_shape.total_tokens;
        self.validate_bucketed_capacity(num_active_tokens, num_total_tokens);
        let mlp_topologies = self
            .layers
            .iter()
            .map(|layer| layer.mlp_replay_topology(num_total_tokens))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Qwen35MainReplayKey::for_bucketed(
            num_total_tokens,
            gqa_shape,
            gqa_topology,
            gdn_shape,
            gdn_topology,
            mlp_topologies,
        )
    }

    fn validate_bucketed_capacity(&self, num_active_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.replay_token_capacity(num_active_tokens),
            num_total_tokens,
            "qwen3.5 Main metadata token capacity must match the stage policy"
        );
        for layer in &self.layers {
            assert_eq!(
                layer.mlp_replay_topology(num_active_tokens),
                layer.mlp_replay_topology(num_total_tokens),
                "qwen3.5 Main replay token capacity must preserve every layer MLP topology"
            );
        }
    }
}

fn main_replay_bucket_policy(max_tokens: u32, mut topology_boundaries: Vec<u32>) -> ReplayBucketPolicy {
    topology_boundaries.sort_unstable();
    topology_boundaries.dedup();
    ReplayBucketPolicy::with_topology_boundaries(max_tokens, &topology_boundaries)
}

fn main_replay_arguments(num_active_tokens: u32) -> ReplayArguments {
    ReplayArguments::new().with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, num_active_tokens)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainReplayKey {
    mode: Qwen35MainReplayMode,
    gqa: Qwen35GQAReplayKey,
    gdn: Qwen35MainGDNReplayKey,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Qwen35MainReplayMode {
    Legacy {
        num_tokens: u32,
    },
    Bucketed {
        num_total_tokens: u32,
        mlp_topologies: Box<[Qwen35MainMLPReplayTopology]>,
    },
}

impl Qwen35MainReplayKey {
    pub fn from_shapes(
        gqa_shape: inference_executor_core::attn::GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
    ) -> Self {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, gdn_shape.num_tokens,
            "qwen3.5 main GQA and GDN replay token counts must match"
        );
        Self {
            mode: Qwen35MainReplayMode::Legacy {
                num_tokens: gqa_shape.num_tokens,
            },
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
            gdn: Qwen35MainGDNReplayKey::new(gdn_shape, gdn_topology),
        }
    }

    fn for_bucketed(
        num_total_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
        mlp_topologies: Box<[Qwen35MainMLPReplayTopology]>,
    ) -> Self {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.total_tokens, num_total_tokens,
            "qwen3.5 Main GQA key capacity must match the stage"
        );
        assert_eq!(
            gdn_shape.total_tokens, num_total_tokens,
            "qwen3.5 Main GDN key capacity must match the stage"
        );
        Self {
            mode: Qwen35MainReplayMode::Bucketed {
                num_total_tokens,
                mlp_topologies,
            },
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
            gdn: Qwen35MainGDNReplayKey::new(gdn_shape, gdn_topology),
        }
    }

    fn bucketed_num_total_tokens(&self) -> u32 {
        match &self.mode {
            Qwen35MainReplayMode::Legacy { .. } => {
                panic!("legacy qwen3.5 Main replay key does not select a bucketed token capacity")
            },
            Qwen35MainReplayMode::Bucketed { num_total_tokens, .. } => *num_total_tokens,
        }
    }

    #[cfg(test)]
    pub fn debug_parts(&self) -> (u32, u32, u32, u32, u32, u32, GQAReplayTopology, GDNReplayTopology) {
        let (total_tokens, total_q_token_tiles, total_task_templates, topology) = self.gqa.debug_parts();
        let num_tokens = match &self.mode {
            Qwen35MainReplayMode::Legacy { num_tokens } => *num_tokens,
            Qwen35MainReplayMode::Bucketed { num_total_tokens, .. } => *num_total_tokens,
        };
        (
            num_tokens,
            total_tokens,
            total_q_token_tiles,
            total_task_templates,
            self.gdn.total_reqs,
            self.gdn.total_tokens,
            topology,
            self.gdn.topology,
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35MainGDNReplayKey {
    total_reqs: u32,
    total_tokens: u32,
    topology: GDNReplayTopology,
}

impl Qwen35MainGDNReplayKey {
    fn new(gdn_shape: GDNReplayShape, topology: GDNReplayTopology) -> Self {
        gdn_shape.validate();
        Self {
            total_reqs: gdn_shape.total_reqs,
            total_tokens: gdn_shape.total_tokens,
            topology,
        }
    }
}

impl ReplayComponent for Qwen35Main {
    type Key = Qwen35MainReplayKey;
    type Input<'a> = Qwen35MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.bucketed_replay_key(
            input.num_tokens,
            input.gqa.replay_shape(),
            input.gqa_replay_topology,
            input.gdn.replay_shape(),
            input.gdn_replay_topology,
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        self.record_bucketed(recorder, key.bucketed_num_total_tokens(), *input);
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::GQAComputePath;
    use inference_backend_metal::components::QuantizedDenseMLPReplayTopology;
    use inference_backend_metal::components::ResidualAddCaptureTarget;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;

    use super::*;
    use crate::attn::gdn::backend::GDN_NUM_ACTIVE_REQUESTS;
    use crate::attn::gdn::backend::add_gdn_private_replay_arguments;
    use crate::attn::gqa::backend::GQA_NUM_ACTIVE_Q_TOKEN_TILES;
    use crate::attn::gqa::backend::GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES;
    use crate::attn::gqa::backend::add_gqa_private_replay_arguments;
    use crate::mlp::moe::backend::GatedMoEComputePath;
    use crate::mlp::moe::backend::GatedMoEReplayTopology;

    struct SelectedLayerCapture {
        model_layer_index: usize,
        capture_output: Buffer,
    }

    impl MainResidualCapture for SelectedLayerCapture {
        fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<ResidualAddCaptureTarget<'_>> {
            (model_layer_index == self.model_layer_index)
                .then(|| ResidualAddCaptureTarget::columns(&self.capture_output, 16, 4..12))
        }
    }

    fn assert_replay_component<T: ReplayComponent>() {}

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

    fn gdn_topology() -> GDNReplayTopology {
        GDNReplayTopology {
            materialize_candidate_states: true,
            qkvabz_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            output_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        }
    }

    fn dense_topology() -> Qwen35MainMLPReplayTopology {
        Qwen35MainMLPReplayTopology::Dense(QuantizedDenseMLPReplayTopology {
            gate_up_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            down_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
        })
    }

    fn moe_topology() -> Qwen35MainMLPReplayTopology {
        Qwen35MainMLPReplayTopology::MoE(GatedMoEReplayTopology {
            compute_path: GatedMoEComputePath::TokenMajor,
            router_affine: AffineQuantizedMatmulKernelKind::QmvBn8Bk32,
            shared_expert_gate_affine: None,
            shared_experts_dense: None,
        })
    }

    fn gqa_shape(num_tokens: u32, total_tokens: u32) -> GQAReplayShape {
        GQAReplayShape::new(num_tokens, total_tokens, 1, 2, 2, 4, false)
    }

    fn gdn_shape(num_reqs: u32, num_tokens: u32, total_tokens: u32) -> GDNReplayShape {
        GDNReplayShape::new(num_reqs, 2, num_tokens, total_tokens)
    }

    #[test]
    fn test_main_policy_composes_base_buckets_and_topology_boundaries() {
        let policy = main_replay_bucket_policy(16, vec![10, 5, 10]);

        assert_eq!(policy.buckets(), [1, 2, 4, 6, 8, 9, 12, 16]);
        assert_eq!(policy.capacity(3), 4);
        assert_eq!(policy.capacity(4), 4);
        assert_eq!(policy.capacity(5), 6);
        assert_eq!(policy.capacity(9), 9);
        assert_eq!(policy.capacity(10), 12);
    }

    #[test]
    fn test_main_arguments_own_one_stage_active_token_value() {
        assert_eq!(
            main_replay_arguments(3),
            ReplayArguments::new().with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, 3)
        );
        assert_eq!(
            main_replay_arguments(4),
            ReplayArguments::new().with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, 4)
        );
    }

    #[test]
    fn test_main_argument_composition_uses_only_stage_token_and_private_attention_values() {
        let shape = gqa_shape(3, 4);
        let gdn_shape = gdn_shape(1, 3, 4);
        let mut single_arguments = main_replay_arguments(3);
        add_gqa_private_replay_arguments(shape, gqa_topology(), &mut single_arguments);
        add_gdn_private_replay_arguments(gdn_shape, &mut single_arguments);
        assert_eq!(
            single_arguments,
            ReplayArguments::new()
                .with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, 3)
                .with_u32(GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 2)
                .with_u32(GDN_NUM_ACTIVE_REQUESTS, 1)
        );

        let tiled_topology = GQAReplayTopology {
            compute_path: GQAComputePath::TiledQueryTokens {
                q_token_tile_size: 8,
                kv_token_tile_size: 16,
                q_head_tile_size: 6,
            },
            ..gqa_topology()
        };
        let mut tiled_arguments = main_replay_arguments(3);
        add_gqa_private_replay_arguments(shape, tiled_topology, &mut tiled_arguments);
        add_gdn_private_replay_arguments(gdn_shape, &mut tiled_arguments);
        assert_eq!(
            tiled_arguments,
            ReplayArguments::new()
                .with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, 3)
                .with_u32(GQA_NUM_ACTIVE_Q_TOKEN_TILES, 1)
                .with_u32(GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 2)
                .with_u32(GDN_NUM_ACTIVE_REQUESTS, 1)
        );
    }

    #[test]
    fn test_bucketed_main_key_ignores_active_counts_and_isolates_legacy_mode() {
        let topologies = vec![dense_topology(), moe_topology()].into_boxed_slice();
        let active_three = Qwen35MainReplayKey::for_bucketed(
            4,
            gqa_shape(3, 4),
            gqa_topology(),
            gdn_shape(1, 3, 4),
            gdn_topology(),
            topologies.clone(),
        );
        let active_four = Qwen35MainReplayKey::for_bucketed(
            4,
            gqa_shape(4, 4),
            gqa_topology(),
            gdn_shape(2, 4, 4),
            gdn_topology(),
            topologies,
        );
        let legacy =
            Qwen35MainReplayKey::from_shapes(gqa_shape(3, 4), gqa_topology(), gdn_shape(1, 3, 4), gdn_topology());

        assert_eq!(active_three, active_four);
        assert_ne!(active_three, legacy);
        assert_eq!(active_three.bucketed_num_total_tokens(), 4);
    }

    #[test]
    fn test_bucketed_main_key_separates_capacity_and_ordered_layer_topology() {
        let base = Qwen35MainReplayKey::for_bucketed(
            4,
            gqa_shape(3, 4),
            gqa_topology(),
            gdn_shape(1, 3, 4),
            gdn_topology(),
            vec![dense_topology(), moe_topology()].into_boxed_slice(),
        );
        let different_capacity = Qwen35MainReplayKey::for_bucketed(
            6,
            gqa_shape(3, 6),
            gqa_topology(),
            gdn_shape(1, 3, 6),
            gdn_topology(),
            vec![dense_topology(), moe_topology()].into_boxed_slice(),
        );
        let different_order = Qwen35MainReplayKey::for_bucketed(
            4,
            gqa_shape(3, 4),
            gqa_topology(),
            gdn_shape(1, 3, 4),
            gdn_topology(),
            vec![moe_topology(), dense_topology()].into_boxed_slice(),
        );
        let different_gqa_capacity = Qwen35MainReplayKey::for_bucketed(
            4,
            GQAReplayShape::new(3, 4, 1, 4, 2, 4, false),
            gqa_topology(),
            gdn_shape(1, 3, 4),
            gdn_topology(),
            vec![dense_topology(), moe_topology()].into_boxed_slice(),
        );
        let different_gdn_capacity = Qwen35MainReplayKey::for_bucketed(
            4,
            gqa_shape(3, 4),
            gqa_topology(),
            GDNReplayShape::new(1, 4, 3, 4),
            gdn_topology(),
            vec![dense_topology(), moe_topology()].into_boxed_slice(),
        );

        assert_ne!(base, different_capacity);
        assert_ne!(base, different_order);
        assert_ne!(base, different_gqa_capacity);
        assert_ne!(base, different_gdn_capacity);
    }

    #[test]
    fn test_gdn_key_uses_capacities_and_topology() {
        let topology = gdn_topology();
        let base = Qwen35MainGDNReplayKey::new(GDNReplayShape::new(1, 2, 3, 4), topology);
        assert_eq!(
            base,
            Qwen35MainGDNReplayKey::new(GDNReplayShape::new(2, 2, 4, 4), topology)
        );
        assert_ne!(
            base,
            Qwen35MainGDNReplayKey::new(GDNReplayShape::new(2, 4, 4, 4), topology)
        );
        assert_ne!(
            base,
            Qwen35MainGDNReplayKey::new(GDNReplayShape::new(1, 2, 3, 6), topology)
        );
        for different in [
            GDNReplayTopology {
                materialize_candidate_states: false,
                ..topology
            },
            GDNReplayTopology {
                qkvabz_affine: AffineQuantizedMatmulKernelKind::QmmBm8Bn32,
                ..topology
            },
            GDNReplayTopology {
                output_affine: AffineQuantizedMatmulKernelKind::QmmBm8Bn32,
                ..topology
            },
        ] {
            assert_ne!(
                base,
                Qwen35MainGDNReplayKey::new(GDNReplayShape::new(1, 2, 3, 4), different)
            );
        }
    }

    #[test]
    fn test_residual_capture_selects_only_the_configured_model_layer() {
        let device = Device::system_default();
        let capture = Rc::new(SelectedLayerCapture {
            model_layer_index: 10,
            capture_output: Buffer::new_zeroed_elements(&device, 64, Dtype::Bfloat16),
        });
        let erased_capture: Rc<dyn MainResidualCapture> = capture.clone();

        assert!(erased_capture.capture_for_model_layer(9).is_none());
        assert!(erased_capture.capture_for_model_layer(10).is_some());
        assert!(erased_capture.capture_for_model_layer(11).is_none());

        assert_replay_component::<Qwen35Main>();
    }

    #[test]
    fn test_absent_main_capture_never_captures() {
        let capture: Option<Rc<dyn MainResidualCapture>> = None;
        for model_layer_index in [0, 1, 10, usize::MAX] {
            assert!(
                capture
                    .as_ref()
                    .and_then(|capture| capture.capture_for_model_layer(model_layer_index))
                    .is_none()
            );
        }

        assert_replay_component::<Qwen35Main>();
    }
}
