use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::num_target_hidden_states;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MainWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::embed_unembed::Embed;
use crate::model::embed_unembed::EmbedInput;
use crate::model::embed_unembed::Unembed;
use crate::model::embed_unembed::UnembedConfig;
use crate::model::embed_unembed::UnembedInput;
use crate::model::gather::Gather;
use crate::model::qwen::v3_5::layer::Qwen35Layer;
use crate::model::qwen::v3_5::layer::Qwen35LayerInput;
use crate::model::qwen::v3_5::layer::scratch::Qwen35LayerScratch;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::state::Qwen35GDNState;
use crate::model::qwen::v3_5::state::Qwen35GQAState;
use crate::model::qwen::v3_5::weight::load_qwen35_norm_weight;
use crate::model::rms_norm::RmsNorm;
use crate::replay::ReplayComponent;

pub struct Qwen35Main {
    layers: Vec<Qwen35Layer>,
    final_norm: RmsNorm,
}

pub struct Qwen35MainEmbed {
    embed: Rc<Embed>,
}

pub struct Qwen35GatherUnembed {
    gather: Gather,
    unembed: Rc<Unembed>,
    hidden_dim: u32,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainEmbedArgs<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gdn: &'a crate::attn::gdn::batch_metadata::GDNMetadataBuffers,
    pub pages: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35GatherUnembedArgs<'a> {
    pub num_rows: u32,
    pub hidden_input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Qwen35MainEmbed {
    pub fn new(embed: Rc<Embed>) -> Self {
        Self { embed }
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainEmbedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        <Embed as ReplayLayer>::record(
            &self.embed,
            recorder,
            EmbedInput {
                num_tokens: args.num_tokens,
                token_ids: args.token_ids,
                output_hidden: args.hidden_output,
            },
        )
    }
}

impl Qwen35Main {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        bindings: Qwen35MainWeightBindings,
        gqa_state: &Qwen35GQAState,
        gdn_state: &Qwen35GDNState,
        layer_scratch: Rc<Qwen35LayerScratch>,
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
        let mut compact_gqa_layer_index = 0;
        let mut compact_gdn_layer_index = 0;
        let mut layers = Vec::with_capacity(layer_bindings.len());
        for (layer_index, bindings) in layer_bindings.into_iter().enumerate() {
            let Qwen35LayerWeightBindings {
                input_norm_weight,
                post_attention_norm_weight,
                attention,
                mlp,
            } = bindings;
            let attention = Qwen35Layer::load_attention(
                device,
                store,
                config,
                defaults,
                layer_index,
                &mut compact_gqa_layer_index,
                &mut compact_gdn_layer_index,
                attention,
                gqa_state.backend(),
                gqa_state.scratch(),
                gqa_state.request_page_table(),
                gdn_state.backend(),
                gdn_state.scratch(),
                gdn_state.request_state_table(),
            )?;
            let mlp = Qwen35Layer::load_mlp(
                device,
                store,
                config,
                defaults,
                layer_index,
                mlp,
                dense_scratch,
                moe_scratch,
            )?;
            layers.push(Qwen35Layer::load(
                device,
                store,
                config,
                layer_index,
                input_norm_weight,
                post_attention_norm_weight,
                attention,
                mlp,
                Rc::clone(&layer_scratch),
            )?);
            store.unload_all();
        }

        let final_norm_weight = load_qwen35_norm_weight(
            device,
            store,
            &final_norm_weight,
            &[config.text_config.hidden_size],
            config.quantization.is_some(),
        )?;
        Ok(Self {
            layers,
            final_norm: RmsNorm::new(
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
                final_norm_weight,
                RmsNorm::kernel(device),
            ),
        })
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MainArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let output = layer.output();
            hidden = <Qwen35Layer as ReplayLayer>::record(
                layer,
                recorder,
                Qwen35LayerInput {
                    gdn: Some(args.gdn),
                    gqa: args.gqa,
                    input: hidden,
                    output,
                    num_tokens,
                    pages: args.pages,
                },
            );
        }
        self.final_norm.record(recorder, num_tokens, hidden, args.hidden_output);
        args.hidden_output
    }
}

impl Qwen35GatherUnembed {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: UnembedConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let unembed = Rc::new(Unembed::load(device, store, config, bindings)?);
        Ok(Self {
            gather: Gather::new(device),
            unembed,
            hidden_dim: config.hidden_dim,
        })
    }

    pub fn unembed(&self) -> &Rc<Unembed> {
        &self.unembed
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35GatherUnembedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.gather.record(
            recorder,
            args.num_rows,
            self.hidden_dim,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        <Unembed as ReplayLayer>::record(
            &self.unembed,
            recorder,
            UnembedInput {
                num_rows: args.num_rows,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainEmbedReplayKey {
    num_tokens: u32,
}

impl Qwen35MainEmbedReplayKey {
    pub fn new(num_tokens: u32) -> Self {
        Self { num_tokens }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35GQAReplayKey {
    num_q_token_tiles: u32,
    total_sdpa_map_task_templates: u32,
}

impl Qwen35GQAReplayKey {
    pub fn from_shape(gqa_shape: inference_executor_core::attn::GQAReplayShape) -> Self {
        gqa_shape.validate();
        Self {
            num_q_token_tiles: gqa_shape.num_q_token_tiles,
            total_sdpa_map_task_templates: gqa_shape.total_sdpa_map_task_templates,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainReplayKey {
    num_tokens: u32,
    gqa: Qwen35GQAReplayKey,
    gdn: Qwen35GDNReplayKey,
}

impl Qwen35MainReplayKey {
    pub fn from_shapes(gqa_shape: inference_executor_core::attn::GQAReplayShape, gdn_shape: GDNReplayShape) -> Self {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, gdn_shape.num_tokens,
            "qwen3.5 main GQA and GDN replay token counts must match"
        );
        Self {
            num_tokens: gqa_shape.num_tokens,
            gqa: Qwen35GQAReplayKey::from_shape(gqa_shape),
            gdn: Qwen35GDNReplayKey::from_shape(gdn_shape),
        }
    }

    #[cfg(test)]
    pub fn debug_parts(&self) -> (u32, u32, u32, u32) {
        (
            self.num_tokens,
            self.gqa.num_q_token_tiles,
            self.gqa.total_sdpa_map_task_templates,
            self.gdn.num_reqs,
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35GDNReplayKey {
    num_reqs: u32,
}

impl Qwen35GDNReplayKey {
    fn from_shape(gdn_shape: GDNReplayShape) -> Self {
        gdn_shape.validate();
        Self {
            num_reqs: gdn_shape.num_reqs,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35GatherUnembedReplayKey {
    num_target_hidden_states: u32,
}

impl Qwen35GatherUnembedReplayKey {
    pub fn from_microbatch(microbatch: &Qwen35Microbatch) -> Self {
        let num_target_hidden_states = num_target_hidden_states(microbatch)
            .try_into()
            .expect("qwen3.5 target hidden-state count must fit u32");
        assert!(
            num_target_hidden_states > 0,
            "qwen3.5 GatherUnembed replay requires target hidden states"
        );
        Self {
            num_target_hidden_states,
        }
    }

    pub fn num_target_hidden_states(&self) -> u32 {
        self.num_target_hidden_states
    }
}

impl ReplayComponent for Qwen35MainEmbed {
    type Key = Qwen35MainEmbedReplayKey;
    type Input<'a> = Qwen35MainEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::Key {
            num_tokens: input.num_tokens,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35MainEmbed::record(self, recorder, *input);
    }
}

impl ReplayComponent for Qwen35Main {
    type Key = Qwen35MainReplayKey;
    type Input<'a> = Qwen35MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen35MainReplayKey::from_shapes(input.gqa.replay_shape(), input.gdn.replay_shape())
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35Main::record(self, recorder, *input);
    }
}

impl ReplayComponent for Qwen35GatherUnembed {
    type Key = Qwen35GatherUnembedReplayKey;
    type Input<'a> = Qwen35GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        assert!(
            input.num_rows > 0,
            "qwen3.5 GatherUnembed requires target hidden states"
        );
        Qwen35GatherUnembedReplayKey {
            num_target_hidden_states: input.num_rows,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35GatherUnembed::record(self, recorder, *input);
    }
}
