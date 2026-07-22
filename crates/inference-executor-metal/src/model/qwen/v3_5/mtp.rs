use std::rc::Rc;

use inference_backend_metal::components::Bf16ConcatRowsBuffers;
use inference_backend_metal::components::Bf16ConcatRowsKernel;
use inference_backend_metal::components::Bf16ConcatRowsShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35AttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPEmbedWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::embed_unembed::Embed;
use crate::model::embed_unembed::EmbedInput;
use crate::model::gather::Gather;
use crate::model::qwen::v3_5::layer::Qwen35Attention;
use crate::model::qwen::v3_5::layer::Qwen35GQA;
use crate::model::qwen::v3_5::layer::Qwen35Layer;
use crate::model::qwen::v3_5::layer::Qwen35LayerInput;
use crate::model::qwen::v3_5::layer::scratch::Qwen35LayerScratch;
use crate::model::qwen::v3_5::model::Qwen35GQAReplayKey;
use crate::model::qwen::v3_5::plan::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::state::Qwen35GQAState;
use crate::model::qwen::v3_5::weight::load_qwen35_norm_weight;
use crate::model::qwen::v3_5::weight::quant_weight;
use crate::model::qwen::v3_5::weight::typed_tensor;
use crate::model::qwen::v3_5::weight::validate_len;
use crate::model::residual::Residual;
use crate::model::rms_norm::RmsNorm;
use crate::replay::ReplayComponent;

pub struct Qwen35MTP {
    hidden_dim: usize,
    layer: Qwen35Layer,
    output_residual: Residual,
    output_norm: RmsNorm,
    request_page_table: Rc<crate::attn::gqa::request_page_table::GQARequestPageTable>,
}

pub struct Qwen35MTPEmbed {
    embed: Rc<Embed>,
    input_gather: Gather,
    hidden_dim: usize,
    hidden_norm: RmsNorm,
    embedding_norm: RmsNorm,
    concat: Bf16ConcatRowsKernel,
    fc_kernel: AffineQuantizedMatmulKernel,
    fc_shape: AffineQuantizedMatmulShape,
    fc_weight: Buffer,
    fc_scales: Buffer,
    fc_biases: Buffer,
    normed_hidden: Buffer,
    normed_embedding: Buffer,
    fused_input: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPEmbedArgs<'a> {
    pub num_tokens: u32,
    pub prev_hidden_source: &'a Buffer,
    pub prev_hidden_indices: &'a Buffer,
    pub prev_hidden_input: &'a Buffer,
    pub token_ids: &'a Buffer,
    pub token_hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub pages: &'a Buffer,
}

impl Qwen35MTPEmbed {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35MTPEmbedWeightBindings,
        embed: Rc<Embed>,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let quant = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP requires quantized checkpoint weights"))?;
        let weights = Qwen35MTPEmbedWeights::load(
            device,
            store,
            &bindings,
            hidden_dim,
            quant.group_size,
            quant.bits,
            config.quantization.is_some(),
        )?;
        let Qwen35MTPEmbedWeights {
            token_hidden_norm_weight,
            prev_hidden_norm_weight,
            fc_weight,
            fc_scales,
            fc_biases,
        } = weights;
        let fused_hidden_dim = hidden_dim
            .checked_mul(2)
            .expect("qwen3.5 MTP fused hidden dimension must fit usize");
        let fc_shape = AffineQuantizedMatmulShape {
            m: max_tokens.try_into().expect("qwen3.5 MTP max_tokens must fit i32"),
            n: hidden_dim
                .try_into()
                .expect("qwen3.5 MTP hidden dimension must fit i32"),
            k: fused_hidden_dim
                .try_into()
                .expect("qwen3.5 MTP fused hidden dimension must fit i32"),
            group_size: quant
                .group_size
                .try_into()
                .expect("qwen3.5 MTP group size must fit i32"),
            bits: quant.bits.try_into().expect("qwen3.5 MTP bits must fit i32"),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 MTP hidden capacity must fit usize");
        u32::try_from(hidden_elements).expect("qwen3.5 MTP hidden capacity must fit shader u32 count");
        let fused_elements = hidden_elements
            .checked_mul(2)
            .expect("qwen3.5 MTP fused input capacity must fit usize");
        u32::try_from(fused_elements).expect("qwen3.5 MTP fused capacity must fit shader u32 count");
        let norm_op = RmsNorm::kernel(device);
        Ok(Self {
            embed,
            input_gather: Gather::new(device),
            hidden_dim,
            hidden_norm: RmsNorm::new(
                hidden_dim,
                config.text_config.rms_norm_eps,
                prev_hidden_norm_weight,
                Rc::clone(&norm_op),
            ),
            embedding_norm: RmsNorm::new(
                hidden_dim,
                config.text_config.rms_norm_eps,
                token_hidden_norm_weight,
                norm_op,
            ),
            concat: Bf16ConcatRowsKernel::new(device),
            fc_kernel: AffineQuantizedMatmulKernel::new(device, fc_shape),
            fc_shape,
            fc_weight,
            fc_scales,
            fc_biases,
            normed_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normed_embedding: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            fused_input: Buffer::new_zeroed_elements(device, fused_elements, Dtype::Bfloat16),
        })
    }
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
        gqa_state: &Qwen35GQAState,
        layer_scratch: Rc<Qwen35LayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let Qwen35LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
        } = bindings;
        let attention = match attention {
            Qwen35AttentionWeightBindings::GQA(bindings) => {
                Qwen35Attention::GQA(Qwen35GQA::load(
                    device,
                    store,
                    config,
                    defaults,
                    main_config.text_config.num_hidden_layers,
                    0,
                    bindings,
                    Rc::clone(gqa_state.backend()),
                    Rc::clone(gqa_state.scratch()),
                    Rc::clone(gqa_state.request_page_table()),
                )?)
            },
            Qwen35AttentionWeightBindings::GDN(_) => {
                panic!("qwen3.5 MTP body must use full attention")
            },
        };
        let mlp = Qwen35Layer::load_mlp(device, store, config, defaults, 0, mlp, dense_scratch, moe_scratch)?;
        let layer = Qwen35Layer::load(
            device,
            store,
            config,
            0,
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
            Rc::clone(&layer_scratch),
        )?;
        let final_norm_weight = load_qwen35_norm_weight(
            device,
            store,
            &final_norm_weight,
            &[hidden_dim],
            config.quantization.is_some(),
        )?;
        Ok(Self {
            hidden_dim,
            layer,
            output_residual: Residual::new(device),
            output_norm: RmsNorm::new(
                hidden_dim,
                config.text_config.rms_norm_eps,
                final_norm_weight,
                RmsNorm::kernel(device),
            ),
            request_page_table: Rc::clone(gqa_state.request_page_table()),
        })
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
        let tokens_per_page = self
            .layer
            .gqa_tokens_per_page()
            .expect("qwen3.5 MTP body must contain GQA");
        assert!(
            max_context_tokens.div_ceil(tokens_per_page.max(1)) <= page_capacity,
            "qwen3.5 MTP GQA request context exceeds page-table capacity"
        );
    }
}

impl Qwen35MTPEmbed {
    fn record_projection<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        previous_hidden: &'a Buffer,
        shifted_embeddings: &'a Buffer,
        output: &'a Buffer,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.hidden_norm
            .record_opaque(recorder, num_tokens, previous_hidden, &self.normed_hidden, true);
        self.embedding_norm
            .record_opaque(recorder, num_tokens, shifted_embeddings, &self.normed_embedding, false);
        let hidden_dim = self
            .hidden_dim
            .try_into()
            .expect("qwen3.5 MTP hidden dimension must fit u32");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.concat.invoke(
            Bf16ConcatRowsShape {
                num_rows: num_tokens,
                num_cols: hidden_dim,
            },
            Bf16ConcatRowsBuffers {
                lhs: &self.normed_embedding,
                rhs: &self.normed_hidden,
                output: &self.fused_input,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.fc_kernel.invoke_with_shape(
            AffineQuantizedMatmulShape {
                m: num_tokens.try_into().expect("qwen3.5 MTP token count must fit i32"),
                ..self.fc_shape
            },
            output,
            0,
            &self.fused_input,
            0,
            &self.fc_weight,
            0,
            &self.fc_scales,
            0,
            &self.fc_biases,
            0,
        )));
        output
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MTPEmbedArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        self.input_gather.record(
            recorder,
            num_tokens,
            self.hidden_dim
                .try_into()
                .expect("qwen3.5 MTP hidden dimension must fit u32"),
            args.prev_hidden_source,
            args.prev_hidden_indices,
            args.prev_hidden_input,
        );
        let _ = <Embed as ReplayLayer>::record(
            &self.embed,
            recorder,
            EmbedInput {
                num_tokens,
                token_ids: args.token_ids,
                output_hidden: args.token_hidden_input,
            },
        );
        self.record_projection(
            recorder,
            num_tokens,
            args.prev_hidden_input,
            args.token_hidden_input,
            args.hidden_output,
        )
    }
}

impl Qwen35MTP {
    pub fn record<'a, R>(&'a self, recorder: &mut R, args: Qwen35MTPArgs<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_tokens = args.num_tokens;
        let residual = self.layer.record_body(
            recorder,
            Qwen35LayerInput {
                gdn: None,
                gqa: args.gqa,
                input: args.hidden_input,
                output: self.layer.output(),
                num_tokens,
                pages: args.pages,
            },
        );
        let num_values = num_tokens
            .checked_mul(
                self.hidden_dim
                    .try_into()
                    .expect("qwen3.5 MTP hidden dimension must fit u32"),
            )
            .expect("qwen3.5 MTP residual element count must fit u32");
        self.output_residual.record(
            recorder,
            num_values,
            residual.lhs,
            residual.rhs,
            self.layer.output(),
            None,
        );
        self.output_norm
            .record(recorder, num_tokens, self.layer.output(), args.hidden_output);
        args.hidden_output
    }
}

struct Qwen35MTPEmbedWeights {
    token_hidden_norm_weight: Buffer,
    prev_hidden_norm_weight: Buffer,
    fc_weight: Buffer,
    fc_scales: Buffer,
    fc_biases: Buffer,
}

impl Qwen35MTPEmbedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35MTPEmbedWeightBindings,
        hidden_dim: usize,
        group_size: usize,
        bits: usize,
        norms_store_actual_scale: bool,
    ) -> Result<Self, ModelExecutorError> {
        let fused_hidden_dim = hidden_dim
            .checked_mul(2)
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP fused hidden dimension overflow"))?;
        let fc_shape = AffineQuantizedMatmulShape {
            m: 1,
            n: hidden_dim
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP hidden_dim must fit i32"))?,
            k: fused_hidden_dim
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP fused hidden dimension must fit i32"))?,
            group_size: group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP group_size must fit i32"))?,
            bits: bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP bits must fit i32"))?,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let fc_weight = quant_weight(store, &bindings.projection.weight)?;
        let fc_scales = typed_tensor(store, &bindings.projection.scales, safetensors::Dtype::BF16)?.into_data();
        let fc_biases = typed_tensor(store, &bindings.projection.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("MTP fc weight", fc_weight.len(), fc_shape.weight_bytes())?;
        validate_len("MTP fc scales", fc_scales.len(), fc_shape.affine_param_bytes())?;
        validate_len("MTP fc biases", fc_biases.len(), fc_shape.affine_param_bytes())?;
        Ok(Self {
            token_hidden_norm_weight: load_qwen35_norm_weight(
                device,
                store,
                &bindings.token_hidden_norm_weight,
                &[hidden_dim],
                norms_store_actual_scale,
            )?,
            prev_hidden_norm_weight: load_qwen35_norm_weight(
                device,
                store,
                &bindings.prev_hidden_norm_weight,
                &[hidden_dim],
                norms_store_actual_scale,
            )?,
            fc_weight: Buffer::from_slice(device, &fc_weight),
            fc_scales: Buffer::from_slice(device, &fc_scales),
            fc_biases: Buffer::from_slice(device, &fc_biases),
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPReplayKey {
    mtp_module_index: usize,
    num_tokens: usize,
    gqa: Qwen35GQAReplayKey,
}

impl Qwen35MTPReplayKey {
    pub fn new(
        mtp_module_index: usize,
        num_tokens: usize,
        gqa_shape: inference_executor_core::attn::GQAReplayShape,
    ) -> Self {
        Self {
            mtp_module_index,
            num_tokens,
            gqa: Qwen35GQAReplayKey::from_shape(gqa_shape),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPEmbedReplayKey {
    mtp_module_index: usize,
    num_tokens: usize,
}

impl Qwen35MTPEmbedReplayKey {
    pub fn new(mtp_module_index: usize, num_tokens: usize) -> Self {
        Self {
            mtp_module_index,
            num_tokens,
        }
    }
}

impl ReplayComponent for Qwen35MTPEmbed {
    type Key = Qwen35MTPEmbedReplayKey;
    type Input<'a> = Qwen35MTPEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::Key {
            mtp_module_index: 0,
            num_tokens: input.num_tokens as usize,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35MTPEmbed::record(self, recorder, *input);
    }
}

impl ReplayComponent for Qwen35MTP {
    type Key = Qwen35MTPReplayKey;
    type Input<'a> = Qwen35MTPArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::Key {
            mtp_module_index: 0,
            num_tokens: input.num_tokens as usize,
            gqa: Qwen35GQAReplayKey::from_shape(input.gqa.replay_shape()),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        Qwen35MTP::record(self, recorder, *input);
    }
}
