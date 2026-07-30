use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::Bf16ConcatRowsBuffers;
use inference_backend_metal::operators::Bf16ConcatRowsConfig;
use inference_backend_metal::operators::Bf16ConcatRowsKernel;
use inference_backend_metal::operators::Bf16ConcatRowsShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPEmbedWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::model::gather::Gather;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::remove_typed_tensor;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub struct Qwen35MTPEmbed {
    embed: Rc<Embed>,
    input_gather: Gather,
    hidden_norm: RMSNorm,
    embedding_norm: RMSNorm,
    concat: Bf16ConcatRowsKernel,
    fc: AffineQuantizedMatmul,
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
        let weights = Qwen35MTPEmbedWeights::load(device, store, &bindings, hidden_dim, quant.group_size, quant.bits)?;
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
        let fc_config = AffineQuantizedMatmulConfig {
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
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 MTP hidden capacity must fit usize");
        u32::try_from(hidden_elements).expect("qwen3.5 MTP hidden capacity must fit shader u32 count");
        let fused_elements = hidden_elements
            .checked_mul(2)
            .expect("qwen3.5 MTP fused input capacity must fit usize");
        u32::try_from(fused_elements).expect("qwen3.5 MTP fused capacity must fit shader u32 count");
        Ok(Self {
            embed,
            input_gather: Gather::new(
                device,
                hidden_dim
                    .try_into()
                    .expect("qwen3.5 MTP hidden dimension must fit u32"),
            ),
            hidden_norm: RMSNorm::new(
                device,
                hidden_dim,
                config.text_config.rms_norm_eps,
                prev_hidden_norm_weight,
            ),
            embedding_norm: RMSNorm::new(
                device,
                hidden_dim,
                config.text_config.rms_norm_eps,
                token_hidden_norm_weight,
            ),
            concat: Bf16ConcatRowsKernel::new(
                device,
                Bf16ConcatRowsConfig {
                    num_cols: hidden_dim
                        .try_into()
                        .expect("qwen3.5 MTP hidden dimension must fit u32"),
                },
            ),
            fc: AffineQuantizedMatmul::new(device, fc_config),
            fc_weight,
            fc_scales,
            fc_biases,
            normed_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normed_embedding: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            fused_input: Buffer::new_zeroed_elements(device, fused_elements, Dtype::Bfloat16),
        })
    }

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
            .record_with_barrier(recorder, num_tokens, previous_hidden, &self.normed_hidden);
        self.embedding_norm
            .record(recorder, num_tokens, shifted_embeddings, &self.normed_embedding);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.concat.invoke(
            Bf16ConcatRowsShape { num_rows: num_tokens },
            Bf16ConcatRowsBuffers {
                lhs: &self.normed_embedding,
                rhs: &self.normed_hidden,
                output: &self.fused_input,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.fc.invoke(
            num_tokens.try_into().expect("qwen3.5 MTP token count must fit i32"),
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
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let fused_hidden_dim = hidden_dim
            .checked_mul(2)
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP fused hidden dimension overflow"))?;
        let fc_config = AffineQuantizedMatmulConfig {
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
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let fc_weight = remove_quant_weight(&mut tensors, &bindings.projection.weight)?;
        let fc_scales =
            remove_typed_tensor(&mut tensors, &bindings.projection.scales, safetensors::Dtype::BF16)?.into_data();
        let fc_biases =
            remove_typed_tensor(&mut tensors, &bindings.projection.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("MTP fc weight", fc_weight.len(), fc_config.weight_bytes())?;
        validate_len("MTP fc scales", fc_scales.len(), fc_config.scale_or_bias_bytes())?;
        validate_len("MTP fc biases", fc_biases.len(), fc_config.scale_or_bias_bytes())?;
        let weights = Self {
            token_hidden_norm_weight: remove_qwen3x_norm_weight(
                device,
                &mut tensors,
                &bindings.token_hidden_norm_weight,
                &[hidden_dim],
            )?,
            prev_hidden_norm_weight: remove_qwen3x_norm_weight(
                device,
                &mut tensors,
                &bindings.prev_hidden_norm_weight,
                &[hidden_dim],
            )?,
            fc_weight: Buffer::from_slice(device, &fc_weight),
            fc_scales: Buffer::from_slice(device, &fc_scales),
            fc_biases: Buffer::from_slice(device, &fc_biases),
        };
        assert!(tensors.is_empty(), "qwen3.5 MTP embed must consume its tensor map");
        Ok(weights)
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
