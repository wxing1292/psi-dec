use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfidenceWeightBindings;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkConfig;
use inference_executor_core::model::qwen::v3_x::dspark::Qwen3xDSparkMarkovWeightBindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::TopKSamplingBounds;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_typed_tensor;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::qwen::v3_x::weight::validate_shape;
use crate::sampling::dspark_markov::DSparkConfidenceInput;
use crate::sampling::dspark_markov::DSparkConfidenceWeights;
use crate::sampling::dspark_markov::DSparkMarkovConfidenceConfig;
use crate::sampling::dspark_markov::DSparkMarkovInput;
use crate::sampling::dspark_markov::DSparkMarkovReplayShape;
use crate::sampling::dspark_markov::DSparkMarkovSampling;
use crate::sampling::dspark_markov::DSparkMarkovSamplingConfig;
use crate::sampling::dspark_markov::DSparkMarkovWeights;
use crate::sampling::dspark_markov::DSparkProposal;
use crate::sampling::spec_probs::SpecProbsStore;

pub struct Qwen3xDSparkMarkov {
    backend: DSparkMarkovSampling,
    weights: Qwen3xDSparkMarkovWeights,
    confidence: Option<Qwen3xDSparkConfidenceWeights>,
}

struct Qwen3xDSparkMarkovWeights {
    w1_weight: Buffer,
    w1_scales: Buffer,
    w1_biases: Buffer,
    w2_weight: Buffer,
    w2_scales: Buffer,
    w2_biases: Buffer,
}

struct Qwen3xDSparkConfidenceWeights {
    weight: Buffer,
    bias: Buffer,
}

impl Qwen3xDSparkMarkov {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        model_config: &Qwen3xDSparkConfig,
        bindings: &Qwen3xDSparkMarkovWeightBindings,
        confidence_bindings: Option<&Qwen3xDSparkConfidenceWeightBindings>,
        max_requests: usize,
        sampler_bounds: TopKSamplingBounds,
    ) -> Result<Self, ModelExecutorError> {
        assert!(max_requests <= sampler_bounds.max_sampling_inputs as usize);
        let quantization = model_config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DSpark Markov requires quantization config"))?;
        let w1_quantization = quantization.resolve_for_tensor(&bindings.w1.weight);
        let w2_quantization = quantization.resolve_for_tensor(&bindings.w2.weight);
        let rank = to_u32("Qwen3x DSpark Markov rank", model_config.markov_rank)?;
        let vocab_size = to_u32("Qwen3x DSpark Markov vocabulary", model_config.vocab_size)?;
        let confidence_config = model_config
            .enable_confidence_head
            .then_some(DSparkMarkovConfidenceConfig {
                hidden_dim: to_u32("Qwen3x DSpark confidence hidden dimension", model_config.hidden_size)?,
                with_markov: model_config.confidence_head_with_markov,
            });
        let config = DSparkMarkovSamplingConfig {
            block_size: model_config.block_size,
            vocab_size,
            rank,
            w1_group_size: to_u32("Qwen3x DSpark Markov W1 group_size", w1_quantization.group_size)?,
            w1_bits: to_u32("Qwen3x DSpark Markov W1 bits", w1_quantization.bits)?,
            w2_group_size: to_u32("Qwen3x DSpark Markov W2 group_size", w2_quantization.group_size)?,
            w2_bits: to_u32("Qwen3x DSpark Markov W2 bits", w2_quantization.bits)?,
            io_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
            confidence: confidence_config,
            sampling: TopKSamplingBounds {
                max_sampling_inputs: max_requests
                    .try_into()
                    .expect("Qwen3x DSpark maximum requests must fit sampling bounds"),
                ..sampler_bounds
            },
        };
        config.validate();
        match (confidence_config, confidence_bindings) {
            (Some(_), Some(_)) | (None, None) => {},
            _ => {
                return Err(ModelExecutorError::custom(
                    "Qwen3x DSpark confidence config and weight bindings must match",
                ));
            },
        }
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        if let Some(bindings) = confidence_bindings {
            bindings.push_tensor_names(&mut tensor_names);
        }
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Qwen3xDSparkMarkovWeights::from_tensors(device, &mut tensors, config, bindings)?;
        let confidence = match (confidence_config, confidence_bindings) {
            (Some(confidence), Some(bindings)) => {
                Some(Qwen3xDSparkConfidenceWeights::from_tensors(
                    device,
                    &mut tensors,
                    confidence,
                    rank,
                    bindings,
                )?)
            },
            (None, None) => None,
            _ => unreachable!("confidence config and bindings were validated"),
        };
        assert!(tensors.is_empty(), "Qwen3x DSpark Markov must consume its tensor map");
        Ok(Self {
            backend: DSparkMarkovSampling::new(device, config),
            weights,
            confidence,
        })
    }

    pub fn prepare(
        &self,
        req_slots: &[u32],
        anchor_token_ids: &[u32],
        anchor_positions: &[u32],
        sampler_configs: &[SamplerConfig],
        distribution_store: &SpecProbsStore,
    ) -> DSparkMarkovReplayShape {
        self.backend.prepare(
            req_slots,
            anchor_token_ids,
            anchor_positions,
            sampler_configs,
            distribution_store,
        )
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: DSparkMarkovReplayShape,
        base_logits: &'a Buffer,
        hidden: &'a Buffer,
        distribution_store: &'a SpecProbsStore,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.backend.record(
            recorder,
            DSparkMarkovInput {
                shape,
                base_logits,
                distribution_store,
                weights: self.weights.as_borrowed(),
                confidence: self.confidence.as_ref().map(|weights| {
                    DSparkConfidenceInput {
                        hidden,
                        weights: weights.as_borrowed(),
                    }
                }),
            },
        );
    }

    pub fn add_replay_arguments(&self, shape: DSparkMarkovReplayShape, arguments: &mut ReplayArguments) {
        self.backend.add_replay_arguments(shape, arguments);
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> DSparkProposal {
        self.backend.read_proposal(req_slots, distribution_store)
    }
}

impl Qwen3xDSparkMarkovWeights {
    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        config: DSparkMarkovSamplingConfig,
        bindings: &Qwen3xDSparkMarkovWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let w1_config = QuantizedEmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_dim: config.rank,
            group_size: config.w1_group_size,
            bits: config.w1_bits,
            scale_bias_dtype: config.scale_bias_dtype,
            output_dtype: config.io_dtype,
        };
        w1_config.validate();
        let w1_weight = remove_quant_weight(tensors, &bindings.w1.weight)?;
        let w1_scales = remove_typed_tensor(tensors, &bindings.w1.scales, safetensors::Dtype::BF16)?.into_data();
        let w1_biases = remove_typed_tensor(tensors, &bindings.w1.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "Qwen3x DSpark Markov W1 weight",
            w1_weight.len(),
            w1_config.weight_bytes(),
        )?;
        let w1_affine_bytes = w1_config
            .num_affine_params()
            .checked_mul(config.scale_bias_dtype.item_size())
            .expect("Qwen3x DSpark Markov W1 affine byte length must fit usize");
        validate_len("Qwen3x DSpark Markov W1 scales", w1_scales.len(), w1_affine_bytes)?;
        validate_len("Qwen3x DSpark Markov W1 biases", w1_biases.len(), w1_affine_bytes)?;

        let w2_config = AffineQuantizedMatmulConfig {
            n: config
                .vocab_size
                .try_into()
                .expect("Qwen3x DSpark vocabulary must fit i32"),
            k: config.rank.try_into().expect("Qwen3x DSpark Markov rank must fit i32"),
            group_size: config
                .w2_group_size
                .try_into()
                .expect("Qwen3x DSpark Markov group size must fit i32"),
            bits: config
                .w2_bits
                .try_into()
                .expect("Qwen3x DSpark Markov bits must fit i32"),
            input_dtype: config.io_dtype,
            output_dtype: config.io_dtype,
            scale_bias_dtype: config.scale_bias_dtype,
        };
        w2_config.validate();
        let w2_weight = remove_quant_weight(tensors, &bindings.w2.weight)?;
        let w2_scales = remove_typed_tensor(tensors, &bindings.w2.scales, safetensors::Dtype::BF16)?.into_data();
        let w2_biases = remove_typed_tensor(tensors, &bindings.w2.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "Qwen3x DSpark Markov W2 weight",
            w2_weight.len(),
            w2_config.weight_bytes(),
        )?;
        validate_len(
            "Qwen3x DSpark Markov W2 scales",
            w2_scales.len(),
            w2_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3x DSpark Markov W2 biases",
            w2_biases.len(),
            w2_config.scale_or_bias_bytes(),
        )?;
        Ok(Self {
            w1_weight: Buffer::from_slice(device, &w1_weight),
            w1_scales: Buffer::from_slice(device, &w1_scales),
            w1_biases: Buffer::from_slice(device, &w1_biases),
            w2_weight: Buffer::from_slice(device, &w2_weight),
            w2_scales: Buffer::from_slice(device, &w2_scales),
            w2_biases: Buffer::from_slice(device, &w2_biases),
        })
    }

    fn as_borrowed(&self) -> DSparkMarkovWeights<'_> {
        DSparkMarkovWeights {
            w1_weight: &self.w1_weight,
            w1_scales: &self.w1_scales,
            w1_biases: &self.w1_biases,
            w2_weight: &self.w2_weight,
            w2_scales: &self.w2_scales,
            w2_biases: &self.w2_biases,
        }
    }
}

impl Qwen3xDSparkConfidenceWeights {
    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        config: DSparkMarkovConfidenceConfig,
        rank: u32,
        bindings: &Qwen3xDSparkConfidenceWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.hidden_dim as usize;
        let input_dim = if config.with_markov {
            hidden_dim
                .checked_add(rank as usize)
                .expect("Qwen3x DSpark confidence input dimension must fit usize")
        } else {
            hidden_dim
        };
        let weight = remove_typed_tensor(tensors, &bindings.weight, safetensors::Dtype::BF16)?;
        validate_shape("Qwen3x DSpark confidence weight", weight.shape(), &[1, input_dim])?;
        let bias = remove_typed_tensor(tensors, &bindings.bias, safetensors::Dtype::BF16)?;
        validate_shape("Qwen3x DSpark confidence bias", bias.shape(), &[1])?;
        Ok(Self {
            weight: Buffer::from_slice(device, weight.data()),
            bias: Buffer::from_slice(device, bias.data()),
        })
    }

    fn as_borrowed(&self) -> DSparkConfidenceWeights<'_> {
        DSparkConfidenceWeights {
            weight: &self.weight,
            bias: &self.bias,
        }
    }
}
