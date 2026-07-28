use inference_backend_metal::components::QuantizedEmbeddingBuffers;
use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::components::QuantizedEmbeddingKernel;
use inference_backend_metal::components::QuantizedEmbeddingShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbedConfig {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub affine_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl EmbedConfig {
    pub fn validate(self) {
        validate_quantized_embedding(
            self.max_tokens,
            self.vocab_size,
            self.hidden_dim,
            self.group_size,
            self.bits,
        );
        self.config().validate();
    }

    fn config(self) -> QuantizedEmbeddingConfig {
        QuantizedEmbeddingConfig {
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            affine_dtype: self.affine_dtype,
            output_dtype: self.output_dtype,
        }
    }
}

pub struct Embed {
    config: EmbedConfig,
    kernel: QuantizedEmbeddingKernel,
    weights: EmbedWeights,
}

struct EmbedWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

#[derive(Clone, Copy)]
pub struct EmbedInput<'a> {
    pub num_tokens: u32,
    pub token_ids: &'a Buffer,
    pub output_hidden: &'a Buffer,
}

impl Embed {
    fn validate_input(&self, input: EmbedInput<'_>) {
        assert!(input.num_tokens > 0, "embedding requires at least one token");
        assert!(
            input.num_tokens <= self.config.max_tokens,
            "embedding num_tokens={} exceed max_tokens={}",
            input.num_tokens,
            self.config.max_tokens
        );
    }

    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: EmbedConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        config.validate();
        let weights = EmbedWeights::load(device, store, config.config(), bindings)?;
        let embedding = Self {
            config,
            kernel: QuantizedEmbeddingKernel::new(device, config.config()),
            weights,
        };
        embedding.validate_weights();
        Ok(embedding)
    }

    fn validate_weights(&self) {
        let config = self.config.config();
        assert_eq!(self.weights.weight.len_bytes(), config.weight_bytes());
        assert_eq!(
            self.weights.scales.len_bytes(),
            config.num_affine_params() * self.config.affine_dtype.item_size()
        );
        assert_eq!(self.weights.biases.len_bytes(), self.weights.scales.len_bytes());
    }
}

impl ReplayLayer for Embed {
    type Input<'a> = EmbedInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(input);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kernel.invoke(
            QuantizedEmbeddingShape {
                num_tokens: input.num_tokens,
            },
            QuantizedEmbeddingBuffers {
                token_ids: input.token_ids,
                weight: &self.weights.weight,
                scales: &self.weights.scales,
                biases: &self.weights.biases,
                output: input.output_hidden,
            },
        )));
        input.output_hidden
    }
}

impl EmbedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: QuantizedEmbeddingConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let weight = store
            .tensor_bytes(&bindings.weight, safetensors::Dtype::U32)?
            .into_data();
        let scales = store
            .tensor_bytes(&bindings.scales, safetensors::Dtype::BF16)?
            .into_data();
        let biases = store
            .tensor_bytes(&bindings.biases, safetensors::Dtype::BF16)?
            .into_data();
        validate_len("embed weight", weight.len(), config.weight_bytes())?;
        validate_len(
            "embed scales",
            scales.len(),
            config.num_affine_params() * config.affine_dtype.item_size(),
        )?;
        validate_len("embed biases", biases.len(), scales.len())?;
        Ok(Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        })
    }
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_quantized_embedding(max_tokens: u32, vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) {
    assert!(max_tokens > 0);
    assert!(vocab_size > 0);
    assert!(hidden_dim > 0);
    assert!(matches!(group_size, 32 | 64 | 128));
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    assert_eq!(hidden_dim % group_size, 0);
}
