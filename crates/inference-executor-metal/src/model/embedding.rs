use std::rc::Rc;

use inference_backend_metal::components::embedding;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::remove_tensor;
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
    pub scale_bias_dtype: Dtype,
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

    fn config(self) -> embedding::Config {
        embedding::Config {
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            scale_bias_dtype: self.scale_bias_dtype,
            output_dtype: self.output_dtype,
        }
    }
}

pub struct Embed {
    config: EmbedConfig,
    kernel: Rc<embedding::Compute>,
    weights: Option<Rc<EmbedWeights>>,
}

struct EmbedWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

#[derive(Clone, Copy)]
pub struct EmbedInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub token_ids: &'a Buffer,
    pub output_hidden: &'a Buffer,
}

impl Embed {
    /// Returns the token-row capacity of the embedding buffers.
    pub fn max_tokens(&self) -> u32 {
        self.config.max_tokens
    }

    /// Returns a token-capacity view that shares the immutable kernel and weights.
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let config = EmbedConfig {
            max_tokens,
            ..self.config
        };
        config.validate();
        Self {
            config,
            kernel: Rc::clone(&self.kernel),
            weights: self.weights.as_ref().map(Rc::clone),
        }
    }

    pub fn new(device: &Device, config: EmbedConfig) -> Self {
        config.validate();
        Self {
            config,
            kernel: Rc::new(embedding::Compute::new(device, config.config())),
            weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: QuantizedTensorBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "embedding weights are already loaded");
        self.weights = Some(Rc::new(EmbedWeights::load(
            device,
            store,
            self.config.config(),
            bindings,
        )?));
        self.validate_weights();
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "embedding weights are not loaded");
        self.weights.take();
    }

    fn validate_weights(&self) {
        let config = self.config.config();
        let weights = self.weights();
        assert_eq!(weights.weight.len_bytes(), config.weight_bytes());
        assert_eq!(
            weights.scales.len_bytes(),
            config.num_affine_params() * self.config.scale_bias_dtype.item_size()
        );
        assert_eq!(weights.biases.len_bytes(), weights.scales.len_bytes());
    }

    fn weights(&self) -> &EmbedWeights {
        self.weights
            .as_deref()
            .expect("embedding weights must be loaded before execution")
    }

    fn validate_num_tokens(&self, num_tokens: u32) {
        assert!(num_tokens > 0, "embedding requires at least one token");
        assert!(
            num_tokens <= self.config.max_tokens,
            "embedding num_tokens={} exceed max_tokens={}",
            num_tokens,
            self.config.max_tokens
        );
    }
}

impl ReplayLayer for Embed {
    type Input<'a> = EmbedInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_num_tokens(input.num_total_tokens);
        let weights = self.weights();
        let shape = embedding::Shape {
            num_total_tokens: input.num_total_tokens,
        };
        let buffers = embedding::Buffers {
            token_ids: input.token_ids,
            weight: &weights.weight,
            scales: &weights.scales,
            biases: &weights.biases,
            output: input.output_hidden,
        };
        let invocation = match input.num_active_tokens {
            ReplayU32::Fixed(num_active_tokens) => {
                assert_eq!(num_active_tokens, input.num_total_tokens);
                self.kernel.invoke(shape, buffers)
            },
            ReplayU32::Parameter(key) => self.kernel.invoke_bucketed(shape, key, buffers),
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
        input.output_hidden
    }
}

impl EmbedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: embedding::Config,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensors = store.load_tensors([
            bindings.weight.as_str(),
            bindings.scales.as_str(),
            bindings.biases.as_str(),
        ])?;
        let weight = remove_tensor(&mut tensors, &bindings.weight, safetensors::Dtype::U32)?.into_data();
        let scales = remove_tensor(&mut tensors, &bindings.scales, safetensors::Dtype::BF16)?.into_data();
        let biases = remove_tensor(&mut tensors, &bindings.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("embed weight", weight.len(), config.weight_bytes())?;
        validate_len(
            "embed scales",
            scales.len(),
            config.num_affine_params() * config.scale_bias_dtype.item_size(),
        )?;
        validate_len("embed biases", biases.len(), scales.len())?;
        let weights = Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        };
        assert!(tensors.is_empty(), "embed must consume its tensor map");
        Ok(weights)
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
