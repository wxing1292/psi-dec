use inference_backend_metal::components::QuantizedEmbeddingBuffers;
use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::components::QuantizedEmbeddingKernel;
use inference_backend_metal::components::QuantizedEmbeddingShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::Layer;
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
        validate_quantized_head(
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

    fn active_shape(&self, num_tokens: u32) -> QuantizedEmbeddingShape {
        QuantizedEmbeddingShape { num_tokens }
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

impl Layer for Embed {
    type Input<'a> = EmbedInput<'a>;
    type Output<'a> = &'a Buffer;

    type InputShape = EmbedConfig;
    type OutputShape = EmbedConfig;

    fn input_shape(&self) -> Self::InputShape {
        self.config
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.config
    }
}

impl ReplayLayer for Embed {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(input);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kernel.invoke(
            self.active_shape(input.num_tokens),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnembedConfig {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub affine_dtype: Dtype,
}

impl UnembedConfig {
    pub fn validate(self) {
        validate_quantized_head(
            self.max_tokens,
            self.vocab_size,
            self.hidden_dim,
            self.group_size,
            self.bits,
        );
        assert_eq!(self.input_dtype, Dtype::Bfloat16);
        assert_eq!(self.output_dtype, Dtype::Bfloat16);
        assert!(matches!(self.affine_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn logits_bytes(self) -> usize {
        self.validate();
        self.affine_shape(self.max_tokens).output_bytes()
    }

    fn affine_shape(self, num_rows: u32) -> AffineQuantizedMatmulShape {
        assert!(num_rows > 0);
        assert!(num_rows <= self.max_tokens);
        AffineQuantizedMatmulShape {
            m: num_rows.try_into().expect("unembed row count must fit i32"),
            n: self.vocab_size.try_into().expect("unembed vocab_size must fit i32"),
            k: self.hidden_dim.try_into().expect("unembed hidden_dim must fit i32"),
            group_size: self.group_size.try_into().expect("unembed group_size must fit i32"),
            bits: self.bits.try_into().expect("unembed bits must fit i32"),
            input_dtype: self.input_dtype,
            output_dtype: self.output_dtype,
            affine_dtype: self.affine_dtype,
        }
    }
}

pub struct Unembed {
    config: UnembedConfig,
    qmv_kernel: AffineQuantizedMatmulKernel,
    qmm_kernel: AffineQuantizedMatmulKernel,
    weights: UnembedWeights,
}

struct UnembedWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

#[derive(Clone, Copy)]
pub struct UnembedInput<'a> {
    pub num_rows: u32,
    pub hidden: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Unembed {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: UnembedConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        config.validate();
        let weights = UnembedWeights::load(device, store, config.affine_shape(config.max_tokens), bindings)?;
        let qmv_shape = config.affine_shape(1);
        let qmm_rows = unembed_qmv_batch_limit(config.hidden_dim, config.vocab_size).min(config.max_tokens);
        let qmm_shape = config.affine_shape(qmm_rows);
        let unembed = Self {
            config,
            qmv_kernel: AffineQuantizedMatmulKernel::new(device, qmv_shape),
            qmm_kernel: AffineQuantizedMatmulKernel::new(device, qmm_shape),
            weights,
        };
        unembed.validate_weights();
        Ok(unembed)
    }

    fn assert_rows_fit(&self, num_rows: u32, op_name: &str) {
        assert!(num_rows > 0, "{op_name} requires at least one row");
        assert!(
            num_rows <= self.config.max_tokens,
            "{op_name} num_rows={} exceed max_tokens={}",
            num_rows,
            self.config.max_tokens
        );
    }

    fn validate_weights(&self) {
        let shape = self.config.affine_shape(self.config.max_tokens);
        assert_eq!(self.weights.weight.len_bytes(), shape.weight_bytes());
        assert_eq!(self.weights.scales.len_bytes(), shape.affine_param_bytes());
        assert_eq!(self.weights.biases.len_bytes(), self.weights.scales.len_bytes());
    }
}

impl Layer for Unembed {
    type Input<'a> = UnembedInput<'a>;
    type Output<'a> = &'a Buffer;

    type InputShape = UnembedConfig;
    type OutputShape = UnembedConfig;

    fn input_shape(&self) -> Self::InputShape {
        self.config
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.config
    }
}

impl ReplayLayer for Unembed {
    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.assert_rows_fit(input.num_rows, "unembed");
        let shape = self.config.affine_shape(input.num_rows);
        let kernel = if input.num_rows < unembed_qmv_batch_limit(self.config.hidden_dim, self.config.vocab_size) {
            &self.qmv_kernel
        } else {
            &self.qmm_kernel
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(kernel.invoke_with_shape(
            shape,
            input.logits,
            0,
            input.hidden,
            0,
            &self.weights.weight,
            0,
            &self.weights.scales,
            0,
            &self.weights.biases,
            0,
        )));
        input.logits
    }
}

impl EmbedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: QuantizedEmbeddingConfig,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let weight = tensor_data(store, &bindings.weight, safetensors::Dtype::U32)?;
        let scales = tensor_data(store, &bindings.scales, safetensors::Dtype::BF16)?;
        let biases = tensor_data(store, &bindings.biases, safetensors::Dtype::BF16)?;
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

impl UnembedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        shape: AffineQuantizedMatmulShape,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let weight = tensor_data(store, &bindings.weight, safetensors::Dtype::U32)?;
        let scales = tensor_data(store, &bindings.scales, safetensors::Dtype::BF16)?;
        let biases = tensor_data(store, &bindings.biases, safetensors::Dtype::BF16)?;
        validate_len("unembed weight", weight.len(), shape.weight_bytes())?;
        validate_len("unembed scales", scales.len(), shape.affine_param_bytes())?;
        validate_len("unembed biases", biases.len(), shape.affine_param_bytes())?;
        Ok(Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        })
    }
}

fn tensor_data(
    store: &mut SafeTensorStore,
    name: &str,
    dtype: safetensors::Dtype,
) -> Result<Vec<u8>, ModelExecutorError> {
    Ok(store.tensor_bytes(name, dtype)?.into_data())
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_quantized_head(max_tokens: u32, vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) {
    assert!(max_tokens > 0);
    assert!(vocab_size > 0);
    assert!(hidden_dim > 0);
    assert!(matches!(group_size, 32 | 64 | 128));
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    assert_eq!(hidden_dim % group_size, 0);
}

fn unembed_qmv_batch_limit(input_dim: u32, output_dim: u32) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}
