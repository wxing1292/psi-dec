use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

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
        validate_quantized_unembedding(
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

    fn validate_weights(&self) {
        let shape = self.config.affine_shape(self.config.max_tokens);
        assert_eq!(self.weights.weight.len_bytes(), shape.weight_bytes());
        assert_eq!(self.weights.scales.len_bytes(), shape.affine_param_bytes());
        assert_eq!(self.weights.biases.len_bytes(), self.weights.scales.len_bytes());
    }
}

impl ReplayLayer for Unembed {
    type Input<'a> = UnembedInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(input.num_rows > 0, "unembed requires at least one row");
        assert!(
            input.num_rows <= self.config.max_tokens,
            "unembed num_rows={} exceed max_tokens={}",
            input.num_rows,
            self.config.max_tokens
        );
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

impl UnembedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        shape: AffineQuantizedMatmulShape,
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

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_quantized_unembedding(max_tokens: u32, vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) {
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
