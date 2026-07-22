use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const QUANTIZED_EMBEDDING_SOURCE: &str = include_str!("metal/quantized_embedding.metal");

#[derive(Clone, Copy, Debug)]
pub struct QuantizedEmbeddingConfig {
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub affine_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl QuantizedEmbeddingConfig {
    pub fn f32_to_bf16(vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            group_size,
            bits,
            affine_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn bf16_to_bf16(vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            group_size,
            bits,
            affine_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.vocab_size > 0);
        assert!(self.hidden_dim > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.affine_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.output_dtype, Dtype::Bfloat16);
        let _ = self.weight_bytes_unchecked();
        let _ = self.num_affine_params_unchecked();
    }

    pub fn packed_cols(self) -> usize {
        self.validate();
        self.packed_cols_unchecked()
    }

    fn packed_cols_unchecked(self) -> usize {
        (self.hidden_dim as usize)
            .checked_mul(self.bits as usize)
            .expect("quantized embedding packed width must fit usize")
            / 32
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        self.weight_bytes_unchecked()
    }

    fn weight_bytes_unchecked(self) -> usize {
        (self.vocab_size as usize)
            .checked_mul(self.packed_cols_unchecked())
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .expect("quantized embedding weight byte length must fit usize")
    }

    pub fn num_affine_params(self) -> usize {
        self.validate();
        self.num_affine_params_unchecked()
    }

    fn num_affine_params_unchecked(self) -> usize {
        (self.vocab_size as usize)
            .checked_mul((self.hidden_dim / self.group_size) as usize)
            .expect("quantized embedding affine parameter count must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizedEmbeddingShape {
    pub num_tokens: u32,
}

impl QuantizedEmbeddingShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
    }

    pub fn num_output_values(self, config: QuantizedEmbeddingConfig) -> usize {
        self.validate();
        self.num_tokens
            .checked_mul(config.hidden_dim)
            .expect("quantized embedding output index must fit the shader u32 domain") as usize
    }
}

#[derive(Clone, Copy)]
pub struct QuantizedEmbeddingBuffers<'a> {
    pub token_ids: &'a Buffer,
    pub weight: &'a Buffer,
    pub scales: &'a Buffer,
    pub biases: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct QuantizedEmbeddingKernel {
    config: QuantizedEmbeddingConfig,
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
}

impl QuantizedEmbeddingKernel {
    pub fn new(device: &Device, config: QuantizedEmbeddingConfig) -> Self {
        config.validate();
        Self {
            config,
            f32_kernel: Kernel::new(device, QUANTIZED_EMBEDDING_SOURCE, "quantized_embedding_f32_to_bf16"),
            bf16_kernel: Kernel::new(device, QUANTIZED_EMBEDDING_SOURCE, "quantized_embedding_bf16_to_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedEmbeddingShape,
        buffers: QuantizedEmbeddingBuffers<'a>,
    ) -> QuantizedEmbeddingInvocation<'a> {
        QuantizedEmbeddingInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct QuantizedEmbeddingInvocation<'a> {
    kernel: &'a QuantizedEmbeddingKernel,
    shape: QuantizedEmbeddingShape,
    buffers: QuantizedEmbeddingBuffers<'a>,
}

impl Operator for QuantizedEmbeddingInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        validate_buffers(self.kernel.config, self.shape, &self.buffers);
        let config = self.kernel.config;
        let kernel = match config.affine_dtype {
            Dtype::Float32 => &self.kernel.f32_kernel,
            Dtype::Bfloat16 => &self.kernel.bf16_kernel,
            other => panic!("unsupported quantized embedding affine dtype: {other:?}"),
        };
        builder.set_kernel(kernel);
        builder.set_buffer_read(0, self.buffers.token_ids, 0);
        builder.set_buffer_read(1, self.buffers.weight, 0);
        builder.set_buffer_read(2, self.buffers.scales, 0);
        builder.set_buffer_read(3, self.buffers.biases, 0);
        builder.set_buffer_write(4, self.buffers.output, 0);
        builder.set_u32(5, self.shape.num_tokens);
        builder.set_u32(6, config.vocab_size);
        builder.set_u32(7, config.hidden_dim);
        builder.set_u32(8, config.group_size);
        builder.set_u32(9, config.bits);
        builder.dispatch_threadblocks((self.shape.num_output_values(config).div_ceil(256), 1, 1), (256, 1, 1));
    }
}

fn validate_buffers(
    config: QuantizedEmbeddingConfig,
    shape: QuantizedEmbeddingShape,
    buffers: &QuantizedEmbeddingBuffers<'_>,
) {
    shape.validate();
    let affine_param_bytes = config
        .num_affine_params_unchecked()
        .checked_mul(config.affine_dtype.item_size())
        .expect("quantized embedding affine parameter bytes must fit usize");
    let output_bytes = shape
        .num_output_values(config)
        .checked_mul(config.output_dtype.item_size())
        .expect("quantized embedding output bytes must fit usize");
    assert!(buffers.token_ids.len_bytes() >= shape.num_tokens as usize * size_of::<i32>());
    assert_eq!(buffers.weight.len_bytes(), config.weight_bytes_unchecked());
    assert_eq!(buffers.scales.len_bytes(), affine_param_bytes);
    assert_eq!(buffers.biases.len_bytes(), affine_param_bytes);
    assert!(buffers.output.len_bytes() >= output_bytes);
}
