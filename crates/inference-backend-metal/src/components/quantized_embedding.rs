use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

const QUANTIZED_EMBEDDING_SOURCE: &str = include_str!("metal/quantized_embedding.metal");

#[derive(Clone, Copy, Debug)]
pub struct QuantizedEmbeddingConfig {
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl QuantizedEmbeddingConfig {
    pub fn validate(self) {
        assert!(self.vocab_size > 0);
        assert!(self.hidden_dim > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        match self.output_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 quantized embedding output is not implemented"),
            dtype => panic!("unsupported quantized embedding output dtype {dtype:?}"),
        }
        let _ = self.weight_bytes_unchecked();
        let _ = self.num_affine_params_unchecked();
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
    kernel: Kernel,
}

impl QuantizedEmbeddingKernel {
    pub fn new(device: &Device, config: QuantizedEmbeddingConfig) -> Self {
        config.validate();
        let function_name = match config.scale_bias_dtype {
            Dtype::Float32 => "quantized_embedding_f32_to_bf16",
            Dtype::Bfloat16 => "quantized_embedding_bf16_to_bf16",
            _ => unreachable!("validated quantized embedding scale/bias dtype"),
        };
        Self {
            config,
            kernel: Kernel::new(device, QUANTIZED_EMBEDDING_SOURCE, function_name),
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
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity grid whose active token count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        capacity_shape: QuantizedEmbeddingShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: QuantizedEmbeddingBuffers<'a>,
    ) -> QuantizedEmbeddingInvocation<'a> {
        QuantizedEmbeddingInvocation {
            kernel: self,
            shape: capacity_shape,
            buffers,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }
}

pub struct QuantizedEmbeddingInvocation<'a> {
    kernel: &'a QuantizedEmbeddingKernel,
    shape: QuantizedEmbeddingShape,
    buffers: QuantizedEmbeddingBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedEmbeddingInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        validate_buffers(self.kernel.config, self.shape, &self.buffers);
        let config = self.kernel.config;
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.buffers.token_ids, 0);
        builder.set_buffer_read(1, self.buffers.weight, 0);
        builder.set_buffer_read(2, self.buffers.scales, 0);
        builder.set_buffer_read(3, self.buffers.biases, 0);
        builder.set_buffer_write(4, self.buffers.output, 0);
        match self.num_active_tokens_key {
            Some(key) => builder.bind_u32(5, key, 1, self.shape.num_tokens),
            None => builder.set_u32(5, self.shape.num_tokens),
        }
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
        .checked_mul(config.scale_bias_dtype.item_size())
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

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::QuantizedEmbeddingBuffers;
    use super::QuantizedEmbeddingConfig;
    use super::QuantizedEmbeddingKernel;
    use super::QuantizedEmbeddingShape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const VOCAB_SIZE: u32 = 2;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;
    const BITS: u32 = 4;
    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.quantized_embedding.num_active_tokens");
    const OUTPUT_CANARY: u16 = 0x7fc1;

    #[test]
    #[should_panic(expected = "F32 quantized embedding output is not implemented")]
    fn test_f32_output_is_explicit_future_work() {
        QuantizedEmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: BITS,
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Float32,
        }
        .validate();
    }

    #[test]
    fn test_f32_scale_bias_reference() {
        let scales = [0.5f32, 0.25];
        let biases = [-1.0f32, 2.0];
        test_reference(
            Dtype::Float32,
            &f32_bytes(&scales),
            &f32_bytes(&biases),
            &scales,
            &biases,
        );
    }

    #[test]
    fn test_bf16_scale_bias_reference() {
        let scales = [bf16::from_f32(0.5), bf16::from_f32(0.25)];
        let biases = [bf16::from_f32(-1.0), bf16::from_f32(2.0)];
        test_reference(
            Dtype::Bfloat16,
            &bf16_bytes(&scales),
            &bf16_bytes(&biases),
            &scales.map(bf16::to_f32),
            &biases.map(bf16::to_f32),
        );
    }

    #[test]
    fn test_bucketed_replay_preserves_inactive_tail_across_grow_and_shrink() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let scales = [0.5f32, 0.25];
        let biases = [-1.0f32, 2.0];
        let config = QuantizedEmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: BITS,
            scale_bias_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedEmbeddingShape { num_tokens: 4 };
        let token_ids = Buffer::from_slice(&device, &[0_u32, 1, 0, u32::MAX]);
        let weight = Buffer::from_slice(&device, &packed_q4_rows());
        let scales_buffer = Buffer::from_slice(&device, &f32_bytes(&scales));
        let biases_buffer = Buffer::from_slice(&device, &f32_bytes(&biases));
        let output = Buffer::from_slice(&device, &vec![OUTPUT_CANARY; shape.num_output_values(config)]);
        let kernel = QuantizedEmbeddingKernel::new(&device, config);

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke_bucketed(
            shape,
            NUM_ACTIVE_TOKENS,
            QuantizedEmbeddingBuffers {
                token_ids: &token_ids,
                weight: &weight,
                scales: &scales_buffer,
                biases: &biases_buffer,
                output: &output,
            },
        ));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 1);

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 3))
            .wait();
        let active_values = 3 * HIDDEN_DIM as usize;
        let expected_active = token_ids_reference(&[0, 1, 0], &scales, &biases);
        assert_eq!(output.read_typed::<u16>(0, active_values), expected_active);
        assert_eq!(
            output.read_typed::<u16>(active_values, HIDDEN_DIM as usize),
            vec![OUTPUT_CANARY; HIDDEN_DIM as usize]
        );

        let exact_output = Buffer::from_slice(&device, &vec![OUTPUT_CANARY; active_values]);
        let mut exact_recorder = stream.create_replay_program();
        exact_recorder.record(kernel.invoke(
            QuantizedEmbeddingShape { num_tokens: 3 },
            QuantizedEmbeddingBuffers {
                token_ids: &token_ids,
                weight: &weight,
                scales: &scales_buffer,
                biases: &biases_buffer,
                output: &exact_output,
            },
        ));
        let exact_replay = exact_recorder.build();
        assert_eq!(exact_replay.stats().parameter_count, 0);
        stream.submit_replay(&exact_replay).wait();
        assert_eq!(exact_output.read_typed::<u16>(0, active_values), expected_active);

        token_ids.write_typed(3, &[1_u32]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 4))
            .wait();
        let expected_full = token_ids_reference(&[0, 1, 0, 1], &scales, &biases);
        let full_output = output.read_typed::<u16>(0, shape.num_output_values(config));
        assert_eq!(full_output, expected_full);

        token_ids.write_typed(3, &[u32::MAX]);
        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(0, active_values), expected_active);
        assert_eq!(
            output.read_typed::<u16>(active_values, HIDDEN_DIM as usize),
            full_output[active_values..]
        );
    }

    #[test]
    fn test_bucketed_replay_validates_arguments_and_total_capacity_buffers() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = QuantizedEmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: BITS,
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedEmbeddingShape { num_tokens: 4 };
        let token_ids = Buffer::from_slice(&device, &[0_i32; 4]);
        let active_token_ids = Buffer::from_slice(&device, &[0_i32; 3]);
        let weight = Buffer::from_slice(&device, &packed_q4_rows());
        let scales = Buffer::from_slice(&device, &[bf16::ONE.to_bits(); 2]);
        let biases = Buffer::from_slice(&device, &[bf16::ZERO.to_bits(); 2]);
        let output = Buffer::new_zeroed_elements(&device, shape.num_output_values(config), Dtype::Bfloat16);
        let active_output = Buffer::new_zeroed_elements(&device, 3 * HIDDEN_DIM as usize, Dtype::Bfloat16);
        let kernel = QuantizedEmbeddingKernel::new(&device, config);
        let buffers = |token_ids, output| {
            QuantizedEmbeddingBuffers {
                token_ids,
                weight: &weight,
                scales: &scales,
                biases: &biases,
                output,
            }
        };

        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke_bucketed(shape, NUM_ACTIVE_TOKENS, buffers(&token_ids, &output)));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 1);
        assert_panics(|| {
            let _ = stream.submit_replay(&replay);
        });
        assert_panics(|| {
            let arguments = ReplayArguments::new().with_i32(NUM_ACTIVE_TOKENS, 3);
            let _ = stream.submit_replay_with_arguments(&replay, &arguments);
        });
        for invalid_num_active_tokens in [0, 5] {
            assert_panics(|| {
                let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, invalid_num_active_tokens);
                let _ = stream.submit_replay_with_arguments(&replay, &arguments);
            });
        }

        assert_panics(|| {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke_bucketed(shape, NUM_ACTIVE_TOKENS, buffers(&active_token_ids, &output)));
        });
        assert_panics(|| {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke_bucketed(shape, NUM_ACTIVE_TOKENS, buffers(&token_ids, &active_output)));
        });
    }

    fn test_reference(
        scale_bias_dtype: Dtype,
        scale_bytes: &[u8],
        bias_bytes: &[u8],
        scales: &[f32; 2],
        biases: &[f32; 2],
    ) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = QuantizedEmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: BITS,
            scale_bias_dtype,
            output_dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedEmbeddingShape { num_tokens: 4 };
        let token_ids = [-1i32, 0, 1, 2];
        let packed_weights = packed_q4_rows();
        let token_ids = Buffer::from_slice(&device, &token_ids);
        let weight = Buffer::from_slice(&device, &packed_weights);
        let scales_buffer = Buffer::from_slice(&device, scale_bytes);
        let biases_buffer = Buffer::from_slice(&device, bias_bytes);
        let output = Buffer::new_zeroed_elements(&device, shape.num_output_values(config), Dtype::Bfloat16);
        let kernel = QuantizedEmbeddingKernel::new(&device, config);
        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke(
            shape,
            QuantizedEmbeddingBuffers {
                token_ids: &token_ids,
                weight: &weight,
                scales: &scales_buffer,
                biases: &biases_buffer,
                output: &output,
            },
        ));
        let replay = recorder.build();
        assert_eq!(replay.stats().parameter_count, 0);
        stream.submit_replay(&replay).wait();

        let actual = output.read_typed::<u16>(0, shape.num_output_values(config));
        let expected = token_ids_reference(&[-1, 0, 1, 2], scales, biases);
        assert_eq!(actual, expected);
    }

    fn packed_q4_rows() -> Vec<u8> {
        (0..VOCAB_SIZE)
            .flat_map(|row| {
                (0..HIDDEN_DIM / 2).map(move |column| {
                    let lo = (row * 3 + column * 2) & 0x0f;
                    let hi = (row * 3 + column * 2 + 1) & 0x0f;
                    (lo | (hi << 4)) as u8
                })
            })
            .collect()
    }

    fn token_ids_reference(token_ids: &[i32], scales: &[f32; 2], biases: &[f32; 2]) -> Vec<u16> {
        token_ids
            .iter()
            .flat_map(|&token_id| {
                (0..HIDDEN_DIM).map(move |hidden_index| {
                    if !(0..VOCAB_SIZE as i32).contains(&token_id) {
                        return bf16::ZERO.to_bits();
                    }
                    let quantized = (token_id as u32 * 3 + hidden_index) & 0x0f;
                    bf16::from_f32(quantized as f32 * scales[token_id as usize] + biases[token_id as usize]).to_bits()
                })
            })
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    fn bf16_bytes(values: &[bf16]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_bits().to_ne_bytes()).collect()
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
