use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::remove_tensor;
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
    pub scale_bias_dtype: Dtype,
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
        assert_eq!(
            self.scale_bias_dtype,
            Dtype::Bfloat16,
            "unembed requires BF16 affine parameters"
        );
    }

    pub fn logits_bytes(self) -> usize {
        self.validate();
        self.affine_config()
            .output_bytes(self.max_tokens.try_into().expect("unembed max_tokens must fit i32"))
    }

    fn affine_config(self) -> AffineQuantizedMatmulConfig {
        AffineQuantizedMatmulConfig {
            n: self.vocab_size.try_into().expect("unembed vocab_size must fit i32"),
            k: self.hidden_dim.try_into().expect("unembed hidden_dim must fit i32"),
            group_size: self.group_size.try_into().expect("unembed group_size must fit i32"),
            bits: self.bits.try_into().expect("unembed bits must fit i32"),
            input_dtype: self.input_dtype,
            output_dtype: self.output_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }
}

pub struct Unembed {
    config: UnembedConfig,
    matmul: Rc<AffineQuantizedMatmul>,
    weights: Rc<UnembedWeights>,
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

#[derive(Clone, Copy)]
pub struct UnembedBucketedInput<'a> {
    pub num_total_rows: u32,
    pub num_active_rows_key: ReplayParameterKey,
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
        let affine_config = config.affine_config();
        let weights = UnembedWeights::load(device, store, affine_config, bindings)?;
        let unembed = Self {
            config,
            matmul: Rc::new(AffineQuantizedMatmul::new(device, affine_config)),
            weights: Rc::new(weights),
        };
        unembed.validate_weights();
        Ok(unembed)
    }

    fn validate_weights(&self) {
        let config = self.config.affine_config();
        assert_eq!(self.weights.weight.len_bytes(), config.weight_bytes());
        assert_eq!(self.weights.scales.len_bytes(), config.scale_or_bias_bytes());
        assert_eq!(self.weights.biases.len_bytes(), self.weights.scales.len_bytes());
    }

    pub fn max_tokens(&self) -> u32 {
        self.config.max_tokens
    }

    /// Returns a row-capacity view that shares the immutable matmul and weights.
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let config = UnembedConfig {
            max_tokens,
            ..self.config
        };
        config.validate();
        Self {
            config,
            matmul: Rc::clone(&self.matmul),
            weights: Rc::clone(&self.weights),
        }
    }

    pub fn replay_topology(&self, num_total_rows: u32) -> AffineQuantizedMatmulKernelKind {
        self.validate_num_rows(num_total_rows);
        self.matmul.topology(num_total_rows)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.matmul.topology_boundaries()
    }

    pub fn record_bucketed<'a, R>(&'a self, recorder: &mut R, input: UnembedBucketedInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_num_rows(input.num_total_rows);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.matmul.invoke_bucketed(
            input.num_total_rows,
            input.num_active_rows_key,
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

    fn validate_num_rows(&self, num_rows: u32) {
        assert!(num_rows > 0, "unembed requires at least one row");
        assert!(
            num_rows <= self.config.max_tokens,
            "unembed num_rows={} exceed max_tokens={}",
            num_rows,
            self.config.max_tokens
        );
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
        recorder.record_with_barrier_before(ReplayOp::opaque(self.matmul.invoke(
            input.num_rows.try_into().expect("unembed row count must fit i32"),
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
        config: AffineQuantizedMatmulConfig,
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
        validate_len("unembed weight", weight.len(), config.weight_bytes())?;
        validate_len("unembed scales", scales.len(), config.scale_or_bias_bytes())?;
        validate_len("unembed biases", biases.len(), config.scale_or_bias_bytes())?;
        let weights = Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        };
        assert!(tensors.is_empty(), "unembed must consume its tensor map");
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

fn validate_quantized_unembedding(max_tokens: u32, vocab_size: u32, hidden_dim: u32, group_size: u32, bits: u32) {
    assert!(max_tokens > 0);
    assert!(vocab_size > 0);
    assert!(hidden_dim > 0);
    assert!(matches!(group_size, 32 | 64 | 128));
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    assert_eq!(hidden_dim % group_size, 0);
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use half::bf16;
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayParameterKey;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::replay::ReplayBucketPolicy;

    use super::Dtype;
    use super::Unembed;
    use super::UnembedBucketedInput;
    use super::UnembedConfig;
    use super::UnembedInput;
    use super::UnembedWeights;
    use crate::def::layer::ReplayLayer;
    use crate::def::replay_op::MetalReplayRuntime;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.unembed.num_active_rows");

    #[test]
    #[should_panic(expected = "unembed requires BF16 affine parameters")]
    fn test_rejects_mixed_scale_bias_dtype() {
        let config = UnembedConfig {
            max_tokens: 16,
            vocab_size: 151_936,
            hidden_dim: 5120,
            group_size: 64,
            bits: 4,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Float32,
        };
        config.validate();
    }

    #[test]
    fn test_topology_boundaries_preserve_recorded_affine_topology() {
        let device = Device::system_default();
        let unembed = test_unembed(&device, test_config(32));
        let boundaries = unembed.replay_topology_boundaries();
        let policy = ReplayBucketPolicy::with_topology_boundaries(32, &boundaries);

        assert_eq!(&*boundaries, &[18]);
        for num_active_rows in 1..=32 {
            let num_total_rows = policy.capacity(num_active_rows);
            assert_eq!(
                unembed.replay_topology(num_active_rows),
                unembed.replay_topology(num_total_rows),
                "num_active_rows={num_active_rows} num_total_rows={num_total_rows}"
            );
        }
    }

    #[test]
    fn test_capacity_view_shares_matmul_and_weights() {
        let device = Device::system_default();
        let unembed = test_unembed(&device, test_config(2));

        let expanded = unembed.with_max_tokens(7);

        assert_eq!(unembed.max_tokens(), 2);
        assert_eq!(expanded.max_tokens(), 7);
        assert!(Rc::ptr_eq(&unembed.matmul, &expanded.matmul));
        assert!(Rc::ptr_eq(&unembed.weights, &expanded.weights));
    }

    #[test]
    fn test_exact_and_bucketed_replay_parity_and_grow_shrink_guards() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let config = test_config(4);
        let unembed = test_unembed(&device, config);
        let num_values_per_hidden_row = config.hidden_dim as usize;
        let num_values_per_logits_row = config.vocab_size as usize;
        let num_total_hidden_values = config.max_tokens as usize * num_values_per_hidden_row;
        let num_total_logits_values = config.max_tokens as usize * num_values_per_logits_row;
        let sentinel = bf16::from_f32(-123.0).to_f32();
        let hidden = bf16_buffer(&device, &vec![0.0; num_total_hidden_values]);
        let exact_logits = bf16_buffer(&device, &vec![sentinel; num_total_logits_values]);

        let mut exact_recorder = runtime.create_recorder();
        let exact_output = unembed.record(
            &mut exact_recorder,
            UnembedInput {
                num_rows: 1,
                hidden: &hidden,
                logits: &exact_logits,
            },
        );
        assert!(std::ptr::eq(exact_output, &exact_logits));
        let exact_replay = exact_recorder.build();
        assert_eq!(exact_replay.stats().parameter_count, 0);
        runtime.submit_replay(&exact_replay).wait();
        let exact_values = read_bf16_values(&exact_logits, num_total_logits_values);
        assert_eq!(
            &exact_values[..num_values_per_logits_row],
            &vec![0.0; num_values_per_logits_row]
        );
        assert_eq!(
            &exact_values[num_values_per_logits_row..],
            &vec![sentinel; num_total_logits_values - num_values_per_logits_row]
        );

        let logits = bf16_buffer(&device, &vec![sentinel; num_total_logits_values]);
        let mut bucketed_recorder = runtime.create_recorder();
        let bucketed_output = unembed.record_bucketed(
            &mut bucketed_recorder,
            UnembedBucketedInput {
                num_total_rows: config.max_tokens,
                num_active_rows_key: NUM_ACTIVE_ROWS,
                hidden: &hidden,
                logits: &logits,
            },
        );
        assert!(std::ptr::eq(bucketed_output, &logits));
        let bucketed_replay = bucketed_recorder.build();
        assert_eq!(bucketed_replay.stats().parameter_count, 1);

        assert_invalid_active_rows(&stream, &bucketed_replay, config.max_tokens);

        let active_one = ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 1);
        write_bf16_values(
            &hidden,
            &poisoned_hidden_values(config.max_tokens, num_values_per_hidden_row),
        );
        runtime
            .submit_replay_with_arguments(&bucketed_replay, &active_one)
            .wait();
        let first_values = read_bf16_values(&logits, num_total_logits_values);
        assert_eq!(
            &first_values[..num_values_per_logits_row],
            &exact_values[..num_values_per_logits_row]
        );
        assert_eq!(
            &first_values[num_values_per_logits_row..],
            &vec![sentinel; num_total_logits_values - num_values_per_logits_row]
        );

        write_bf16_values(&hidden, &vec![0.0; num_total_hidden_values]);
        runtime
            .submit_replay_with_arguments(
                &bucketed_replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, config.max_tokens),
            )
            .wait();
        assert_eq!(
            read_bf16_values(&logits, num_total_logits_values),
            vec![0.0; num_total_logits_values]
        );

        write_bf16_values(&logits, &vec![sentinel; num_total_logits_values]);
        write_bf16_values(
            &hidden,
            &poisoned_hidden_values(config.max_tokens, num_values_per_hidden_row),
        );
        runtime
            .submit_replay_with_arguments(&bucketed_replay, &active_one)
            .wait();
        let shrunk_values = read_bf16_values(&logits, num_total_logits_values);
        assert_eq!(
            &shrunk_values[..num_values_per_logits_row],
            &exact_values[..num_values_per_logits_row]
        );
        assert_eq!(
            &shrunk_values[num_values_per_logits_row..],
            &vec![sentinel; num_total_logits_values - num_values_per_logits_row]
        );

        for num_total_rows in [0, config.max_tokens + 1] {
            assert_panics(|| {
                let mut recorder = runtime.create_recorder();
                unembed.record_bucketed(
                    &mut recorder,
                    UnembedBucketedInput {
                        num_total_rows,
                        num_active_rows_key: NUM_ACTIVE_ROWS,
                        hidden: &hidden,
                        logits: &logits,
                    },
                );
            });
        }

        let short_hidden = bf16_buffer(&device, &vec![0.0; num_values_per_hidden_row]);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            unembed.record_bucketed(
                &mut recorder,
                UnembedBucketedInput {
                    num_total_rows: config.max_tokens,
                    num_active_rows_key: NUM_ACTIVE_ROWS,
                    hidden: &short_hidden,
                    logits: &logits,
                },
            );
        });

        let short_logits = bf16_buffer(&device, &vec![sentinel; num_values_per_logits_row]);
        assert_panics(|| {
            let mut recorder = runtime.create_recorder();
            unembed.record_bucketed(
                &mut recorder,
                UnembedBucketedInput {
                    num_total_rows: config.max_tokens,
                    num_active_rows_key: NUM_ACTIVE_ROWS,
                    hidden: &hidden,
                    logits: &short_logits,
                },
            );
        });
    }

    fn test_config(max_tokens: u32) -> UnembedConfig {
        UnembedConfig {
            max_tokens,
            vocab_size: 32,
            hidden_dim: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        }
    }

    fn test_unembed(device: &Device, config: UnembedConfig) -> Unembed {
        config.validate();
        let affine_config = config.affine_config();
        let unembed = Unembed {
            config,
            matmul: Rc::new(super::AffineQuantizedMatmul::new(device, affine_config)),
            weights: Rc::new(UnembedWeights {
                weight: Buffer::new_zeroed(device, affine_config.weight_bytes()),
                scales: Buffer::new_zeroed(device, affine_config.scale_or_bias_bytes()),
                biases: Buffer::new_zeroed(device, affine_config.scale_or_bias_bytes()),
            }),
        };
        unembed.validate_weights();
        unembed
    }

    fn poisoned_hidden_values(num_rows: u32, num_values_per_row: usize) -> Vec<f32> {
        let mut values = vec![f32::NAN; num_rows as usize * num_values_per_row];
        values[..num_values_per_row].fill(0.0);
        values
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }

    fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        buffer.write_typed(0, &bits);
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_invalid_active_rows(
        stream: &Stream,
        replay: &inference_backend_metal::metal::ReplayProgram,
        num_total_rows: u32,
    ) {
        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(replay, &ReplayArguments::new());
        });
        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(replay, &ReplayArguments::new().with_i32(NUM_ACTIVE_ROWS, 1));
        });
        for num_active_rows in [0, num_total_rows + 1] {
            assert_panics(|| {
                let _ = stream.submit_replay_with_arguments(
                    replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows),
                );
            });
        }
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
