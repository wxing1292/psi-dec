use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::operators::AffineQuantizedMatmul;
use crate::operators::AffineQuantizedMatmulConfig;
use crate::operators::AffineQuantizedMatmulKernelKind;

const DENSE_MLP_SWIGLU_SOURCE: &str = include_str!("metal/quantized_dense_mlp_swiglu.metal");

const ELEMENTWISE_NUM_THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct QuantizedDenseMLPConfig {
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub dtype: Dtype,
}

impl QuantizedDenseMLPConfig {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        self.stacked_intermediate_dim();
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        i32::try_from(self.hidden_dim).expect("dense MLP hidden_dim must fit i32");
        i32::try_from(self.intermediate_dim).expect("dense MLP intermediate_dim must fit i32");
        i32::try_from(self.stacked_intermediate_dim()).expect("dense MLP stacked intermediate_dim must fit i32");
        i32::try_from(self.group_size).expect("dense MLP group_size must fit i32");
        i32::try_from(self.bits).expect("dense MLP bits must fit i32");
    }

    pub fn gate_up_config(self) -> AffineQuantizedMatmulConfig {
        self.validate();
        self.affine_config_unchecked(self.stacked_intermediate_dim(), self.hidden_dim)
    }

    pub fn down_config(self) -> AffineQuantizedMatmulConfig {
        self.validate();
        self.affine_config_unchecked(self.hidden_dim, self.intermediate_dim)
    }

    pub fn swiglu_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.validate();
        shape.validate();
        (self.swiglu_num_values_unchecked(shape) as usize)
            .checked_mul(self.dtype.item_size())
            .expect("dense MLP swiglu byte length must fit usize")
    }

    fn swiglu_num_values_unchecked(self, shape: QuantizedDenseMLPShape) -> u32 {
        self.intermediate_dim
            .checked_mul(shape.num_tokens)
            .expect("dense MLP swiglu num_values must fit u32")
    }

    pub fn input_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.validate();
        shape.validate();
        self.input_bytes_unchecked(shape)
    }

    fn input_bytes_unchecked(self, shape: QuantizedDenseMLPShape) -> usize {
        (shape.num_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|count| count.checked_mul(self.dtype.item_size()))
            .expect("dense MLP input byte length must fit usize")
    }

    fn gate_up_output_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.gate_up_config()
            .output_bytes(shape.num_tokens.try_into().expect("dense MLP token count must fit i32"))
    }

    fn output_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.down_config()
            .output_bytes(shape.num_tokens.try_into().expect("dense MLP token count must fit i32"))
    }

    fn affine_config_unchecked(self, n: u32, k: u32) -> AffineQuantizedMatmulConfig {
        AffineQuantizedMatmulConfig {
            n: n.try_into().expect("dense MLP output dimension must fit i32"),
            k: k.try_into().expect("dense MLP input dimension must fit i32"),
            group_size: self.group_size.try_into().expect("dense MLP group_size must fit i32"),
            bits: self.bits.try_into().expect("dense MLP bits must fit i32"),
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            scale_bias_dtype: self.dtype,
        }
    }

    fn stacked_intermediate_dim(self) -> u32 {
        self.intermediate_dim
            .checked_mul(2)
            .expect("dense MLP stacked gate/up dim must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedDenseMLPShape {
    pub num_tokens: u32,
}

impl QuantizedDenseMLPShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        i32::try_from(self.num_tokens).expect("dense MLP token count must fit i32");
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QuantizedDenseMLPReplayTopology {
    pub gate_up_affine: AffineQuantizedMatmulKernelKind,
    pub down_affine: AffineQuantizedMatmulKernelKind,
}

#[derive(Clone, Copy)]
pub struct QuantizedDenseMLPBuffers<'a> {
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedDenseMLPWeights<'a> {
    pub gate_up_weight: &'a Buffer,
    pub gate_up_scales: &'a Buffer,
    pub gate_up_biases: &'a Buffer,
    pub down_weight: &'a Buffer,
    pub down_scales: &'a Buffer,
    pub down_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QuantizedDenseMLPScratch<'a> {
    pub gate_up: &'a Buffer,
    pub swiglu: &'a Buffer,
}

/// Records one quantized dense MLP:
///
/// ```text
/// hidden_state [T, H]
///        |
///        v
///     gate_up
///        |
///        v
/// scratch.gate_up [T, 2I]
///        |
///        v
///      swiglu
///        |
///        v
/// scratch.swiglu [T, I]
///        |
///        v
///       down
///        |
///        v
/// next_hidden_state [T, H]
/// ```
pub struct QuantizedDenseMLP {
    config: QuantizedDenseMLPConfig,
    gate_up: AffineQuantizedMatmul,
    down: AffineQuantizedMatmul,
    swiglu: QuantizedDenseMLPSwiGLUKernel,
}

impl QuantizedDenseMLP {
    pub fn new(device: &Device, config: QuantizedDenseMLPConfig) -> Self {
        config.validate();
        Self {
            config,
            gate_up: AffineQuantizedMatmul::new(device, config.gate_up_config()),
            down: AffineQuantizedMatmul::new(device, config.down_config()),
            swiglu: QuantizedDenseMLPSwiGLUKernel::new(device, config.dtype),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        buffers: QuantizedDenseMLPBuffers<'a>,
        scratch: QuantizedDenseMLPScratch<'a>,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPInvocation<'a> {
        QuantizedDenseMLPInvocation {
            compute: self,
            shape,
            buffers,
            scratch,
            weights,
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity dense MLP whose active token count is supplied at submission.
    pub fn invoke_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        buffers: QuantizedDenseMLPBuffers<'a>,
        scratch: QuantizedDenseMLPScratch<'a>,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPInvocation<'a> {
        let shape = capacity_shape(num_total_tokens);
        QuantizedDenseMLPInvocation {
            compute: self,
            shape,
            buffers,
            scratch,
            weights,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    pub fn topology(&self, num_total_tokens: u32) -> QuantizedDenseMLPReplayTopology {
        let shape = capacity_shape(num_total_tokens);
        QuantizedDenseMLPReplayTopology {
            gate_up_affine: self.gate_up.topology(shape.num_tokens),
            down_affine: self.down.topology(shape.num_tokens),
        }
    }

    pub fn topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.gate_up.topology_boundaries().into_vec();
        boundaries.extend(self.down.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    pub fn invoke_gate_up<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        hidden_state: &'a Buffer,
        gate_up: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPGateUpInvocation<'a> {
        QuantizedDenseMLPGateUpInvocation {
            compute: self,
            shape,
            hidden_state,
            gate_up,
            weights,
            num_active_tokens_key: None,
        }
    }

    pub fn invoke_gate_up_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        hidden_state: &'a Buffer,
        gate_up: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPGateUpInvocation<'a> {
        QuantizedDenseMLPGateUpInvocation {
            compute: self,
            shape: capacity_shape(num_total_tokens),
            hidden_state,
            gate_up,
            weights,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    pub fn invoke_swiglu<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        gate_up: &'a Buffer,
        swiglu: &'a Buffer,
    ) -> QuantizedDenseMLPSwiGLUInvocation<'a> {
        QuantizedDenseMLPSwiGLUInvocation {
            compute: self,
            shape,
            gate_up,
            swiglu,
            num_active_tokens_key: None,
        }
    }

    pub fn invoke_swiglu_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        gate_up: &'a Buffer,
        swiglu: &'a Buffer,
    ) -> QuantizedDenseMLPSwiGLUInvocation<'a> {
        QuantizedDenseMLPSwiGLUInvocation {
            compute: self,
            shape: capacity_shape(num_total_tokens),
            gate_up,
            swiglu,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        swiglu: &'a Buffer,
        next_hidden_state: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPDownInvocation<'a> {
        QuantizedDenseMLPDownInvocation {
            compute: self,
            shape,
            swiglu,
            next_hidden_state,
            weights,
            num_active_tokens_key: None,
        }
    }

    pub fn invoke_down_bucketed<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens_key: ReplayParameterKey,
        swiglu: &'a Buffer,
        next_hidden_state: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPDownInvocation<'a> {
        QuantizedDenseMLPDownInvocation {
            compute: self,
            shape: capacity_shape(num_total_tokens),
            swiglu,
            next_hidden_state,
            weights,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }
}

fn capacity_shape(num_total_tokens: u32) -> QuantizedDenseMLPShape {
    let shape = QuantizedDenseMLPShape {
        num_tokens: num_total_tokens,
    };
    shape.validate();
    shape
}

pub struct QuantizedDenseMLPInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    buffers: QuantizedDenseMLPBuffers<'a>,
    scratch: QuantizedDenseMLPScratch<'a>,
    weights: QuantizedDenseMLPWeights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        QuantizedDenseMLPGateUpInvocation {
            compute: self.compute,
            shape: self.shape,
            hidden_state: self.buffers.hidden_state,
            gate_up: self.scratch.gate_up,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        }
        .record(builder);
        builder.record_with_barrier_before(QuantizedDenseMLPSwiGLUInvocation {
            compute: self.compute,
            shape: self.shape,
            gate_up: self.scratch.gate_up,
            swiglu: self.scratch.swiglu,
            num_active_tokens_key: self.num_active_tokens_key,
        });
        builder.record_with_barrier_before(QuantizedDenseMLPDownInvocation {
            compute: self.compute,
            shape: self.shape,
            swiglu: self.scratch.swiglu,
            next_hidden_state: self.buffers.next_hidden_state,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        });
    }
}

pub struct QuantizedDenseMLPGateUpInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    hidden_state: &'a Buffer,
    gate_up: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPGateUpInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let invocation = match self.num_active_tokens_key {
            Some(key) => {
                self.compute.gate_up.invoke_bucketed(
                    self.shape.num_tokens,
                    key,
                    self.gate_up,
                    0,
                    self.hidden_state,
                    0,
                    self.weights.gate_up_weight,
                    0,
                    self.weights.gate_up_scales,
                    0,
                    self.weights.gate_up_biases,
                    0,
                )
            },
            None => {
                self.compute.gate_up.invoke(
                    self.shape
                        .num_tokens
                        .try_into()
                        .expect("dense MLP token count must fit i32"),
                    self.gate_up,
                    0,
                    self.hidden_state,
                    0,
                    self.weights.gate_up_weight,
                    0,
                    self.weights.gate_up_scales,
                    0,
                    self.weights.gate_up_biases,
                    0,
                )
            },
        };
        invocation.record(builder);
    }
}

pub struct QuantizedDenseMLPSwiGLUInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPSwiGLUInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.compute
            .swiglu
            .invoke(
                self.compute.config,
                self.shape,
                self.gate_up,
                self.swiglu,
                self.num_active_tokens_key,
            )
            .record(builder);
    }
}

pub struct QuantizedDenseMLPDownInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    swiglu: &'a Buffer,
    next_hidden_state: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPDownInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let invocation = match self.num_active_tokens_key {
            Some(key) => {
                self.compute.down.invoke_bucketed(
                    self.shape.num_tokens,
                    key,
                    self.next_hidden_state,
                    0,
                    self.swiglu,
                    0,
                    self.weights.down_weight,
                    0,
                    self.weights.down_scales,
                    0,
                    self.weights.down_biases,
                    0,
                )
            },
            None => {
                self.compute.down.invoke(
                    self.shape
                        .num_tokens
                        .try_into()
                        .expect("dense MLP token count must fit i32"),
                    self.next_hidden_state,
                    0,
                    self.swiglu,
                    0,
                    self.weights.down_weight,
                    0,
                    self.weights.down_scales,
                    0,
                    self.weights.down_biases,
                    0,
                )
            },
        };
        invocation.record(builder);
    }
}

struct QuantizedDenseMLPSwiGLUKernel {
    kernel: Kernel,
}

impl QuantizedDenseMLPSwiGLUKernel {
    fn new(device: &Device, dtype: Dtype) -> Self {
        let function_name = match dtype {
            Dtype::Float32 => "dense_mlp_swiglu_f32",
            Dtype::Bfloat16 => "dense_mlp_swiglu_bf16",
            dtype => panic!("unsupported dense MLP swiglu dtype {dtype:?}"),
        };
        Self {
            kernel: Kernel::new(device, DENSE_MLP_SWIGLU_SOURCE, function_name),
        }
    }

    fn invoke<'a>(
        &'a self,
        config: QuantizedDenseMLPConfig,
        shape: QuantizedDenseMLPShape,
        gate_up: &'a Buffer,
        swiglu: &'a Buffer,
        num_active_tokens_key: Option<ReplayParameterKey>,
    ) -> QuantizedDenseMLPSwiGLURowMajorInvocation<'a> {
        QuantizedDenseMLPSwiGLURowMajorInvocation {
            kernel: &self.kernel,
            config,
            shape,
            gate_up,
            swiglu,
            num_active_tokens_key,
        }
    }
}

struct QuantizedDenseMLPSwiGLURowMajorInvocation<'a> {
    kernel: &'a Kernel,
    config: QuantizedDenseMLPConfig,
    shape: QuantizedDenseMLPShape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPSwiGLURowMajorInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.gate_up, 0);
        builder.set_buffer_write(1, self.swiglu, 0);
        match self.num_active_tokens_key {
            Some(key) => builder.bind_u32(2, key, 1, self.shape.num_tokens),
            None => builder.set_u32(2, self.shape.num_tokens),
        }
        builder.set_u32(3, self.config.intermediate_dim);
        let num_values = self.config.swiglu_num_values_unchecked(self.shape) as usize;
        builder.dispatch_1d(num_values, ELEMENTWISE_NUM_THREADS_PER_THREADBLOCK);
    }
}

impl QuantizedDenseMLPSwiGLURowMajorInvocation<'_> {
    fn validate(&self) {
        self.shape.validate();
        let gate_up_output_bytes = self.config.gate_up_output_bytes(self.shape);
        let swiglu_bytes = self.config.swiglu_bytes(self.shape);
        assert!(
            self.gate_up.len_bytes() >= gate_up_output_bytes,
            "dense MLP gate/up projection buffer is too small"
        );
        assert!(
            self.swiglu.len_bytes() >= swiglu_bytes,
            "dense MLP swiglu buffer is too small"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use half::bf16;
    use inference_executor_core::mlp::dense::DenseMLPCore;
    use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceWeights;
    use inference_executor_core::mlp::dense::reference::quantized_dense_mlp_reference;
    use inference_executor_core::replay::ReplayBucketPolicy;

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayProgram;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.dense_mlp.num_active_tokens");
    const HIDDEN_POISON: u16 = 0x7fc1;
    const GATE_UP_CANARY: u16 = 0x3555;
    const SWIGLU_CANARY: u16 = 0x3aaa;
    const OUTPUT_CANARY: u16 = 0x3c00;

    #[test]
    fn test_fixed() {
        let config = QuantizedDenseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 64,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedDenseMLPShape { num_tokens: 4 };
        let (device, compute) = create_dense_mlp_compute(config);
        let stream = Stream::new(&device);
        let gate_up_config = config.gate_up_config();
        let down_config = config.down_config();
        let hidden_values = hidden_fixture(shape.num_tokens as usize, config.hidden_dim as usize);
        let hidden_state = bf16_buffer(&device, &hidden_values);
        let gate_up_weight_values = quantized_weight_values(gate_up_config.weight_bytes());
        let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
        let gate_up_scale_values = affine_param_fixture(gate_up_config.scale_or_bias_bytes() / size_of::<u16>());
        let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
        let gate_up_bias_values = zero_fixture(gate_up_config.scale_or_bias_bytes() / size_of::<u16>());
        let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
        let down_weight_values = quantized_weight_values(down_config.weight_bytes());
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values = affine_param_fixture(down_config.scale_or_bias_bytes() / size_of::<u16>());
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values = zero_fixture(down_config.scale_or_bias_bytes() / size_of::<u16>());
        let down_biases = bf16_buffer(&device, &down_bias_values);
        let weights = QuantizedDenseMLPWeights {
            gate_up_weight: &gate_up_weight,
            gate_up_scales: &gate_up_scales,
            gate_up_biases: &gate_up_biases,
            down_weight: &down_weight,
            down_scales: &down_scales,
            down_biases: &down_biases,
        };

        let replay_output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let replay_gate_up = Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape));
        let replay_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape));
        let mut builder = stream.create_replay_program();
        builder.record(compute.invoke(
            shape,
            QuantizedDenseMLPBuffers {
                hidden_state: &hidden_state,
                next_hidden_state: &replay_output,
            },
            QuantizedDenseMLPScratch {
                gate_up: &replay_gate_up,
                swiglu: &replay_swiglu,
            },
            weights,
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let expected = quantized_dense_mlp_reference(
            &DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: config.hidden_dim as usize,
                intermediate_dim: config.intermediate_dim as usize,
            },
            &hidden_values
                .iter()
                .map(|value| bf16::from_f32(*value).to_f32())
                .collect::<Vec<_>>(),
            shape.num_tokens as usize,
            config.group_size as usize,
            config.bits as usize,
            QuantizedDenseMLPReferenceWeights {
                gate_up_weight: &gate_up_weight_values,
                gate_up_scales: &bf16_values(&gate_up_scale_values),
                gate_up_biases: &bf16_values(&gate_up_bias_values),
                down_weight: &down_weight_values,
                down_scales: &bf16_values(&down_scale_values),
                down_biases: &bf16_values(&down_bias_values),
            },
        );
        let expected = expected
            .into_iter()
            .map(|value| bf16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let actual = replay_output
            .read_typed::<u16>(0, config.output_bytes(shape) / size_of::<u16>())
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
    }

    #[test]
    fn test_random() {
        let random_seed = 0x5D2A_91C7;
        let config = QuantizedDenseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 4160,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedDenseMLPShape { num_tokens: 7 };
        let (device, compute) = create_dense_mlp_compute(config);
        let stream = Stream::new(&device);
        let gate_up_config = config.gate_up_config();
        let down_config = config.down_config();
        let hidden_values = generated_values(shape.num_tokens as usize * config.hidden_dim as usize, random_seed);
        let hidden_state = bf16_buffer(&device, &hidden_values);
        let gate_up_weight_values = generated_bytes(gate_up_config.weight_bytes(), random_seed.wrapping_add(1));
        let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
        let gate_up_scale_values = generated_scales(
            gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(2),
        );
        let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
        let gate_up_bias_values = generated_biases(
            gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(3),
        );
        let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
        let down_weight_values = generated_bytes(down_config.weight_bytes(), random_seed.wrapping_add(4));
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values = generated_scales(
            down_config.scale_or_bias_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(5),
        );
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values = generated_biases(
            down_config.scale_or_bias_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(6),
        );
        let down_biases = bf16_buffer(&device, &down_bias_values);

        let replay_output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let replay_gate_up = Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape));
        let replay_swiglu = Buffer::new_zeroed(&device, config.swiglu_bytes(shape));
        let mut builder = stream.create_replay_program();
        builder.record(compute.invoke(
            shape,
            QuantizedDenseMLPBuffers {
                hidden_state: &hidden_state,
                next_hidden_state: &replay_output,
            },
            QuantizedDenseMLPScratch {
                gate_up: &replay_gate_up,
                swiglu: &replay_swiglu,
            },
            QuantizedDenseMLPWeights {
                gate_up_weight: &gate_up_weight,
                gate_up_scales: &gate_up_scales,
                gate_up_biases: &gate_up_biases,
                down_weight: &down_weight,
                down_scales: &down_scales,
                down_biases: &down_biases,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let expected = quantized_dense_mlp_reference(
            &DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: config.hidden_dim as usize,
                intermediate_dim: config.intermediate_dim as usize,
            },
            &bf16_values(&hidden_values),
            shape.num_tokens as usize,
            config.group_size as usize,
            config.bits as usize,
            QuantizedDenseMLPReferenceWeights {
                gate_up_weight: &gate_up_weight_values,
                gate_up_scales: &bf16_values(&gate_up_scale_values),
                gate_up_biases: &bf16_values(&gate_up_bias_values),
                down_weight: &down_weight_values,
                down_scales: &bf16_values(&down_scale_values),
                down_biases: &bf16_values(&down_bias_values),
            },
        );
        let expected = expected
            .into_iter()
            .map(|value| bf16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let actual = replay_output
            .read_typed::<u16>(0, config.output_bytes(shape) / size_of::<u16>())
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
    }

    #[test]
    fn test_bucket_topology_unions_both_affines_and_preserves_active_identity() {
        let device = Device::system_default();
        let matching_boundaries = QuantizedDenseMLP::new(&device, bucket_test_config());
        assert_eq!(&*matching_boundaries.topology_boundaries(), &[6, 9, 17]);
        assert_policy_preserves_topology(&matching_boundaries, 64);

        let different_boundaries = QuantizedDenseMLP::new(
            &device,
            QuantizedDenseMLPConfig {
                hidden_dim: 128,
                intermediate_dim: 4096,
                group_size: 32,
                bits: 4,
                dtype: Dtype::Bfloat16,
            },
        );
        assert_eq!(&*different_boundaries.topology_boundaries(), &[6, 9, 12, 17]);
        assert_policy_preserves_topology(&different_boundaries, 64);
    }

    #[test]
    fn test_exact_and_bucketed_parameter_counts_and_validation() {
        let fixture = BucketedDenseMLPFixture::new(20);
        assert_eq!(fixture.exact_replay(20).stats().parameter_count, 0);
        let replay = fixture.bucketed_replay(20);
        assert_eq!(replay.stats().parameter_count, 1);

        assert_panics(|| {
            let _ = fixture.stream.submit_replay(&replay);
        });
        assert_panics(|| {
            let arguments = ReplayArguments::new().with_i32(NUM_ACTIVE_TOKENS, 17);
            let _ = fixture.stream.submit_replay_with_arguments(&replay, &arguments);
        });
        for invalid_num_active_tokens in [0, 21] {
            assert_panics(|| {
                let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, invalid_num_active_tokens);
                let _ = fixture.stream.submit_replay_with_arguments(&replay, &arguments);
            });
        }

        fixture.assert_total_buffer_validation();
    }

    #[test]
    fn test_bucketed_replay_preserves_poisoned_tails_across_topologies_and_shrink() {
        let fixture = BucketedDenseMLPFixture::new(20);
        for (num_total_tokens, num_active_tokens, seed) in [
            (4, 3, 0x1000_0001),
            (8, 7, 0x1000_0002),
            (12, 9, 0x1000_0003),
            (20, 17, 0x1000_0004),
        ] {
            fixture.reset_canaries();
            let hidden = fixture.write_hidden(num_active_tokens, seed);
            let replay = fixture.bucketed_replay(num_total_tokens);
            assert_eq!(replay.stats().parameter_count, 1);
            fixture.submit(&replay, num_active_tokens);
            fixture.assert_active_output(&hidden, num_active_tokens);
            fixture.assert_canary_tails(num_active_tokens);
        }

        let replay = fixture.bucketed_replay(20);
        fixture.reset_canaries();
        let first_hidden = fixture.write_hidden(18, 0x2000_0001);
        fixture.submit(&replay, 18);
        fixture.assert_active_output(&first_hidden, 18);
        fixture.assert_canary_tails(18);

        let full_hidden = fixture.write_hidden(20, 0x2000_0002);
        fixture.submit(&replay, 20);
        fixture.assert_active_output(&full_hidden, 20);
        let full_gate_up = fixture.read_gate_up();
        let full_swiglu = fixture.read_swiglu();
        let full_output = fixture.read_output();

        let smaller_hidden = fixture.write_hidden(17, 0x2000_0003);
        fixture.submit(&replay, 17);
        fixture.assert_active_output(&smaller_hidden, 17);
        fixture.assert_tails_equal(17, &full_gate_up, &full_swiglu, &full_output);
    }

    struct BucketedDenseMLPFixture {
        device: Device,
        stream: Stream,
        config: QuantizedDenseMLPConfig,
        compute: QuantizedDenseMLP,
        num_allocated_tokens: u32,
        hidden_state: Buffer,
        next_hidden_state: Buffer,
        gate_up: Buffer,
        swiglu: Buffer,
        weights: BucketedDenseMLPWeights,
    }

    impl BucketedDenseMLPFixture {
        fn new(num_allocated_tokens: u32) -> Self {
            let config = bucket_test_config();
            let shape = QuantizedDenseMLPShape {
                num_tokens: num_allocated_tokens,
            };
            let device = Device::system_default();
            let stream = Stream::new(&device);
            let compute = QuantizedDenseMLP::new(&device, config);
            let weights = BucketedDenseMLPWeights::new(&device, config);
            Self {
                hidden_state: Buffer::new_zeroed(&device, config.input_bytes(shape)),
                next_hidden_state: Buffer::new_zeroed(&device, config.output_bytes(shape)),
                gate_up: Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape)),
                swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(shape)),
                device,
                stream,
                config,
                compute,
                num_allocated_tokens,
                weights,
            }
        }

        fn exact_replay(&self, num_tokens: u32) -> ReplayProgram {
            let mut builder = self.stream.create_replay_program();
            builder.record(self.compute.invoke(
                QuantizedDenseMLPShape { num_tokens },
                self.buffers(),
                self.scratch(),
                self.weights.as_borrowed(),
            ));
            builder.build()
        }

        fn bucketed_replay(&self, num_total_tokens: u32) -> ReplayProgram {
            let mut builder = self.stream.create_replay_program();
            builder.record(self.compute.invoke_bucketed(
                num_total_tokens,
                NUM_ACTIVE_TOKENS,
                self.buffers(),
                self.scratch(),
                self.weights.as_borrowed(),
            ));
            builder.build()
        }

        fn buffers(&self) -> QuantizedDenseMLPBuffers<'_> {
            QuantizedDenseMLPBuffers {
                hidden_state: &self.hidden_state,
                next_hidden_state: &self.next_hidden_state,
            }
        }

        fn scratch(&self) -> QuantizedDenseMLPScratch<'_> {
            QuantizedDenseMLPScratch {
                gate_up: &self.gate_up,
                swiglu: &self.swiglu,
            }
        }

        fn reset_canaries(&self) {
            self.gate_up.write_typed(
                0,
                &vec![GATE_UP_CANARY; self.num_allocated_tokens as usize * self.config.intermediate_dim as usize * 2],
            );
            self.swiglu.write_typed(
                0,
                &vec![SWIGLU_CANARY; self.num_allocated_tokens as usize * self.config.intermediate_dim as usize],
            );
            self.next_hidden_state.write_typed(
                0,
                &vec![OUTPUT_CANARY; self.num_allocated_tokens as usize * self.config.hidden_dim as usize],
            );
        }

        fn write_hidden(&self, num_active_tokens: u32, seed: u32) -> Vec<f32> {
            assert!(num_active_tokens <= self.num_allocated_tokens);
            let num_active_values = num_active_tokens as usize * self.config.hidden_dim as usize;
            let active_values = bf16_values(&generated_values(num_active_values, seed));
            let mut all_bits =
                vec![HIDDEN_POISON; self.num_allocated_tokens as usize * self.config.hidden_dim as usize];
            for (bits, value) in all_bits.iter_mut().zip(&active_values) {
                *bits = bf16::from_f32(*value).to_bits();
            }
            self.hidden_state.write_typed(0, &all_bits);
            active_values
        }

        fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
            let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
            self.stream.submit_replay_with_arguments(replay, &arguments).wait();
        }

        fn assert_active_output(&self, hidden: &[f32], num_active_tokens: u32) {
            let num_output_values = num_active_tokens as usize * self.config.hidden_dim as usize;
            let expected = quantized_dense_mlp_reference(
                &DenseMLPCore {
                    model_layer_index: 0,
                    hidden_dim: self.config.hidden_dim as usize,
                    intermediate_dim: self.config.intermediate_dim as usize,
                },
                hidden,
                num_active_tokens as usize,
                self.config.group_size as usize,
                self.config.bits as usize,
                self.weights.as_reference(),
            )
            .into_iter()
            .map(|value| bf16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
            let actual = self
                .next_hidden_state
                .read_typed::<u16>(0, num_output_values)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect::<Vec<_>>();
            assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
        }

        fn assert_canary_tails(&self, num_active_tokens: u32) {
            let gate_up_tail = num_active_tokens as usize * self.config.intermediate_dim as usize * 2;
            let swiglu_tail = num_active_tokens as usize * self.config.intermediate_dim as usize;
            let output_tail = num_active_tokens as usize * self.config.hidden_dim as usize;
            assert!(
                self.read_gate_up()[gate_up_tail..]
                    .iter()
                    .all(|&bits| bits == GATE_UP_CANARY)
            );
            assert!(
                self.read_swiglu()[swiglu_tail..]
                    .iter()
                    .all(|&bits| bits == SWIGLU_CANARY)
            );
            assert!(
                self.read_output()[output_tail..]
                    .iter()
                    .all(|&bits| bits == OUTPUT_CANARY)
            );
        }

        fn assert_tails_equal(
            &self,
            num_active_tokens: u32,
            expected_gate_up: &[u16],
            expected_swiglu: &[u16],
            expected_output: &[u16],
        ) {
            let gate_up_tail = num_active_tokens as usize * self.config.intermediate_dim as usize * 2;
            let swiglu_tail = num_active_tokens as usize * self.config.intermediate_dim as usize;
            let output_tail = num_active_tokens as usize * self.config.hidden_dim as usize;
            assert_eq!(&self.read_gate_up()[gate_up_tail..], &expected_gate_up[gate_up_tail..]);
            assert_eq!(&self.read_swiglu()[swiglu_tail..], &expected_swiglu[swiglu_tail..]);
            assert_eq!(&self.read_output()[output_tail..], &expected_output[output_tail..]);
        }

        fn read_gate_up(&self) -> Vec<u16> {
            self.gate_up.read_typed(
                0,
                self.num_allocated_tokens as usize * self.config.intermediate_dim as usize * 2,
            )
        }

        fn read_swiglu(&self) -> Vec<u16> {
            self.swiglu.read_typed(
                0,
                self.num_allocated_tokens as usize * self.config.intermediate_dim as usize,
            )
        }

        fn read_output(&self) -> Vec<u16> {
            self.next_hidden_state
                .read_typed(0, self.num_allocated_tokens as usize * self.config.hidden_dim as usize)
        }

        fn assert_total_buffer_validation(&self) {
            let short_shape = QuantizedDenseMLPShape {
                num_tokens: self.num_allocated_tokens - 1,
            };
            let short_hidden = Buffer::new_zeroed(&self.device, self.config.input_bytes(short_shape));
            let short_gate_up = Buffer::new_zeroed(&self.device, self.config.gate_up_output_bytes(short_shape));
            let short_swiglu = Buffer::new_zeroed(&self.device, self.config.swiglu_bytes(short_shape));
            let short_output = Buffer::new_zeroed(&self.device, self.config.output_bytes(short_shape));

            assert_panics(|| {
                let mut builder = self.stream.create_replay_program();
                builder.record(self.compute.invoke_bucketed(
                    self.num_allocated_tokens,
                    NUM_ACTIVE_TOKENS,
                    QuantizedDenseMLPBuffers {
                        hidden_state: &short_hidden,
                        next_hidden_state: &self.next_hidden_state,
                    },
                    self.scratch(),
                    self.weights.as_borrowed(),
                ));
            });
            assert_panics(|| {
                let mut builder = self.stream.create_replay_program();
                builder.record(self.compute.invoke_bucketed(
                    self.num_allocated_tokens,
                    NUM_ACTIVE_TOKENS,
                    self.buffers(),
                    QuantizedDenseMLPScratch {
                        gate_up: &short_gate_up,
                        swiglu: &self.swiglu,
                    },
                    self.weights.as_borrowed(),
                ));
            });
            assert_panics(|| {
                let mut builder = self.stream.create_replay_program();
                builder.record(self.compute.invoke_bucketed(
                    self.num_allocated_tokens,
                    NUM_ACTIVE_TOKENS,
                    self.buffers(),
                    QuantizedDenseMLPScratch {
                        gate_up: &self.gate_up,
                        swiglu: &short_swiglu,
                    },
                    self.weights.as_borrowed(),
                ));
            });
            assert_panics(|| {
                let mut builder = self.stream.create_replay_program();
                builder.record(self.compute.invoke_bucketed(
                    self.num_allocated_tokens,
                    NUM_ACTIVE_TOKENS,
                    QuantizedDenseMLPBuffers {
                        hidden_state: &self.hidden_state,
                        next_hidden_state: &short_output,
                    },
                    self.scratch(),
                    self.weights.as_borrowed(),
                ));
            });
        }
    }

    struct BucketedDenseMLPWeights {
        gate_up_weight: Buffer,
        gate_up_scales: Buffer,
        gate_up_biases: Buffer,
        down_weight: Buffer,
        down_scales: Buffer,
        down_biases: Buffer,
        gate_up_weight_values: Vec<u8>,
        gate_up_scale_values: Vec<f32>,
        gate_up_bias_values: Vec<f32>,
        down_weight_values: Vec<u8>,
        down_scale_values: Vec<f32>,
        down_bias_values: Vec<f32>,
    }

    impl BucketedDenseMLPWeights {
        fn new(device: &Device, config: QuantizedDenseMLPConfig) -> Self {
            let gate_up_config = config.gate_up_config();
            let down_config = config.down_config();
            let gate_up_weight_values = generated_bytes(gate_up_config.weight_bytes(), 0x3000_0001);
            let gate_up_scale_values = bf16_values(&generated_scales(
                gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
                0x3000_0002,
            ));
            let gate_up_bias_values = bf16_values(&generated_biases(
                gate_up_config.scale_or_bias_bytes() / size_of::<u16>(),
                0x3000_0003,
            ));
            let down_weight_values = generated_bytes(down_config.weight_bytes(), 0x3000_0004);
            let down_scale_values = bf16_values(&generated_scales(
                down_config.scale_or_bias_bytes() / size_of::<u16>(),
                0x3000_0005,
            ));
            let down_bias_values = bf16_values(&generated_biases(
                down_config.scale_or_bias_bytes() / size_of::<u16>(),
                0x3000_0006,
            ));
            Self {
                gate_up_weight: Buffer::from_slice(device, &gate_up_weight_values),
                gate_up_scales: bf16_buffer(device, &gate_up_scale_values),
                gate_up_biases: bf16_buffer(device, &gate_up_bias_values),
                down_weight: Buffer::from_slice(device, &down_weight_values),
                down_scales: bf16_buffer(device, &down_scale_values),
                down_biases: bf16_buffer(device, &down_bias_values),
                gate_up_weight_values,
                gate_up_scale_values,
                gate_up_bias_values,
                down_weight_values,
                down_scale_values,
                down_bias_values,
            }
        }

        fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
            QuantizedDenseMLPWeights {
                gate_up_weight: &self.gate_up_weight,
                gate_up_scales: &self.gate_up_scales,
                gate_up_biases: &self.gate_up_biases,
                down_weight: &self.down_weight,
                down_scales: &self.down_scales,
                down_biases: &self.down_biases,
            }
        }

        fn as_reference(&self) -> QuantizedDenseMLPReferenceWeights<'_> {
            QuantizedDenseMLPReferenceWeights {
                gate_up_weight: &self.gate_up_weight_values,
                gate_up_scales: &self.gate_up_scale_values,
                gate_up_biases: &self.gate_up_bias_values,
                down_weight: &self.down_weight_values,
                down_scales: &self.down_scale_values,
                down_biases: &self.down_bias_values,
            }
        }
    }

    fn bucket_test_config() -> QuantizedDenseMLPConfig {
        QuantizedDenseMLPConfig {
            hidden_dim: 64,
            intermediate_dim: 4160,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        }
    }

    fn assert_policy_preserves_topology(compute: &QuantizedDenseMLP, max_tokens: u32) {
        let boundaries = compute.topology_boundaries();
        let policy = ReplayBucketPolicy::with_topology_boundaries(max_tokens, &boundaries);
        for num_active_tokens in 1..=max_tokens {
            let num_total_tokens = policy.capacity(num_active_tokens);
            assert_eq!(
                compute.topology(num_active_tokens),
                compute.topology(num_total_tokens),
                "num_active_tokens={num_active_tokens} num_total_tokens={num_total_tokens}"
            );
        }
    }

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }

    fn create_dense_mlp_compute(config: QuantizedDenseMLPConfig) -> (Device, QuantizedDenseMLP) {
        let device = Device::system_default();
        let compute = QuantizedDenseMLP::new(&device, config);
        (device, compute)
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }

    fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<f32> {
        (0..num_tokens * hidden_dim)
            .map(|index| ((index * 13 + 5) % 31) as f32 * 0.0625 - 1.0)
            .collect()
    }

    fn bf16_values(values: &[f32]) -> Vec<f32> {
        values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
    }

    fn quantized_weight_values(len: usize) -> Vec<u8> {
        (0..len).map(|index| ((index * 13 + 17) & 0xff) as u8).collect()
    }

    fn affine_param_fixture(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| 0.001 + ((index * 3) % 7) as f32 * 0.0001)
            .collect()
    }

    fn zero_fixture(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    fn generated_scales(count: usize, random_seed: u32) -> Vec<f32> {
        generated_values(count, random_seed)
            .into_iter()
            .map(|value| 0.0005 + value.abs() * 0.001)
            .collect()
    }

    fn generated_biases(count: usize, random_seed: u32) -> Vec<f32> {
        generated_values(count, random_seed)
            .into_iter()
            .map(|value| value * 0.0002)
            .collect()
    }

    fn generated_bytes(count: usize, random_seed: u32) -> Vec<u8> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect()
    }

    fn assert_close_rel(actual: &[f32], expected: &[f32], abs_tolerance: f32, rel_tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
            assert!(
                diff <= tolerance,
                "dense MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
                 tolerance={tolerance}"
            );
        }
    }
}
