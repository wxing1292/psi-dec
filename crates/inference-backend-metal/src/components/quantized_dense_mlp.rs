use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::operators::AffineQuantizedMatmul;
use crate::operators::AffineQuantizedMatmulConfig;

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
        }
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
        }
    }
}

pub struct QuantizedDenseMLPInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    buffers: QuantizedDenseMLPBuffers<'a>,
    scratch: QuantizedDenseMLPScratch<'a>,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.compute
            .invoke_gate_up(
                self.shape,
                self.buffers.hidden_state,
                self.scratch.gate_up,
                self.weights,
            )
            .record(builder);
        builder.record_with_barrier_before(self.compute.invoke_swiglu(
            self.shape,
            self.scratch.gate_up,
            self.scratch.swiglu,
        ));
        builder.record_with_barrier_before(self.compute.invoke_down(
            self.shape,
            self.scratch.swiglu,
            self.buffers.next_hidden_state,
            self.weights,
        ));
    }
}

pub struct QuantizedDenseMLPGateUpInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    hidden_state: &'a Buffer,
    gate_up: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPGateUpInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.compute
            .gate_up
            .invoke(
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
            .record(builder);
    }
}

pub struct QuantizedDenseMLPSwiGLUInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
}

impl Operator for QuantizedDenseMLPSwiGLUInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.compute
            .swiglu
            .invoke(self.compute.config, self.shape, self.gate_up, self.swiglu)
            .record(builder);
    }
}

pub struct QuantizedDenseMLPDownInvocation<'a> {
    compute: &'a QuantizedDenseMLP,
    shape: QuantizedDenseMLPShape,
    swiglu: &'a Buffer,
    next_hidden_state: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPDownInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.compute
            .down
            .invoke(
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
            .record(builder);
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
    ) -> QuantizedDenseMLPSwiGLURowMajorInvocation<'a> {
        QuantizedDenseMLPSwiGLURowMajorInvocation {
            kernel: &self.kernel,
            config,
            shape,
            gate_up,
            swiglu,
        }
    }
}

struct QuantizedDenseMLPSwiGLURowMajorInvocation<'a> {
    kernel: &'a Kernel,
    config: QuantizedDenseMLPConfig,
    shape: QuantizedDenseMLPShape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
}

impl Operator for QuantizedDenseMLPSwiGLURowMajorInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.gate_up, 0);
        builder.set_buffer_write(1, self.swiglu, 0);
        builder.set_u32(2, self.shape.num_tokens);
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

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::Stream;

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
