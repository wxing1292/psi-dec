use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::operators::AffineQuantizedMatmulKernel;
use crate::operators::AffineQuantizedMatmulShape;

const DENSE_MLP_ACTIVATION_SOURCE: &str = include_str!("metal/quantized_dense_mlp_activation.metal");

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

    pub fn gate_up_shape(self, shape: QuantizedDenseMLPShape) -> AffineQuantizedMatmulShape {
        self.validate();
        shape.validate();
        self.gate_up_affine_shape_unchecked(shape)
    }

    pub fn down_shape(self, shape: QuantizedDenseMLPShape) -> AffineQuantizedMatmulShape {
        self.validate();
        shape.validate();
        self.down_shape_unchecked(shape)
    }

    fn down_shape_unchecked(self, shape: QuantizedDenseMLPShape) -> AffineQuantizedMatmulShape {
        self.affine_shape_unchecked(shape, self.hidden_dim, self.intermediate_dim)
    }

    pub fn activation_shape(self, shape: QuantizedDenseMLPShape) -> QuantizedDenseMLPActivationShape {
        self.validate();
        shape.validate();
        self.activation_shape_unchecked(shape)
    }

    fn activation_shape_unchecked(self, shape: QuantizedDenseMLPShape) -> QuantizedDenseMLPActivationShape {
        let num_values = self
            .intermediate_dim
            .checked_mul(shape.num_tokens)
            .expect("dense MLP activation num_values must fit u32");
        match self.dtype {
            Dtype::Float32 => QuantizedDenseMLPActivationShape::f32(num_values),
            Dtype::Bfloat16 => QuantizedDenseMLPActivationShape::bf16(num_values),
            dtype => panic!("unsupported dense MLP activation dtype {dtype:?}"),
        }
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
        self.gate_up_affine_shape_unchecked(shape).output_bytes()
    }

    fn output_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.down_shape_unchecked(shape).output_bytes()
    }

    fn gate_up_affine_shape_unchecked(self, shape: QuantizedDenseMLPShape) -> AffineQuantizedMatmulShape {
        self.affine_shape_unchecked(shape, self.stacked_intermediate_dim(), self.hidden_dim)
    }

    fn affine_shape_unchecked(self, shape: QuantizedDenseMLPShape, n: u32, k: u32) -> AffineQuantizedMatmulShape {
        AffineQuantizedMatmulShape {
            m: shape.num_tokens.try_into().expect("dense MLP token count must fit i32"),
            n: n.try_into().expect("dense MLP output dimension must fit i32"),
            k: k.try_into().expect("dense MLP input dimension must fit i32"),
            group_size: self.group_size.try_into().expect("dense MLP group_size must fit i32"),
            bits: self.bits.try_into().expect("dense MLP bits must fit i32"),
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
        }
    }

    fn compile_shape(self, m: u32, n: u32, k: u32) -> AffineQuantizedMatmulShape {
        self.validate();
        AffineQuantizedMatmulShape {
            m: m.try_into().expect("dense MLP compile rows must fit i32"),
            n: n.try_into().expect("dense MLP compile output dimension must fit i32"),
            k: k.try_into().expect("dense MLP compile input dimension must fit i32"),
            group_size: self.group_size.try_into().expect("dense MLP group_size must fit i32"),
            bits: self.bits.try_into().expect("dense MLP bits must fit i32"),
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            affine_dtype: self.dtype,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizedDenseMLPActivationShape {
    pub num_values: u32,
    pub dtype: Dtype,
}

impl QuantizedDenseMLPActivationShape {
    pub fn f32(num_values: u32) -> Self {
        Self {
            num_values,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_values: u32) -> Self {
        Self {
            num_values,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_values > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn bytes(self) -> usize {
        (self.num_values as usize)
            .checked_mul(self.dtype.item_size())
            .expect("dense MLP activation byte length must fit usize")
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
    pub gate_up_proj: &'a Buffer,
    pub activation: &'a Buffer,
}

pub struct QuantizedDenseMLPKernels {
    config: QuantizedDenseMLPConfig,
    gate_up_proj_qmv: AffineQuantizedMatmulKernel,
    gate_up_proj_qmm: AffineQuantizedMatmulKernel,
    down_proj_qmv: AffineQuantizedMatmulKernel,
    down_proj_qmm: AffineQuantizedMatmulKernel,
    activation: QuantizedDenseMLPActivationKernel,
}

impl QuantizedDenseMLPKernels {
    pub fn new(device: &Device, config: QuantizedDenseMLPConfig) -> Self {
        config.validate();
        let gate_up_n = config.stacked_intermediate_dim();
        let gate_up_qmm_m = qmv_batch_limit(config.hidden_dim, gate_up_n);
        let down_qmm_m = qmv_batch_limit(config.intermediate_dim, config.hidden_dim);
        Self {
            config,
            gate_up_proj_qmv: AffineQuantizedMatmulKernel::new(
                device,
                config.compile_shape(1, gate_up_n, config.hidden_dim),
            ),
            gate_up_proj_qmm: AffineQuantizedMatmulKernel::new(
                device,
                config.compile_shape(gate_up_qmm_m, gate_up_n, config.hidden_dim),
            ),
            down_proj_qmv: AffineQuantizedMatmulKernel::new(
                device,
                config.compile_shape(1, config.hidden_dim, config.intermediate_dim),
            ),
            down_proj_qmm: AffineQuantizedMatmulKernel::new(
                device,
                config.compile_shape(down_qmm_m, config.hidden_dim, config.intermediate_dim),
            ),
            activation: QuantizedDenseMLPActivationKernel::new(device),
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
            kernels: self,
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
        gate_up_proj: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPGateUpInvocation<'a> {
        QuantizedDenseMLPGateUpInvocation {
            kernels: self,
            shape,
            hidden_state,
            gate_up_proj,
            weights,
        }
    }

    pub fn invoke_activation<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        gate_up_proj: &'a Buffer,
        activation: &'a Buffer,
    ) -> QuantizedDenseMLPActivationInvocation<'a> {
        QuantizedDenseMLPActivationInvocation {
            kernels: self,
            shape,
            gate_up_proj,
            activation,
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: QuantizedDenseMLPShape,
        activation: &'a Buffer,
        next_hidden_state: &'a Buffer,
        weights: QuantizedDenseMLPWeights<'a>,
    ) -> QuantizedDenseMLPDownInvocation<'a> {
        QuantizedDenseMLPDownInvocation {
            kernels: self,
            shape,
            activation,
            next_hidden_state,
            weights,
        }
    }

    fn gate_up_proj(&self, shape: QuantizedDenseMLPShape) -> &AffineQuantizedMatmulKernel {
        let threshold = qmv_batch_limit(self.config.hidden_dim, self.config.stacked_intermediate_dim());
        if shape.num_tokens >= threshold {
            &self.gate_up_proj_qmm
        } else {
            &self.gate_up_proj_qmv
        }
    }

    fn down_proj(&self, shape: QuantizedDenseMLPShape) -> &AffineQuantizedMatmulKernel {
        let threshold = qmv_batch_limit(self.config.intermediate_dim, self.config.hidden_dim);
        if shape.num_tokens >= threshold {
            &self.down_proj_qmm
        } else {
            &self.down_proj_qmv
        }
    }
}

pub struct QuantizedDenseMLPInvocation<'a> {
    kernels: &'a QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    buffers: QuantizedDenseMLPBuffers<'a>,
    scratch: QuantizedDenseMLPScratch<'a>,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels
            .invoke_gate_up(
                self.shape,
                self.buffers.hidden_state,
                self.scratch.gate_up_proj,
                self.weights,
            )
            .record(builder);
        builder.record_with_barrier_before(self.kernels.invoke_activation(
            self.shape,
            self.scratch.gate_up_proj,
            self.scratch.activation,
        ));
        builder.record_with_barrier_before(self.kernels.invoke_down(
            self.shape,
            self.scratch.activation,
            self.buffers.next_hidden_state,
            self.weights,
        ));
    }
}

pub struct QuantizedDenseMLPGateUpInvocation<'a> {
    kernels: &'a QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    hidden_state: &'a Buffer,
    gate_up_proj: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPGateUpInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels
            .gate_up_proj(self.shape)
            .invoke_with_shape(
                self.kernels.config.gate_up_affine_shape_unchecked(self.shape),
                self.gate_up_proj,
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

pub struct QuantizedDenseMLPActivationInvocation<'a> {
    kernels: &'a QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    gate_up_proj: &'a Buffer,
    activation: &'a Buffer,
}

impl Operator for QuantizedDenseMLPActivationInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels
            .activation
            .invoke(self.kernels.config, self.shape, self.gate_up_proj, self.activation)
            .record(builder);
    }
}

pub struct QuantizedDenseMLPDownInvocation<'a> {
    kernels: &'a QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    activation: &'a Buffer,
    next_hidden_state: &'a Buffer,
    weights: QuantizedDenseMLPWeights<'a>,
}

impl Operator for QuantizedDenseMLPDownInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.kernels
            .down_proj(self.shape)
            .invoke_with_shape(
                self.kernels.config.down_shape_unchecked(self.shape),
                self.next_hidden_state,
                0,
                self.activation,
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

struct QuantizedDenseMLPActivationKernel {
    f32_kernel: Kernel,
    bf16_kernel: Kernel,
}

impl QuantizedDenseMLPActivationKernel {
    fn new(device: &Device) -> Self {
        Self {
            f32_kernel: Kernel::new(device, DENSE_MLP_ACTIVATION_SOURCE, "dense_mlp_activation_f32"),
            bf16_kernel: Kernel::new(device, DENSE_MLP_ACTIVATION_SOURCE, "dense_mlp_activation_bf16"),
        }
    }

    fn invoke<'a>(
        &'a self,
        config: QuantizedDenseMLPConfig,
        shape: QuantizedDenseMLPShape,
        gate_up_proj: &'a Buffer,
        activation: &'a Buffer,
    ) -> QuantizedDenseMLPActivationRowMajorInvocation<'a> {
        QuantizedDenseMLPActivationRowMajorInvocation {
            kernel: self.kernel(config),
            config,
            shape,
            gate_up_proj,
            activation,
        }
    }

    fn kernel(&self, config: QuantizedDenseMLPConfig) -> &Kernel {
        match config.dtype {
            Dtype::Float32 => &self.f32_kernel,
            Dtype::Bfloat16 => &self.bf16_kernel,
            dtype => panic!("unsupported dense MLP activation dtype {dtype:?}"),
        }
    }
}

struct QuantizedDenseMLPActivationRowMajorInvocation<'a> {
    kernel: &'a Kernel,
    config: QuantizedDenseMLPConfig,
    shape: QuantizedDenseMLPShape,
    gate_up_proj: &'a Buffer,
    activation: &'a Buffer,
}

impl Operator for QuantizedDenseMLPActivationRowMajorInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl QuantizedDenseMLPActivationRowMajorInvocation<'_> {
    fn validate(&self) {
        self.shape.validate();
        let gate_up_output_bytes = self.config.gate_up_output_bytes(self.shape);
        let activation_bytes = self.config.activation_shape_unchecked(self.shape).bytes();
        assert!(
            self.gate_up_proj.len_bytes() >= gate_up_output_bytes,
            "dense MLP gate/up projection buffer is too small"
        );
        assert!(
            self.activation.len_bytes() >= activation_bytes,
            "dense MLP activation buffer is too small"
        );
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.gate_up_proj, 0);
        builder.set_buffer_write(1, self.activation, 0);
        builder.set_u32(2, self.shape.num_tokens);
        builder.set_u32(3, self.config.intermediate_dim);
        let num_values = self.config.activation_shape_unchecked(self.shape).num_values as usize;
        builder.dispatch_1d(num_values, ELEMENTWISE_NUM_THREADS_PER_THREADBLOCK);
    }
}

fn qmv_batch_limit(input_dim: u32, output_dim: u32) -> u32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
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
        let (device, kernels) = create_dense_mlp_kernels(config);
        let stream = Stream::new(&device);
        let gate_up_shape = config.gate_up_shape(shape);
        let down_shape = config.down_shape(shape);
        let hidden_values = hidden_fixture(shape.num_tokens as usize, config.hidden_dim as usize);
        let hidden_state = bf16_buffer(&device, &hidden_values);
        let gate_up_weight_values = quantized_weight_values(gate_up_shape.weight_bytes());
        let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
        let gate_up_scale_values = affine_param_fixture(gate_up_shape.affine_param_bytes() / size_of::<u16>());
        let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
        let gate_up_bias_values = zero_fixture(gate_up_shape.affine_param_bytes() / size_of::<u16>());
        let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
        let down_weight_values = quantized_weight_values(down_shape.weight_bytes());
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values = affine_param_fixture(down_shape.affine_param_bytes() / size_of::<u16>());
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values = zero_fixture(down_shape.affine_param_bytes() / size_of::<u16>());
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
        let replay_activation = Buffer::new_zeroed(&device, config.activation_shape(shape).bytes());
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke(
            shape,
            QuantizedDenseMLPBuffers {
                hidden_state: &hidden_state,
                next_hidden_state: &replay_output,
            },
            QuantizedDenseMLPScratch {
                gate_up_proj: &replay_gate_up,
                activation: &replay_activation,
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
            intermediate_dim: 64,
            group_size: 32,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedDenseMLPShape { num_tokens: 5 };
        let (device, kernels) = create_dense_mlp_kernels(config);
        let stream = Stream::new(&device);
        let gate_up_shape = config.gate_up_shape(shape);
        let down_shape = config.down_shape(shape);
        let hidden_values = generated_values(shape.num_tokens as usize * config.hidden_dim as usize, random_seed);
        let hidden_state = bf16_buffer(&device, &hidden_values);
        let gate_up_weight_values = generated_bytes(gate_up_shape.weight_bytes(), random_seed.wrapping_add(1));
        let gate_up_weight = Buffer::from_slice(&device, &gate_up_weight_values);
        let gate_up_scale_values = generated_scales(
            gate_up_shape.affine_param_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(2),
        );
        let gate_up_scales = bf16_buffer(&device, &gate_up_scale_values);
        let gate_up_bias_values = generated_biases(
            gate_up_shape.affine_param_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(3),
        );
        let gate_up_biases = bf16_buffer(&device, &gate_up_bias_values);
        let down_weight_values = generated_bytes(down_shape.weight_bytes(), random_seed.wrapping_add(4));
        let down_weight = Buffer::from_slice(&device, &down_weight_values);
        let down_scale_values = generated_scales(
            down_shape.affine_param_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(5),
        );
        let down_scales = bf16_buffer(&device, &down_scale_values);
        let down_bias_values = generated_biases(
            down_shape.affine_param_bytes() / size_of::<u16>(),
            random_seed.wrapping_add(6),
        );
        let down_biases = bf16_buffer(&device, &down_bias_values);

        let replay_output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let replay_gate_up = Buffer::new_zeroed(&device, config.gate_up_output_bytes(shape));
        let replay_activation = Buffer::new_zeroed(&device, config.activation_shape(shape).bytes());
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke(
            shape,
            QuantizedDenseMLPBuffers {
                hidden_state: &hidden_state,
                next_hidden_state: &replay_output,
            },
            QuantizedDenseMLPScratch {
                gate_up_proj: &replay_gate_up,
                activation: &replay_activation,
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

    fn create_dense_mlp_kernels(config: QuantizedDenseMLPConfig) -> (Device, QuantizedDenseMLPKernels) {
        let device = Device::system_default();
        let kernels = QuantizedDenseMLPKernels::new(&device, config);
        (device, kernels)
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
