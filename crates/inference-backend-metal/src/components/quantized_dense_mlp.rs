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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DenseMLPSwiGLUThreadBlockSpecialization {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DenseMLPSwiGLUKernelSpecialization {
    io_dtype: Dtype,
    thread_block: DenseMLPSwiGLUThreadBlockSpecialization,
}

impl DenseMLPSwiGLUKernelSpecialization {
    fn new(io_dtype: Dtype) -> Self {
        let specialization = Self {
            io_dtype,
            thread_block: DenseMLPSwiGLUThreadBlockSpecialization { required_threads: 256 },
        };
        specialization.validate();
        specialization
    }

    fn validate(self) {
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(self.thread_block.required_threads > 0);
    }
}

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
            .checked_mul(shape.num_total_tokens)
            .expect("dense MLP swiglu num_values must fit u32")
    }

    pub fn input_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.validate();
        shape.validate();
        self.input_bytes_unchecked(shape)
    }

    fn input_bytes_unchecked(self, shape: QuantizedDenseMLPShape) -> usize {
        (shape.num_total_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|count| count.checked_mul(self.dtype.item_size()))
            .expect("dense MLP input byte length must fit usize")
    }

    fn gate_up_output_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.gate_up_config().output_bytes(
            shape
                .num_total_tokens
                .try_into()
                .expect("dense MLP token count must fit i32"),
        )
    }

    fn output_bytes(self, shape: QuantizedDenseMLPShape) -> usize {
        self.down_config().output_bytes(
            shape
                .num_total_tokens
                .try_into()
                .expect("dense MLP token count must fit i32"),
        )
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
    pub num_total_tokens: u32,
}

impl QuantizedDenseMLPShape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
        i32::try_from(self.num_total_tokens).expect("dense MLP token count must fit i32");
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
            gate_up_affine: self.gate_up.topology(shape.num_total_tokens),
            down_affine: self.down.topology(shape.num_total_tokens),
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
    let shape = QuantizedDenseMLPShape { num_total_tokens };
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        QuantizedDenseMLPGateUpInvocation {
            compute: self.compute,
            shape: self.shape,
            hidden_state: self.buffers.hidden_state,
            gate_up: self.scratch.gate_up,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        }
        .record(recorder);
        recorder.record_with_barrier_before(QuantizedDenseMLPSwiGLUInvocation {
            compute: self.compute,
            shape: self.shape,
            gate_up: self.scratch.gate_up,
            swiglu: self.scratch.swiglu,
            num_active_tokens_key: self.num_active_tokens_key,
        });
        recorder.record_with_barrier_before(QuantizedDenseMLPDownInvocation {
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = match self.num_active_tokens_key {
            Some(key) => {
                self.compute.gate_up.invoke_bucketed(
                    self.shape.num_total_tokens,
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
                        .num_total_tokens
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
        invocation.record(recorder);
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.compute
            .swiglu
            .invoke(
                self.compute.config,
                self.shape,
                self.gate_up,
                self.swiglu,
                self.num_active_tokens_key,
            )
            .record(recorder);
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = match self.num_active_tokens_key {
            Some(key) => {
                self.compute.down.invoke_bucketed(
                    self.shape.num_total_tokens,
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
                        .num_total_tokens
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
        invocation.record(recorder);
    }
}

struct QuantizedDenseMLPSwiGLUKernel {
    specialization: DenseMLPSwiGLUKernelSpecialization,
    kernel: Kernel,
}

impl QuantizedDenseMLPSwiGLUKernel {
    fn new(device: &Device, dtype: Dtype) -> Self {
        let specialization = DenseMLPSwiGLUKernelSpecialization::new(dtype);
        let function_name = match specialization.io_dtype {
            Dtype::Float32 => "dense_mlp_swiglu_f32",
            Dtype::Bfloat16 => "dense_mlp_swiglu_bf16",
            dtype => panic!("unsupported dense MLP swiglu dtype {dtype:?}"),
        };
        Self {
            specialization,
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
            kernel: self,
            config,
            shape,
            gate_up,
            swiglu,
            num_active_tokens_key,
        }
    }
}

struct QuantizedDenseMLPSwiGLURowMajorInvocation<'a> {
    kernel: &'a QuantizedDenseMLPSwiGLUKernel,
    config: QuantizedDenseMLPConfig,
    shape: QuantizedDenseMLPShape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for QuantizedDenseMLPSwiGLURowMajorInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.gate_up, 0);
        recorder.set_buffer_write(1, self.swiglu, 0);
        match self.num_active_tokens_key {
            Some(key) => recorder.bind_u32(2, key, 1, self.shape.num_total_tokens),
            None => recorder.set_u32(2, self.shape.num_total_tokens),
        }
        recorder.set_u32(3, self.config.intermediate_dim);
        let num_values = self.config.swiglu_num_values_unchecked(self.shape) as usize;
        recorder.dispatch_1d(
            num_values,
            self.kernel.specialization.thread_block.required_threads as usize,
        );
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
#[path = "quantized_dense_mlp_test.rs"]
mod tests;
