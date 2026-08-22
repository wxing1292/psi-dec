use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::operators::affine_quantized;

const DENSE_MLP_SWIGLU_SOURCE: &str = include_str!("metal/quantized_dense_mlp_swiglu.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SwiGLUThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SwiGLUKernelConstants {
    io_dtype: Dtype,
    thread_block: SwiGLUThreadBlockConstants,
}

impl SwiGLUKernelConstants {
    fn new(io_dtype: Dtype) -> Self {
        let constants = Self {
            io_dtype,
            thread_block: SwiGLUThreadBlockConstants { required_threads: 256 },
        };
        constants.validate();
        constants
    }

    fn validate(self) {
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(self.thread_block.required_threads > 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub gate_up_group_size: u32,
    pub gate_up_bits: u32,
    pub gate_up_scale_bias_dtype: Dtype,
    pub down_group_size: u32,
    pub down_bits: u32,
    pub down_scale_bias_dtype: Dtype,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        self.stacked_intermediate_dim();
        assert!(matches!(self.gate_up_group_size, 32 | 64 | 128));
        assert!(matches!(self.gate_up_bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(
            self.gate_up_scale_bias_dtype,
            Dtype::Float32 | Dtype::Bfloat16
        ));
        assert!(matches!(self.down_group_size, 32 | 64 | 128));
        assert!(matches!(self.down_bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.down_scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        i32::try_from(self.hidden_dim).expect("dense MLP hidden_dim must fit i32");
        i32::try_from(self.intermediate_dim).expect("dense MLP intermediate_dim must fit i32");
        i32::try_from(self.stacked_intermediate_dim()).expect("dense MLP stacked intermediate_dim must fit i32");
        i32::try_from(self.gate_up_group_size).expect("dense MLP gate/up group_size must fit i32");
        i32::try_from(self.gate_up_bits).expect("dense MLP gate/up bits must fit i32");
        i32::try_from(self.down_group_size).expect("dense MLP down group_size must fit i32");
        i32::try_from(self.down_bits).expect("dense MLP down bits must fit i32");
    }

    pub fn gate_up_config(self) -> affine_quantized::Config {
        self.validate();
        self.affine_config_unchecked(
            self.stacked_intermediate_dim(),
            self.hidden_dim,
            self.gate_up_group_size,
            self.gate_up_bits,
            self.gate_up_scale_bias_dtype,
        )
    }

    pub fn down_config(self) -> affine_quantized::Config {
        self.validate();
        self.affine_config_unchecked(
            self.hidden_dim,
            self.intermediate_dim,
            self.down_group_size,
            self.down_bits,
            self.down_scale_bias_dtype,
        )
    }

    pub fn swiglu_bytes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (self.swiglu_num_values_unchecked(shape) as usize)
            .checked_mul(self.dtype.item_size())
            .expect("dense MLP swiglu byte length must fit usize")
    }

    fn swiglu_num_values_unchecked(self, shape: Shape) -> u32 {
        self.intermediate_dim
            .checked_mul(shape.num_total_tokens)
            .expect("dense MLP swiglu num_values must fit u32")
    }

    pub fn input_bytes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        self.input_bytes_unchecked(shape)
    }

    fn input_bytes_unchecked(self, shape: Shape) -> usize {
        (shape.num_total_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|count| count.checked_mul(self.dtype.item_size()))
            .expect("dense MLP input byte length must fit usize")
    }

    fn gate_up_output_bytes(self, shape: Shape) -> usize {
        self.gate_up_config().output_bytes(
            shape
                .num_total_tokens
                .try_into()
                .expect("dense MLP token count must fit i32"),
        )
    }

    fn output_bytes(self, shape: Shape) -> usize {
        self.down_config().output_bytes(
            shape
                .num_total_tokens
                .try_into()
                .expect("dense MLP token count must fit i32"),
        )
    }

    fn affine_config_unchecked(
        self,
        n: u32,
        k: u32,
        group_size: u32,
        bits: u32,
        scale_bias_dtype: Dtype,
    ) -> affine_quantized::Config {
        affine_quantized::Config {
            n: n.try_into().expect("dense MLP output dimension must fit i32"),
            k: k.try_into().expect("dense MLP input dimension must fit i32"),
            group_size: group_size.try_into().expect("dense MLP group_size must fit i32"),
            bits: bits.try_into().expect("dense MLP bits must fit i32"),
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            scale_bias_dtype,
        }
    }

    fn stacked_intermediate_dim(self) -> u32 {
        self.intermediate_dim
            .checked_mul(2)
            .expect("dense MLP stacked gate/up dim must fit u32")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
        i32::try_from(self.num_total_tokens).expect("dense MLP token count must fit i32");
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayTopology {
    pub gate_up_affine: affine_quantized::KernelKind,
    pub down_affine: affine_quantized::KernelKind,
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Weights<'a> {
    pub gate_up_weight: &'a Buffer,
    pub gate_up_scales: &'a Buffer,
    pub gate_up_biases: &'a Buffer,
    pub down_weight: &'a Buffer,
    pub down_scales: &'a Buffer,
    pub down_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Scratch<'a> {
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
pub struct Compute {
    config: Config,
    gate_up: affine_quantized::Matmul,
    down: affine_quantized::Matmul,
    swiglu: SwiGLUKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            gate_up: affine_quantized::Matmul::new(device, config.gate_up_config()),
            down: affine_quantized::Matmul::new(device, config.down_config()),
            swiglu: SwiGLUKernel::new(device, config.dtype),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: Buffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> Invocation<'a> {
        shape.validate();
        Invocation {
            compute: self,
            shape,
            buffers,
            scratch,
            weights,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn topology(&self, num_total_tokens: u32) -> ReplayTopology {
        let shape = capacity_shape(num_total_tokens);
        ReplayTopology {
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
        shape: Shape,
        num_active_tokens: ReplayU32,
        hidden_state: &'a Buffer,
        gate_up: &'a Buffer,
        weights: Weights<'a>,
    ) -> GateUpInvocation<'a> {
        shape.validate();
        GateUpInvocation {
            compute: self,
            shape,
            hidden_state,
            gate_up,
            weights,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn invoke_swiglu<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        gate_up: &'a Buffer,
        swiglu: &'a Buffer,
    ) -> SwiGLUInvocation<'a> {
        shape.validate();
        SwiGLUInvocation {
            compute: self,
            shape,
            gate_up,
            swiglu,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        swiglu: &'a Buffer,
        next_hidden_state: &'a Buffer,
        weights: Weights<'a>,
    ) -> DownInvocation<'a> {
        shape.validate();
        DownInvocation {
            compute: self,
            shape,
            swiglu,
            next_hidden_state,
            weights,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }
}

fn capacity_shape(num_total_tokens: u32) -> Shape {
    let shape = Shape { num_total_tokens };
    shape.validate();
    shape
}

fn active_key(num_total_tokens: u32, num_active_tokens: ReplayU32) -> Option<ReplayParameterKey> {
    match num_active_tokens {
        ReplayU32::Fixed(num_active_tokens) => {
            assert_eq!(num_active_tokens, num_total_tokens);
            None
        },
        ReplayU32::Parameter(key) => Some(key),
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    scratch: Scratch<'a>,
    weights: Weights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        GateUpInvocation {
            compute: self.compute,
            shape: self.shape,
            hidden_state: self.buffers.hidden_state,
            gate_up: self.scratch.gate_up,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        }
        .record(recorder);
        recorder.record_with_barrier_before(SwiGLUInvocation {
            compute: self.compute,
            shape: self.shape,
            gate_up: self.scratch.gate_up,
            swiglu: self.scratch.swiglu,
            num_active_tokens_key: self.num_active_tokens_key,
        });
        recorder.record_with_barrier_before(DownInvocation {
            compute: self.compute,
            shape: self.shape,
            swiglu: self.scratch.swiglu,
            next_hidden_state: self.buffers.next_hidden_state,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        });
    }
}

pub struct GateUpInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    hidden_state: &'a Buffer,
    gate_up: &'a Buffer,
    weights: Weights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for GateUpInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_active_tokens = match self.num_active_tokens_key {
            Some(key) => ReplayU32::Parameter(key),
            None => ReplayU32::Fixed(self.shape.num_total_tokens),
        };
        let invocation = self.compute.gate_up.invoke(
            self.shape.num_total_tokens,
            num_active_tokens,
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
        );
        invocation.record(recorder);
    }
}

pub struct SwiGLUInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for SwiGLUInvocation<'_> {
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

pub struct DownInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    swiglu: &'a Buffer,
    next_hidden_state: &'a Buffer,
    weights: Weights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for DownInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_active_tokens = match self.num_active_tokens_key {
            Some(key) => ReplayU32::Parameter(key),
            None => ReplayU32::Fixed(self.shape.num_total_tokens),
        };
        let invocation = self.compute.down.invoke(
            self.shape.num_total_tokens,
            num_active_tokens,
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
        );
        invocation.record(recorder);
    }
}

struct SwiGLUKernel {
    constants: SwiGLUKernelConstants,
    kernel: CompiledKernel,
}

impl SwiGLUKernel {
    fn new(device: &Device, dtype: Dtype) -> Self {
        let constants = SwiGLUKernelConstants::new(dtype);
        let function_name = match constants.io_dtype {
            Dtype::Float32 => "dense_mlp_swiglu_f32",
            Dtype::Bfloat16 => "dense_mlp_swiglu_bf16",
            dtype => panic!("unsupported dense MLP swiglu dtype {dtype:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, DENSE_MLP_SWIGLU_SOURCE, function_name),
        }
    }

    fn invoke<'a>(
        &'a self,
        config: Config,
        shape: Shape,
        gate_up: &'a Buffer,
        swiglu: &'a Buffer,
        num_active_tokens_key: Option<ReplayParameterKey>,
    ) -> SwiGLURowMajorInvocation<'a> {
        SwiGLURowMajorInvocation {
            kernel: self,
            config,
            shape,
            gate_up,
            swiglu,
            num_active_tokens_key,
        }
    }
}

struct SwiGLURowMajorInvocation<'a> {
    kernel: &'a SwiGLUKernel,
    config: Config,
    shape: Shape,
    gate_up: &'a Buffer,
    swiglu: &'a Buffer,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for SwiGLURowMajorInvocation<'_> {
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
        recorder.dispatch_1d(num_values, self.kernel.constants.thread_block.required_threads as usize);
    }
}

impl SwiGLURowMajorInvocation<'_> {
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
#[path = "dense_mlp_test.rs"]
mod tests;
