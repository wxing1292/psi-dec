use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::operators::affine_quantized;

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
        assert!(
            self.intermediate_dim <= i32::MAX as u32 / 2,
            "dense MLP stacked intermediate_dim must fit i32"
        );
        assert_eq!(self.hidden_dim % self.gate_up_group_size, 0);
        assert_eq!(self.intermediate_dim % self.down_group_size, 0);
    }

    pub fn gate_up_config(self) -> affine_quantized::Config {
        self.validate();
        self.affine_config_unchecked(
            self.intermediate_dim * 2,
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
        (shape.num_total_tokens as usize)
            .checked_mul(self.intermediate_dim as usize)
            .and_then(|count| count.checked_mul(self.dtype.item_size()))
            .expect("dense MLP swiglu byte length must fit usize")
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
            n: n as i32,
            k: k as i32,
            group_size: group_size as i32,
            bits: bits as i32,
            input_dtype: self.dtype,
            output_dtype: self.dtype,
            scale_bias_dtype,
        }
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
    pub gate_up_swiglu_affine: affine_quantized::KernelKind,
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
    pub swiglu: &'a Buffer,
}

/// Records fused gate/up/SwiGLU, then down projection.
/// The only intermediate tensor is `scratch.swiglu [T, I]`.
pub struct Compute {
    gate_up_swiglu: affine_quantized::Matmul,
    down: affine_quantized::Matmul,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            gate_up_swiglu: affine_quantized::Matmul::new_gate_up_swiglu(device, config.gate_up_config()),
            down: affine_quantized::Matmul::new(device, config.down_config()),
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
            gate_up_swiglu_affine: self.gate_up_swiglu.topology(shape.num_total_tokens),
            down_affine: self.down.topology(shape.num_total_tokens),
        }
    }

    pub fn topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.gate_up_swiglu.topology_boundaries().into_vec();
        boundaries.extend(self.down.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    pub fn invoke_gate_up_swiglu<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        hidden_state: &'a Buffer,
        swiglu: &'a Buffer,
        weights: Weights<'a>,
    ) -> GateUpSwiGLUInvocation<'a> {
        shape.validate();
        GateUpSwiGLUInvocation {
            compute: self,
            shape,
            hidden_state,
            swiglu,
            weights,
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
        GateUpSwiGLUInvocation {
            compute: self.compute,
            shape: self.shape,
            hidden_state: self.buffers.hidden_state,
            swiglu: self.scratch.swiglu,
            weights: self.weights,
            num_active_tokens_key: self.num_active_tokens_key,
        }
        .record(recorder);
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

pub struct GateUpSwiGLUInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    hidden_state: &'a Buffer,
    swiglu: &'a Buffer,
    weights: Weights<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for GateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_active_tokens = match self.num_active_tokens_key {
            Some(key) => ReplayU32::Parameter(key),
            None => ReplayU32::Fixed(self.shape.num_total_tokens),
        };
        let invocation = self.compute.gate_up_swiglu.invoke(
            self.shape.num_total_tokens,
            num_active_tokens,
            self.swiglu,
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

#[cfg(test)]
#[path = "dense_mlp_test.rs"]
mod tests;
