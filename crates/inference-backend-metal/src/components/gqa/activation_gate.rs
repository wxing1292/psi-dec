use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_activation_gate.metal");

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_q_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn f32(num_q_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            head_dim,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_q_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            head_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.head_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_values(self, shape: Shape) -> usize {
        checked_product(
            "GQA activation/gate element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn bytes(self, shape: Shape) -> usize {
        self.num_values(shape)
            .checked_mul(self.dtype.item_size())
            .expect("GQA activation/gate byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert_u32_count_domain(config.num_values(self), "GQA activation/gate");
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub attention_output: &'a Buffer,
    pub g: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    config: Config,
    kernel: Kernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let source = source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_activation_gate_f32",
            Dtype::Bfloat16 => "gqa_activation_gate_bf16",
            dtype => panic!("unsupported GQA activation gate dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        Invocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens: ReplayU32::Fixed(shape.num_total_tokens),
        }
    }

    pub fn invoke_bucketed<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_tokens: ReplayU32,
    ) -> Invocation<'a> {
        Invocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        }
    }
}

fn source(config: Config) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct Invocation<'a> {
    config: Config,
    kernel: &'a Kernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.attention_output, 0);
        recorder.set_buffer_read(1, self.buffers.g, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            3,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA activation-gate active token count",
        );
        recorder.dispatch_1d(self.config.num_values(shape), 256);
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.attention_output.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::Shape;

    #[test]
    #[should_panic(expected = "GQA activation/gate exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::bf16(2, 2));
    }
}
