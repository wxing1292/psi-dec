use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_activation_gate.metal");

#[derive(Clone, Copy, Debug, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        config.validate();
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        let source = source(constants);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_activation_gate_f32",
            Dtype::Bfloat16 => "gqa_activation_gate_bf16",
            dtype => panic!("unsupported GQA activation gate dtype {dtype:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>, num_active_tokens: ReplayU32) -> Invocation<'a> {
        Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        }
    }
}

fn source(constants: KernelConstants) -> String {
    let config = constants.config;
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
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
        recorder.dispatch_1d(
            self.constants.config.num_values(shape),
            self.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        assert!(self.buffers.attention_output.len_bytes() >= config.bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= config.bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= config.bytes(self.shape));
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
    use super::Buffers;
    use super::Compute;
    use super::Config;
    use super::Shape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;
    use crate::test_support::ReplayTestCache;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa_activation_gate.num_active_tokens");

    #[test]
    fn test_replay_matches_reference_across_active_counts() {
        const NUM_TOTAL_TOKENS: u32 = 8;

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::f32(2, 2);
        let shape = Shape {
            num_total_tokens: NUM_TOTAL_TOKENS,
        };
        let values_per_token = config.num_q_heads as usize * config.head_dim as usize;
        let attention_values = (0..NUM_TOTAL_TOKENS as usize * values_per_token)
            .map(|index| index as f32 * 0.03125 - 0.5)
            .collect::<Vec<_>>();
        let gate_values = (0..NUM_TOTAL_TOKENS as usize * values_per_token)
            .map(|index| index as f32 * -0.0625 + 0.75)
            .collect::<Vec<_>>();
        let attention = Buffer::from_slice(&device, &attention_values);
        let gates = Buffer::from_slice(&device, &gate_values);
        let output = Buffer::new_zeroed_elements(&device, config.num_values(shape), Dtype::Float32);
        let compute = Compute::new(&device, config);
        let mut cache = ReplayTestCache::new();
        let (_, cache_hit) = cache.record(shape.num_total_tokens, || {
            let mut builder = stream.create_replay_program();
            builder.record(compute.invoke(
                shape,
                Buffers {
                    attention_output: &attention,
                    g: &gates,
                    output: &output,
                },
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ));
            builder.build()
        });
        assert!(!cache_hit);

        for num_active_tokens in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            let (replay, cache_hit) = cache.record(shape.num_total_tokens, || unreachable!());
            assert!(cache_hit);
            stream
                .submit_replay_with_arguments(
                    replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
                )
                .wait();
            let num_active_values = num_active_tokens * values_per_token;
            let expected = attention_values[..num_active_values]
                .iter()
                .zip(&gate_values[..num_active_values])
                .map(|(&attention, &gate)| attention / (1.0 + (-gate).exp()))
                .collect::<Vec<_>>();
            let actual = output.read_typed::<f32>(0, num_active_values);
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 2.0e-5);
            }
        }
    }

    #[test]
    #[should_panic(expected = "GQA activation/gate exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::bf16(2, 2));
    }
}
