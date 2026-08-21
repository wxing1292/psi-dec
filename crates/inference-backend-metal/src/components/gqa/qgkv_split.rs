use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_qgkv_split.metal");

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
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn f32(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert!(self.head_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        let _ = self.qgkv_width();
    }

    pub fn num_qgkv_slots(self, shape: Shape) -> usize {
        checked_product(
            "GQA projection element count",
            &[shape.num_total_tokens as usize, self.qgkv_width()],
        )
    }

    pub fn num_q_slots(self, shape: Shape) -> usize {
        checked_product(
            "GQA query element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_kv_slots(self, shape: Shape) -> usize {
        checked_product(
            "GQA key/value element count",
            &[
                shape.num_total_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn qgkv_bytes(self, shape: Shape) -> usize {
        checked_product(
            "GQA projection byte length",
            &[self.num_qgkv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn q_bytes(self, shape: Shape) -> usize {
        checked_product(
            "GQA query byte length",
            &[self.num_q_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn kv_bytes(self, shape: Shape) -> usize {
        checked_product(
            "GQA key/value byte length",
            &[self.num_kv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn qgkv_width(self) -> usize {
        let num_qgkv_heads = self
            .num_q_heads
            .checked_mul(2)
            .and_then(|num_q_heads| {
                self.num_kv_heads
                    .checked_mul(2)
                    .and_then(|num_kv_heads| num_q_heads.checked_add(num_kv_heads))
            })
            .expect("GQA fused projection head count must fit u32");
        checked_product(
            "GQA fused projection width",
            &[num_qgkv_heads as usize, self.head_dim as usize],
        )
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
        assert_u32_count_domain(config.num_qgkv_slots(self), "GQA projection elements");
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::KernelConstants;
    use super::Shape;
    use super::ThreadBlockConstants;

    #[test]
    fn test_constants_have_explicit_thread_block_scope() {
        let config = Config::bf16(2, 1, 128);
        assert_eq!(
            KernelConstants::current(config),
            KernelConstants {
                config,
                thread_block: ThreadBlockConstants { required_threads: 256 },
            }
        );
    }

    #[test]
    #[should_panic(expected = "GQA projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::f32(1, 1, 1));
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub qgkv: &'a Buffer,
    pub q: &'a Buffer,
    pub g: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        let source = qgkv_split_source(constants);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_qgkv_split_f32",
            Dtype::Bfloat16 => "gqa_qgkv_split_bf16",
            dtype => panic!("unsupported GQA projection split dtype {dtype:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        Invocation {
            constants: self.constants,
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
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        }
    }
}

fn qgkv_split_source(constants: KernelConstants) -> String {
    let config = constants.config;
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint num_kv_heads = {}u;\nconstant uint \
         head_dim = {}u;",
        config.num_q_heads, config.num_kv_heads, config.head_dim,
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
        recorder.set_buffer_read(0, self.buffers.qgkv, 0);
        recorder.set_buffer_write(1, self.buffers.q, 0);
        recorder.set_buffer_write(2, self.buffers.g, 0);
        recorder.set_buffer_write(3, self.buffers.k, 0);
        recorder.set_buffer_write(4, self.buffers.v, 0);
        set_replay_u32(
            recorder,
            5,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA projection-split active token count",
        );
        recorder.dispatch_1d(
            self.constants.config.num_qgkv_slots(shape),
            self.constants.thread_block.required_threads as usize,
        );
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

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        assert!(self.buffers.qgkv.len_bytes() >= config.qgkv_bytes(self.shape));
        assert!(self.buffers.q.len_bytes() >= config.q_bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= config.q_bytes(self.shape));
        assert!(self.buffers.k.len_bytes() >= config.kv_bytes(self.shape));
        assert!(self.buffers.v.len_bytes() >= config.kv_bytes(self.shape));
    }
}
