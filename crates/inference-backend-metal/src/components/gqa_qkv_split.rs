use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_QKV_SPLIT_SOURCE: &str = include_str!("metal/gqa_qkv_split.metal");

#[derive(Clone, Copy, Debug)]
pub struct GQAQKVSplitConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl GQAQKVSplitConfig {
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
        let _ = self.qkv_width();
    }

    pub fn num_qkv_slots(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA projection element count",
            &[shape.num_tokens as usize, self.qkv_width()],
        )
    }

    pub fn num_q_slots(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA query element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_kv_slots(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA key/value element count",
            &[
                shape.num_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn qkv_bytes(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA projection byte length",
            &[self.num_qkv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn q_bytes(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA query byte length",
            &[self.num_q_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn kv_bytes(self, shape: GQAQKVSplitShape) -> usize {
        checked_product(
            "ungated GQA key/value byte length",
            &[self.num_kv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn qkv_width(self) -> usize {
        let num_qkv_heads = self
            .num_kv_heads
            .checked_mul(2)
            .and_then(|num_kv_heads| self.num_q_heads.checked_add(num_kv_heads))
            .expect("ungated GQA fused projection head count must fit u32");
        checked_product(
            "ungated GQA fused projection width",
            &[num_qkv_heads as usize, self.head_dim as usize],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GQAQKVSplitShape {
    pub num_tokens: u32,
}

impl GQAQKVSplitShape {
    pub fn validate(self, config: GQAQKVSplitConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_u32_count_domain(config.num_qkv_slots(self), "ungated GQA projection elements");
    }
}

#[derive(Clone, Copy)]
pub struct GQAQKVSplitBuffers<'a> {
    pub qkv: &'a Buffer,
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
}

pub struct GQAQKVSplitKernel {
    config: GQAQKVSplitConfig,
    kernel: Kernel,
}

impl GQAQKVSplitKernel {
    pub fn new(device: &Device, config: GQAQKVSplitConfig) -> Self {
        config.validate();
        let source = qkv_split_source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_qkv_split_f32",
            Dtype::Bfloat16 => "gqa_qkv_split_bf16",
            dtype => panic!("unsupported ungated GQA projection split dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: GQAQKVSplitShape, buffers: GQAQKVSplitBuffers<'a>) -> GQAQKVSplitInvocation<'a> {
        GQAQKVSplitInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn qkv_split_source(config: GQAQKVSplitConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint num_kv_heads = {}u;\nconstant uint \
         head_dim = {}u;",
        config.num_q_heads, config.num_kv_heads, config.head_dim,
    );
    GQA_QKV_SPLIT_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQAQKVSplitInvocation<'a> {
    config: GQAQKVSplitConfig,
    kernel: &'a Kernel,
    shape: GQAQKVSplitShape,
    buffers: GQAQKVSplitBuffers<'a>,
}

impl Operator for GQAQKVSplitInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.qkv, 0);
        builder.set_buffer_write(1, self.buffers.q, 0);
        builder.set_buffer_write(2, self.buffers.k, 0);
        builder.set_buffer_write(3, self.buffers.v, 0);
        builder.set_u32(4, shape.num_tokens);
        builder.dispatch_1d(self.config.num_qkv_slots(shape), 256);
    }
}

impl GQAQKVSplitInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.qkv.len_bytes() >= self.config.qkv_bytes(self.shape));
        assert!(self.buffers.q.len_bytes() >= self.config.q_bytes(self.shape));
        assert!(self.buffers.k.len_bytes() >= self.config.kv_bytes(self.shape));
        assert!(self.buffers.v.len_bytes() >= self.config.kv_bytes(self.shape));
    }
}

#[cfg(test)]
mod tests {
    use super::GQAQKVSplitBuffers;
    use super::GQAQKVSplitConfig;
    use super::GQAQKVSplitKernel;
    use super::GQAQKVSplitShape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::Stream;

    #[test]
    fn test_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = GQAQKVSplitConfig::f32(2, 1, 2);
        let shape = GQAQKVSplitShape { num_tokens: 2 };
        let qkv = Buffer::from_slice(&device, &(0..16).map(|value| value as f32).collect::<Vec<_>>());
        let q = Buffer::new_zeroed_elements(&device, config.num_q_slots(shape), Dtype::Float32);
        let k = Buffer::new_zeroed_elements(&device, config.num_kv_slots(shape), Dtype::Float32);
        let v = Buffer::new_zeroed_elements(&device, config.num_kv_slots(shape), Dtype::Float32);
        let kernel = GQAQKVSplitKernel::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            GQAQKVSplitBuffers {
                qkv: &qkv,
                q: &q,
                k: &k,
                v: &v,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            q.read_typed::<f32>(0, 8),
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(k.read_typed::<f32>(0, 4), vec![4.0, 5.0, 12.0, 13.0]);
        assert_eq!(v.read_typed::<f32>(0, 4), vec![6.0, 7.0, 14.0, 15.0]);
    }

    #[test]
    #[should_panic(expected = "ungated GQA projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        GQAQKVSplitShape { num_tokens: 1 << 30 }.validate(GQAQKVSplitConfig::f32(2, 1, 1));
    }
}
