use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_QGKV_SPLIT_SOURCE: &str = include_str!("metal/gqa_qgkv_split.metal");

#[derive(Clone, Copy, Debug)]
pub struct GQAQGKVSplitConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl GQAQGKVSplitConfig {
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

    pub fn num_qgkv_slots(self, shape: GQAQGKVSplitShape) -> usize {
        checked_product(
            "GQA projection element count",
            &[shape.num_tokens as usize, self.qgkv_width()],
        )
    }

    pub fn num_q_slots(self, shape: GQAQGKVSplitShape) -> usize {
        checked_product(
            "GQA query element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_kv_slots(self, shape: GQAQGKVSplitShape) -> usize {
        checked_product(
            "GQA key/value element count",
            &[
                shape.num_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn qgkv_bytes(self, shape: GQAQGKVSplitShape) -> usize {
        checked_product(
            "GQA projection byte length",
            &[self.num_qgkv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn q_bytes(self, shape: GQAQGKVSplitShape) -> usize {
        checked_product(
            "GQA query byte length",
            &[self.num_q_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn kv_bytes(self, shape: GQAQGKVSplitShape) -> usize {
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
pub struct GQAQGKVSplitShape {
    pub num_tokens: u32,
}

impl GQAQGKVSplitShape {
    pub fn validate(self, config: GQAQGKVSplitConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_u32_count_domain(config.num_qgkv_slots(self), "GQA projection elements");
    }
}

#[cfg(test)]
mod tests {
    use super::GQAQGKVSplitConfig;
    use super::GQAQGKVSplitShape;

    #[test]
    #[should_panic(expected = "GQA projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        GQAQGKVSplitShape { num_tokens: 1 << 30 }.validate(GQAQGKVSplitConfig::f32(1, 1, 1));
    }
}

#[derive(Clone, Copy)]
pub struct GQAQGKVSplitBuffers<'a> {
    pub qgkv: &'a Buffer,
    pub q: &'a Buffer,
    pub g: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
}

pub struct GQAQGKVSplitKernel {
    config: GQAQGKVSplitConfig,
    kernel: Kernel,
}

impl GQAQGKVSplitKernel {
    pub fn new(device: &Device, config: GQAQGKVSplitConfig) -> Self {
        config.validate();
        let source = qgkv_split_source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_qgkv_split_f32",
            Dtype::Bfloat16 => "gqa_qgkv_split_bf16",
            dtype => panic!("unsupported GQA projection split dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQAQGKVSplitShape,
        buffers: GQAQGKVSplitBuffers<'a>,
    ) -> GQAQGKVSplitInvocation<'a> {
        GQAQGKVSplitInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn qgkv_split_source(config: GQAQGKVSplitConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint num_kv_heads = {}u;\nconstant uint \
         head_dim = {}u;",
        config.num_q_heads, config.num_kv_heads, config.head_dim,
    );
    GQA_QGKV_SPLIT_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQAQGKVSplitInvocation<'a> {
    config: GQAQGKVSplitConfig,
    kernel: &'a Kernel,
    shape: GQAQGKVSplitShape,
    buffers: GQAQGKVSplitBuffers<'a>,
}

impl Operator for GQAQGKVSplitInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.qgkv, 0);
        builder.set_buffer_write(1, self.buffers.q, 0);
        builder.set_buffer_write(2, self.buffers.g, 0);
        builder.set_buffer_write(3, self.buffers.k, 0);
        builder.set_buffer_write(4, self.buffers.v, 0);
        builder.set_u32(5, shape.num_tokens);
        builder.dispatch_1d(self.config.num_qgkv_slots(shape), 256);
    }
}

impl GQAQGKVSplitInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.qgkv.len_bytes() >= self.config.qgkv_bytes(self.shape));
        assert!(self.buffers.q.len_bytes() >= self.config.q_bytes(self.shape));
        assert!(self.buffers.g.len_bytes() >= self.config.q_bytes(self.shape));
        assert!(self.buffers.k.len_bytes() >= self.config.kv_bytes(self.shape));
        assert!(self.buffers.v.len_bytes() >= self.config.kv_bytes(self.shape));
    }
}
