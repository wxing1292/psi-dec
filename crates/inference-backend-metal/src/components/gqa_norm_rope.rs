use super::assert_u32_count_domain;
use super::assert_u32_index_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_NORM_ROPE_SOURCE: &str = include_str!("metal/gqa_norm_rope.metal");

const NUM_THREADS_PER_THREADBLOCK: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQANormRopeConfig {
    pub num_heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub dtype: Dtype,
}

impl GQANormRopeConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn f32(num_heads: u32, head_dim: u32, rope_dim: u32, eps: f32, rope_theta: f32, rope_scale: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            rope_scale,
            dtype: Dtype::Float32,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bf16(num_heads: u32, head_dim: u32, rope_dim: u32, eps: f32, rope_theta: f32, rope_scale: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            rope_scale,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.rope_dim > 0);
        assert!(self.rope_dim <= self.head_dim);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.eps > 0.0);
        assert!(self.rope_theta > 0.0);
        assert!(self.rope_scale > 0.0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_slots(self, shape: GQANormRopeShape) -> usize {
        checked_product(
            "GQA norm/RoPE element count",
            &[
                shape.num_tokens as usize,
                self.num_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn bytes(self, shape: GQANormRopeShape) -> usize {
        checked_product(
            "GQA norm/RoPE byte length",
            &[self.num_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn norm_weight_bytes(self) -> usize {
        checked_product(
            "GQA norm/RoPE weight byte length",
            &[self.head_dim as usize, size_of::<f32>()],
        )
    }

    pub fn flat_token_indices_bytes(self, shape: GQANormRopeShape) -> usize {
        checked_product(
            "GQA norm/RoPE token-index byte length",
            &[shape.num_tokens as usize, size_of::<u32>()],
        )
    }

    fn num_token_heads(self, shape: GQANormRopeShape) -> usize {
        checked_product(
            "GQA norm/RoPE token-head row count",
            &[shape.num_tokens as usize, self.num_heads as usize],
        )
    }

    fn num_threads(self, shape: GQANormRopeShape) -> usize {
        checked_product(
            "GQA norm/RoPE thread count",
            &[self.num_token_heads(shape), NUM_THREADS_PER_THREADBLOCK as usize],
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQANormRopeShape {
    pub num_tokens: u32,
}

impl GQANormRopeShape {
    pub fn validate(self, config: GQANormRopeConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_u32_count_domain(config.num_token_heads(self), "GQA norm/RoPE token-head rows");
        assert_u32_index_domain(config.num_slots(self), "GQA norm/RoPE elements");
    }
}

#[cfg(test)]
mod tests {
    use super::GQANormRopeConfig;
    use super::GQANormRopeShape;

    #[test]
    #[should_panic(expected = "GQA norm/RoPE elements exceeds the shader u32 element-index domain")]
    fn test_shape_rejects_shader_index_overflow() {
        GQANormRopeShape { num_tokens: 1 << 30 }.validate(GQANormRopeConfig::bf16(2, 4, 4, 1e-6, 1_000_000.0, 1.0));
    }
}

#[derive(Clone, Copy)]
pub struct GQANormRopeBuffers<'a> {
    pub input: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct GQANormRopeKernel {
    config: GQANormRopeConfig,
    kernel: Kernel,
}

impl GQANormRopeKernel {
    pub fn new(device: &Device, config: GQANormRopeConfig) -> Self {
        config.validate();
        let source = norm_rope_source(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_norm_rope_f32",
            Dtype::Bfloat16 => "gqa_norm_rope_bf16",
            dtype => panic!("unsupported GQA norm/RoPE dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: GQANormRopeShape, buffers: GQANormRopeBuffers<'a>) -> GQANormRopeInvocation<'a> {
        GQANormRopeInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn norm_rope_source(config: GQANormRopeConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint \
         rope_dim = {}u;\nconstant float eps = {:.9e}f;\nconstant float rope_theta = {:.9e}f;\nconstant float \
         rope_scale = {:.9e}f;",
        config.num_heads, config.head_dim, config.rope_dim, config.eps, config.rope_theta, config.rope_scale,
    );
    GQA_NORM_ROPE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQANormRopeInvocation<'a> {
    config: GQANormRopeConfig,
    kernel: &'a Kernel,
    shape: GQANormRopeShape,
    buffers: GQANormRopeBuffers<'a>,
}

impl Operator for GQANormRopeInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQANormRopeInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.input.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.norm_weight.len_bytes() >= self.config.norm_weight_bytes());
        assert!(self.buffers.flat_token_indices.len_bytes() >= self.config.flat_token_indices_bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }

    fn record_compute(self, builder: &CommandRecorder) {
        let shape = self.shape;
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.norm_weight, 0);
        builder.set_buffer_read(2, self.buffers.flat_token_indices, 0);
        builder.set_buffer_write(3, self.buffers.output, 0);
        builder.set_u32(4, shape.num_tokens);
        builder.dispatch_1d(self.config.num_threads(shape), NUM_THREADS_PER_THREADBLOCK as usize);
    }
}
