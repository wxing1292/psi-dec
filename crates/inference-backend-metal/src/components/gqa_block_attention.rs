use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_BLOCK_SDPA_SOURCE: &str = include_str!("metal/gqa_block_sdpa.metal");
const SIMD_WIDTH: u32 = 32;
const NUM_THREADS_PER_THREADBLOCK: u32 = SIMD_WIDTH;

/// Dense request-block SDPA that writes one `SDPAPartialOutput` into the
/// `SDPAMapTaskTemplate` slot selected for each Q token. One threadblock owns
/// one Q-token/Q-head block-SDPA task. The shared paged reducer later combines
/// the bidirectional block and persistent-history partial outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQABlockSDPAConfig {
    pub block_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub dtype: Dtype,
}

impl GQABlockSDPAConfig {
    pub fn validate(self) {
        assert!(self.block_size > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.scale.is_finite() && self.scale > 0.0);
        assert_eq!(
            self.head_dim % SIMD_WIDTH,
            0,
            "GQA block SDPA head_dim must be divisible by the SIMD width"
        );
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    fn q_elements(self, shape: GQABlockSDPAShape) -> usize {
        checked_product(
            "GQA block SDPA Q element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn kv_elements(self, shape: GQABlockSDPAShape) -> usize {
        checked_product(
            "GQA block SDPA K/V element count",
            &[
                shape.num_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn partial_output_stat_elements(self, shape: GQABlockSDPAShape) -> usize {
        checked_product(
            "GQA block SDPA partial-output statistic element count",
            &[
                shape.num_total_sdpa_map_task_templates as usize,
                self.num_q_heads as usize,
            ],
        )
    }

    fn partial_output_values(self, shape: GQABlockSDPAShape) -> usize {
        self.partial_output_stat_elements(shape)
            .checked_mul(self.head_dim as usize)
            .expect("GQA block SDPA partial-output element count must fit usize")
    }

    fn dispatch_threads(self, shape: GQABlockSDPAShape) -> usize {
        checked_product(
            "GQA block SDPA thread count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                NUM_THREADS_PER_THREADBLOCK as usize,
            ],
        )
    }

    fn threadblock_memory_bytes(self) -> usize {
        (self.block_size as usize)
            .checked_mul(size_of::<f32>())
            .expect("GQA block SDPA threadblock memory must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQABlockSDPAShape {
    pub num_tokens: u32,
    pub num_total_sdpa_map_task_templates: u32,
}

impl GQABlockSDPAShape {
    pub fn validate(self, config: GQABlockSDPAConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_eq!(
            self.num_tokens % config.block_size,
            0,
            "GQA block SDPA tokens must contain complete request blocks"
        );
        assert!(self.num_total_sdpa_map_task_templates >= self.num_tokens);
        assert_u32_count_domain(config.q_elements(self), "GQA block SDPA Q");
        assert_u32_count_domain(config.kv_elements(self), "GQA block SDPA K/V");
        assert_u32_index_domain(
            config.partial_output_stat_elements(self),
            "GQA block SDPA partial-output statistics",
        );
        assert_u32_index_domain(config.partial_output_values(self), "GQA block SDPA partial output");
        assert_u32_count_domain(config.dispatch_threads(self), "GQA block SDPA threads");
    }
}

#[derive(Clone, Copy)]
pub struct GQABlockSDPABuffers<'a> {
    pub q: &'a Buffer,
    pub local_k: &'a Buffer,
    pub local_v: &'a Buffer,
    pub block_sdpa_map_task_template_indices: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
}

pub struct GQABlockSDPAKernel {
    config: GQABlockSDPAConfig,
    kernel: Kernel,
}

impl GQABlockSDPAKernel {
    pub fn new(device: &Device, config: GQABlockSDPAConfig) -> Self {
        config.validate();
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_block_sdpa_f32",
            Dtype::Bfloat16 => "gqa_block_sdpa_bf16",
            dtype => panic!("unsupported GQA block SDPA dtype {dtype:?}"),
        };
        let kernel = Kernel::new(device, &block_sdpa_source(config), function_name);
        assert_eq!(
            kernel.thread_execution_width(),
            SIMD_WIDTH as usize,
            "GQA block SDPA requires a 32-thread SIMDgroup"
        );
        assert!(
            NUM_THREADS_PER_THREADBLOCK as usize <= kernel.max_total_threads_per_threadblock(),
            "GQA block SDPA requires {} threads per threadblock but the pipeline supports {}",
            NUM_THREADS_PER_THREADBLOCK,
            kernel.max_total_threads_per_threadblock()
        );
        let max_threadblock_memory_length = device.max_threadblock_memory_length();
        assert!(
            config.threadblock_memory_bytes() <= max_threadblock_memory_length,
            "GQA block SDPA requires {} bytes of threadblock memory but the device supports {}",
            config.threadblock_memory_bytes(),
            max_threadblock_memory_length
        );
        assert!(
            kernel.static_threadblock_memory_length() <= max_threadblock_memory_length,
            "GQA block SDPA pipeline uses {} bytes of static threadblock memory but the device supports {}",
            kernel.static_threadblock_memory_length(),
            max_threadblock_memory_length
        );
        Self { config, kernel }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQABlockSDPAShape,
        buffers: GQABlockSDPABuffers<'a>,
    ) -> GQABlockSDPAInvocation<'a> {
        GQABlockSDPAInvocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn block_sdpa_source(config: GQABlockSDPAConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint block_size = {}u;\nconstant uint num_q_heads = {}u;\nconstant uint \
         num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant float attention_scale = {:.9e}f;\nconstant uint \
         num_threads_per_threadblock = {}u;\nconstant uint simd_width = {}u;\nconstant uint q_values_per_thread = \
         head_dim / simd_width;",
        config.block_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.scale,
        NUM_THREADS_PER_THREADBLOCK,
        SIMD_WIDTH,
    );
    GQA_BLOCK_SDPA_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQABlockSDPAInvocation<'a> {
    config: GQABlockSDPAConfig,
    kernel: &'a Kernel,
    shape: GQABlockSDPAShape,
    buffers: GQABlockSDPABuffers<'a>,
}

impl Operator for GQABlockSDPAInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.local_k, 0);
        recorder.set_buffer_read(2, self.buffers.local_v, 0);
        recorder.set_buffer_read(3, self.buffers.block_sdpa_map_task_template_indices, 0);
        recorder.set_buffer_write(4, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(5, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_write(6, self.buffers.partial_output, 0);
        recorder.set_u32(7, self.shape.num_tokens);
        recorder.dispatch_1d(
            self.config.dispatch_threads(self.shape),
            NUM_THREADS_PER_THREADBLOCK as usize,
        );
    }
}

impl GQABlockSDPAInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.q.len_bytes() >= bytes(self.config.q_elements(self.shape), self.config.dtype));
        assert!(self.buffers.local_k.len_bytes() >= bytes(self.config.kv_elements(self.shape), self.config.dtype));
        assert!(self.buffers.local_v.len_bytes() >= bytes(self.config.kv_elements(self.shape), self.config.dtype));
        assert!(
            self.buffers.block_sdpa_map_task_template_indices.len_bytes()
                >= (self.shape.num_tokens as usize)
                    .checked_mul(size_of::<u32>())
                    .expect("GQA block SDPA map TaskTemplate index bytes must fit usize")
        );
        let partial_output_stat_bytes = self
            .config
            .partial_output_stat_elements(self.shape)
            .checked_mul(size_of::<f32>())
            .expect("GQA block SDPA partial-output statistic bytes must fit usize");
        assert!(self.buffers.partial_exp_sums.len_bytes() >= partial_output_stat_bytes);
        assert!(self.buffers.partial_max_logits.len_bytes() >= partial_output_stat_bytes);
        assert!(
            self.buffers.partial_output.len_bytes()
                >= bytes(self.config.partial_output_values(self.shape), self.config.dtype,)
        );
    }
}

fn bytes(num_elements: usize, dtype: Dtype) -> usize {
    num_elements
        .checked_mul(dtype.item_size())
        .expect("GQA block SDPA buffer byte length must fit usize")
}

#[cfg(test)]
#[path = "gqa_block_attention_test.rs"]
mod tests;
