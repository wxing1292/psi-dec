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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GQABlockSDPAThreadBlockSpecialization {
    required_threads: u32,
    simdgroup_width: u32,
}

impl GQABlockSDPAThreadBlockSpecialization {
    fn current() -> Self {
        Self {
            required_threads: 32,
            simdgroup_width: 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GQABlockSDPAKernelSpecialization {
    config: GQABlockSDPAConfig,
    thread_block: GQABlockSDPAThreadBlockSpecialization,
}

impl GQABlockSDPAKernelSpecialization {
    fn current(config: GQABlockSDPAConfig) -> Self {
        Self {
            config,
            thread_block: GQABlockSDPAThreadBlockSpecialization::current(),
        }
    }
}

/// Dense request-block SDPA that writes one `SDPAPartialOutput` into the
/// `SDPAMapTaskTemplate` slot selected for each Q token. One thread block owns
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
        let thread_block = GQABlockSDPAThreadBlockSpecialization::current();
        assert!(self.block_size > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.scale.is_finite() && self.scale > 0.0);
        assert_eq!(
            self.head_dim % thread_block.simdgroup_width,
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

    fn dispatch_threads(self, shape: GQABlockSDPAShape, thread_block: GQABlockSDPAThreadBlockSpecialization) -> usize {
        checked_product(
            "GQA block SDPA thread count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                thread_block.required_threads as usize,
            ],
        )
    }

    fn thread_block_memory_bytes(self) -> usize {
        (self.block_size as usize)
            .checked_mul(size_of::<f32>())
            .expect("GQA block SDPA thread-block memory must fit usize")
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
        assert_u32_count_domain(
            config.dispatch_threads(self, GQABlockSDPAThreadBlockSpecialization::current()),
            "GQA block SDPA threads",
        );
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
    specialization: GQABlockSDPAKernelSpecialization,
    kernel: Kernel,
}

impl GQABlockSDPAKernel {
    pub fn new(device: &Device, config: GQABlockSDPAConfig) -> Self {
        config.validate();
        let specialization = GQABlockSDPAKernelSpecialization::current(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_block_sdpa_f32",
            Dtype::Bfloat16 => "gqa_block_sdpa_bf16",
            dtype => panic!("unsupported GQA block SDPA dtype {dtype:?}"),
        };
        let kernel = Kernel::new(device, &block_sdpa_source(specialization), function_name);
        assert_eq!(
            kernel.thread_execution_width(),
            specialization.thread_block.simdgroup_width as usize,
            "GQA block SDPA requires a 32-thread SIMDgroup"
        );
        assert!(
            specialization.thread_block.required_threads as usize <= kernel.max_total_threads_per_threadblock(),
            "GQA block SDPA requires {} threads per thread block but the pipeline supports {}",
            specialization.thread_block.required_threads,
            kernel.max_total_threads_per_threadblock()
        );
        let max_thread_block_memory_length = device.max_threadblock_memory_length();
        assert!(
            config.thread_block_memory_bytes() <= max_thread_block_memory_length,
            "GQA block SDPA requires {} bytes of thread-block memory but the device supports {}",
            config.thread_block_memory_bytes(),
            max_thread_block_memory_length
        );
        assert!(
            kernel.static_threadblock_memory_length() <= max_thread_block_memory_length,
            "GQA block SDPA pipeline uses {} bytes of static thread-block memory but the device supports {}",
            kernel.static_threadblock_memory_length(),
            max_thread_block_memory_length
        );
        Self { specialization, kernel }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQABlockSDPAShape,
        buffers: GQABlockSDPABuffers<'a>,
    ) -> GQABlockSDPAInvocation<'a> {
        GQABlockSDPAInvocation {
            specialization: self.specialization,
            kernel: &self.kernel,
            shape,
            buffers,
        }
    }
}

fn block_sdpa_source(specialization: GQABlockSDPAKernelSpecialization) -> String {
    let config = specialization.config;
    let constants = format!(
        "using namespace metal;\n\nconstant uint block_size = {}u;\nconstant uint num_q_heads = {}u;\nconstant uint \
         num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant float attention_scale = {:.9e}f;\nconstant uint \
         simd_width = {}u;\nconstant uint q_values_per_thread = head_dim / simd_width;",
        config.block_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.scale,
        specialization.thread_block.simdgroup_width,
    );
    GQA_BLOCK_SDPA_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQABlockSDPAInvocation<'a> {
    specialization: GQABlockSDPAKernelSpecialization,
    kernel: &'a Kernel,
    shape: GQABlockSDPAShape,
    buffers: GQABlockSDPABuffers<'a>,
}

impl Operator for GQABlockSDPAInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let config = self.specialization.config;
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
            config.dispatch_threads(self.shape, self.specialization.thread_block),
            self.specialization.thread_block.required_threads as usize,
        );
    }
}

impl GQABlockSDPAInvocation<'_> {
    fn validate(&self) {
        let config = self.specialization.config;
        self.shape.validate(config);
        assert!(self.buffers.q.len_bytes() >= bytes(config.q_elements(self.shape), config.dtype));
        assert!(self.buffers.local_k.len_bytes() >= bytes(config.kv_elements(self.shape), config.dtype));
        assert!(self.buffers.local_v.len_bytes() >= bytes(config.kv_elements(self.shape), config.dtype));
        assert!(
            self.buffers.block_sdpa_map_task_template_indices.len_bytes()
                >= (self.shape.num_tokens as usize)
                    .checked_mul(size_of::<u32>())
                    .expect("GQA block SDPA map TaskTemplate index bytes must fit usize")
        );
        let partial_output_stat_bytes = self
            .specialization
            .config
            .partial_output_stat_elements(self.shape)
            .checked_mul(size_of::<f32>())
            .expect("GQA block SDPA partial-output statistic bytes must fit usize");
        assert!(self.buffers.partial_exp_sums.len_bytes() >= partial_output_stat_bytes);
        assert!(self.buffers.partial_max_logits.len_bytes() >= partial_output_stat_bytes);
        assert!(
            self.buffers.partial_output.len_bytes() >= bytes(config.partial_output_values(self.shape), config.dtype,)
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
