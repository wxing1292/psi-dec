use super::assert_u32_count_domain;
use super::assert_u32_index_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const GQA_LOCAL_SDPA_SOURCE: &str = include_str!("metal/gqa_local_sdpa.metal");

/// Dense request-local SDPA that writes one `SDPAPartialOutput` into the
/// `SDPAMapTaskTemplate` slot selected for each Q token. One threadblock owns
/// one Q-token/Q-head local-SDPA task; the shared paged reducer later combines
/// the local and persistent partial outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQALocalSDPAConfig {
    pub local_block_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub num_threads_per_threadblock: u32,
    pub dtype: Dtype,
}

impl GQALocalSDPAConfig {
    pub fn validate(self) {
        assert!(self.local_block_size > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.scale.is_finite() && self.scale > 0.0);
        assert!(self.num_threads_per_threadblock.is_power_of_two() && self.num_threads_per_threadblock <= 256);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    fn q_elements(self, shape: GQALocalSDPAShape) -> usize {
        checked_product(
            "GQA local SDPA Q element count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn kv_elements(self, shape: GQALocalSDPAShape) -> usize {
        checked_product(
            "GQA local SDPA K/V element count",
            &[
                shape.num_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn partial_output_stat_elements(self, shape: GQALocalSDPAShape) -> usize {
        checked_product(
            "GQA local SDPA partial-output statistic element count",
            &[shape.total_sdpa_map_task_templates as usize, self.num_q_heads as usize],
        )
    }

    fn partial_output_values(self, shape: GQALocalSDPAShape) -> usize {
        self.partial_output_stat_elements(shape)
            .checked_mul(self.head_dim as usize)
            .expect("GQA local SDPA partial-output element count must fit usize")
    }

    fn dispatch_threads(self, shape: GQALocalSDPAShape) -> usize {
        checked_product(
            "GQA local SDPA thread count",
            &[
                shape.num_tokens as usize,
                self.num_q_heads as usize,
                self.num_threads_per_threadblock as usize,
            ],
        )
    }

    fn threadblock_memory_bytes(self) -> usize {
        (self.local_block_size as usize + self.num_threads_per_threadblock as usize)
            .checked_mul(size_of::<f32>())
            .expect("GQA local SDPA threadblock memory must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQALocalSDPAShape {
    pub num_tokens: u32,
    pub total_sdpa_map_task_templates: u32,
}

impl GQALocalSDPAShape {
    pub fn validate(self, config: GQALocalSDPAConfig) {
        config.validate();
        assert!(self.num_tokens > 0);
        assert_eq!(
            self.num_tokens % config.local_block_size,
            0,
            "GQA local SDPA tokens must contain complete local blocks"
        );
        assert!(self.total_sdpa_map_task_templates >= self.num_tokens);
        assert_u32_count_domain(config.q_elements(self), "GQA local SDPA Q");
        assert_u32_count_domain(config.kv_elements(self), "GQA local SDPA K/V");
        assert_u32_index_domain(
            config.partial_output_stat_elements(self),
            "GQA local SDPA partial-output statistics",
        );
        assert_u32_index_domain(config.partial_output_values(self), "GQA local SDPA partial output");
        assert_u32_count_domain(config.dispatch_threads(self), "GQA local SDPA threads");
    }
}

#[derive(Clone, Copy)]
pub struct GQALocalSDPABuffers<'a> {
    pub q: &'a Buffer,
    pub local_k: &'a Buffer,
    pub local_v: &'a Buffer,
    pub local_sdpa_map_task_template_indices: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
}

pub struct GQALocalSDPAKernel {
    config: GQALocalSDPAConfig,
    kernel: Kernel,
    max_threadblock_memory_length: usize,
}

impl GQALocalSDPAKernel {
    pub fn new(device: &Device, config: GQALocalSDPAConfig) -> Self {
        config.validate();
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_local_sdpa_f32",
            Dtype::Bfloat16 => "gqa_local_sdpa_bf16",
            dtype => panic!("unsupported GQA local SDPA dtype {dtype:?}"),
        };
        Self {
            config,
            kernel: Kernel::new(device, &local_sdpa_source(config), function_name),
            max_threadblock_memory_length: device.max_threadblock_memory_length(),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: GQALocalSDPAShape,
        buffers: GQALocalSDPABuffers<'a>,
    ) -> GQALocalSDPAInvocation<'a> {
        GQALocalSDPAInvocation {
            config: self.config,
            kernel: &self.kernel,
            max_threadblock_memory_length: self.max_threadblock_memory_length,
            shape,
            buffers,
        }
    }
}

fn local_sdpa_source(config: GQALocalSDPAConfig) -> String {
    let constants = format!(
        "using namespace metal;\n\nconstant uint local_block_size = {}u;\nconstant uint num_q_heads = {}u;\nconstant \
         uint num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant float attention_scale = {:.9e}f;\nconstant \
         uint num_threads_per_threadblock = {}u;",
        config.local_block_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.scale,
        config.num_threads_per_threadblock,
    );
    GQA_LOCAL_SDPA_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct GQALocalSDPAInvocation<'a> {
    config: GQALocalSDPAConfig,
    kernel: &'a Kernel,
    max_threadblock_memory_length: usize,
    shape: GQALocalSDPAShape,
    buffers: GQALocalSDPABuffers<'a>,
}

impl Operator for GQALocalSDPAInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.validate();
        self.record_compute(builder);
    }
}

impl GQALocalSDPAInvocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        assert!(self.buffers.q.len_bytes() >= bytes(self.config.q_elements(self.shape), self.config.dtype));
        assert!(self.buffers.local_k.len_bytes() >= bytes(self.config.kv_elements(self.shape), self.config.dtype));
        assert!(self.buffers.local_v.len_bytes() >= bytes(self.config.kv_elements(self.shape), self.config.dtype));
        assert!(
            self.buffers.local_sdpa_map_task_template_indices.len_bytes()
                >= (self.shape.num_tokens as usize)
                    .checked_mul(size_of::<u32>())
                    .expect("GQA local SDPA map TaskTemplate index bytes must fit usize")
        );
        let partial_output_stat_bytes = self
            .config
            .partial_output_stat_elements(self.shape)
            .checked_mul(size_of::<f32>())
            .expect("GQA local SDPA partial-output statistic bytes must fit usize");
        assert!(self.buffers.partial_exp_sums.len_bytes() >= partial_output_stat_bytes);
        assert!(self.buffers.partial_max_logits.len_bytes() >= partial_output_stat_bytes);
        assert!(
            self.buffers.partial_output.len_bytes()
                >= bytes(self.config.partial_output_values(self.shape), self.config.dtype,)
        );
        assert!(
            self.config.threadblock_memory_bytes() <= self.max_threadblock_memory_length,
            "GQA local SDPA requires {} bytes of threadblock memory but device only supports {}",
            self.config.threadblock_memory_bytes(),
            self.max_threadblock_memory_length
        );
    }

    fn record_compute(self, builder: &CommandRecorder) {
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.q, 0);
        builder.set_buffer_read(1, self.buffers.local_k, 0);
        builder.set_buffer_read(2, self.buffers.local_v, 0);
        builder.set_buffer_read(3, self.buffers.local_sdpa_map_task_template_indices, 0);
        builder.set_buffer_write(4, self.buffers.partial_exp_sums, 0);
        builder.set_buffer_write(5, self.buffers.partial_max_logits, 0);
        builder.set_buffer_write(6, self.buffers.partial_output, 0);
        builder.set_u32(7, self.shape.num_tokens);
        builder.dispatch_1d(
            self.config.dispatch_threads(self.shape),
            self.config.num_threads_per_threadblock as usize,
        );
    }
}

fn bytes(num_elements: usize, dtype: Dtype) -> usize {
    num_elements
        .checked_mul(dtype.item_size())
        .expect("GQA local SDPA buffer byte length must fit usize")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::Stream;

    fn attention_partial(scores: &[f32], values: &[[f32; 2]]) -> (f32, f32, [f32; 2]) {
        assert_eq!(scores.len(), values.len());
        let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|&score| (score - max_logit).exp())
            .collect::<Vec<_>>();
        let exp_sum = weights.iter().sum::<f32>();
        let mut output = [0.0; 2];
        for (&weight, value) in weights.iter().zip(values) {
            for dim in 0..2 {
                output[dim] += weight * value[dim] / exp_sum;
            }
        }
        (max_logit, exp_sum, output)
    }

    fn reduce_partials(partials: &[(f32, f32, [f32; 2])]) -> [f32; 2] {
        let global_max = partials
            .iter()
            .map(|&(max_logit, ..)| max_logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut global_exp_sum = 0.0;
        let mut output = [0.0; 2];
        for &(max_logit, exp_sum, partial_output) in partials {
            let weight = (max_logit - global_max).exp() * exp_sum;
            global_exp_sum += weight;
            for dim in 0..2 {
                output[dim] += weight * partial_output[dim];
            }
        }
        for value in &mut output {
            *value /= global_exp_sum;
        }
        output
    }

    #[test]
    fn local_shape_models_complete_bidirectional_blocks() {
        let config = GQALocalSDPAConfig {
            local_block_size: 9,
            num_q_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            scale: 128.0_f32.sqrt().recip(),
            num_threads_per_threadblock: 128,
            dtype: Dtype::Bfloat16,
        };
        let shape = GQALocalSDPAShape {
            num_tokens: 18,
            total_sdpa_map_task_templates: 64,
        };
        shape.validate(config);
        assert_eq!(config.q_elements(shape), 92_160);
        assert_eq!(config.kv_elements(shape), 18_432);
        assert_eq!(config.partial_output_stat_elements(shape), 2560);
    }

    #[test]
    #[should_panic(expected = "complete local blocks")]
    fn local_shape_rejects_partial_request_block() {
        GQALocalSDPAShape {
            num_tokens: 10,
            total_sdpa_map_task_templates: 16,
        }
        .validate(GQALocalSDPAConfig {
            local_block_size: 9,
            num_q_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            scale: 1.0,
            num_threads_per_threadblock: 128,
            dtype: Dtype::Bfloat16,
        });
    }

    #[test]
    fn partial_output_reduction_matches_joint_persistent_and_bidirectional_local_attention() {
        let persistent_scores = [-1.0, 0.25, 1.5];
        let persistent_values = [[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]];
        let local_scores = [0.75, -0.5, 2.0, 0.1];
        let local_values = [[5.0, 1.0], [0.0, 7.0], [2.0, -3.0], [9.0, 6.0]];

        let merged = reduce_partials(&[
            attention_partial(&persistent_scores, &persistent_values),
            attention_partial(&local_scores, &local_values),
        ]);
        let joint_scores = [persistent_scores.as_slice(), local_scores.as_slice()].concat();
        let joint_values = [persistent_values.as_slice(), local_values.as_slice()].concat();
        let direct = attention_partial(&joint_scores, &joint_values).2;

        for (actual, expected) in merged.into_iter().zip(direct) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn f32_kernel_matches_bidirectional_request_local_reference() {
        assert_kernel_matches_bidirectional_request_local_reference(Dtype::Float32, 1.0e-5);
    }

    #[test]
    fn bf16_kernel_matches_bidirectional_request_local_reference() {
        assert_kernel_matches_bidirectional_request_local_reference(Dtype::Bfloat16, 1.0e-2);
    }

    fn assert_kernel_matches_bidirectional_request_local_reference(dtype: Dtype, output_tolerance: f32) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = GQALocalSDPAConfig {
            local_block_size: 3,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            scale: 0.5,
            num_threads_per_threadblock: 4,
            dtype,
        };
        let shape = GQALocalSDPAShape {
            num_tokens: 6,
            total_sdpa_map_task_templates: 8,
        };
        let q_values = round_for_dtype(
            (0..config.q_elements(shape))
                .map(|index| index as f32 * 0.03125 - 0.4)
                .collect::<Vec<_>>(),
            dtype,
        );
        let k_values = round_for_dtype(
            (0..config.kv_elements(shape))
                .map(|index| index as f32 * -0.025 + 0.3)
                .collect::<Vec<_>>(),
            dtype,
        );
        let v_values = round_for_dtype(
            (0..config.kv_elements(shape))
                .map(|index| index as f32 * 0.05 - 0.2)
                .collect::<Vec<_>>(),
            dtype,
        );
        let q = buffer(&device, &q_values, dtype);
        let local_k = buffer(&device, &k_values, dtype);
        let local_v = buffer(&device, &v_values, dtype);
        let local_sdpa_map_task_template_indices = Buffer::from_slice(&device, &[0_u32, 1, 2, 3, 4, 5]);
        let partial_exp_sums =
            Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
        let partial_max_logits =
            Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
        let partial_output = Buffer::new_zeroed_elements(&device, config.partial_output_values(shape), dtype);
        let kernel = GQALocalSDPAKernel::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            GQALocalSDPABuffers {
                q: &q,
                local_k: &local_k,
                local_v: &local_v,
                local_sdpa_map_task_template_indices: &local_sdpa_map_task_template_indices,
                partial_exp_sums: &partial_exp_sums,
                partial_max_logits: &partial_max_logits,
                partial_output: &partial_output,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        let actual_partial_exp_sums = partial_exp_sums.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
        let actual_partial_max_logits =
            partial_max_logits.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
        let actual_output = read_values(&partial_output, config.partial_output_values(shape), dtype);
        for q_token_index in 0..shape.num_tokens as usize {
            let local_kv_token_begin =
                q_token_index / config.local_block_size as usize * config.local_block_size as usize;
            for q_head in 0..config.num_q_heads as usize {
                let q_start = (q_token_index * config.num_q_heads as usize + q_head) * config.head_dim as usize;
                let q_row = &q_values[q_start..q_start + config.head_dim as usize];
                let mut scores = Vec::new();
                let mut values = Vec::new();
                for local_kv_offset in 0..config.local_block_size as usize {
                    let kv_token_index = local_kv_token_begin + local_kv_offset;
                    let kv_start = kv_token_index * config.head_dim as usize;
                    let key = &k_values[kv_start..kv_start + config.head_dim as usize];
                    scores.push(q_row.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>() * config.scale);
                    values.push([
                        v_values[kv_start],
                        v_values[kv_start + 1],
                        v_values[kv_start + 2],
                        v_values[kv_start + 3],
                    ]);
                }
                let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|&score| (score - max_logit).exp())
                    .collect::<Vec<_>>();
                let exp_sum = weights.iter().sum::<f32>();
                let partial_output_index = q_token_index * config.num_q_heads as usize + q_head;
                assert!((actual_partial_max_logits[partial_output_index] - max_logit).abs() < 1.0e-5);
                assert!((actual_partial_exp_sums[partial_output_index] - exp_sum).abs() < 1.0e-5);
                for dim in 0..config.head_dim as usize {
                    let expected = weights
                        .iter()
                        .zip(&values)
                        .map(|(&weight, value)| weight * value[dim])
                        .sum::<f32>()
                        / exp_sum;
                    let actual = actual_output[partial_output_index * config.head_dim as usize + dim];
                    assert!(
                        (actual - expected).abs() < output_tolerance,
                        "actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    fn round_for_dtype(values: Vec<f32>, dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::Float32 => values,
            Dtype::Bfloat16 => values.into_iter().map(|value| bf16::from_f32(value).to_f32()).collect(),
            dtype => panic!("unsupported test dtype {dtype:?}"),
        }
    }

    fn buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
        match dtype {
            Dtype::Float32 => Buffer::from_slice(device, values),
            Dtype::Bfloat16 => {
                let bits = values
                    .iter()
                    .map(|value| bf16::from_f32(*value).to_bits())
                    .collect::<Vec<_>>();
                Buffer::from_slice(device, &bits)
            },
            dtype => panic!("unsupported test dtype {dtype:?}"),
        }
    }

    fn read_values(buffer: &Buffer, count: usize, dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::Float32 => buffer.read_typed::<f32>(0, count),
            Dtype::Bfloat16 => {
                buffer
                    .read_typed::<u16>(0, count)
                    .into_iter()
                    .map(|bits| bf16::from_bits(bits).to_f32())
                    .collect()
            },
            dtype => panic!("unsupported test dtype {dtype:?}"),
        }
    }
}
