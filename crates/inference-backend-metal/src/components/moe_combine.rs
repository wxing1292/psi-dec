use std::mem::size_of;

use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const MOE_COMBINE_SOURCE: &str = include_str!("metal/moe_combine.metal");

#[derive(Clone, Copy, Debug)]
pub struct MoECombineConfig {
    pub num_experts_per_token: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl MoECombineConfig {
    pub fn bf16(num_experts_per_token: u32, hidden_dim: u32) -> Self {
        Self {
            num_experts_per_token,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_experts_per_token > 0);
        assert!(self.hidden_dim > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
    }

    pub fn validate_shape(self, shape: MoECombineShape) {
        self.validate();
        shape.validate();
        assert_u32_index_domain(self.num_routed_elements(shape), "MoE combine routed elements");
        assert_u32_count_domain(self.num_output_elements(shape), "MoE combine output elements");
    }

    fn num_routes(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine route count",
            &[shape.num_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_routed_elements(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine routed element count",
            &[self.num_routes(shape), self.hidden_dim as usize],
        )
    }

    fn num_output_elements(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine output element count",
            &[shape.num_tokens as usize, self.hidden_dim as usize],
        )
    }

    pub fn routed_output_bytes(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine routed-output byte length",
            &[self.num_routed_elements(shape), self.dtype.item_size()],
        )
    }

    pub fn routed_probs_bytes(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine routed-probability byte length",
            &[self.num_routes(shape), size_of::<f32>()],
        )
    }

    pub fn output_bytes(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine output byte length",
            &[self.num_output_elements(shape), self.dtype.item_size()],
        )
    }

    pub fn shared_expert_gate_logits_bytes(self, shape: MoECombineShape) -> usize {
        checked_product(
            "MoE combine shared-gate byte length",
            &[shape.num_tokens as usize, self.dtype.item_size()],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoECombineShape {
    pub num_tokens: u32,
}

impl MoECombineShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
    }
}

pub struct MoECombineWithoutSharedExpertsBuffers<'a> {
    pub routed_hidden: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoECombineWithSharedExpertsBuffers<'a> {
    pub routed_hidden: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub shared_hidden: &'a Buffer,
    pub shared_expert_gate_logits: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoECombineKernels {
    config: MoECombineConfig,
    without_shared_experts: Kernel,
    with_shared_experts: Kernel,
}

impl MoECombineKernels {
    pub fn new(device: &crate::metal::Device, config: MoECombineConfig) -> Self {
        config.validate();
        Self {
            config,
            without_shared_experts: Kernel::new(device, MOE_COMBINE_SOURCE, "moe_combine_without_shared_experts"),
            with_shared_experts: Kernel::new(device, MOE_COMBINE_SOURCE, "moe_combine_with_shared_experts"),
        }
    }

    pub fn invoke_without_shared_experts<'a>(
        &'a self,
        shape: MoECombineShape,
        buffers: MoECombineWithoutSharedExpertsBuffers<'a>,
    ) -> MoECombineWithoutSharedExpertsInvocation<'a> {
        MoECombineWithoutSharedExpertsInvocation {
            config: self.config,
            kernel: &self.without_shared_experts,
            shape,
            buffers,
        }
    }

    pub fn invoke_with_shared_experts<'a>(
        &'a self,
        shape: MoECombineShape,
        buffers: MoECombineWithSharedExpertsBuffers<'a>,
    ) -> MoECombineWithSharedExpertsInvocation<'a> {
        MoECombineWithSharedExpertsInvocation {
            config: self.config,
            kernel: &self.with_shared_experts,
            shape,
            buffers,
        }
    }
}

pub struct MoECombineWithoutSharedExpertsInvocation<'a> {
    config: MoECombineConfig,
    kernel: &'a Kernel,
    shape: MoECombineShape,
    buffers: MoECombineWithoutSharedExpertsBuffers<'a>,
}

impl Operator for MoECombineWithoutSharedExpertsInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.config;
        config.validate_shape(self.shape);
        debug_validate_without_shared_experts_buffers(config, self.shape, &self.buffers);
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        builder.set_buffer_read(1, self.buffers.routed_probs, 0);
        builder.set_buffer_write(2, self.buffers.output, 0);
        builder.set_u32(3, self.shape.num_tokens);
        builder.set_u32(4, config.num_experts_per_token);
        builder.set_u32(5, config.hidden_dim);
        builder.dispatch_1d(config.num_output_elements(self.shape), 256);
    }
}

pub struct MoECombineWithSharedExpertsInvocation<'a> {
    config: MoECombineConfig,
    kernel: &'a Kernel,
    shape: MoECombineShape,
    buffers: MoECombineWithSharedExpertsBuffers<'a>,
}

impl Operator for MoECombineWithSharedExpertsInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        let config = self.config;
        config.validate_shape(self.shape);
        debug_validate_with_shared_experts_buffers(config, self.shape, &self.buffers);
        builder.set_kernel(self.kernel);
        builder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        builder.set_buffer_read(1, self.buffers.routed_probs, 0);
        builder.set_buffer_read(2, self.buffers.shared_hidden, 0);
        builder.set_buffer_read(3, self.buffers.shared_expert_gate_logits, 0);
        builder.set_buffer_write(4, self.buffers.output, 0);
        builder.set_u32(5, self.shape.num_tokens);
        builder.set_u32(6, config.num_experts_per_token);
        builder.set_u32(7, config.hidden_dim);
        builder.dispatch_1d(config.num_output_elements(self.shape), 256);
    }
}

fn debug_validate_without_shared_experts_buffers(
    config: MoECombineConfig,
    shape: MoECombineShape,
    buffers: &MoECombineWithoutSharedExpertsBuffers<'_>,
) {
    #[cfg(debug_assertions)]
    validate_without_shared_experts_buffers(config, shape, buffers);
}

fn debug_validate_with_shared_experts_buffers(
    config: MoECombineConfig,
    shape: MoECombineShape,
    buffers: &MoECombineWithSharedExpertsBuffers<'_>,
) {
    #[cfg(debug_assertions)]
    validate_with_shared_experts_buffers(config, shape, buffers);
}

fn validate_without_shared_experts_buffers(
    config: MoECombineConfig,
    shape: MoECombineShape,
    buffers: &MoECombineWithoutSharedExpertsBuffers<'_>,
) {
    let routed_output_bytes = config.routed_output_bytes(shape);
    let routed_probs_bytes = config.routed_probs_bytes(shape);
    let output_bytes = config.output_bytes(shape);
    assert!(
        buffers.routed_hidden.len_bytes() >= routed_output_bytes,
        "MoE combine without shared routed_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_output_bytes,
        buffers.routed_hidden.len_bytes()
    );
    assert!(
        buffers.routed_probs.len_bytes() >= routed_probs_bytes,
        "MoE combine without shared routed_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_probs_bytes,
        buffers.routed_probs.len_bytes()
    );
    assert!(
        buffers.output.len_bytes() >= output_bytes,
        "MoE combine without shared output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.output.len_bytes()
    );
}

fn validate_with_shared_experts_buffers(
    config: MoECombineConfig,
    shape: MoECombineShape,
    buffers: &MoECombineWithSharedExpertsBuffers<'_>,
) {
    let routed_output_bytes = config.routed_output_bytes(shape);
    let routed_probs_bytes = config.routed_probs_bytes(shape);
    let output_bytes = config.output_bytes(shape);
    let shared_expert_gate_logits_bytes = config.shared_expert_gate_logits_bytes(shape);
    assert!(
        buffers.routed_hidden.len_bytes() >= routed_output_bytes,
        "MoE combine with shared routed_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_output_bytes,
        buffers.routed_hidden.len_bytes()
    );
    assert!(
        buffers.routed_probs.len_bytes() >= routed_probs_bytes,
        "MoE combine with shared routed_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        routed_probs_bytes,
        buffers.routed_probs.len_bytes()
    );
    assert!(
        buffers.shared_hidden.len_bytes() >= output_bytes,
        "MoE combine with shared shared_hidden buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.shared_hidden.len_bytes()
    );
    assert!(
        buffers.shared_expert_gate_logits.len_bytes() >= shared_expert_gate_logits_bytes,
        "MoE combine with shared shared_expert_gate_logits buffer too short: shape={shape:?} required_bytes={} \
         buffer_bytes={}",
        shared_expert_gate_logits_bytes,
        buffers.shared_expert_gate_logits.len_bytes()
    );
    assert!(
        buffers.output.len_bytes() >= output_bytes,
        "MoE combine with shared output buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        output_bytes,
        buffers.output.len_bytes()
    );
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_executor_core::mlp::moe::reference::moe_combine_with_shared_experts_bf16_reference;
    use inference_executor_core::mlp::moe::reference::moe_combine_without_shared_experts_bf16_reference;

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "MoE combine output elements exceeds the shader u32 count domain")]
    fn test_without_shared_experts_shape_rejects_shader_count_overflow() {
        MoECombineConfig::bf16(1, 4).validate_shape(MoECombineShape { num_tokens: 1 << 30 });
    }

    #[test]
    #[should_panic(expected = "MoE combine output elements exceeds the shader u32 count domain")]
    fn test_with_shared_experts_shape_rejects_shader_count_overflow() {
        MoECombineConfig::bf16(1, 4).validate_shape(MoECombineShape { num_tokens: 1 << 30 });
    }

    #[test]
    fn test_without_shared_experts_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(2, 3);
        let shape = MoECombineShape { num_tokens: 2 };
        let routed_hidden_values = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            -1.0, 0.5, 2.0, //
            0.25, -0.75, 1.5,
        ];
        let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let kernels = MoECombineKernels::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_without_shared_experts(
            shape,
            MoECombineWithoutSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let actual = output.read_typed::<u16>(0, 6);
        let expected =
            moe_combine_without_shared_experts_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    #[test]
    fn test_with_shared_experts_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(2, 3);
        let shape = MoECombineShape { num_tokens: 2 };
        let routed_hidden_values = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            -1.0, 0.5, 2.0, //
            0.25, -0.75, 1.5,
        ];
        let routed_probs_values = [0.25, 0.75, 0.5, 0.5];
        let shared_hidden_values = [0.5, 1.0, -2.0, 1.5, -0.5, 0.25];
        let shared_expert_gate_logits_values = [-1.0, 2.0];
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let shared_hidden = bf16_buffer(&device, &shared_hidden_values);
        let shared_expert_gate_logits = bf16_buffer(&device, &shared_expert_gate_logits_values);
        let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let kernels = MoECombineKernels::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_with_shared_experts(
            shape,
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &shared_expert_gate_logits,
                output: &output,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let actual = output.read_typed::<u16>(0, 6);
        let routed =
            moe_combine_without_shared_experts_bf16_reference(&routed_hidden_values, &routed_probs_values, 2, 2, 3);
        let expected = moe_combine_with_shared_experts_bf16_reference(
            &routed,
            &shared_hidden_values,
            &shared_expert_gate_logits_values,
            2,
            3,
        );
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    #[test]
    fn test_with_shared_experts_random() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(3, 5);
        let shape = MoECombineShape { num_tokens: 3 };
        let random_seed = 0xC461_8E2B;
        let routed_hidden_values = generated_values(
            shape.num_tokens as usize * config.num_experts_per_token as usize * config.hidden_dim as usize,
            random_seed,
        );
        let routed_probs_values = generated_probs(
            shape.num_tokens as usize,
            config.num_experts_per_token as usize,
            random_seed.wrapping_add(1),
        );
        let shared_hidden_values = generated_values(
            shape.num_tokens as usize * config.hidden_dim as usize,
            random_seed.wrapping_add(2),
        );
        let shared_expert_gate_logits_values = generated_values(shape.num_tokens as usize, random_seed.wrapping_add(3));
        let routed_hidden = bf16_buffer(&device, &routed_hidden_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let shared_hidden = bf16_buffer(&device, &shared_hidden_values);
        let shared_expert_gate_logits = bf16_buffer(&device, &shared_expert_gate_logits_values);
        let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let kernels = MoECombineKernels::new(&device, config);

        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_with_shared_experts(
            shape,
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &shared_expert_gate_logits,
                output: &output,
            },
        ));
        let program = builder.build();
        stream.submit_replay(&program).wait();

        let routed = moe_combine_without_shared_experts_bf16_reference(
            &routed_hidden_values,
            &routed_probs_values,
            shape.num_tokens as usize,
            config.num_experts_per_token as usize,
            config.hidden_dim as usize,
        );
        let expected = moe_combine_with_shared_experts_bf16_reference(
            &routed,
            &shared_hidden_values,
            &shared_expert_gate_logits_values,
            shape.num_tokens as usize,
            config.hidden_dim as usize,
        );
        let actual = output.read_typed::<u16>(0, shape.num_tokens as usize * config.hidden_dim as usize);
        assert_close_bits(&actual, &expected, 1.0e-3);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn generated_probs(num_tokens: usize, num_experts_per_token: usize, random_seed: u32) -> Vec<f32> {
        let mut values = generated_values(num_tokens * num_experts_per_token, random_seed)
            .into_iter()
            .map(|value| value.abs() + 0.05)
            .collect::<Vec<_>>();
        for row in values.chunks_mut(num_experts_per_token) {
            let sum = row.iter().sum::<f32>();
            for value in row {
                *value /= sum;
            }
        }
        values
    }

    fn assert_close_bits(actual: &[u16], expected: &[u16], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = bf16::from_bits(actual).to_f32();
            let expected = bf16::from_bits(expected).to_f32();
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
