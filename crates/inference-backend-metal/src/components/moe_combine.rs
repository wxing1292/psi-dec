use std::mem::size_of;

use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;

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
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity combine whose active token count is supplied at submission.
    pub fn invoke_without_shared_experts_bucketed<'a>(
        &'a self,
        shape: MoECombineShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: MoECombineWithoutSharedExpertsBuffers<'a>,
    ) -> MoECombineWithoutSharedExpertsInvocation<'a> {
        MoECombineWithoutSharedExpertsInvocation {
            config: self.config,
            kernel: &self.without_shared_experts,
            shape,
            buffers,
            num_active_tokens_key: Some(num_active_tokens_key),
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
            num_active_tokens_key: None,
        }
    }

    /// Records a fixed-capacity combine whose active token count is supplied at submission.
    pub fn invoke_with_shared_experts_bucketed<'a>(
        &'a self,
        shape: MoECombineShape,
        num_active_tokens_key: ReplayParameterKey,
        buffers: MoECombineWithSharedExpertsBuffers<'a>,
    ) -> MoECombineWithSharedExpertsInvocation<'a> {
        MoECombineWithSharedExpertsInvocation {
            config: self.config,
            kernel: &self.with_shared_experts,
            shape,
            buffers,
            num_active_tokens_key: Some(num_active_tokens_key),
        }
    }
}

pub struct MoECombineWithoutSharedExpertsInvocation<'a> {
    config: MoECombineConfig,
    kernel: &'a Kernel,
    shape: MoECombineShape,
    buffers: MoECombineWithoutSharedExpertsBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
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
        record_num_active_tokens(builder, 3, self.shape.num_tokens, self.num_active_tokens_key);
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
    num_active_tokens_key: Option<ReplayParameterKey>,
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
        record_num_active_tokens(builder, 5, self.shape.num_tokens, self.num_active_tokens_key);
        builder.set_u32(6, config.num_experts_per_token);
        builder.set_u32(7, config.hidden_dim);
        builder.dispatch_1d(config.num_output_elements(self.shape), 256);
    }
}

fn record_num_active_tokens(
    builder: &CommandRecorder<'_>,
    binding_index: usize,
    num_total_tokens: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => builder.bind_u32(binding_index, key, 1, num_total_tokens),
        None => builder.set_u32(binding_index, num_total_tokens),
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
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.combine.num_active_tokens");

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
        assert_eq!(program.stats().parameter_count, 0);
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
        assert_eq!(program.stats().parameter_count, 0);
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

    #[test]
    fn test_bucketed_capacity_is_reusable_with_and_without_shared_experts() {
        for with_shared_experts in [false, true] {
            run_bucketed_capacity_case(with_shared_experts);
        }
    }

    #[test]
    fn test_bucketed_submission_validates_active_tokens() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(2, 3);
        let shape = MoECombineShape { num_tokens: 4 };
        let routed_hidden = Buffer::new_zeroed(&device, config.routed_output_bytes(shape));
        let routed_probs = Buffer::new_zeroed(&device, config.routed_probs_bytes(shape));
        let output = Buffer::new_zeroed(&device, config.output_bytes(shape));
        let kernels = MoECombineKernels::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_without_shared_experts_bucketed(
            shape,
            NUM_ACTIVE_TOKENS,
            MoECombineWithoutSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
        let replay = builder.build();

        assert_panics(|| {
            let _ = stream.submit_replay_with_arguments(&replay, &ReplayArguments::new());
        });
        assert_panics(|| {
            let _ =
                stream.submit_replay_with_arguments(&replay, &ReplayArguments::new().with_i32(NUM_ACTIVE_TOKENS, 3));
        });
        for value in [0, shape.num_tokens + 1] {
            assert_panics(|| {
                let _ = stream
                    .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, value));
            });
        }
    }

    #[test]
    fn test_bucketed_buffer_validation_uses_total_tokens() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(2, 3);
        let total_shape = MoECombineShape { num_tokens: 4 };
        let active_shape = MoECombineShape { num_tokens: 3 };
        let full_routed_hidden = Buffer::new_zeroed(&device, config.routed_output_bytes(total_shape));
        let full_routed_probs = Buffer::new_zeroed(&device, config.routed_probs_bytes(total_shape));
        let full_shared_hidden = Buffer::new_zeroed(&device, config.output_bytes(total_shape));
        let full_gate_logits = Buffer::new_zeroed(&device, config.shared_expert_gate_logits_bytes(total_shape));
        let full_output = Buffer::new_zeroed(&device, config.output_bytes(total_shape));
        let short_routed_hidden = Buffer::new_zeroed(&device, config.routed_output_bytes(active_shape));
        let short_routed_probs = Buffer::new_zeroed(&device, config.routed_probs_bytes(active_shape));
        let short_shared_hidden = Buffer::new_zeroed(&device, config.output_bytes(active_shape));
        let short_gate_logits = Buffer::new_zeroed(&device, config.shared_expert_gate_logits_bytes(active_shape));
        let short_output = Buffer::new_zeroed(&device, config.output_bytes(active_shape));
        let kernels = MoECombineKernels::new(&device, config);

        for buffers in [
            MoECombineWithoutSharedExpertsBuffers {
                routed_hidden: &short_routed_hidden,
                routed_probs: &full_routed_probs,
                output: &full_output,
            },
            MoECombineWithoutSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &short_routed_probs,
                output: &full_output,
            },
            MoECombineWithoutSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &full_routed_probs,
                output: &short_output,
            },
        ] {
            assert_panics(|| {
                let mut builder = stream.create_replay_program();
                builder.record(kernels.invoke_without_shared_experts_bucketed(total_shape, NUM_ACTIVE_TOKENS, buffers));
            });
        }

        for buffers in [
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &short_routed_hidden,
                routed_probs: &full_routed_probs,
                shared_hidden: &full_shared_hidden,
                shared_expert_gate_logits: &full_gate_logits,
                output: &full_output,
            },
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &short_routed_probs,
                shared_hidden: &full_shared_hidden,
                shared_expert_gate_logits: &full_gate_logits,
                output: &full_output,
            },
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &full_routed_probs,
                shared_hidden: &short_shared_hidden,
                shared_expert_gate_logits: &full_gate_logits,
                output: &full_output,
            },
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &full_routed_probs,
                shared_hidden: &full_shared_hidden,
                shared_expert_gate_logits: &short_gate_logits,
                output: &full_output,
            },
            MoECombineWithSharedExpertsBuffers {
                routed_hidden: &full_routed_hidden,
                routed_probs: &full_routed_probs,
                shared_hidden: &full_shared_hidden,
                shared_expert_gate_logits: &full_gate_logits,
                output: &short_output,
            },
        ] {
            assert_panics(|| {
                let mut builder = stream.create_replay_program();
                builder.record(kernels.invoke_with_shared_experts_bucketed(total_shape, NUM_ACTIVE_TOKENS, buffers));
            });
        }
    }

    fn run_bucketed_capacity_case(with_shared_experts: bool) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = MoECombineConfig::bf16(2, 3);
        let shape = MoECombineShape { num_tokens: 4 };
        let num_total_tokens = shape.num_tokens as usize;
        let num_active_tokens = 3_usize;
        let topk = config.num_experts_per_token as usize;
        let hidden_dim = config.hidden_dim as usize;
        let all_routed_hidden = generated_values(num_total_tokens * topk * hidden_dim, 0x2148_937A);
        let all_routed_probs = generated_probs(num_total_tokens, topk, 0x672D_A9B4);
        let all_shared_hidden = generated_values(num_total_tokens * hidden_dim, 0x153F_72C8);
        let all_gate_logits = generated_values(num_total_tokens, 0xB307_4D16);
        let active_routes = num_active_tokens * topk;
        let active_routed_values = active_routes * hidden_dim;
        let active_output_values = num_active_tokens * hidden_dim;
        let mut active_routed_hidden = all_routed_hidden.clone();
        active_routed_hidden[active_routed_values..].fill(f32::NAN);
        let mut active_routed_probs = all_routed_probs.clone();
        active_routed_probs[active_routes..].fill(f32::NAN);
        let mut active_shared_hidden = all_shared_hidden.clone();
        active_shared_hidden[active_output_values..].fill(f32::NAN);
        let mut active_gate_logits = all_gate_logits.clone();
        active_gate_logits[num_active_tokens..].fill(f32::NAN);

        let routed_hidden = bf16_buffer(&device, &active_routed_hidden);
        let routed_probs = Buffer::from_slice(&device, &active_routed_probs);
        let shared_hidden = bf16_buffer(&device, &active_shared_hidden);
        let gate_logits = bf16_buffer(&device, &active_gate_logits);
        let output_sentinel = bf16::from_f32(91.0).to_bits();
        let output = Buffer::from_slice(&device, &vec![output_sentinel; num_total_tokens * hidden_dim]);
        let kernels = MoECombineKernels::new(&device, config);
        let mut builder = stream.create_replay_program();
        if with_shared_experts {
            builder.record(kernels.invoke_with_shared_experts_bucketed(
                shape,
                NUM_ACTIVE_TOKENS,
                MoECombineWithSharedExpertsBuffers {
                    routed_hidden: &routed_hidden,
                    routed_probs: &routed_probs,
                    shared_hidden: &shared_hidden,
                    shared_expert_gate_logits: &gate_logits,
                    output: &output,
                },
            ));
        } else {
            builder.record(kernels.invoke_without_shared_experts_bucketed(
                shape,
                NUM_ACTIVE_TOKENS,
                MoECombineWithoutSharedExpertsBuffers {
                    routed_hidden: &routed_hidden,
                    routed_probs: &routed_probs,
                    output: &output,
                },
            ));
        }
        let replay = builder.build();
        assert_eq!(replay.stats().parameter_count, 1);

        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
            )
            .wait();
        let expected_active = expected_output(
            &all_routed_hidden,
            &all_routed_probs,
            &all_shared_hidden,
            &all_gate_logits,
            num_active_tokens,
            topk,
            hidden_dim,
            with_shared_experts,
        );
        let first = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
        assert_close_bits(&first[..active_output_values], &expected_active, 1.0e-3);
        assert_eq!(
            &first[active_output_values..],
            &vec![output_sentinel; hidden_dim],
            "inactive output tail must preserve its canary"
        );

        write_bf16_values(&routed_hidden, &all_routed_hidden);
        routed_probs.write_typed(0, &all_routed_probs);
        write_bf16_values(&shared_hidden, &all_shared_hidden);
        write_bf16_values(&gate_logits, &all_gate_logits);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, shape.num_tokens),
            )
            .wait();
        let expected_full = expected_output(
            &all_routed_hidden,
            &all_routed_probs,
            &all_shared_hidden,
            &all_gate_logits,
            num_total_tokens,
            topk,
            hidden_dim,
            with_shared_experts,
        );
        let full = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
        assert_close_bits(&full, &expected_full, 1.0e-3);

        write_bf16_values(&routed_hidden, &active_routed_hidden);
        routed_probs.write_typed(0, &active_routed_probs);
        write_bf16_values(&shared_hidden, &active_shared_hidden);
        write_bf16_values(&gate_logits, &active_gate_logits);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
            )
            .wait();
        let shrunk = output.read_typed::<u16>(0, num_total_tokens * hidden_dim);
        assert_close_bits(&shrunk[..active_output_values], &expected_active, 1.0e-3);
        assert_eq!(
            &shrunk[active_output_values..],
            &full[active_output_values..],
            "shrinking the active prefix must not rewrite the previous full tail"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_output(
        routed_hidden: &[f32],
        routed_probs: &[f32],
        shared_hidden: &[f32],
        gate_logits: &[f32],
        num_tokens: usize,
        topk: usize,
        hidden_dim: usize,
        with_shared_experts: bool,
    ) -> Vec<u16> {
        let num_routes = num_tokens * topk;
        let routed = moe_combine_without_shared_experts_bf16_reference(
            &routed_hidden[..num_routes * hidden_dim],
            &routed_probs[..num_routes],
            num_tokens,
            topk,
            hidden_dim,
        );
        if with_shared_experts {
            moe_combine_with_shared_experts_bf16_reference(
                &routed,
                &shared_hidden[..num_tokens * hidden_dim],
                &gate_logits[..num_tokens],
                num_tokens,
                hidden_dim,
            )
        } else {
            routed
        }
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        buffer.write_typed(0, &bits);
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

    fn assert_panics(f: impl FnOnce()) {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    }
}
