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
            &[shape.num_total_tokens as usize, self.num_experts_per_token as usize],
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
            &[shape.num_total_tokens as usize, self.hidden_dim as usize],
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
            &[shape.num_total_tokens as usize, self.dtype.item_size()],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoECombineShape {
    pub num_total_tokens: u32,
}

impl MoECombineShape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.config;
        config.validate_shape(self.shape);
        debug_validate_without_shared_experts_buffers(config, self.shape, &self.buffers);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        recorder.set_buffer_read(1, self.buffers.routed_probs, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        record_num_active_tokens(recorder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(4, config.num_experts_per_token);
        recorder.set_u32(5, config.hidden_dim);
        recorder.dispatch_1d(config.num_output_elements(self.shape), 256);
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
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.config;
        config.validate_shape(self.shape);
        debug_validate_with_shared_experts_buffers(config, self.shape, &self.buffers);
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.routed_hidden, 0);
        recorder.set_buffer_read(1, self.buffers.routed_probs, 0);
        recorder.set_buffer_read(2, self.buffers.shared_hidden, 0);
        recorder.set_buffer_read(3, self.buffers.shared_expert_gate_logits, 0);
        recorder.set_buffer_write(4, self.buffers.output, 0);
        record_num_active_tokens(recorder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(6, config.num_experts_per_token);
        recorder.set_u32(7, config.hidden_dim);
        recorder.dispatch_1d(config.num_output_elements(self.shape), 256);
    }
}

fn record_num_active_tokens(
    recorder: &CommandRecorder<'_>,
    binding_index: usize,
    num_total_tokens: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_tokens),
        None => recorder.set_u32(binding_index, num_total_tokens),
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
#[path = "moe_combine_test.rs"]
mod tests;
