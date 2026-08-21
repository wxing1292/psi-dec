use std::mem::size_of;

use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const MOE_ROUTING_SOURCE: &str = include_str!("../metal/moe_routing.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current() -> Self {
        Self {
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

/// Routes each token to its top-k experts from router probabilities.
///
/// The caller owns the preceding softmax stage. This kernel selects top-k
/// experts by the bf16 softmax probabilities and optionally renormalizes the
/// selected probabilities across the top-k set.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub norm_topk_prob: bool,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.num_experts <= 256, "MoE routing supports at most 256 experts");
        assert!(self.num_experts_per_token > 0);
        assert!(self.num_experts_per_token <= self.num_experts);
        assert!(
            self.num_experts_per_token <= 16,
            "MoE routing supports at most 16 experts per token"
        );
    }

    pub fn num_routes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        checked_product(
            "MoE routing route count",
            &[shape.num_total_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_router_prob_elements(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        checked_product(
            "MoE routing probability element count",
            &[shape.num_total_tokens as usize, self.num_experts as usize],
        )
    }

    pub fn validate_shape(self, shape: Shape) {
        self.validate();
        shape.validate();
        assert_u32_index_domain(self.num_router_prob_elements(shape), "MoE routing probability elements");
        assert_u32_index_domain(self.num_routes(shape), "MoE routing routes");
    }

    pub fn router_probs_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing probability byte length",
            &[self.num_router_prob_elements(shape), size_of::<u16>()],
        )
    }

    pub fn expert_indices_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing expert-index byte length",
            &[self.num_routes(shape), size_of::<u32>()],
        )
    }

    pub fn expert_probs_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing expert-probability byte length",
            &[self.num_routes(shape), size_of::<f32>()],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
    }
}

pub struct Buffers<'a> {
    pub router_probs: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub expert_probs: &'a Buffer,
}

pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &crate::metal::Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: CompiledKernel::new(device, MOE_ROUTING_SOURCE, "moe_route_topk"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_tokens: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate();
        Invocation {
            kernel: self,
            shape,
            buffers,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }
}

fn active_key(num_total_tokens: u32, num_active_tokens: ReplayU32) -> Option<ReplayParameterKey> {
    match num_active_tokens {
        ReplayU32::Fixed(num_active_tokens) => {
            assert_eq!(num_active_tokens, num_total_tokens);
            None
        },
        ReplayU32::Parameter(key) => Some(key),
    }
}

pub struct Invocation<'a> {
    kernel: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.kernel.config.validate_shape(self.shape);
        debug_validate_buffers(self.kernel.config, self.shape, &self.buffers);
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.router_probs, 0);
        recorder.set_buffer_write(1, self.buffers.expert_indices, 0);
        recorder.set_buffer_write(2, self.buffers.expert_probs, 0);
        match self.num_active_tokens_key {
            Some(key) => recorder.bind_u32(3, key, 1, self.shape.num_total_tokens),
            None => recorder.set_u32(3, self.shape.num_total_tokens),
        }
        recorder.set_u32(4, self.kernel.config.num_experts);
        recorder.set_u32(5, self.kernel.config.num_experts_per_token);
        recorder.set_u32(6, u32::from(self.kernel.config.norm_topk_prob));
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

fn debug_validate_buffers(config: Config, shape: Shape, buffers: &Buffers<'_>) {
    #[cfg(debug_assertions)]
    validate_buffers(config, shape, buffers);
}

fn validate_buffers(config: Config, shape: Shape, buffers: &Buffers<'_>) {
    let router_probs_bytes = config.router_probs_bytes(shape);
    let expert_indices_bytes = config.expert_indices_bytes(shape);
    let expert_probs_bytes = config.expert_probs_bytes(shape);
    assert!(
        buffers.router_probs.len_bytes() >= router_probs_bytes,
        "MoE routing router_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        router_probs_bytes,
        buffers.router_probs.len_bytes()
    );
    assert!(
        buffers.expert_indices.len_bytes() >= expert_indices_bytes,
        "MoE routing expert_indices buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        expert_indices_bytes,
        buffers.expert_indices.len_bytes()
    );
    assert!(
        buffers.expert_probs.len_bytes() >= expert_probs_bytes,
        "MoE routing expert_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        expert_probs_bytes,
        buffers.expert_probs.len_bytes()
    );
}

#[cfg(test)]
#[path = "routing_test.rs"]
mod tests;
