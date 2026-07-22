use std::mem::size_of;

use super::assert_u32_index_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Kernel;
use crate::metal::Operator;

const MOE_ROUTING_SOURCE: &str = include_str!("metal/moe_routing.metal");

/// Routes each token to its top-k experts from router probabilities.
///
/// The caller owns the preceding softmax stage. This kernel selects top-k
/// experts by the bf16 softmax probabilities and optionally renormalizes the
/// selected probabilities across the top-k set.
#[derive(Clone, Copy, Debug)]
pub struct MoERoutingShape {
    pub num_tokens: u32,
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub norm_topk_prob: bool,
}

impl MoERoutingShape {
    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_experts > 0);
        assert!(self.num_experts <= 256);
        assert!(self.num_experts_per_token > 0);
        assert!(self.num_experts_per_token <= self.num_experts);
        assert!(self.num_experts_per_token <= 16);
        assert_u32_index_domain(self.num_router_prob_elements(), "MoE routing probability elements");
        assert_u32_index_domain(self.num_routes(), "MoE routing routes");
    }

    pub fn num_routes(self) -> usize {
        checked_product(
            "MoE routing route count",
            &[self.num_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_router_prob_elements(self) -> usize {
        checked_product(
            "MoE routing probability element count",
            &[self.num_tokens as usize, self.num_experts as usize],
        )
    }

    pub fn router_probs_bytes(self) -> usize {
        checked_product(
            "MoE routing probability byte length",
            &[self.num_router_prob_elements(), size_of::<u16>()],
        )
    }

    pub fn expert_indices_bytes(self) -> usize {
        checked_product(
            "MoE routing expert-index byte length",
            &[self.num_routes(), size_of::<u32>()],
        )
    }

    pub fn expert_probs_bytes(self) -> usize {
        checked_product(
            "MoE routing expert-probability byte length",
            &[self.num_routes(), size_of::<f32>()],
        )
    }
}

pub struct MoERoutingBuffers<'a> {
    pub router_probs: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub expert_probs: &'a Buffer,
}

pub struct MoERoutingKernel {
    kernel: Kernel,
}

impl MoERoutingKernel {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            kernel: Kernel::new(device, MOE_ROUTING_SOURCE, "moe_route_topk"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: MoERoutingShape, buffers: MoERoutingBuffers<'a>) -> MoERoutingInvocation<'a> {
        MoERoutingInvocation {
            kernel: self,
            shape,
            buffers,
        }
    }
}

pub struct MoERoutingInvocation<'a> {
    kernel: &'a MoERoutingKernel,
    shape: MoERoutingShape,
    buffers: MoERoutingBuffers<'a>,
}

impl Operator for MoERoutingInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernel.kernel);
        builder.set_buffer_read(0, self.buffers.router_probs, 0);
        builder.set_buffer_write(1, self.buffers.expert_indices, 0);
        builder.set_buffer_write(2, self.buffers.expert_probs, 0);
        builder.set_u32(3, self.shape.num_tokens);
        builder.set_u32(4, self.shape.num_experts);
        builder.set_u32(5, self.shape.num_experts_per_token);
        builder.set_u32(6, u32::from(self.shape.norm_topk_prob));
        builder.dispatch_threadblocks((self.shape.num_tokens as usize, 1, 1), (256, 1, 1));
    }
}

fn debug_validate_buffers(shape: MoERoutingShape, buffers: &MoERoutingBuffers<'_>) {
    #[cfg(debug_assertions)]
    validate_buffers(shape, buffers);
}

fn validate_buffers(shape: MoERoutingShape, buffers: &MoERoutingBuffers<'_>) {
    let router_probs_bytes = shape.router_probs_bytes();
    let expert_indices_bytes = shape.expert_indices_bytes();
    let expert_probs_bytes = shape.expert_probs_bytes();
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
mod tests {
    use std::mem::size_of;

    use half::bf16;
    use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "MoE routing probability elements exceeds the shader u32 element-index domain")]
    fn test_shape_rejects_shader_index_overflow() {
        MoERoutingShape {
            num_tokens: (u32::MAX / 256) + 2,
            num_experts: 256,
            num_experts_per_token: 1,
            norm_topk_prob: false,
        }
        .validate();
    }

    #[test]
    fn test_topk_renorm() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoERoutingShape {
            num_tokens: 2,
            num_experts: 4,
            num_experts_per_token: 2,
            norm_topk_prob: true,
        };
        let router_probs_values = [
            softmax_prob(0.25, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(2.0, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(-1.0, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(1.0, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(3.0, &[3.0, 3.0, 0.5, -2.0]),
            softmax_prob(3.0, &[3.0, 3.0, 0.5, -2.0]),
            softmax_prob(0.5, &[3.0, 3.0, 0.5, -2.0]),
            softmax_prob(-2.0, &[3.0, 3.0, 0.5, -2.0]),
        ];
        let router_probs = bf16_buffer(&device, &router_probs_values);
        let expert_indices = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<u32>());
        let expert_probs = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<f32>());
        let kernel = MoERoutingKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            MoERoutingBuffers {
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let expected = moe_routing_from_bf16_probs_reference(&router_probs_values, 2, 4, 2, true);
        assert_eq!(expert_indices.read_typed::<u32>(0, 4), expected.expert_indices);
        let actual = expert_probs.read_typed::<f32>(0, 4);
        assert_close(&actual, &expected.expert_probs, 1.0e-3);
    }

    #[test]
    fn test_no_topk_renorm() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoERoutingShape {
            num_tokens: 1,
            num_experts: 4,
            num_experts_per_token: 2,
            norm_topk_prob: false,
        };
        let router_probs_values = [
            softmax_prob(0.25, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(2.0, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(-1.0, &[0.25, 2.0, -1.0, 1.0]),
            softmax_prob(1.0, &[0.25, 2.0, -1.0, 1.0]),
        ];
        let router_probs = bf16_buffer(&device, &router_probs_values);
        let expert_indices = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<u32>());
        let expert_probs = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<f32>());
        let kernel = MoERoutingKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            MoERoutingBuffers {
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
            },
        ));
        let program = builder.build();
        let submitted = stream.submit_replay(&program);
        submitted.wait();

        let expected = moe_routing_from_bf16_probs_reference(&router_probs_values, 1, 4, 2, false);
        assert_eq!(expert_indices.read_typed::<u32>(0, 2), expected.expert_indices);
        let actual = expert_probs.read_typed::<f32>(0, 2);
        assert_close(&actual, &expected.expert_probs, 1.0e-6);
    }

    #[test]
    fn test_random() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoERoutingShape {
            num_tokens: 5,
            num_experts: 8,
            num_experts_per_token: 3,
            norm_topk_prob: true,
        };
        let random_seed = 0x91E4_63BA;
        let router_probs_values = generated_probs(shape.num_tokens as usize, shape.num_experts as usize, random_seed);
        let router_probs = bf16_buffer(&device, &router_probs_values);
        let expert_indices = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<u32>());
        let expert_probs = Buffer::new_zeroed(&device, shape.num_routes() * size_of::<f32>());
        let kernel = MoERoutingKernel::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            MoERoutingBuffers {
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
            },
        ));
        let program = builder.build();
        stream.submit_replay(&program).wait();

        let expected = moe_routing_from_bf16_probs_reference(
            &router_probs_values,
            shape.num_tokens as usize,
            shape.num_experts as usize,
            shape.num_experts_per_token as usize,
            shape.norm_topk_prob,
        );
        let actual_probs = expert_probs.read_typed::<f32>(0, shape.num_routes());
        assert_eq!(
            expert_indices.read_typed::<u32>(0, shape.num_routes()),
            expected.expert_indices
        );
        assert_close(&actual_probs, &expected.expert_probs, 1.0e-3);
    }

    fn softmax_prob(logit: f32, all_logits: &[f32]) -> f32 {
        let all_logits: Vec<f32> = all_logits.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
        let max_logit = all_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let all_exp_sum: f32 = all_logits.iter().map(|value| (*value - max_logit).exp()).sum();
        bf16::from_f32((bf16::from_f32(logit).to_f32() - max_logit).exp() / all_exp_sum).to_f32()
    }

    fn generated_probs(num_tokens: usize, num_experts: usize, random_seed: u32) -> Vec<f32> {
        let mut state = random_seed;
        let mut probs = Vec::with_capacity(num_tokens * num_experts);
        for _ in 0..num_tokens {
            let mut row = Vec::with_capacity(num_experts);
            let mut sum = 0.0f32;
            for _ in 0..num_experts {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) as f32 / 16_777_216.0) + 0.01;
                row.push(value);
                sum += value;
            }
            probs.extend(row.into_iter().map(|value| value / sum));
        }
        probs
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
