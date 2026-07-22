use std::mem::size_of;

use super::assert_u32_count_domain;
use super::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Dtype;
use crate::metal::Kernel;
use crate::metal::Operator;

const MOE_EXPERT_MAJOR_SOURCE: &str = include_str!("metal/moe_expert_major.metal");

#[derive(Clone, Copy, Debug)]
pub struct MoEExpertMajorShape {
    pub num_tokens: u32,
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl MoEExpertMajorShape {
    pub fn bf16(num_tokens: u32, num_experts: u32, num_experts_per_token: u32, hidden_dim: u32) -> Self {
        Self {
            num_tokens,
            num_experts,
            num_experts_per_token,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_experts > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(self.num_experts_per_token <= self.num_experts);
        assert!(self.hidden_dim > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
        self.num_routes();
        assert_u32_count_domain(
            self.num_route_hidden_elements(),
            "MoE expert-major routed-hidden elements",
        );
        assert_u32_count_domain(
            self.num_token_hidden_elements(),
            "MoE expert-major token-hidden elements",
        );
    }

    pub fn num_routes(self) -> u32 {
        self.num_tokens
            .checked_mul(self.num_experts_per_token)
            .expect("MoE expert-major route count must fit u32")
    }

    fn num_route_hidden_elements(self) -> usize {
        checked_product(
            "MoE expert-major routed-hidden element count",
            &[self.num_routes() as usize, self.hidden_dim as usize],
        )
    }

    fn num_token_hidden_elements(self) -> usize {
        checked_product(
            "MoE expert-major token-hidden element count",
            &[self.num_tokens as usize, self.hidden_dim as usize],
        )
    }

    pub fn route_indices_bytes(self) -> usize {
        checked_product(
            "MoE expert-major route-index byte length",
            &[self.num_routes() as usize, size_of::<u32>()],
        )
    }

    pub fn expert_counts_bytes(self) -> usize {
        checked_product(
            "MoE expert-major expert-count byte length",
            &[self.num_experts as usize, size_of::<u32>()],
        )
    }

    pub fn expert_offsets_bytes(self) -> usize {
        checked_product(
            "MoE expert-major expert-offset byte length",
            &[self.num_experts as usize + 1, size_of::<u32>()],
        )
    }

    pub fn route_probs_bytes(self) -> usize {
        checked_product(
            "MoE expert-major route-probability byte length",
            &[self.num_routes() as usize, size_of::<f32>()],
        )
    }

    pub fn route_hidden_bytes(self) -> usize {
        checked_product(
            "MoE expert-major routed-hidden byte length",
            &[self.num_route_hidden_elements(), self.dtype.item_size()],
        )
    }

    pub fn token_hidden_bytes(self) -> usize {
        checked_product(
            "MoE expert-major token-hidden byte length",
            &[self.num_token_hidden_elements(), self.dtype.item_size()],
        )
    }

    pub fn common_gate_logits_bytes(self) -> usize {
        checked_product(
            "MoE expert-major common-gate byte length",
            &[self.num_tokens as usize, self.dtype.item_size()],
        )
    }
}

pub struct MoEExpertMajorLayoutBuffers<'a> {
    pub expert_indices: &'a Buffer,
    pub expert_counts: &'a Buffer,
    pub expert_offsets: &'a Buffer,
    pub expert_cursors: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub experts_by_route: &'a Buffer,
}

pub struct MoEExpertMajorPackInputBuffers<'a> {
    pub input: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub packed_input: &'a Buffer,
}

pub struct MoEExpertMajorScatterWithoutCommonBuffers<'a> {
    pub route_output: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoEExpertMajorScatterWithCommonBuffers<'a> {
    pub route_output: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub common_hidden: &'a Buffer,
    pub common_gate_logits: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct MoEExpertMajorKernels {
    layout_clear: Kernel,
    layout_count: Kernel,
    layout_prefix: Kernel,
    layout_scatter: Kernel,
    pack_input: Kernel,
    scatter_without_common: Kernel,
    scatter_with_common: Kernel,
}

impl MoEExpertMajorKernels {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            layout_clear: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_clear"),
            layout_count: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_count"),
            layout_prefix: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_prefix"),
            layout_scatter: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_scatter"),
            pack_input: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_pack_input"),
            scatter_without_common: Kernel::new(
                device,
                MOE_EXPERT_MAJOR_SOURCE,
                "moe_expert_major_scatter_without_common",
            ),
            scatter_with_common: Kernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_scatter_with_common"),
        }
    }

    pub fn invoke_layout<'a>(
        &'a self,
        shape: MoEExpertMajorShape,
        buffers: MoEExpertMajorLayoutBuffers<'a>,
    ) -> MoEExpertMajorLayoutInvocation<'a> {
        MoEExpertMajorLayoutInvocation {
            kernels: self,
            shape,
            buffers,
        }
    }

    pub fn invoke_pack_input<'a>(
        &'a self,
        shape: MoEExpertMajorShape,
        buffers: MoEExpertMajorPackInputBuffers<'a>,
    ) -> MoEExpertMajorPackInputInvocation<'a> {
        MoEExpertMajorPackInputInvocation {
            kernels: self,
            shape,
            buffers,
        }
    }

    pub fn invoke_scatter_without_common<'a>(
        &'a self,
        shape: MoEExpertMajorShape,
        buffers: MoEExpertMajorScatterWithoutCommonBuffers<'a>,
    ) -> MoEExpertMajorScatterWithoutCommonInvocation<'a> {
        MoEExpertMajorScatterWithoutCommonInvocation {
            kernels: self,
            shape,
            buffers,
        }
    }

    pub fn invoke_scatter_with_common<'a>(
        &'a self,
        shape: MoEExpertMajorShape,
        buffers: MoEExpertMajorScatterWithCommonBuffers<'a>,
    ) -> MoEExpertMajorScatterWithCommonInvocation<'a> {
        MoEExpertMajorScatterWithCommonInvocation {
            kernels: self,
            shape,
            buffers,
        }
    }
}

pub struct MoEExpertMajorLayoutInvocation<'a> {
    kernels: &'a MoEExpertMajorKernels,
    shape: MoEExpertMajorShape,
    buffers: MoEExpertMajorLayoutBuffers<'a>,
}

impl Operator for MoEExpertMajorLayoutInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_layout_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernels.layout_clear);
        builder.set_buffer_write(0, self.buffers.expert_counts, 0);
        builder.set_buffer_write(1, self.buffers.expert_cursors, 0);
        builder.set_u32(2, self.shape.num_experts);
        builder.dispatch_1d(self.shape.num_experts as usize, 256);

        builder.set_kernel(&self.kernels.layout_count);
        builder.set_barrier_before();
        builder.set_buffer_read(0, self.buffers.expert_indices, 0);
        builder.set_buffer_read_write(1, self.buffers.expert_counts, 0);
        builder.set_u32(2, self.shape.num_routes());
        builder.set_u32(3, self.shape.num_experts);
        builder.dispatch_1d(self.shape.num_routes() as usize, 256);

        builder.set_kernel(&self.kernels.layout_prefix);
        builder.set_barrier_before();
        builder.set_buffer_read(0, self.buffers.expert_counts, 0);
        builder.set_buffer_write(1, self.buffers.expert_offsets, 0);
        builder.set_buffer_write(2, self.buffers.expert_cursors, 0);
        builder.set_u32(3, self.shape.num_experts);
        builder.dispatch_1d(1, 1);

        builder.set_kernel(&self.kernels.layout_scatter);
        builder.set_barrier_before();
        builder.set_buffer_read(0, self.buffers.expert_indices, 0);
        builder.set_buffer_read_write(1, self.buffers.expert_cursors, 0);
        builder.set_buffer_write(2, self.buffers.routes_by_expert, 0);
        builder.set_buffer_write(3, self.buffers.routes_by_token, 0);
        builder.set_buffer_write(4, self.buffers.experts_by_route, 0);
        builder.set_u32(5, self.shape.num_routes());
        builder.set_u32(6, self.shape.num_experts);
        builder.dispatch_1d(self.shape.num_routes() as usize, 256);
    }
}

pub struct MoEExpertMajorPackInputInvocation<'a> {
    kernels: &'a MoEExpertMajorKernels,
    shape: MoEExpertMajorShape,
    buffers: MoEExpertMajorPackInputBuffers<'a>,
}

impl Operator for MoEExpertMajorPackInputInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_pack_input_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernels.pack_input);
        builder.set_buffer_read(0, self.buffers.input, 0);
        builder.set_buffer_read(1, self.buffers.routes_by_expert, 0);
        builder.set_buffer_write(2, self.buffers.packed_input, 0);
        builder.set_u32(3, self.shape.num_routes());
        builder.set_u32(4, self.shape.num_experts_per_token);
        builder.set_u32(5, self.shape.hidden_dim);
        builder.dispatch_1d(self.shape.num_route_hidden_elements(), 256);
    }
}

pub struct MoEExpertMajorScatterWithoutCommonInvocation<'a> {
    kernels: &'a MoEExpertMajorKernels,
    shape: MoEExpertMajorShape,
    buffers: MoEExpertMajorScatterWithoutCommonBuffers<'a>,
}

impl Operator for MoEExpertMajorScatterWithoutCommonInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_scatter_without_common_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernels.scatter_without_common);
        builder.set_buffer_read(0, self.buffers.route_output, 0);
        builder.set_buffer_read(1, self.buffers.routes_by_token, 0);
        builder.set_buffer_read(2, self.buffers.routed_probs, 0);
        builder.set_buffer_write(3, self.buffers.output, 0);
        builder.set_u32(4, self.shape.num_tokens);
        builder.set_u32(5, self.shape.num_experts_per_token);
        builder.set_u32(6, self.shape.hidden_dim);
        builder.dispatch_1d(self.shape.num_token_hidden_elements(), 256);
    }
}

pub struct MoEExpertMajorScatterWithCommonInvocation<'a> {
    kernels: &'a MoEExpertMajorKernels,
    shape: MoEExpertMajorShape,
    buffers: MoEExpertMajorScatterWithCommonBuffers<'a>,
}

impl Operator for MoEExpertMajorScatterWithCommonInvocation<'_> {
    fn record(self, builder: &CommandRecorder<'_>) {
        self.shape.validate();
        debug_validate_scatter_with_common_buffers(self.shape, &self.buffers);
        builder.set_kernel(&self.kernels.scatter_with_common);
        builder.set_buffer_read(0, self.buffers.route_output, 0);
        builder.set_buffer_read(1, self.buffers.routes_by_token, 0);
        builder.set_buffer_read(2, self.buffers.routed_probs, 0);
        builder.set_buffer_read(3, self.buffers.common_hidden, 0);
        builder.set_buffer_read(4, self.buffers.common_gate_logits, 0);
        builder.set_buffer_write(5, self.buffers.output, 0);
        builder.set_u32(6, self.shape.num_tokens);
        builder.set_u32(7, self.shape.num_experts_per_token);
        builder.set_u32(8, self.shape.hidden_dim);
        builder.dispatch_1d(self.shape.num_token_hidden_elements(), 256);
    }
}

fn debug_validate_layout_buffers(shape: MoEExpertMajorShape, buffers: &MoEExpertMajorLayoutBuffers<'_>) {
    #[cfg(debug_assertions)]
    validate_layout_buffers(shape, buffers);
}

fn debug_validate_pack_input_buffers(shape: MoEExpertMajorShape, buffers: &MoEExpertMajorPackInputBuffers<'_>) {
    #[cfg(debug_assertions)]
    validate_pack_input_buffers(shape, buffers);
}

fn debug_validate_scatter_without_common_buffers(
    shape: MoEExpertMajorShape,
    buffers: &MoEExpertMajorScatterWithoutCommonBuffers<'_>,
) {
    #[cfg(debug_assertions)]
    validate_scatter_without_common_buffers(shape, buffers);
}

fn debug_validate_scatter_with_common_buffers(
    shape: MoEExpertMajorShape,
    buffers: &MoEExpertMajorScatterWithCommonBuffers<'_>,
) {
    #[cfg(debug_assertions)]
    validate_scatter_with_common_buffers(shape, buffers);
}

fn validate_layout_buffers(shape: MoEExpertMajorShape, buffers: &MoEExpertMajorLayoutBuffers<'_>) {
    let bytes = shape.route_indices_bytes();
    assert!(buffers.expert_indices.len_bytes() >= bytes);
    assert!(buffers.expert_counts.len_bytes() >= shape.expert_counts_bytes());
    assert!(buffers.expert_offsets.len_bytes() >= shape.expert_offsets_bytes());
    assert!(buffers.expert_cursors.len_bytes() >= shape.expert_counts_bytes());
    assert!(buffers.routes_by_expert.len_bytes() >= bytes);
    assert!(buffers.routes_by_token.len_bytes() >= bytes);
    assert!(buffers.experts_by_route.len_bytes() >= bytes);
}

fn validate_pack_input_buffers(shape: MoEExpertMajorShape, buffers: &MoEExpertMajorPackInputBuffers<'_>) {
    assert!(buffers.input.len_bytes() >= shape.token_hidden_bytes());
    assert!(buffers.routes_by_expert.len_bytes() >= shape.route_indices_bytes());
    assert!(buffers.packed_input.len_bytes() >= shape.route_hidden_bytes());
}

fn validate_scatter_without_common_buffers(
    shape: MoEExpertMajorShape,
    buffers: &MoEExpertMajorScatterWithoutCommonBuffers<'_>,
) {
    assert!(buffers.route_output.len_bytes() >= shape.route_hidden_bytes());
    assert!(buffers.routes_by_token.len_bytes() >= shape.route_indices_bytes());
    assert!(buffers.routed_probs.len_bytes() >= shape.route_probs_bytes());
    assert!(buffers.output.len_bytes() >= shape.token_hidden_bytes());
}

fn validate_scatter_with_common_buffers(
    shape: MoEExpertMajorShape,
    buffers: &MoEExpertMajorScatterWithCommonBuffers<'_>,
) {
    assert!(buffers.route_output.len_bytes() >= shape.route_hidden_bytes());
    assert!(buffers.routes_by_token.len_bytes() >= shape.route_indices_bytes());
    assert!(buffers.routed_probs.len_bytes() >= shape.route_probs_bytes());
    assert!(buffers.common_hidden.len_bytes() >= shape.token_hidden_bytes());
    assert!(buffers.common_gate_logits.len_bytes() >= shape.common_gate_logits_bytes());
    assert!(buffers.output.len_bytes() >= shape.token_hidden_bytes());
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "MoE expert-major routed-hidden elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        MoEExpertMajorShape::bf16(1 << 30, 1, 1, 4).validate();
    }

    #[test]
    fn test_layout_pack_scatter() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = MoEExpertMajorShape::bf16(4, 6, 3, 3);
        let input_values = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, -1.0, -2.0, -3.0,
        ];
        let expert_indices_values = [5_u32, 2, 2, 0, 5, 2, 3, 2, 5, 0, 2, 5];
        let routed_probs_values = [
            0.25_f32, 0.50, 0.25, //
            0.125, 0.625, 0.25, //
            0.75, 0.125, 0.125, //
            0.20, 0.30, 0.50,
        ];
        let common_hidden_values = [
            0.5, 1.0, -0.5, //
            1.5, -1.0, 0.25, //
            -0.25, 0.75, 1.25, //
            2.0, -1.5, 0.5,
        ];
        let common_gate_logits_values = [-1.0, 0.0, 1.0, 2.0];
        let input = bf16_buffer(&device, &input_values);
        let expert_indices = Buffer::from_slice(&device, &expert_indices_values);
        let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
        let common_hidden = bf16_buffer(&device, &common_hidden_values);
        let common_gate_logits = bf16_buffer(&device, &common_gate_logits_values);
        let expert_counts = Buffer::new_zeroed(&device, shape.expert_counts_bytes());
        let expert_offsets = Buffer::new_zeroed(&device, shape.expert_offsets_bytes());
        let expert_cursors = Buffer::new_zeroed(&device, shape.expert_counts_bytes());
        let routes_by_expert = Buffer::new_zeroed(&device, shape.route_indices_bytes());
        let routes_by_token = Buffer::new_zeroed(&device, shape.route_indices_bytes());
        let experts_by_route = Buffer::new_zeroed(&device, shape.route_indices_bytes());
        let packed_input = Buffer::new_zeroed(&device, shape.route_hidden_bytes());
        let output = Buffer::new_zeroed(&device, shape.token_hidden_bytes());
        let output_with_common = Buffer::new_zeroed(&device, shape.token_hidden_bytes());
        let kernels = MoEExpertMajorKernels::new(&device);

        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_layout(
            shape,
            MoEExpertMajorLayoutBuffers {
                expert_indices: &expert_indices,
                expert_counts: &expert_counts,
                expert_offsets: &expert_offsets,
                expert_cursors: &expert_cursors,
                routes_by_expert: &routes_by_expert,
                routes_by_token: &routes_by_token,
                experts_by_route: &experts_by_route,
            },
        ));
        builder.record_with_barrier_before(kernels.invoke_pack_input(
            shape,
            MoEExpertMajorPackInputBuffers {
                input: &input,
                routes_by_expert: &routes_by_expert,
                packed_input: &packed_input,
            },
        ));
        builder.record_with_barrier_before(kernels.invoke_scatter_without_common(
            shape,
            MoEExpertMajorScatterWithoutCommonBuffers {
                route_output: &packed_input,
                routes_by_token: &routes_by_token,
                routed_probs: &routed_probs,
                output: &output,
            },
        ));
        builder.record_with_barrier_before(kernels.invoke_scatter_with_common(
            shape,
            MoEExpertMajorScatterWithCommonBuffers {
                route_output: &packed_input,
                routes_by_token: &routes_by_token,
                routed_probs: &routed_probs,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &output_with_common,
            },
        ));
        let replay = builder.build();
        stream.submit_replay(&replay).wait();

        let routes_by_expert_values = routes_by_expert.read_typed::<u32>(0, 12);
        let routes_by_token_values = routes_by_token.read_typed::<u32>(0, 12);
        let experts_by_route_values = experts_by_route.read_typed::<u32>(0, 12);
        assert_eq!(expert_counts.read_typed::<u32>(0, 6), vec![2, 0, 5, 1, 0, 4]);
        assert_eq!(expert_offsets.read_typed::<u32>(0, 7), vec![0, 2, 2, 7, 8, 8, 12]);
        assert_expert_major_maps(
            &expert_indices_values,
            &routes_by_expert_values,
            &routes_by_token_values,
            &experts_by_route_values,
        );
        assert_packed_input_matches_routes(
            &input_values,
            &packed_input.read_typed::<u16>(0, 36),
            &routes_by_expert_values,
            3,
            3,
        );
        let expected = cpu_scatter(&input_values, &routed_probs_values, 4, 3, 3);
        assert_eq!(output.read_typed::<u16>(0, 12), expected);
        let expected_with_common =
            cpu_scatter_with_common(&expected, &common_hidden_values, &common_gate_logits_values, 4, 3);
        assert_eq!(output_with_common.read_typed::<u16>(0, 12), expected_with_common);
    }

    fn cpu_scatter(input: &[f32], probs: &[f32], num_tokens: usize, topk: usize, hidden: usize) -> Vec<u16> {
        let mut out = Vec::new();
        for token in 0..num_tokens {
            for dim in 0..hidden {
                let mut acc = 0.0_f32;
                for slot in 0..topk {
                    let route = token * topk + slot;
                    let route_weight = bf16::from_f32(probs[route]).to_f32();
                    let hidden_value = bf16::from_f32(input[token * hidden + dim]).to_f32();
                    let weighted = bf16::from_f32(route_weight * hidden_value).to_f32();
                    acc = bf16::from_f32(acc + weighted).to_f32();
                }
                out.push(bf16::from_f32(acc).to_bits());
            }
        }
        out
    }

    fn assert_expert_major_maps(
        expert_indices: &[u32],
        routes_by_expert: &[u32],
        routes_by_token: &[u32],
        experts_by_route: &[u32],
    ) {
        for (expert_route, original_route) in routes_by_expert.iter().enumerate() {
            let original_route = *original_route as usize;
            assert_eq!(routes_by_token[original_route] as usize, expert_route);
            assert_eq!(experts_by_route[expert_route], expert_indices[original_route]);
        }
    }

    fn assert_packed_input_matches_routes(
        input: &[f32],
        packed_input: &[u16],
        routes_by_expert: &[u32],
        topk: usize,
        hidden: usize,
    ) {
        for (expert_route, original_route) in routes_by_expert.iter().enumerate() {
            let token = *original_route as usize / topk;
            for dim in 0..hidden {
                assert_eq!(
                    packed_input[expert_route * hidden + dim],
                    bf16::from_f32(input[token * hidden + dim]).to_bits()
                );
            }
        }
    }

    fn cpu_scatter_with_common(
        routed_output: &[u16],
        common_hidden: &[f32],
        common_gate_logits: &[f32],
        num_tokens: usize,
        hidden: usize,
    ) -> Vec<u16> {
        let mut out = Vec::new();
        for (token, gate_logit) in common_gate_logits.iter().enumerate().take(num_tokens) {
            let gate_logit = bf16::from_f32(*gate_logit).to_f32();
            let common_gate = 1.0 / (1.0 + (-gate_logit).exp());
            for dim in 0..hidden {
                let gid = token * hidden + dim;
                let routed = bf16::from_bits(routed_output[gid]).to_f32();
                let common = bf16::from_f32(common_hidden[gid]).to_f32();
                out.push(bf16::from_f32(routed + common_gate * common).to_bits());
            }
        }
        out
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }
}
