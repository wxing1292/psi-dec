use half::bf16;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayProgram;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.expert_major.num_active_tokens");
const U32_CANARY: u32 = 0xA5A5_5A5A;
const BF16_CANARY: u16 = 0x42B6;

#[test]
fn test_constants_have_phase_scoped_thread_blocks() {
    let constants = KernelConstants::current();
    assert_eq!(constants.layout_clear.required_threads, 256);
    assert_eq!(constants.layout_count.required_threads, 256);
    assert_eq!(constants.layout_prefix.required_threads, 1);
    assert_eq!(constants.layout_scatter.required_threads, 256);
    assert_eq!(constants.pack_input.required_threads, 256);
    assert_eq!(constants.scatter_output.required_threads, 256);
}

#[test]
#[should_panic(expected = "MoE expert-major routed-hidden elements exceeds the shader u32 count domain")]
fn test_shape_rejects_shader_count_overflow() {
    Config::bf16(1, 1, 4).validate_shape(Shape {
        num_total_tokens: 1 << 30,
    });
}

#[test]
fn test_layout_pack_scatter() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config::bf16(6, 3, 3);
    let shape = Shape { num_total_tokens: 4 };
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
    let shared_hidden_values = [
        0.5, 1.0, -0.5, //
        1.5, -1.0, 0.25, //
        -0.25, 0.75, 1.25, //
        2.0, -1.5, 0.5,
    ];
    let shared_expert_gate_logits_values = [-1.0, 0.0, 1.0, 2.0];
    let input = bf16_buffer(&device, &input_values);
    let expert_indices = Buffer::from_slice(&device, &expert_indices_values);
    let routed_probs = Buffer::from_slice(&device, &routed_probs_values);
    let shared_hidden = bf16_buffer(&device, &shared_hidden_values);
    let shared_expert_gate_logits = bf16_buffer(&device, &shared_expert_gate_logits_values);
    let expert_counts = Buffer::new_zeroed(&device, config.expert_counts_bytes());
    let expert_offsets = Buffer::new_zeroed(&device, config.expert_offsets_bytes());
    let expert_cursors = Buffer::new_zeroed(&device, config.expert_counts_bytes());
    let routes_by_expert = Buffer::new_zeroed(&device, config.route_indices_bytes(shape));
    let routes_by_token = Buffer::new_zeroed(&device, config.route_indices_bytes(shape));
    let experts_by_route = Buffer::new_zeroed(&device, config.route_indices_bytes(shape));
    let packed_input = Buffer::new_zeroed(&device, config.route_hidden_bytes(shape));
    let output = Buffer::new_zeroed(&device, config.token_hidden_bytes(shape));
    let output_with_shared_experts = Buffer::new_zeroed(&device, config.token_hidden_bytes(shape));
    let kernels = Compute::new(&device, config);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_layout(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        LayoutBuffers {
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
        ReplayU32::Fixed(shape.num_total_tokens),
        PackInputBuffers {
            input: &input,
            routes_by_expert: &routes_by_expert,
            packed_input: &packed_input,
        },
    ));
    builder.record_with_barrier_before(kernels.invoke_scatter_without_shared_experts(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        ScatterWithoutSharedExpertsBuffers {
            packed_output: &packed_input,
            routes_by_token: &routes_by_token,
            routed_probs: &routed_probs,
            output: &output,
        },
    ));
    builder.record_with_barrier_before(kernels.invoke_scatter_with_shared_experts(
        shape,
        ReplayU32::Fixed(shape.num_total_tokens),
        ScatterWithSharedExpertsBuffers {
            packed_output: &packed_input,
            routes_by_token: &routes_by_token,
            routed_probs: &routed_probs,
            shared_hidden: &shared_hidden,
            shared_expert_gate_logits: &shared_expert_gate_logits,
            output: &output_with_shared_experts,
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
    let expected_with_shared_experts = cpu_scatter_with_shared_experts(
        &expected,
        &shared_hidden_values,
        &shared_expert_gate_logits_values,
        4,
        3,
    );
    assert_eq!(
        output_with_shared_experts.read_typed::<u16>(0, 12),
        expected_with_shared_experts
    );
}

#[test]
fn test_bucketed_layout_pack_scatter_preserves_inactive_capacity_and_shrink() {
    let fixture = BucketedFixture::new();
    let replay = fixture.bucketed_replay();

    let first = fixture.write_work(5, 1);
    fixture.submit(&replay, 5);
    fixture.assert_active_work(&first, 5);
    fixture.assert_canary_tails(5);

    let full = fixture.write_work(6, 2);
    fixture.submit(&replay, 6);
    fixture.assert_active_work(&full, 6);
    let full_routes_by_expert = fixture
        .routes_by_expert
        .read_typed::<u32>(0, fixture.num_total_routes());
    let full_routes_by_token = fixture.routes_by_token.read_typed::<u32>(0, fixture.num_total_routes());
    let full_experts_by_route = fixture
        .experts_by_route
        .read_typed::<u32>(0, fixture.num_total_routes());
    let full_packed_input = fixture
        .packed_input
        .read_typed::<u16>(0, fixture.num_total_routes() * fixture.hidden_dim());
    let full_output = fixture
        .output
        .read_typed::<u16>(0, fixture.num_total_tokens() * fixture.hidden_dim());
    let full_output_with_shared_experts = fixture
        .output_with_shared_experts
        .read_typed::<u16>(0, fixture.num_total_tokens() * fixture.hidden_dim());

    let smaller = fixture.write_work(5, 3);
    fixture.submit(&replay, 5);
    fixture.assert_active_work(&smaller, 5);
    fixture.assert_preserved_tails(
        5,
        &full_routes_by_expert,
        &full_routes_by_token,
        &full_experts_by_route,
        &full_packed_input,
        &full_output,
        &full_output_with_shared_experts,
    );
}

struct BucketedWork {
    input: Vec<f32>,
    expert_indices: Vec<u32>,
    routed_probs: Vec<f32>,
    shared_hidden: Vec<f32>,
    shared_expert_gate_logits: Vec<f32>,
}

struct BucketedFixture {
    stream: Stream,
    config: Config,
    shape: Shape,
    kernels: Compute,
    input: Buffer,
    expert_indices: Buffer,
    routed_probs: Buffer,
    shared_hidden: Buffer,
    shared_expert_gate_logits: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
    output: Buffer,
    output_with_shared_experts: Buffer,
}

impl BucketedFixture {
    fn new() -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::bf16(6, 3, 3);
        let shape = Shape { num_total_tokens: 6 };
        let num_total_routes = config.num_routes(shape) as usize;
        let num_total_tokens = shape.num_total_tokens as usize;
        let hidden_dim = config.hidden_dim as usize;
        let kernels = Compute::new(&device, config);
        Self {
            input: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            expert_indices: Buffer::new_zeroed(&device, config.route_indices_bytes(shape)),
            routed_probs: Buffer::new_zeroed(&device, config.route_probs_bytes(shape)),
            shared_hidden: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            shared_expert_gate_logits: Buffer::new_zeroed(&device, config.shared_expert_gate_logits_bytes(shape)),
            expert_counts: Buffer::new_zeroed(&device, config.expert_counts_bytes()),
            expert_offsets: Buffer::new_zeroed(&device, config.expert_offsets_bytes()),
            expert_cursors: Buffer::new_zeroed(&device, config.expert_counts_bytes()),
            routes_by_expert: Buffer::from_slice(&device, &vec![U32_CANARY; num_total_routes]),
            routes_by_token: Buffer::from_slice(&device, &vec![U32_CANARY; num_total_routes]),
            experts_by_route: Buffer::from_slice(&device, &vec![U32_CANARY; num_total_routes]),
            packed_input: Buffer::from_slice(&device, &vec![BF16_CANARY; num_total_routes * hidden_dim]),
            output: Buffer::from_slice(&device, &vec![BF16_CANARY; num_total_tokens * hidden_dim]),
            output_with_shared_experts: Buffer::from_slice(&device, &vec![BF16_CANARY; num_total_tokens * hidden_dim]),
            stream,
            config,
            shape,
            kernels,
        }
    }

    fn bucketed_replay(&self) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.kernels.invoke_layout(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            self.layout_buffers(),
        ));
        builder.record_with_barrier_before(self.kernels.invoke_pack_input(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            self.pack_input_buffers(),
        ));
        builder.record_with_barrier_before(self.kernels.invoke_scatter_without_shared_experts(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            self.scatter_without_shared_experts_buffers(),
        ));
        builder.record_with_barrier_before(self.kernels.invoke_scatter_with_shared_experts(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            self.scatter_with_shared_experts_buffers(),
        ));
        builder.build()
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        self.stream
            .submit_replay_with_arguments(
                replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();
    }

    fn write_work(&self, num_active_tokens: usize, generation: u32) -> BucketedWork {
        let mut input = vec![f32::NAN; self.num_total_tokens() * self.hidden_dim()];
        let mut expert_indices = vec![u32::MAX; self.num_total_routes()];
        let mut routed_probs = vec![f32::NAN; self.num_total_routes()];
        let mut shared_hidden = vec![f32::NAN; self.num_total_tokens() * self.hidden_dim()];
        let mut shared_expert_gate_logits = vec![f32::NAN; self.num_total_tokens()];
        for (token, gate_logit) in shared_expert_gate_logits.iter_mut().enumerate().take(num_active_tokens) {
            *gate_logit = generation as f32 * 0.25 + token as f32 * 0.125 - 0.5;
            for dim in 0..self.hidden_dim() {
                let index = token * self.hidden_dim() + dim;
                input[index] = generation as f32 * 3.0 + token as f32 * 0.75 + dim as f32 * 0.25;
                shared_hidden[index] = generation as f32 * -0.5 + token as f32 * 0.375 - dim as f32 * 0.125;
            }
            for slot in 0..self.num_experts_per_token() {
                let route = token * self.num_experts_per_token() + slot;
                expert_indices[route] = ((route as u32).wrapping_add(generation)) % self.config.num_experts;
                routed_probs[route] = [0.25, 0.50, 0.25][slot];
            }
        }
        write_bf16_values(&self.input, &input);
        self.expert_indices.write_typed(0, &expert_indices);
        self.routed_probs.write_typed(0, &routed_probs);
        write_bf16_values(&self.shared_hidden, &shared_hidden);
        write_bf16_values(&self.shared_expert_gate_logits, &shared_expert_gate_logits);
        BucketedWork {
            input,
            expert_indices,
            routed_probs,
            shared_hidden,
            shared_expert_gate_logits,
        }
    }

    fn assert_active_work(&self, work: &BucketedWork, num_active_tokens: usize) {
        let num_active_routes = num_active_tokens * self.num_experts_per_token();
        let routes_by_expert = self.routes_by_expert.read_typed::<u32>(0, self.num_total_routes());
        let routes_by_token = self.routes_by_token.read_typed::<u32>(0, self.num_total_routes());
        let experts_by_route = self.experts_by_route.read_typed::<u32>(0, self.num_total_routes());
        assert_expert_major_maps(
            &work.expert_indices[..num_active_routes],
            &routes_by_expert[..num_active_routes],
            &routes_by_token[..num_active_routes],
            &experts_by_route[..num_active_routes],
        );
        assert_eq!(
            self.expert_counts
                .read_typed::<u32>(0, self.config.num_experts as usize)
                .into_iter()
                .sum::<u32>(),
            num_active_routes as u32
        );
        assert_eq!(
            self.expert_offsets
                .read_typed::<u32>(0, self.config.num_experts as usize + 1)
                .last(),
            Some(&(num_active_routes as u32))
        );
        assert_packed_input_matches_routes(
            &work.input,
            &self
                .packed_input
                .read_typed::<u16>(0, self.num_total_routes() * self.hidden_dim())
                [..num_active_routes * self.hidden_dim()],
            &routes_by_expert[..num_active_routes],
            self.num_experts_per_token(),
            self.hidden_dim(),
        );
        let expected = cpu_scatter(
            &work.input,
            &work.routed_probs,
            num_active_tokens,
            self.num_experts_per_token(),
            self.hidden_dim(),
        );
        let num_active_output_elements = num_active_tokens * self.hidden_dim();
        assert_eq!(
            &self
                .output
                .read_typed::<u16>(0, self.num_total_tokens() * self.hidden_dim())[..num_active_output_elements],
            expected
        );
        let expected_with_shared_experts = cpu_scatter_with_shared_experts(
            &expected,
            &work.shared_hidden,
            &work.shared_expert_gate_logits,
            num_active_tokens,
            self.hidden_dim(),
        );
        assert_eq!(
            &self
                .output_with_shared_experts
                .read_typed::<u16>(0, self.num_total_tokens() * self.hidden_dim())[..num_active_output_elements],
            expected_with_shared_experts
        );
    }

    fn assert_canary_tails(&self, num_active_tokens: usize) {
        let num_active_routes = num_active_tokens * self.num_experts_per_token();
        let num_active_route_elements = num_active_routes * self.hidden_dim();
        let num_active_output_elements = num_active_tokens * self.hidden_dim();
        assert_eq!(
            &self.routes_by_expert.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &vec![U32_CANARY; self.num_total_routes() - num_active_routes]
        );
        assert_eq!(
            &self.routes_by_token.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &vec![U32_CANARY; self.num_total_routes() - num_active_routes]
        );
        assert_eq!(
            &self.experts_by_route.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &vec![U32_CANARY; self.num_total_routes() - num_active_routes]
        );
        assert_eq!(
            &self
                .packed_input
                .read_typed::<u16>(0, self.num_total_routes() * self.hidden_dim())[num_active_route_elements..],
            &vec![BF16_CANARY; self.num_total_routes() * self.hidden_dim() - num_active_route_elements]
        );
        for output in [&self.output, &self.output_with_shared_experts] {
            assert_eq!(
                &output.read_typed::<u16>(0, self.num_total_tokens() * self.hidden_dim())[num_active_output_elements..],
                &vec![BF16_CANARY; self.num_total_tokens() * self.hidden_dim() - num_active_output_elements]
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_preserved_tails(
        &self,
        num_active_tokens: usize,
        routes_by_expert: &[u32],
        routes_by_token: &[u32],
        experts_by_route: &[u32],
        packed_input: &[u16],
        output: &[u16],
        output_with_shared_experts: &[u16],
    ) {
        let num_active_routes = num_active_tokens * self.num_experts_per_token();
        let num_active_route_elements = num_active_routes * self.hidden_dim();
        let num_active_output_elements = num_active_tokens * self.hidden_dim();
        assert_eq!(
            &self.routes_by_expert.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &routes_by_expert[num_active_routes..]
        );
        assert_eq!(
            &self.routes_by_token.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &routes_by_token[num_active_routes..]
        );
        assert_eq!(
            &self.experts_by_route.read_typed::<u32>(0, self.num_total_routes())[num_active_routes..],
            &experts_by_route[num_active_routes..]
        );
        assert_eq!(
            &self
                .packed_input
                .read_typed::<u16>(0, self.num_total_routes() * self.hidden_dim())[num_active_route_elements..],
            &packed_input[num_active_route_elements..]
        );
        assert_eq!(
            &self
                .output
                .read_typed::<u16>(0, self.num_total_tokens() * self.hidden_dim())[num_active_output_elements..],
            &output[num_active_output_elements..]
        );
        assert_eq!(
            &self
                .output_with_shared_experts
                .read_typed::<u16>(0, self.num_total_tokens() * self.hidden_dim())[num_active_output_elements..],
            &output_with_shared_experts[num_active_output_elements..]
        );
    }

    fn layout_buffers(&self) -> LayoutBuffers<'_> {
        LayoutBuffers {
            expert_indices: &self.expert_indices,
            expert_counts: &self.expert_counts,
            expert_offsets: &self.expert_offsets,
            expert_cursors: &self.expert_cursors,
            routes_by_expert: &self.routes_by_expert,
            routes_by_token: &self.routes_by_token,
            experts_by_route: &self.experts_by_route,
        }
    }

    fn pack_input_buffers(&self) -> PackInputBuffers<'_> {
        PackInputBuffers {
            input: &self.input,
            routes_by_expert: &self.routes_by_expert,
            packed_input: &self.packed_input,
        }
    }

    fn scatter_without_shared_experts_buffers(&self) -> ScatterWithoutSharedExpertsBuffers<'_> {
        ScatterWithoutSharedExpertsBuffers {
            packed_output: &self.packed_input,
            routes_by_token: &self.routes_by_token,
            routed_probs: &self.routed_probs,
            output: &self.output,
        }
    }

    fn scatter_with_shared_experts_buffers(&self) -> ScatterWithSharedExpertsBuffers<'_> {
        ScatterWithSharedExpertsBuffers {
            packed_output: &self.packed_input,
            routes_by_token: &self.routes_by_token,
            routed_probs: &self.routed_probs,
            shared_hidden: &self.shared_hidden,
            shared_expert_gate_logits: &self.shared_expert_gate_logits,
            output: &self.output_with_shared_experts,
        }
    }

    fn num_total_tokens(&self) -> usize {
        self.shape.num_total_tokens as usize
    }

    fn num_total_routes(&self) -> usize {
        self.config.num_routes(self.shape) as usize
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token as usize
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim as usize
    }
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

fn cpu_scatter_with_shared_experts(
    routed_output: &[u16],
    shared_hidden: &[f32],
    shared_expert_gate_logits: &[f32],
    num_tokens: usize,
    hidden: usize,
) -> Vec<u16> {
    let mut out = Vec::new();
    for (token, gate_logit) in shared_expert_gate_logits.iter().enumerate().take(num_tokens) {
        let gate_logit = bf16::from_f32(*gate_logit).to_f32();
        let shared_expert_gate = 1.0 / (1.0 + (-gate_logit).exp());
        for dim in 0..hidden {
            let gid = token * hidden + dim;
            let routed = bf16::from_bits(routed_output[gid]).to_f32();
            let shared = bf16::from_f32(shared_hidden[gid]).to_f32();
            out.push(bf16::from_f32(routed + shared_expert_gate * shared).to_bits());
        }
    }
    out
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    Buffer::from_slice(device, &bits)
}

fn write_bf16_values(buffer: &Buffer, values: &[f32]) {
    let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
    buffer.write_typed(0, &bits);
}
