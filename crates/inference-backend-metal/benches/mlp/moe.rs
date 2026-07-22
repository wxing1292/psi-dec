use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::MoECombineKernel;
use inference_backend_metal::components::MoECombineWithCommonBuffers;
use inference_backend_metal::components::MoECombineWithCommonShape;
use inference_backend_metal::components::MoECombineWithoutCommonBuffers;
use inference_backend_metal::components::MoECombineWithoutCommonShape;
use inference_backend_metal::components::MoEExpertMajorKernels;
use inference_backend_metal::components::MoEExpertMajorLayoutBuffers;
use inference_backend_metal::components::MoEExpertMajorPackInputBuffers;
use inference_backend_metal::components::MoEExpertMajorScatterWithCommonBuffers;
use inference_backend_metal::components::MoEExpertMajorShape;
use inference_backend_metal::components::MoERoutingBuffers;
use inference_backend_metal::components::MoERoutingKernel;
use inference_backend_metal::components::MoERoutingShape;
use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPKernels;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::components::QuantizedSparseMLP;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorScratch;
use inference_backend_metal::components::QuantizedSparseMLPExpertMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorScratch;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayProgramBuilder;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;
use inference_backend_metal::operators::SoftmaxKernel;
use inference_backend_metal::operators::SoftmaxShape;

#[path = "../support.rs"]
mod support;
use support::affine_param_fixture;
use support::bf16_buffer;
use support::gate_fixture;
use support::hidden_fixture;
use support::identity_indices;
use support::quantized_weight;
use support::quantized_weight_stack_for_experts;
use support::route_probs_fixture;
use support::token_route_indices;
use support::zero_fixture;

const NUM_EXPERTS: u32 = 256;
const TOPK_EXPERTS: u32 = 8;
const HIDDEN_DIM: u32 = 2048;
const INTERMEDIATE_DIM: u32 = 512;
const GROUP_SIZE: u32 = 64;
const EXPERT_BITS: u32 = 4;
const ROUTER_BITS: u32 = 8;
const MOE_PROFILE: &str = "qwen36-35b-a3b";
const BENCH_TOKENS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const AUTO_EXPERT_MAJOR_MIN_TOKENS: u32 = 32;

fn bench_moe(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/moe");

    for num_tokens in BENCH_TOKENS {
        let routing = RoutingFixture::new(&device, num_tokens);
        group.throughput(Throughput::Elements(num_tokens as u64));
        group.bench_function(
            format!("{MOE_PROFILE}/route/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}"),
            |b| {
                b.iter(|| {
                    routing.replay();
                    black_box(&routing.expert_probs);
                });
            },
        );

        let combine = CombineFixture::new(&device, num_tokens);
        group.throughput(Throughput::Elements(num_tokens as u64 * HIDDEN_DIM as u64));
        group.bench_function(
            format!(
                "{MOE_PROFILE}/combine/without_common/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/hidden{HIDDEN_DIM}"
            ),
            |b| {
                b.iter(|| {
                    combine.replay_without_common();
                    black_box(&combine.output);
                });
            },
        );
        group.bench_function(
            format!("{MOE_PROFILE}/combine/with_common/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/hidden{HIDDEN_DIM}"),
            |b| {
                b.iter(|| {
                    combine.replay_with_common();
                    black_box(&combine.output);
                });
            },
        );
    }

    for num_tokens in BENCH_TOKENS {
        let forward = MoEForwardFixture::new(&device, num_tokens);
        group.throughput(Throughput::Elements(num_tokens as u64 * HIDDEN_DIM as u64));
        group.bench_function(
            format!(
                "{MOE_PROFILE}/forward/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}/intermediate{INTERMEDIATE_DIM}"
            ),
            |b| {
                b.iter(|| {
                    forward.run_auto_replay();
                    black_box(forward.auto_output());
                });
            },
        );
        group.bench_function(
            format!(
                "{MOE_PROFILE}/forward/token_major/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}/intermediate{INTERMEDIATE_DIM}"
            ),
            |b| {
                b.iter(|| {
                    forward.run_token_major_replay();
                    black_box(&forward.replay_output);
                });
            },
        );
        group.bench_function(
            format!(
                "{MOE_PROFILE}/forward/expert_major/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}/intermediate{INTERMEDIATE_DIM}"
            ),
            |b| {
                b.iter(|| {
                    forward.run_expert_major_replay();
                    black_box(&forward.expert_major_output);
                });
            },
        );
    }

    group.finish();
}

struct RoutingFixture {
    stream: Stream,
    kernel: MoERoutingKernel,
    shape: MoERoutingShape,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
    replay: ReplayProgram,
}

impl RoutingFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        let shape = MoERoutingShape {
            num_tokens,
            num_experts: NUM_EXPERTS,
            num_experts_per_token: TOPK_EXPERTS,
            norm_topk_prob: true,
        };
        let router_probs = bf16_buffer(device, &route_probs_fixture(num_tokens as usize, NUM_EXPERTS as usize));
        let expert_indices = Buffer::new_zeroed(device, shape.expert_indices_bytes());
        let expert_probs = Buffer::new_zeroed(device, shape.expert_probs_bytes());
        let stream = Stream::new(device);
        let kernel = MoERoutingKernel::new(device);
        let replay = build_routing_replay(&stream, &kernel, shape, &router_probs, &expert_indices, &expert_probs);
        let fixture = Self {
            stream,
            kernel,
            shape,
            router_probs,
            expert_indices,
            expert_probs,
            replay,
        };
        fixture.replay();
        fixture
    }

    fn replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

struct CombineFixture {
    stream: Stream,
    kernel: MoECombineKernel,
    without_common_shape: MoECombineWithoutCommonShape,
    with_common_shape: MoECombineWithCommonShape,
    routed_hidden: Buffer,
    routed_probs: Buffer,
    common_hidden: Buffer,
    common_gate_logits: Buffer,
    output: Buffer,
    without_common_replay: ReplayProgram,
    with_common_replay: ReplayProgram,
}

impl CombineFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        let without_common_shape = MoECombineWithoutCommonShape::bf16(num_tokens, TOPK_EXPERTS, HIDDEN_DIM);
        let with_common_shape = MoECombineWithCommonShape::bf16(num_tokens, TOPK_EXPERTS, HIDDEN_DIM);
        let routed_hidden = bf16_buffer(
            device,
            &hidden_fixture(num_tokens as usize * TOPK_EXPERTS as usize, HIDDEN_DIM as usize),
        );
        let routed_probs = Buffer::from_slice(device, &route_probs_fixture(num_tokens as usize, TOPK_EXPERTS as usize));
        let common_hidden = bf16_buffer(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let common_gate_logits = bf16_buffer(device, &gate_fixture(num_tokens as usize));
        let output = Buffer::new_zeroed(device, num_tokens as usize * HIDDEN_DIM as usize * size_of::<u16>());
        let stream = Stream::new(device);
        let kernel = MoECombineKernel::new(device);
        let without_common_replay = build_combine_without_common_replay(
            &stream,
            &kernel,
            without_common_shape,
            &routed_hidden,
            &routed_probs,
            &output,
        );
        let with_common_replay = build_combine_with_common_replay(
            &stream,
            &kernel,
            with_common_shape,
            MoECombineWithCommonBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &output,
            },
        );
        let fixture = Self {
            stream,
            kernel,
            without_common_shape,
            with_common_shape,
            routed_hidden,
            routed_probs,
            common_hidden,
            common_gate_logits,
            output,
            without_common_replay,
            with_common_replay,
        };
        fixture.replay_without_common();
        fixture.replay_with_common();
        fixture
    }

    fn replay_without_common(&self) {
        self.stream.submit_replay(&self.without_common_replay).wait();
    }

    fn replay_with_common(&self) {
        self.stream.submit_replay(&self.with_common_replay).wait();
    }
}

fn build_routing_replay(
    stream: &Stream,
    kernel: &MoERoutingKernel,
    shape: MoERoutingShape,
    router_probs: &Buffer,
    expert_indices: &Buffer,
    expert_probs: &Buffer,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        MoERoutingBuffers {
            router_probs,
            expert_indices,
            expert_probs,
        },
    ));
    builder.build()
}

fn build_combine_without_common_replay(
    stream: &Stream,
    kernel: &MoECombineKernel,
    shape: MoECombineWithoutCommonShape,
    routed_hidden: &Buffer,
    routed_probs: &Buffer,
    output: &Buffer,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke_without_common(
        shape,
        MoECombineWithoutCommonBuffers {
            routed_hidden,
            routed_probs,
            output,
        },
    ));
    builder.build()
}

fn build_combine_with_common_replay(
    stream: &Stream,
    kernel: &MoECombineKernel,
    shape: MoECombineWithCommonShape,
    buffers: MoECombineWithCommonBuffers<'_>,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke_with_common(shape, buffers));
    builder.build()
}

struct MoEForwardFixture {
    stream: Stream,
    router_projection_shape: AffineQuantizedMatmulShape,
    common_gate_shape: AffineQuantizedMatmulShape,
    routing_shape: MoERoutingShape,
    sparse_shape: QuantizedSparseMLPTokenMajorShape,
    expert_major_shape: MoEExpertMajorShape,
    dense_shape: QuantizedDenseMLPShape,
    combine_shape: MoECombineWithCommonShape,
    router_projection: AffineQuantizedMatmulKernel,
    router_softmax: SoftmaxKernel,
    common_gate_projection: AffineQuantizedMatmulKernel,
    routing: MoERoutingKernel,
    expert_major: MoEExpertMajorKernels,
    sparse_mlp: QuantizedSparseMLP,
    common_mlp: QuantizedDenseMLPKernels,
    combine: MoECombineKernel,
    input: Buffer,
    router_logits: Buffer,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
    token_indices: Buffer,
    route_indices: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
    routed_hidden: Buffer,
    expert_major_routed_hidden: Buffer,
    sparse_activation: Buffer,
    expert_major_activation: Buffer,
    common_hidden: Buffer,
    common_gate_logits: Buffer,
    common_gate_weight: Buffer,
    common_gate_scales: Buffer,
    common_gate_biases: Buffer,
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    sparse_weights: SparseMLPWeights,
    common_weights: DenseMLPWeights,
    common_scratch: DenseMLPScratch,
    replay_output: Buffer,
    expert_major_output: Buffer,
    replay: ReplayProgram,
    expert_major_replay: ReplayProgram,
}

impl MoEForwardFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        let stream = Stream::new(device);
        let router_projection_shape = AffineQuantizedMatmulShape {
            m: num_tokens as i32,
            n: NUM_EXPERTS as i32,
            k: HIDDEN_DIM as i32,
            group_size: GROUP_SIZE as i32,
            bits: ROUTER_BITS as i32,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let common_gate_shape = AffineQuantizedMatmulShape {
            m: num_tokens as i32,
            n: 1,
            k: HIDDEN_DIM as i32,
            group_size: GROUP_SIZE as i32,
            bits: ROUTER_BITS as i32,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let routing_shape = MoERoutingShape {
            num_tokens,
            num_experts: NUM_EXPERTS,
            num_experts_per_token: TOPK_EXPERTS,
            norm_topk_prob: true,
        };
        let sparse_config = QuantizedSparseMLPConfig {
            hidden_dim: HIDDEN_DIM,
            intermediate_dim: INTERMEDIATE_DIM,
            group_size: GROUP_SIZE,
            bits: EXPERT_BITS,
            dtype: Dtype::Bfloat16,
        };
        let sparse_shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: num_tokens * TOPK_EXPERTS,
            num_tokens,
        };
        let expert_major_shape = MoEExpertMajorShape::bf16(num_tokens, NUM_EXPERTS, TOPK_EXPERTS, HIDDEN_DIM);
        let expert_major_sparse_shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: expert_major_shape.num_routes(),
            num_tokens: expert_major_shape.num_routes(),
        };
        let dense_config = QuantizedDenseMLPConfig {
            hidden_dim: HIDDEN_DIM,
            intermediate_dim: INTERMEDIATE_DIM,
            group_size: GROUP_SIZE,
            bits: EXPERT_BITS,
            dtype: Dtype::Bfloat16,
        };
        let dense_shape = QuantizedDenseMLPShape { num_tokens };
        let combine_shape = MoECombineWithCommonShape::bf16(num_tokens, TOPK_EXPERTS, HIDDEN_DIM);
        let sparse_gate_up_shape = sparse_config.token_major_fused_gate_up_silu_shape(sparse_shape);
        let sparse_down_shape = sparse_config.token_major_down_shape(sparse_shape);
        let dense_gate_up_shape = dense_config.gate_up_shape(dense_shape);
        let dense_down_shape = dense_config.down_shape(dense_shape);
        let num_routes = num_tokens as usize * TOPK_EXPERTS as usize;
        let input = bf16_buffer(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let router_logits = Buffer::new_zeroed(device, router_projection_shape.output_bytes());
        let router_probs = Buffer::new_zeroed(device, router_projection_shape.output_bytes());
        let expert_indices = Buffer::new_zeroed(device, routing_shape.expert_indices_bytes());
        let expert_probs = Buffer::new_zeroed(device, routing_shape.expert_probs_bytes());
        let token_indices =
            Buffer::from_slice(device, &token_route_indices(num_tokens as usize, TOPK_EXPERTS as usize));
        let route_indices = Buffer::from_slice(device, &identity_indices(num_routes));
        let expert_counts = Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes());
        let expert_offsets = Buffer::new_zeroed(device, expert_major_shape.expert_offsets_bytes());
        let expert_cursors = Buffer::new_zeroed(device, expert_major_shape.expert_counts_bytes());
        let routes_by_expert = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let routes_by_token = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let experts_by_route = Buffer::new_zeroed(device, expert_major_shape.route_indices_bytes());
        let packed_input = Buffer::new_zeroed(device, expert_major_shape.route_hidden_bytes());
        let routed_hidden = Buffer::new_zeroed(device, sparse_config.token_major_output_bytes(sparse_shape));
        let expert_major_routed_hidden = Buffer::new_zeroed(
            device,
            sparse_config.token_major_output_bytes(expert_major_sparse_shape),
        );
        let sparse_activation = Buffer::new_zeroed(device, sparse_config.activation_bytes(sparse_shape.num_routes));
        let expert_major_activation = Buffer::new_zeroed(
            device,
            sparse_config.activation_bytes(expert_major_sparse_shape.num_routes),
        );
        let common_hidden = Buffer::new_zeroed(device, dense_down_shape.output_bytes());
        let common_gate_logits = Buffer::new_zeroed(device, common_gate_shape.output_bytes());
        let replay_output = Buffer::new_zeroed(device, combine_shape.output_bytes());
        let expert_major_output = Buffer::new_zeroed(device, combine_shape.output_bytes());
        let router_projection = AffineQuantizedMatmulKernel::new(device, router_projection_shape);
        let router_softmax = SoftmaxKernel::new(
            device,
            SoftmaxShape {
                num_rows: 1,
                num_values_per_row: NUM_EXPERTS,
                dtype: Dtype::Bfloat16,
            },
        );
        let common_gate_projection = AffineQuantizedMatmulKernel::new(device, common_gate_shape);
        let routing = MoERoutingKernel::new(device);
        let expert_major = MoEExpertMajorKernels::new(device);
        let sparse_mlp = QuantizedSparseMLP::new(device, sparse_config);
        let common_mlp = QuantizedDenseMLPKernels::new(device, dense_config);
        let combine = MoECombineKernel::new(device);
        let router_weight = quantized_weight(device, router_projection_shape.weight_bytes());
        let router_scales = bf16_buffer(
            device,
            &affine_param_fixture(router_projection_shape.affine_param_bytes() / size_of::<u16>()),
        );
        let router_biases = bf16_buffer(
            device,
            &zero_fixture(router_projection_shape.affine_param_bytes() / size_of::<u16>()),
        );
        let common_gate_weight = quantized_weight(device, common_gate_shape.weight_bytes());
        let common_gate_scales = bf16_buffer(
            device,
            &affine_param_fixture(common_gate_shape.affine_param_bytes() / size_of::<u16>()),
        );
        let common_gate_biases = bf16_buffer(
            device,
            &zero_fixture(common_gate_shape.affine_param_bytes() / size_of::<u16>()),
        );
        let sparse_weights = SparseMLPWeights {
            gate_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_gate_up_shape.weight_bytes_per_expert(),
            ),
            gate_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            gate_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            up_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_gate_up_shape.weight_bytes_per_expert(),
            ),
            up_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            up_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            down_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_down_shape.weight_bytes_per_expert(),
            ),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
        };
        let common_weights = DenseMLPWeights {
            gate_up_weight: quantized_weight(device, dense_gate_up_shape.weight_bytes()),
            gate_up_scales: bf16_buffer(
                device,
                &affine_param_fixture(dense_gate_up_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            gate_up_biases: bf16_buffer(
                device,
                &zero_fixture(dense_gate_up_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            down_weight: quantized_weight(device, dense_down_shape.weight_bytes()),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(dense_down_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(dense_down_shape.affine_param_bytes() / size_of::<u16>()),
            ),
        };
        let common_scratch = DenseMLPScratch {
            gate_up_proj: Buffer::new_zeroed(
                device,
                num_tokens as usize * INTERMEDIATE_DIM as usize * 2 * Dtype::Bfloat16.item_size(),
            ),
            activation: Buffer::new_zeroed(device, dense_config.activation_shape(dense_shape).bytes()),
        };
        let replay = build_moe_forward_replay(
            &stream,
            MoEForwardRecord {
                router_projection_shape,
                common_gate_shape,
                routing_shape,
                sparse_shape,
                dense_shape,
                combine_shape,
                router_projection: &router_projection,
                router_softmax: &router_softmax,
                common_gate_projection: &common_gate_projection,
                routing: &routing,
                sparse_mlp: &sparse_mlp,
                common_mlp: &common_mlp,
                combine: &combine,
                input: &input,
                router_logits: &router_logits,
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
                token_indices: &token_indices,
                route_indices: &route_indices,
                routed_hidden: &routed_hidden,
                sparse_activation: &sparse_activation,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &replay_output,
                router_weight: &router_weight,
                router_scales: &router_scales,
                router_biases: &router_biases,
                common_gate_weight: &common_gate_weight,
                common_gate_scales: &common_gate_scales,
                common_gate_biases: &common_gate_biases,
                sparse_weights: sparse_weights.as_borrowed(),
                common_weights: common_weights.as_borrowed(),
                common_scratch: common_scratch.as_borrowed(),
            },
        );
        let expert_major_replay = build_moe_expert_major_forward_replay(
            &stream,
            MoEExpertMajorForwardRecord {
                router_projection_shape,
                common_gate_shape,
                routing_shape,
                sparse_shape: expert_major_sparse_shape,
                expert_major_shape,
                dense_shape,
                router_projection: &router_projection,
                router_softmax: &router_softmax,
                common_gate_projection: &common_gate_projection,
                routing: &routing,
                expert_major: &expert_major,
                sparse_mlp: &sparse_mlp,
                common_mlp: &common_mlp,
                input: &input,
                router_logits: &router_logits,
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
                route_indices: &route_indices,
                expert_counts: &expert_counts,
                expert_offsets: &expert_offsets,
                expert_cursors: &expert_cursors,
                routes_by_expert: &routes_by_expert,
                routes_by_token: &routes_by_token,
                experts_by_route: &experts_by_route,
                packed_input: &packed_input,
                routed_hidden: &expert_major_routed_hidden,
                sparse_activation: &expert_major_activation,
                common_hidden: &common_hidden,
                common_gate_logits: &common_gate_logits,
                output: &expert_major_output,
                router_weight: &router_weight,
                router_scales: &router_scales,
                router_biases: &router_biases,
                common_gate_weight: &common_gate_weight,
                common_gate_scales: &common_gate_scales,
                common_gate_biases: &common_gate_biases,
                sparse_weights: sparse_weights.as_borrowed(),
                common_weights: common_weights.as_borrowed(),
                common_scratch: common_scratch.as_borrowed(),
            },
        );
        let fixture = Self {
            stream,
            router_projection_shape,
            common_gate_shape,
            routing_shape,
            sparse_shape,
            expert_major_shape,
            dense_shape,
            combine_shape,
            router_projection,
            router_softmax,
            common_gate_projection,
            routing,
            expert_major,
            sparse_mlp,
            common_mlp,
            combine,
            input,
            router_logits,
            router_probs,
            expert_indices,
            expert_probs,
            token_indices,
            route_indices,
            expert_counts,
            expert_offsets,
            expert_cursors,
            routes_by_expert,
            routes_by_token,
            experts_by_route,
            packed_input,
            routed_hidden,
            expert_major_routed_hidden,
            sparse_activation,
            expert_major_activation,
            common_hidden,
            common_gate_logits,
            common_gate_weight,
            common_gate_scales,
            common_gate_biases,
            router_weight,
            router_scales,
            router_biases,
            sparse_weights,
            common_weights,
            common_scratch,
            replay_output,
            expert_major_output,
            replay,
            expert_major_replay,
        };
        fixture.assert_token_major_and_expert_major_replay_match_bitwise();
        fixture
    }

    fn run_token_major_replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }

    fn run_auto_replay(&self) {
        if self.routing_shape.num_tokens >= AUTO_EXPERT_MAJOR_MIN_TOKENS {
            self.run_expert_major_replay();
        } else {
            self.run_token_major_replay();
        }
    }

    fn run_expert_major_replay(&self) {
        self.stream.submit_replay(&self.expert_major_replay).wait();
    }

    fn auto_output(&self) -> &Buffer {
        if self.routing_shape.num_tokens >= AUTO_EXPERT_MAJOR_MIN_TOKENS {
            &self.expert_major_output
        } else {
            &self.replay_output
        }
    }

    fn record<'a>(&'a self, output: &'a Buffer) -> MoEForwardRecord<'a> {
        MoEForwardRecord {
            router_projection_shape: self.router_projection_shape,
            common_gate_shape: self.common_gate_shape,
            routing_shape: self.routing_shape,
            sparse_shape: self.sparse_shape,
            dense_shape: self.dense_shape,
            combine_shape: self.combine_shape,
            router_projection: &self.router_projection,
            router_softmax: &self.router_softmax,
            common_gate_projection: &self.common_gate_projection,
            routing: &self.routing,
            sparse_mlp: &self.sparse_mlp,
            common_mlp: &self.common_mlp,
            combine: &self.combine,
            input: &self.input,
            router_logits: &self.router_logits,
            router_probs: &self.router_probs,
            expert_indices: &self.expert_indices,
            expert_probs: &self.expert_probs,
            token_indices: &self.token_indices,
            route_indices: &self.route_indices,
            routed_hidden: &self.routed_hidden,
            sparse_activation: &self.sparse_activation,
            common_hidden: &self.common_hidden,
            common_gate_logits: &self.common_gate_logits,
            output,
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            common_gate_weight: &self.common_gate_weight,
            common_gate_scales: &self.common_gate_scales,
            common_gate_biases: &self.common_gate_biases,
            sparse_weights: self.sparse_weights.as_borrowed(),
            common_weights: self.common_weights.as_borrowed(),
            common_scratch: self.common_scratch.as_borrowed(),
        }
    }

    fn expert_major_record<'a>(&'a self, output: &'a Buffer) -> MoEExpertMajorForwardRecord<'a> {
        MoEExpertMajorForwardRecord {
            router_projection_shape: self.router_projection_shape,
            common_gate_shape: self.common_gate_shape,
            routing_shape: self.routing_shape,
            sparse_shape: QuantizedSparseMLPTokenMajorShape {
                num_routes: self.expert_major_shape.num_routes(),
                num_tokens: self.expert_major_shape.num_routes(),
            },
            expert_major_shape: self.expert_major_shape,
            dense_shape: self.dense_shape,
            router_projection: &self.router_projection,
            router_softmax: &self.router_softmax,
            common_gate_projection: &self.common_gate_projection,
            routing: &self.routing,
            expert_major: &self.expert_major,
            sparse_mlp: &self.sparse_mlp,
            common_mlp: &self.common_mlp,
            input: &self.input,
            router_logits: &self.router_logits,
            router_probs: &self.router_probs,
            expert_indices: &self.expert_indices,
            expert_probs: &self.expert_probs,
            route_indices: &self.route_indices,
            expert_counts: &self.expert_counts,
            expert_offsets: &self.expert_offsets,
            expert_cursors: &self.expert_cursors,
            routes_by_expert: &self.routes_by_expert,
            routes_by_token: &self.routes_by_token,
            experts_by_route: &self.experts_by_route,
            packed_input: &self.packed_input,
            routed_hidden: &self.expert_major_routed_hidden,
            sparse_activation: &self.expert_major_activation,
            common_hidden: &self.common_hidden,
            common_gate_logits: &self.common_gate_logits,
            output,
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            common_gate_weight: &self.common_gate_weight,
            common_gate_scales: &self.common_gate_scales,
            common_gate_biases: &self.common_gate_biases,
            sparse_weights: self.sparse_weights.as_borrowed(),
            common_weights: self.common_weights.as_borrowed(),
            common_scratch: self.common_scratch.as_borrowed(),
        }
    }

    fn assert_token_major_and_expert_major_replay_match_bitwise(&self) {
        self.run_token_major_replay();
        self.run_expert_major_replay();
        let replay = self
            .replay_output
            .read_typed::<u16>(0, self.replay_output.len_bytes() / size_of::<u16>());
        let expert_major = self
            .expert_major_output
            .read_typed::<u16>(0, self.expert_major_output.len_bytes() / size_of::<u16>());
        assert_eq!(
            replay, expert_major,
            "MoE forward token_major and expert_major output bits must match"
        );
    }
}

struct MoEForwardRecord<'a> {
    router_projection_shape: AffineQuantizedMatmulShape,
    common_gate_shape: AffineQuantizedMatmulShape,
    routing_shape: MoERoutingShape,
    sparse_shape: QuantizedSparseMLPTokenMajorShape,
    dense_shape: QuantizedDenseMLPShape,
    combine_shape: MoECombineWithCommonShape,
    router_projection: &'a AffineQuantizedMatmulKernel,
    router_softmax: &'a SoftmaxKernel,
    common_gate_projection: &'a AffineQuantizedMatmulKernel,
    routing: &'a MoERoutingKernel,
    sparse_mlp: &'a QuantizedSparseMLP,
    common_mlp: &'a QuantizedDenseMLPKernels,
    combine: &'a MoECombineKernel,
    input: &'a Buffer,
    router_logits: &'a Buffer,
    router_probs: &'a Buffer,
    expert_indices: &'a Buffer,
    expert_probs: &'a Buffer,
    token_indices: &'a Buffer,
    route_indices: &'a Buffer,
    routed_hidden: &'a Buffer,
    sparse_activation: &'a Buffer,
    common_hidden: &'a Buffer,
    common_gate_logits: &'a Buffer,
    output: &'a Buffer,
    router_weight: &'a Buffer,
    router_scales: &'a Buffer,
    router_biases: &'a Buffer,
    common_gate_weight: &'a Buffer,
    common_gate_scales: &'a Buffer,
    common_gate_biases: &'a Buffer,
    sparse_weights: QuantizedSparseMLPWeights<'a>,
    common_weights: QuantizedDenseMLPWeights<'a>,
    common_scratch: QuantizedDenseMLPScratch<'a>,
}

fn build_moe_forward_replay(stream: &Stream, record: MoEForwardRecord<'_>) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    record_moe_forward(&mut builder, record);
    builder.build()
}

struct MoEExpertMajorForwardRecord<'a> {
    router_projection_shape: AffineQuantizedMatmulShape,
    common_gate_shape: AffineQuantizedMatmulShape,
    routing_shape: MoERoutingShape,
    sparse_shape: QuantizedSparseMLPTokenMajorShape,
    expert_major_shape: MoEExpertMajorShape,
    dense_shape: QuantizedDenseMLPShape,
    router_projection: &'a AffineQuantizedMatmulKernel,
    router_softmax: &'a SoftmaxKernel,
    common_gate_projection: &'a AffineQuantizedMatmulKernel,
    routing: &'a MoERoutingKernel,
    expert_major: &'a MoEExpertMajorKernels,
    sparse_mlp: &'a QuantizedSparseMLP,
    common_mlp: &'a QuantizedDenseMLPKernels,
    input: &'a Buffer,
    router_logits: &'a Buffer,
    router_probs: &'a Buffer,
    expert_indices: &'a Buffer,
    expert_probs: &'a Buffer,
    route_indices: &'a Buffer,
    expert_counts: &'a Buffer,
    expert_offsets: &'a Buffer,
    expert_cursors: &'a Buffer,
    routes_by_expert: &'a Buffer,
    routes_by_token: &'a Buffer,
    experts_by_route: &'a Buffer,
    packed_input: &'a Buffer,
    routed_hidden: &'a Buffer,
    sparse_activation: &'a Buffer,
    common_hidden: &'a Buffer,
    common_gate_logits: &'a Buffer,
    output: &'a Buffer,
    router_weight: &'a Buffer,
    router_scales: &'a Buffer,
    router_biases: &'a Buffer,
    common_gate_weight: &'a Buffer,
    common_gate_scales: &'a Buffer,
    common_gate_biases: &'a Buffer,
    sparse_weights: QuantizedSparseMLPWeights<'a>,
    common_weights: QuantizedDenseMLPWeights<'a>,
    common_scratch: QuantizedDenseMLPScratch<'a>,
}

fn build_moe_expert_major_forward_replay(stream: &Stream, record: MoEExpertMajorForwardRecord<'_>) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    record_moe_expert_major_forward(&mut builder, record);
    builder.build()
}

fn record_moe_forward<I>(builder: &mut I, record: MoEForwardRecord<'_>)
where
    I: MoEForwardBuilder,
{
    builder.record(record.router_projection.invoke_with_shape(
        record.router_projection_shape,
        record.router_logits,
        0,
        record.input,
        0,
        record.router_weight,
        0,
        record.router_scales,
        0,
        record.router_biases,
        0,
    ));
    builder.record_with_barrier_before(record.router_softmax.invoke_with_shape(
        SoftmaxShape {
            num_rows: record.routing_shape.num_tokens,
            num_values_per_row: record.routing_shape.num_experts,
            dtype: Dtype::Bfloat16,
        },
        record.router_probs,
        record.router_logits,
    ));
    builder.record_with_barrier_before(record.routing.invoke(
        record.routing_shape,
        MoERoutingBuffers {
            router_probs: record.router_probs,
            expert_indices: record.expert_indices,
            expert_probs: record.expert_probs,
        },
    ));
    builder.record_with_barrier_before(record.sparse_mlp.invoke_token_major(
        record.sparse_shape,
        QuantizedSparseMLPTokenMajorBuffers {
            input: record.input,
            token_indices: record.token_indices,
            expert_indices: record.expert_indices,
            route_indices: record.route_indices,
            output: record.routed_hidden,
        },
        QuantizedSparseMLPTokenMajorScratch {
            activation: record.sparse_activation,
        },
        record.sparse_weights,
    ));
    builder.record(record.common_mlp.invoke(
        record.dense_shape,
        QuantizedDenseMLPBuffers {
            hidden_state: record.input,
            next_hidden_state: record.common_hidden,
        },
        record.common_scratch,
        record.common_weights,
    ));
    builder.record(record.common_gate_projection.invoke_with_shape(
        record.common_gate_shape,
        record.common_gate_logits,
        0,
        record.input,
        0,
        record.common_gate_weight,
        0,
        record.common_gate_scales,
        0,
        record.common_gate_biases,
        0,
    ));
    builder.record_with_barrier_before(record.combine.invoke_with_common(
        record.combine_shape,
        MoECombineWithCommonBuffers {
            routed_hidden: record.routed_hidden,
            routed_probs: record.expert_probs,
            common_hidden: record.common_hidden,
            common_gate_logits: record.common_gate_logits,
            output: record.output,
        },
    ));
}

fn record_moe_expert_major_forward<I>(builder: &mut I, record: MoEExpertMajorForwardRecord<'_>)
where
    I: MoEForwardBuilder,
{
    builder.record(record.router_projection.invoke_with_shape(
        record.router_projection_shape,
        record.router_logits,
        0,
        record.input,
        0,
        record.router_weight,
        0,
        record.router_scales,
        0,
        record.router_biases,
        0,
    ));
    builder.record_with_barrier_before(record.router_softmax.invoke_with_shape(
        SoftmaxShape {
            num_rows: record.routing_shape.num_tokens,
            num_values_per_row: record.routing_shape.num_experts,
            dtype: Dtype::Bfloat16,
        },
        record.router_probs,
        record.router_logits,
    ));
    builder.record_with_barrier_before(record.routing.invoke(
        record.routing_shape,
        MoERoutingBuffers {
            router_probs: record.router_probs,
            expert_indices: record.expert_indices,
            expert_probs: record.expert_probs,
        },
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_layout(
        record.expert_major_shape,
        MoEExpertMajorLayoutBuffers {
            expert_indices: record.expert_indices,
            expert_counts: record.expert_counts,
            expert_offsets: record.expert_offsets,
            expert_cursors: record.expert_cursors,
            routes_by_expert: record.routes_by_expert,
            routes_by_token: record.routes_by_token,
            experts_by_route: record.experts_by_route,
        },
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_pack_input(
        record.expert_major_shape,
        MoEExpertMajorPackInputBuffers {
            input: record.input,
            routes_by_expert: record.routes_by_expert,
            packed_input: record.packed_input,
        },
    ));
    builder.record_with_barrier_before(record.sparse_mlp.invoke_expert_major(
        QuantizedSparseMLPExpertMajorShape {
            num_experts: record.expert_major_shape.num_experts,
            num_routes: record.expert_major_shape.num_routes(),
        },
        QuantizedSparseMLPExpertMajorBuffers {
            packed_input: record.packed_input,
            experts_by_route: record.experts_by_route,
            route_output: record.routed_hidden,
        },
        QuantizedSparseMLPExpertMajorScratch {
            activation: record.sparse_activation,
        },
        record.sparse_weights,
    ));
    builder.record(record.common_mlp.invoke(
        record.dense_shape,
        QuantizedDenseMLPBuffers {
            hidden_state: record.input,
            next_hidden_state: record.common_hidden,
        },
        record.common_scratch,
        record.common_weights,
    ));
    builder.record(record.common_gate_projection.invoke_with_shape(
        record.common_gate_shape,
        record.common_gate_logits,
        0,
        record.input,
        0,
        record.common_gate_weight,
        0,
        record.common_gate_scales,
        0,
        record.common_gate_biases,
        0,
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_scatter_with_common(
        record.expert_major_shape,
        MoEExpertMajorScatterWithCommonBuffers {
            route_output: record.routed_hidden,
            routes_by_token: record.routes_by_token,
            routed_probs: record.expert_probs,
            common_hidden: record.common_hidden,
            common_gate_logits: record.common_gate_logits,
            output: record.output,
        },
    ));
}

trait MoEForwardBuilder {
    fn record<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T);
    fn record_with_barrier_before<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T);
}

impl MoEForwardBuilder for inference_backend_metal::metal::ReplayProgramBuilder {
    fn record<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T) {
        ReplayProgramBuilder::record(self, invocation);
    }

    fn record_with_barrier_before<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T) {
        ReplayProgramBuilder::record_with_barrier_before(self, invocation);
    }
}

struct SparseMLPWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl SparseMLPWeights {
    fn as_borrowed(&self) -> QuantizedSparseMLPWeights<'_> {
        QuantizedSparseMLPWeights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct DenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl DenseMLPWeights {
    fn as_borrowed(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct DenseMLPScratch {
    gate_up_proj: Buffer,
    activation: Buffer,
}

impl DenseMLPScratch {
    fn as_borrowed(&self) -> QuantizedDenseMLPScratch<'_> {
        QuantizedDenseMLPScratch {
            gate_up_proj: &self.gate_up_proj,
            activation: &self.activation,
        }
    }
}

criterion_group!(benches, bench_moe);
criterion_main!(benches);
