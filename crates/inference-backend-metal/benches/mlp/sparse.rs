use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::QuantizedSparseMLPConfig;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorBuffers;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorKernels;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorScratch;
use inference_backend_metal::components::QuantizedSparseMLPTokenMajorShape;
use inference_backend_metal::components::QuantizedSparseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

#[path = "../support.rs"]
mod support;
use support::affine_param_fixture;
use support::bf16_buffer;
use support::expert_route_indices;
use support::hidden_fixture;
use support::identity_indices;
use support::quantized_weight_stack_for_experts;
use support::repeated_topk_expert_indices;
use support::token_route_indices;
use support::zero_fixture;

const TOPK_EXPERTS: u32 = 8;
const BENCH_TOKENS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const SPARSE_MLP_PROFILE: &str = "qwen36-35b-a3b";
const SPARSE_MLP_NUM_EXPERTS: u32 = 256;
const SPARSE_MLP_HIDDEN_DIM: u32 = 2048;
const SPARSE_MLP_INTERMEDIATE_DIM: u32 = 512;
const SPARSE_MLP_GROUP_SIZE: u32 = 64;
const SPARSE_MLP_BITS: u32 = 4;

fn bench_sparse_mlp(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/sparse-mlp");

    for num_tokens in BENCH_TOKENS {
        let sparse_mlp = QuantizedSparseMLPFixture::new(&device, num_tokens);
        let fixed_topk_sparse_mlp = QuantizedSparseMLPFixture::new_fixed_topk(&device, num_tokens);
        group.throughput(Throughput::Elements(
            num_tokens as u64 * TOPK_EXPERTS as u64 * SPARSE_MLP_HIDDEN_DIM as u64,
        ));
        group.bench_function(
            format!(
                "{SPARSE_MLP_PROFILE}/token_major/fused_gate_up_silu/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/\
                 hidden{SPARSE_MLP_HIDDEN_DIM}/intermediate{SPARSE_MLP_INTERMEDIATE_DIM}/\
                 experts{SPARSE_MLP_NUM_EXPERTS}"
            ),
            |b| {
                b.iter(|| {
                    sparse_mlp.forward_gate_up_silu();
                    black_box(&sparse_mlp.scratch.activation);
                });
            },
        );
        group.bench_function(
            format!(
                "{SPARSE_MLP_PROFILE}/token_major/forward/replay/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/\
                 hidden{SPARSE_MLP_HIDDEN_DIM}/intermediate{SPARSE_MLP_INTERMEDIATE_DIM}/\
                 experts{SPARSE_MLP_NUM_EXPERTS}"
            ),
            |b| {
                b.iter(|| {
                    sparse_mlp.replay_token_major();
                    black_box(&sparse_mlp.buffers.output);
                });
            },
        );
        group.bench_function(
            format!(
                "{SPARSE_MLP_PROFILE}/token_major_fixed_topk/forward/replay/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/\
                 hidden{SPARSE_MLP_HIDDEN_DIM}/intermediate{SPARSE_MLP_INTERMEDIATE_DIM}/\
                 experts{SPARSE_MLP_NUM_EXPERTS}"
            ),
            |b| {
                b.iter(|| {
                    fixed_topk_sparse_mlp.replay_token_major();
                    black_box(&fixed_topk_sparse_mlp.buffers.output);
                });
            },
        );
    }

    group.finish();
}

struct QuantizedSparseMLPFixture {
    stream: Stream,
    shape: QuantizedSparseMLPTokenMajorShape,
    sparse_mlp: QuantizedSparseMLPTokenMajorKernels,
    buffers: QuantizedSparseMLPOwnedBuffers,
    scratch: QuantizedSparseMLPOwnedScratch,
    weights: QuantizedSparseMLPOwnedWeights,
    gate_up_silu_replay: ReplayProgram,
    token_major_replay: ReplayProgram,
}

impl QuantizedSparseMLPFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        Self::new_with_expert_indices(
            device,
            num_tokens,
            expert_route_indices(
                num_tokens as usize,
                TOPK_EXPERTS as usize,
                SPARSE_MLP_NUM_EXPERTS as usize,
            ),
        )
    }

    fn new_fixed_topk(device: &Device, num_tokens: u32) -> Self {
        Self::new_with_expert_indices(
            device,
            num_tokens,
            repeated_topk_expert_indices(num_tokens as usize, TOPK_EXPERTS as usize),
        )
    }

    fn new_with_expert_indices(device: &Device, num_tokens: u32, expert_indices_values: Vec<u32>) -> Self {
        let config = QuantizedSparseMLPConfig {
            hidden_dim: SPARSE_MLP_HIDDEN_DIM,
            intermediate_dim: SPARSE_MLP_INTERMEDIATE_DIM,
            group_size: SPARSE_MLP_GROUP_SIZE,
            bits: SPARSE_MLP_BITS,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedSparseMLPTokenMajorShape {
            num_routes: num_tokens * TOPK_EXPERTS,
            num_tokens,
        };
        let fused_gate_up_silu_shape = config.token_major_fused_gate_up_silu_shape(shape);
        let down_shape = config.token_major_down_shape(shape);
        let stream = Stream::new(device);
        let sparse_mlp = QuantizedSparseMLPTokenMajorKernels::new(device, config);
        let buffers = QuantizedSparseMLPOwnedBuffers {
            input: bf16_buffer(
                device,
                &hidden_fixture(num_tokens as usize, SPARSE_MLP_HIDDEN_DIM as usize),
            ),
            token_indices: Buffer::from_slice(device, &token_route_indices(num_tokens as usize, TOPK_EXPERTS as usize)),
            expert_indices: Buffer::from_slice(device, &expert_indices_values),
            route_indices: Buffer::from_slice(device, &identity_indices((num_tokens * TOPK_EXPERTS) as usize)),
            output: Buffer::new_zeroed(device, config.token_major_output_bytes(shape)),
        };
        let scratch = QuantizedSparseMLPOwnedScratch {
            activation: Buffer::new_zeroed(device, config.activation_bytes(shape.num_routes)),
        };
        let weights = QuantizedSparseMLPOwnedWeights {
            gate_weight: quantized_weight_stack(device, fused_gate_up_silu_shape.weight_bytes_per_expert()),
            gate_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert()
                        / size_of::<u16>(),
                ),
            ),
            gate_biases: bf16_buffer(
                device,
                &zero_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert()
                        / size_of::<u16>(),
                ),
            ),
            up_weight: quantized_weight_stack(device, fused_gate_up_silu_shape.weight_bytes_per_expert()),
            up_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert()
                        / size_of::<u16>(),
                ),
            ),
            up_biases: bf16_buffer(
                device,
                &zero_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * fused_gate_up_silu_shape.affine_param_bytes_per_expert()
                        / size_of::<u16>(),
                ),
            ),
            down_weight: quantized_weight_stack(device, down_shape.weight_bytes_per_expert()),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(
                    SPARSE_MLP_NUM_EXPERTS as usize * down_shape.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
        };
        let gate_up_silu_replay = build_gate_up_silu_replay(&stream, &sparse_mlp, shape, &buffers, &scratch, &weights);
        let token_major_replay = build_token_major_replay(&stream, &sparse_mlp, shape, &buffers, &scratch, &weights);
        let fixture = Self {
            stream,
            shape,
            sparse_mlp,
            buffers,
            scratch,
            weights,
            gate_up_silu_replay,
            token_major_replay,
        };
        fixture.replay_token_major();
        fixture
    }

    fn forward_gate_up_silu(&self) {
        self.stream.submit_replay(&self.gate_up_silu_replay).wait();
    }

    fn replay_token_major(&self) {
        self.stream.submit_replay(&self.token_major_replay).wait();
    }
}

struct QuantizedSparseMLPOwnedBuffers {
    input: Buffer,
    token_indices: Buffer,
    expert_indices: Buffer,
    route_indices: Buffer,
    output: Buffer,
}

struct QuantizedSparseMLPOwnedScratch {
    activation: Buffer,
}

struct QuantizedSparseMLPOwnedWeights {
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

fn quantized_weight_stack(device: &Device, bytes_per_expert: usize) -> Buffer {
    quantized_weight_stack_for_experts(device, SPARSE_MLP_NUM_EXPERTS as usize, bytes_per_expert)
}

fn build_gate_up_silu_replay(
    stream: &Stream,
    sparse_mlp: &QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPOwnedBuffers,
    scratch: &QuantizedSparseMLPOwnedScratch,
    weights: &QuantizedSparseMLPOwnedWeights,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(sparse_mlp.invoke_fused_gate_up_silu(
        shape,
        QuantizedSparseMLPTokenMajorBuffers {
            input: &buffers.input,
            token_indices: &buffers.token_indices,
            expert_indices: &buffers.expert_indices,
            route_indices: &buffers.route_indices,
            output: &buffers.output,
        },
        QuantizedSparseMLPTokenMajorScratch {
            activation: &scratch.activation,
        },
        owned_weights(weights),
    ));
    builder.build()
}

fn build_token_major_replay(
    stream: &Stream,
    sparse_mlp: &QuantizedSparseMLPTokenMajorKernels,
    shape: QuantizedSparseMLPTokenMajorShape,
    buffers: &QuantizedSparseMLPOwnedBuffers,
    scratch: &QuantizedSparseMLPOwnedScratch,
    weights: &QuantizedSparseMLPOwnedWeights,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(sparse_mlp.invoke(
        shape,
        QuantizedSparseMLPTokenMajorBuffers {
            input: &buffers.input,
            token_indices: &buffers.token_indices,
            expert_indices: &buffers.expert_indices,
            route_indices: &buffers.route_indices,
            output: &buffers.output,
        },
        QuantizedSparseMLPTokenMajorScratch {
            activation: &scratch.activation,
        },
        owned_weights(weights),
    ));
    builder.build()
}

fn owned_weights(weights: &QuantizedSparseMLPOwnedWeights) -> QuantizedSparseMLPWeights<'_> {
    QuantizedSparseMLPWeights {
        gate_weight: &weights.gate_weight,
        gate_scales: &weights.gate_scales,
        gate_biases: &weights.gate_biases,
        up_weight: &weights.up_weight,
        up_scales: &weights.up_scales,
        up_biases: &weights.up_biases,
        down_weight: &weights.down_weight,
        down_scales: &weights.down_scales,
        down_biases: &weights.down_biases,
    }
}

criterion_group!(benches, bench_sparse_mlp);
criterion_main!(benches);
