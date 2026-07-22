use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::QuantizedDenseMLPBuffers;
use inference_backend_metal::components::QuantizedDenseMLPConfig;
use inference_backend_metal::components::QuantizedDenseMLPKernels;
use inference_backend_metal::components::QuantizedDenseMLPScratch;
use inference_backend_metal::components::QuantizedDenseMLPShape;
use inference_backend_metal::components::QuantizedDenseMLPWeights;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

#[path = "../support.rs"]
mod support;
use support::affine_param_fixture;
use support::bf16_buffer;
use support::hidden_fixture;
use support::quantized_weight;
use support::zero_fixture;

const BENCH_TOKENS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const DENSE_MLP_PROFILES: [DenseMLPProfile; 2] = [
    DenseMLPProfile {
        name: "qwen36-27b",
        hidden_dim: 5120,
        intermediate_dim: 17_408,
        group_size: 64,
        bits: 4,
    },
    DenseMLPProfile {
        name: "qwen36-35b-a3b-common",
        hidden_dim: 2048,
        intermediate_dim: 512,
        group_size: 64,
        bits: 4,
    },
];

#[derive(Clone, Copy)]
struct DenseMLPProfile {
    name: &'static str,
    hidden_dim: u32,
    intermediate_dim: u32,
    group_size: u32,
    bits: u32,
}

fn bench_dense_mlp(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/dense-mlp");

    for profile in DENSE_MLP_PROFILES {
        for tokens in BENCH_TOKENS {
            let dense_mlp = QuantizedDenseMLPFixture::new(&device, profile, tokens);
            group.throughput(Throughput::Elements(tokens as u64 * profile.hidden_dim as u64));
            group.bench_function(
                format!(
                    "{}/gate_up_activation/tokens{tokens}/hidden{}/intermediate{}",
                    profile.name, profile.hidden_dim, profile.intermediate_dim
                ),
                |b| {
                    b.iter(|| {
                        dense_mlp.replay_gate_up_activation();
                        black_box(&dense_mlp.scratch.activation);
                    });
                },
            );
            group.bench_function(
                format!(
                    "{}/forward/tokens{tokens}/hidden{}/intermediate{}",
                    profile.name, profile.hidden_dim, profile.intermediate_dim
                ),
                |b| {
                    b.iter(|| {
                        dense_mlp.replay_forward();
                        black_box(&dense_mlp.buffers.replay_next_hidden_state);
                    });
                },
            );
        }
    }

    group.finish();
}

struct QuantizedDenseMLPFixture {
    stream: Stream,
    shape: QuantizedDenseMLPShape,
    kernels: QuantizedDenseMLPKernels,
    buffers: QuantizedDenseMLPOwnedBuffers,
    scratch: QuantizedDenseMLPOwnedScratch,
    weights: QuantizedDenseMLPOwnedWeights,
    gate_up_activation_replay: ReplayProgram,
    forward_replay: ReplayProgram,
}

impl QuantizedDenseMLPFixture {
    fn new(device: &Device, profile: DenseMLPProfile, tokens: u32) -> Self {
        let config = QuantizedDenseMLPConfig {
            hidden_dim: profile.hidden_dim,
            intermediate_dim: profile.intermediate_dim,
            group_size: profile.group_size,
            bits: profile.bits,
            dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedDenseMLPShape { num_tokens: tokens };
        let gate_up_shape = config.gate_up_shape(shape);
        let down_shape = config.down_shape(shape);
        let stream = Stream::new(device);
        let kernels = QuantizedDenseMLPKernels::new(device, config);
        let buffers = QuantizedDenseMLPOwnedBuffers {
            hidden_state: bf16_buffer(device, &hidden_fixture(tokens as usize, profile.hidden_dim as usize)),
            replay_next_hidden_state: Buffer::new_zeroed(device, down_shape.output_bytes()),
        };
        let scratch = QuantizedDenseMLPOwnedScratch {
            gate_up_proj: Buffer::new_zeroed(device, gate_up_shape.output_bytes()),
            activation: Buffer::new_zeroed(device, config.activation_shape(shape).bytes()),
            replay_activation: Buffer::new_zeroed(device, config.activation_shape(shape).bytes()),
        };
        let weights = QuantizedDenseMLPOwnedWeights {
            gate_up_weight: quantized_weight(device, gate_up_shape.weight_bytes()),
            gate_up_scales: bf16_buffer(
                device,
                &affine_param_fixture(gate_up_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            gate_up_biases: bf16_buffer(
                device,
                &zero_fixture(gate_up_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            down_weight: quantized_weight(device, down_shape.weight_bytes()),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(down_shape.affine_param_bytes() / size_of::<u16>()),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(down_shape.affine_param_bytes() / size_of::<u16>()),
            ),
        };
        let gate_up_activation_replay =
            build_gate_up_activation_replay(&stream, &kernels, shape, &buffers, &scratch, &weights);
        let forward_replay = build_forward_replay(&stream, &kernels, shape, &buffers, &scratch, &weights);
        let fixture = Self {
            stream,
            shape,
            kernels,
            buffers,
            scratch,
            weights,
            gate_up_activation_replay,
            forward_replay,
        };
        fixture.replay_forward();
        fixture
    }

    fn replay_gate_up_activation(&self) {
        self.stream.submit_replay(&self.gate_up_activation_replay).wait();
    }

    fn replay_forward(&self) {
        self.stream.submit_replay(&self.forward_replay).wait();
    }

    fn weights(&self) -> QuantizedDenseMLPWeights<'_> {
        QuantizedDenseMLPWeights {
            gate_up_weight: &self.weights.gate_up_weight,
            gate_up_scales: &self.weights.gate_up_scales,
            gate_up_biases: &self.weights.gate_up_biases,
            down_weight: &self.weights.down_weight,
            down_scales: &self.weights.down_scales,
            down_biases: &self.weights.down_biases,
        }
    }
}

fn build_gate_up_activation_replay(
    stream: &Stream,
    kernels: &QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    buffers: &QuantizedDenseMLPOwnedBuffers,
    scratch: &QuantizedDenseMLPOwnedScratch,
    weights: &QuantizedDenseMLPOwnedWeights,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_gate_up(
        shape,
        &buffers.hidden_state,
        &scratch.gate_up_proj,
        QuantizedDenseMLPWeights {
            gate_up_weight: &weights.gate_up_weight,
            gate_up_scales: &weights.gate_up_scales,
            gate_up_biases: &weights.gate_up_biases,
            down_weight: &weights.down_weight,
            down_scales: &weights.down_scales,
            down_biases: &weights.down_biases,
        },
    ));
    builder.record_with_barrier_before(kernels.invoke_activation(shape, &scratch.gate_up_proj, &scratch.activation));
    builder.build()
}

fn build_forward_replay(
    stream: &Stream,
    kernels: &QuantizedDenseMLPKernels,
    shape: QuantizedDenseMLPShape,
    buffers: &QuantizedDenseMLPOwnedBuffers,
    scratch: &QuantizedDenseMLPOwnedScratch,
    weights: &QuantizedDenseMLPOwnedWeights,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke(
        shape,
        QuantizedDenseMLPBuffers {
            hidden_state: &buffers.hidden_state,
            next_hidden_state: &buffers.replay_next_hidden_state,
        },
        QuantizedDenseMLPScratch {
            gate_up_proj: &scratch.gate_up_proj,
            activation: &scratch.replay_activation,
        },
        QuantizedDenseMLPWeights {
            gate_up_weight: &weights.gate_up_weight,
            gate_up_scales: &weights.gate_up_scales,
            gate_up_biases: &weights.gate_up_biases,
            down_weight: &weights.down_weight,
            down_scales: &weights.down_scales,
            down_biases: &weights.down_biases,
        },
    ));
    builder.build()
}

struct QuantizedDenseMLPOwnedBuffers {
    hidden_state: Buffer,
    replay_next_hidden_state: Buffer,
}

struct QuantizedDenseMLPOwnedScratch {
    gate_up_proj: Buffer,
    activation: Buffer,
    replay_activation: Buffer,
}

struct QuantizedDenseMLPOwnedWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

criterion_group!(benches, bench_dense_mlp);
criterion_main!(benches);
