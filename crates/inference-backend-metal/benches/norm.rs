use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormConfig;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::components::ResidualAddBuffers;
use inference_backend_metal::components::ResidualAddConfig;
use inference_backend_metal::components::ResidualAddKernel;
use inference_backend_metal::components::ResidualAddRMSNormBuffers;
use inference_backend_metal::components::ResidualAddRMSNormConfig;
use inference_backend_metal::components::ResidualAddRMSNormKernel;
use inference_backend_metal::components::ResidualAddRMSNormKernelKind;
use inference_backend_metal::components::ResidualAddRMSNormShape;
use inference_backend_metal::components::ResidualAddShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

#[path = "support.rs"]
mod support;
use support::affine_param_fixture;
use support::bf16_buffer;
use support::hidden_fixture;

const BENCH_TOKENS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const HIDDEN_DIMS: [u32; 2] = [2048, 5120];
const EPS: f32 = 1.0e-6;
const RMS_ONLY_COMMANDS: usize = 64;

fn bench_residual_add_rms_norm(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/residual-add-rms-norm");

    for hidden_dim in HIDDEN_DIMS {
        for tokens in BENCH_TOKENS {
            let fixture = ResidualAddRMSNormFixture::new(&device, tokens, hidden_dim);
            group.throughput(Throughput::Elements(
                tokens as u64 * hidden_dim as u64 * RMS_ONLY_COMMANDS as u64,
            ));
            group.bench_function(
                format!("rms-only/replay{RMS_ONLY_COMMANDS}/tokens{tokens}/hidden{hidden_dim}"),
                |b| {
                    b.iter(|| {
                        fixture.replay_rms_only();
                        black_box(&fixture.rms_only_output);
                    });
                },
            );
            group.throughput(Throughput::Elements(tokens as u64 * hidden_dim as u64));
            group.bench_function(format!("unfused/replay/tokens{tokens}/hidden{hidden_dim}"), |b| {
                b.iter(|| {
                    fixture.replay_unfused();
                    black_box(&fixture.unfused_norm_output);
                });
            });
            group.bench_function(format!("fused_scalar/replay/tokens{tokens}/hidden{hidden_dim}"), |b| {
                b.iter(|| {
                    fixture.replay_fused_scalar();
                    black_box(&fixture.fused_scalar_norm_output);
                });
            });
            group.bench_function(format!("fused_vec4/replay/tokens{tokens}/hidden{hidden_dim}"), |b| {
                b.iter(|| {
                    fixture.replay_fused_vec4();
                    black_box(&fixture.fused_vec4_norm_output);
                });
            });
        }
    }

    group.finish();
}

struct ResidualAddRMSNormFixture {
    stream: Stream,
    rms_only_replay: ReplayProgram,
    unfused_replay: ReplayProgram,
    fused_scalar_replay: ReplayProgram,
    fused_vec4_replay: ReplayProgram,
    rms_only_output: Buffer,
    unfused_norm_output: Buffer,
    fused_scalar_norm_output: Buffer,
    fused_vec4_norm_output: Buffer,
}

impl ResidualAddRMSNormFixture {
    fn new(device: &Device, tokens: u32, hidden_dim: u32) -> Self {
        let stream = Stream::new(device);
        let rms_norm = RMSNormKernel::new(device, RMSNormConfig::bf16(hidden_dim, EPS));
        let residual_add = ResidualAddKernel::new(device, ResidualAddConfig::bf16());
        let fused_config = ResidualAddRMSNormConfig::bf16(hidden_dim, EPS);
        let fused_scalar =
            ResidualAddRMSNormKernel::new_with_kind(device, fused_config, ResidualAddRMSNormKernelKind::Scalar);
        let fused_vec4 =
            ResidualAddRMSNormKernel::new_with_kind(device, fused_config, ResidualAddRMSNormKernelKind::Bf16Vectorized);
        let shape = ResidualAddRMSNormShape {
            num_total_tokens: tokens,
        };
        let num_values = tokens as usize * hidden_dim as usize;
        let lhs = bf16_buffer(device, &hidden_fixture(tokens as usize, hidden_dim as usize));
        let rhs = bf16_buffer(device, &residual_fixture(num_values));
        let weight = bf16_buffer(device, &affine_param_fixture(hidden_dim as usize));
        let unfused_residual_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let unfused_norm_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let fused_scalar_residual_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let fused_scalar_norm_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let fused_vec4_residual_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let fused_vec4_norm_output = Buffer::new_zeroed(device, num_values * size_of::<u16>());
        let rms_only_a = bf16_buffer(device, &hidden_fixture(tokens as usize, hidden_dim as usize));
        let rms_only_b = Buffer::new_zeroed(device, num_values * size_of::<u16>());

        let rms_only_replay = {
            let mut builder = stream.create_replay_program();
            for command_index in 0..RMS_ONLY_COMMANDS {
                let (input, output) = if command_index.is_multiple_of(2) {
                    (&rms_only_a, &rms_only_b)
                } else {
                    (&rms_only_b, &rms_only_a)
                };
                builder.record_with_barrier_before(rms_norm.invoke(
                    RMSNormShape {
                        num_total_tokens: tokens,
                    },
                    RMSNormBuffers {
                        input,
                        weight: &weight,
                        output,
                    },
                ));
            }
            builder.build()
        };

        let unfused_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(residual_add.invoke(
                ResidualAddShape {
                    num_values: num_values as u32,
                },
                ResidualAddBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &unfused_residual_output,
                },
            ));
            builder.record_with_barrier_before(rms_norm.invoke(
                RMSNormShape {
                    num_total_tokens: tokens,
                },
                RMSNormBuffers {
                    input: &unfused_residual_output,
                    weight: &weight,
                    output: &unfused_norm_output,
                },
            ));
            builder.build()
        };
        let fused_scalar_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(fused_scalar.invoke(
                shape,
                ResidualAddRMSNormBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    weight: &weight,
                    residual_output: &fused_scalar_residual_output,
                    norm_output: &fused_scalar_norm_output,
                },
            ));
            builder.build()
        };
        let fused_vec4_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(fused_vec4.invoke(
                shape,
                ResidualAddRMSNormBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    weight: &weight,
                    residual_output: &fused_vec4_residual_output,
                    norm_output: &fused_vec4_norm_output,
                },
            ));
            builder.build()
        };

        let fixture = Self {
            stream,
            rms_only_replay,
            unfused_replay,
            fused_scalar_replay,
            fused_vec4_replay,
            rms_only_output: rms_only_a,
            unfused_norm_output,
            fused_scalar_norm_output,
            fused_vec4_norm_output,
        };
        fixture.replay_rms_only();
        fixture.replay_unfused();
        fixture.replay_fused_scalar();
        fixture.replay_fused_vec4();
        assert_eq!(
            fixture.unfused_norm_output.read_typed::<u16>(0, num_values),
            fixture.fused_scalar_norm_output.read_typed::<u16>(0, num_values),
            "scalar fused residual-add RMSNorm must match residual add -> RMSNorm"
        );
        assert_eq!(
            fixture.unfused_norm_output.read_typed::<u16>(0, num_values),
            fixture.fused_vec4_norm_output.read_typed::<u16>(0, num_values),
            "vectorized fused residual-add RMSNorm must match residual add -> RMSNorm"
        );
        fixture
    }

    fn replay_rms_only(&self) {
        self.stream.submit_replay(&self.rms_only_replay).wait();
    }

    fn replay_unfused(&self) {
        self.stream.submit_replay(&self.unfused_replay).wait();
    }

    fn replay_fused_scalar(&self) {
        self.stream.submit_replay(&self.fused_scalar_replay).wait();
    }

    fn replay_fused_vec4(&self) {
        self.stream.submit_replay(&self.fused_vec4_replay).wait();
    }
}

fn residual_fixture(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index * 19 + 3) % 37) as f32 * 0.03125 - 0.5)
        .collect()
}

criterion_group!(benches, bench_residual_add_rms_norm);
criterion_main!(benches);
