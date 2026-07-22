use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::RMSNormBuffers;
use inference_backend_metal::components::RMSNormKernel;
use inference_backend_metal::components::RMSNormShape;
use inference_backend_metal::components::ResidualBuffers;
use inference_backend_metal::components::ResidualKernel;
use inference_backend_metal::components::ResidualRMSNormBuffers;
use inference_backend_metal::components::ResidualRMSNormKernel;
use inference_backend_metal::components::ResidualRMSNormShape;
use inference_backend_metal::components::ResidualShape;
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

fn bench_residual_rms_norm(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/residual-rms-norm");

    for hidden_dim in HIDDEN_DIMS {
        for tokens in BENCH_TOKENS {
            let fixture = ResidualRMSNormFixture::new(&device, tokens, hidden_dim);
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

struct ResidualRMSNormFixture {
    stream: Stream,
    unfused_replay: ReplayProgram,
    fused_scalar_replay: ReplayProgram,
    fused_vec4_replay: ReplayProgram,
    unfused_norm_output: Buffer,
    fused_scalar_norm_output: Buffer,
    fused_vec4_norm_output: Buffer,
}

impl ResidualRMSNormFixture {
    fn new(device: &Device, tokens: u32, hidden_dim: u32) -> Self {
        let stream = Stream::new(device);
        let rms_norm = RMSNormKernel::new(device);
        let residual = ResidualKernel::new(device);
        let fused = ResidualRMSNormKernel::new(device);
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

        let unfused_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(residual.invoke(
                ResidualShape::bf16(num_values as u32),
                ResidualBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    output: &unfused_residual_output,
                },
            ));
            builder.record_with_barrier_before(rms_norm.invoke(
                RMSNormShape::bf16(tokens, hidden_dim),
                RMSNormBuffers {
                    input: &unfused_residual_output,
                    weight: &weight,
                    output: &unfused_norm_output,
                },
                EPS,
            ));
            builder.build()
        };
        let fused_scalar_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(fused.invoke_bf16_scalar(
                ResidualRMSNormShape::bf16(tokens, hidden_dim),
                ResidualRMSNormBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    weight: &weight,
                    residual_output: &fused_scalar_residual_output,
                    norm_output: &fused_scalar_norm_output,
                },
                EPS,
            ));
            builder.build()
        };
        let fused_vec4_replay = {
            let mut builder = stream.create_replay_program();
            builder.record(fused.invoke_bf16_vectorized(
                ResidualRMSNormShape::bf16(tokens, hidden_dim),
                ResidualRMSNormBuffers {
                    lhs: &lhs,
                    rhs: &rhs,
                    weight: &weight,
                    residual_output: &fused_vec4_residual_output,
                    norm_output: &fused_vec4_norm_output,
                },
                EPS,
            ));
            builder.build()
        };

        let fixture = Self {
            stream,
            unfused_replay,
            fused_scalar_replay,
            fused_vec4_replay,
            unfused_norm_output,
            fused_scalar_norm_output,
            fused_vec4_norm_output,
        };
        fixture.replay_unfused();
        fixture.replay_fused_scalar();
        fixture.replay_fused_vec4();
        assert_eq!(
            fixture.unfused_norm_output.read_typed::<u16>(0, num_values),
            fixture.fused_scalar_norm_output.read_typed::<u16>(0, num_values),
            "scalar fused residual RMSNorm must match residual -> RMSNorm"
        );
        assert_eq!(
            fixture.unfused_norm_output.read_typed::<u16>(0, num_values),
            fixture.fused_vec4_norm_output.read_typed::<u16>(0, num_values),
            "vectorized fused residual RMSNorm must match residual -> RMSNorm"
        );
        fixture
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

criterion_group!(benches, bench_residual_rms_norm);
criterion_main!(benches);
