use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::matmul_bf16;

const ROWS: [u32; 5] = [1, 2, 16, 64, 390];
const PROFILES: [Profile; 3] = [
    Profile {
        name: "hidden",
        input_dim: 1024,
        output_dim: 1024,
    },
    Profile {
        name: "ffn-up",
        input_dim: 1024,
        output_dim: 4096,
    },
    Profile {
        name: "conv-out",
        input_dim: 7680,
        output_dim: 1024,
    },
];

fn bench_matmul_bf16(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/matmul-bf16");

    for profile in PROFILES {
        for num_rows in ROWS {
            let fixture = MatmulFixture::new(&device, profile, num_rows);
            group.throughput(Throughput::Elements(num_rows as u64 * profile.output_dim as u64));
            group.bench_function(format!("{}/rows{num_rows}", profile.name), |b| {
                b.iter(|| {
                    fixture.run();
                    black_box(&fixture.output);
                });
            });
        }
    }

    group.finish();
}

#[derive(Clone, Copy)]
struct Profile {
    name: &'static str,
    input_dim: u32,
    output_dim: u32,
}

struct MatmulFixture {
    stream: Stream,
    replay: ReplayProgram,
    output: Buffer,
}

impl MatmulFixture {
    fn new(device: &Device, profile: Profile, num_rows: u32) -> Self {
        let config = matmul_bf16::Config {
            input_dim: profile.input_dim,
            output_dim: profile.output_dim,
        };
        let input = Buffer::new_zeroed_elements(device, (num_rows * profile.input_dim) as usize, Dtype::Bfloat16);
        let output = Buffer::new_zeroed_elements(device, (num_rows * profile.output_dim) as usize, Dtype::Bfloat16);
        let weight = Buffer::new_zeroed_elements(
            device,
            (profile.output_dim * profile.input_dim) as usize,
            Dtype::Bfloat16,
        );
        let matmul = matmul_bf16::Matmul::new(device, config);
        let stream = Stream::new(device);
        let mut recorder = stream.create_replay_program();
        recorder.record(matmul.invoke(num_rows, &output, 0, &input, 0, &weight, 0));
        let replay = recorder.build();
        let fixture = Self { stream, replay, output };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_matmul_bf16);
criterion_main!(benches);
