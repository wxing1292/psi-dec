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
use inference_backend_metal::operators::AffineQuantizedMatmul;
use inference_backend_metal::operators::AffineQuantizedMatmulConfig;
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulKernelKind;

const ROWS: [i32; 5] = [1, 6, 8, 16, 32];
const PROFILES: [Profile; 2] = [
    Profile {
        name: "bf16-bf16-bf16",
        n: 5_120,
        k: 5_120,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "f32-bf16-f32",
        n: 6_144,
        k: 2_048,
        input_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
    },
];
const KERNELS: [BenchKernel; 5] = [
    BenchKernel::Auto,
    BenchKernel::Exact(AffineQuantizedMatmulKernelKind::QmvBn8Bk32),
    BenchKernel::Exact(AffineQuantizedMatmulKernelKind::QmmBm8Bn32),
    BenchKernel::Exact(AffineQuantizedMatmulKernelKind::QmmBm16Bn32),
    BenchKernel::Exact(AffineQuantizedMatmulKernelKind::QmmBm32Bn32),
];

fn bench_affine_quantized_matmul(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/affine-quantized-matmul");

    for profile in PROFILES {
        for rows in ROWS {
            for kernel in KERNELS {
                let fixture = AffineQuantizedMatmulFixture::new(&device, profile, rows, kernel);
                group.throughput(Throughput::Elements(rows as u64 * profile.n as u64));
                group.bench_function(format!("{}/rows{rows}/{}", profile.name, kernel.name()), |b| {
                    b.iter(|| {
                        fixture.run();
                        black_box(&fixture.output);
                    });
                });
            }
        }
    }

    group.finish();
}

#[derive(Clone, Copy)]
struct Profile {
    name: &'static str,
    n: i32,
    k: i32,
    input_dtype: Dtype,
    scale_bias_dtype: Dtype,
    output_dtype: Dtype,
}

impl Profile {
    fn config(self) -> AffineQuantizedMatmulConfig {
        AffineQuantizedMatmulConfig {
            n: self.n,
            k: self.k,
            group_size: 64,
            bits: 4,
            input_dtype: self.input_dtype,
            output_dtype: self.output_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }
}

#[derive(Clone, Copy)]
enum BenchKernel {
    Auto,
    Exact(AffineQuantizedMatmulKernelKind),
}

impl BenchKernel {
    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Exact(AffineQuantizedMatmulKernelKind::QmvBn8Bk32) => "qmv-bn8-bk32",
            Self::Exact(AffineQuantizedMatmulKernelKind::QmvQuadBn64) => "qmv-quad-bn64",
            Self::Exact(AffineQuantizedMatmulKernelKind::QmmBm8Bn32) => "qmm-bm8-bn32",
            Self::Exact(AffineQuantizedMatmulKernelKind::QmmBm16Bn32) => "qmm-bm16-bn32",
            Self::Exact(AffineQuantizedMatmulKernelKind::QmmBm32Bn32) => "qmm-bm32-bn32",
        }
    }
}

struct AffineQuantizedMatmulFixture {
    stream: Stream,
    replay: ReplayProgram,
    output: Buffer,
}

impl AffineQuantizedMatmulFixture {
    fn new(device: &Device, profile: Profile, rows: i32, bench_kernel: BenchKernel) -> Self {
        let config = profile.config();
        let input = Buffer::new_zeroed(device, config.input_bytes(rows));
        let output = Buffer::new_zeroed(device, config.output_bytes(rows));
        let weight = Buffer::new_zeroed(device, config.weight_bytes());
        let scales = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let biases = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let stream = Stream::new(device);
        let mut recorder = stream.create_replay_program();
        match bench_kernel {
            BenchKernel::Auto => {
                let matmul = AffineQuantizedMatmul::new(device, config);
                recorder.record(matmul.invoke(rows, &output, 0, &input, 0, &weight, 0, &scales, 0, &biases, 0));
            },
            BenchKernel::Exact(kind) => {
                let kernel = AffineQuantizedMatmulKernel::new(device, config, kind);
                recorder.record(kernel.invoke(rows, &output, 0, &input, 0, &weight, 0, &scales, 0, &biases, 0));
            },
        }
        let replay = recorder.build();
        let fixture = Self { stream, replay, output };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_affine_quantized_matmul);
criterion_main!(benches);
