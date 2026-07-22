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
use inference_backend_metal::operators::AffineQuantizedMatmulKernel;
use inference_backend_metal::operators::AffineQuantizedMatmulShape;

const VOCAB_SIZE: i32 = 151_936;
const TOKENS: [i32; 2] = [1, 16];
const PROFILES: [Profile; 2] = [Profile::Qwen27, Profile::Qwen35];

fn bench_unembedding(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/unembedding");
    for profile in PROFILES {
        for tokens in TOKENS {
            let fixture = UnembeddingFixture::new(&device, profile, tokens);
            group.throughput(Throughput::Elements(tokens as u64 * VOCAB_SIZE as u64));
            group.bench_function(format!("{}/replay/tokens{tokens}", profile.key()), |b| {
                b.iter(|| {
                    fixture.run();
                    black_box(&fixture.logits);
                });
            });
        }
    }
    group.finish();
}

#[derive(Clone, Copy)]
enum Profile {
    Qwen27,
    Qwen35,
}

impl Profile {
    fn key(self) -> &'static str {
        match self {
            Self::Qwen27 => "qwen27",
            Self::Qwen35 => "qwen35",
        }
    }

    fn hidden_dim(self) -> i32 {
        match self {
            Self::Qwen27 => 5120,
            Self::Qwen35 => 2048,
        }
    }
}

struct UnembeddingFixture {
    stream: Stream,
    replay: ReplayProgram,
    logits: Buffer,
}

impl UnembeddingFixture {
    fn new(device: &Device, profile: Profile, tokens: i32) -> Self {
        let shape = AffineQuantizedMatmulShape {
            m: tokens,
            n: VOCAB_SIZE,
            k: profile.hidden_dim(),
            group_size: 64,
            bits: 4,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            affine_dtype: Dtype::Bfloat16,
        };
        let kernel = AffineQuantizedMatmulKernel::new(device, shape);
        let hidden = Buffer::new_zeroed(device, shape.input_bytes());
        let weight = Buffer::new_zeroed(device, shape.weight_bytes());
        let scales = Buffer::new_zeroed(device, shape.affine_param_bytes());
        let biases = Buffer::new_zeroed(device, shape.affine_param_bytes());
        let logits = Buffer::new_zeroed(device, shape.output_bytes());
        let stream = Stream::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(&logits, 0, &hidden, 0, &weight, 0, &scales, 0, &biases, 0));
        let replay = builder.build();
        let fixture = Self { stream, replay, logits };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_unembedding);
criterion_main!(benches);
