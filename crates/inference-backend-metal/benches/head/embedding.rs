use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::QuantizedEmbeddingBuffers;
use inference_backend_metal::components::QuantizedEmbeddingConfig;
use inference_backend_metal::components::QuantizedEmbeddingKernel;
use inference_backend_metal::components::QuantizedEmbeddingShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

const VOCAB_SIZE: u32 = 151_936;
const TOKENS: [u32; 2] = [1, 16];
const PROFILES: [Profile; 2] = [Profile::Qwen27, Profile::Qwen35];

fn bench_embedding(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/embedding");
    for profile in PROFILES {
        for tokens in TOKENS {
            let fixture = EmbeddingFixture::new(&device, profile, tokens);
            group.throughput(Throughput::Elements(tokens as u64 * profile.hidden_dim() as u64));
            group.bench_function(format!("{}/replay/tokens{tokens}", profile.key()), |b| {
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

    fn hidden_dim(self) -> u32 {
        match self {
            Self::Qwen27 => 5120,
            Self::Qwen35 => 2048,
        }
    }
}

struct EmbeddingFixture {
    stream: Stream,
    replay: ReplayProgram,
    output: Buffer,
}

impl EmbeddingFixture {
    fn new(device: &Device, profile: Profile, tokens: u32) -> Self {
        let config = QuantizedEmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            hidden_dim: profile.hidden_dim(),
            group_size: 64,
            bits: 4,
            affine_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        let shape = QuantizedEmbeddingShape { num_tokens: tokens };
        let kernel = QuantizedEmbeddingKernel::new(device, config);
        let token_ids = Buffer::from_slice(
            device,
            &(0..tokens).map(|index| (index % VOCAB_SIZE) as i32).collect::<Vec<_>>(),
        );
        let weight = Buffer::new_zeroed(device, config.weight_bytes());
        let scales = Buffer::new_zeroed(device, config.num_affine_params() * Dtype::Bfloat16.item_size());
        let biases = Buffer::new_zeroed(device, config.num_affine_params() * Dtype::Bfloat16.item_size());
        let output = Buffer::new_zeroed(device, shape.num_output_values(config) * Dtype::Bfloat16.item_size());
        let stream = Stream::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            QuantizedEmbeddingBuffers {
                token_ids: &token_ids,
                weight: &weight,
                scales: &scales,
                biases: &biases,
                output: &output,
            },
        ));
        let replay = builder.build();
        let fixture = Self { stream, replay, output };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_embedding);
criterion_main!(benches);
