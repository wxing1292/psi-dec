use std::mem::size_of;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::sampling::top_k_sampling::TopKSampling;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingInputs;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingSparseDistributionOutput;

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    for &rows in &args.rows {
        let fixture = SamplingFixture::new(&device, args.mode, rows, args.vocab, args.top_k);
        let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run());
        let median = median_duration(&samples);
        println!(
            "mode={} rows={} vocab={} top_k={} iters={} median_us={:.3} per_iter_us={:.3}",
            args.mode.as_str(),
            rows,
            args.vocab,
            args.top_k,
            args.iters,
            median.as_secs_f64() * 1.0e6,
            median.as_secs_f64() * 1.0e6 / args.iters as f64,
        );
    }
}

#[derive(Clone, Copy)]
enum BenchMode {
    Sample,
    SampleAndSparseDistribution,
}

impl BenchMode {
    fn parse(value: &str) -> Self {
        match value {
            "top-k-sample" => Self::Sample,
            "top-k-sample-and-sparse-distribution" => Self::SampleAndSparseDistribution,
            other => panic!("unknown --mode {other:?}; expected top-k-sample or top-k-sample-and-sparse-distribution"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Sample => "top-k-sample",
            Self::SampleAndSparseDistribution => "top-k-sample-and-sparse-distribution",
        }
    }
}

struct Args {
    mode: BenchMode,
    rows: Vec<u32>,
    vocab: u32,
    top_k: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            mode: BenchMode::Sample,
            rows: vec![1, 4],
            vocab: 32_768,
            top_k: 32,
            iters: 100,
            warmup_iters: 20,
            runs: 7,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--mode" => args.mode = BenchMode::parse(&next_arg(&mut iter, &arg)),
                "--rows" => args.rows = parse_u32_list(&next_arg(&mut iter, &arg)),
                "--vocab" => args.vocab = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--top-k" => args.top_k = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.rows.is_empty(), "--rows must include at least one value");
        assert!(
            args.rows.iter().all(|&rows| rows > 0),
            "--rows entries must be positive"
        );
        assert!(args.vocab > 0, "--vocab must be positive");
        assert!(args.top_k > 0, "--top-k must be positive");
        assert!(args.top_k <= args.vocab, "--top-k must not exceed --vocab");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

fn print_help_and_exit() -> ! {
    println!(
        r#"qwen35_sampling bench
--mode top-k-sample|top-k-sample-and-sparse-distribution
--rows 1,4
--vocab 32768
--top-k 32
--iters N
--warmup-iters N
--runs N"#
    );
    std::process::exit(0);
}

struct SamplingFixture {
    stream: Stream,
    replay: ReplayProgram,
    replay_arguments: ReplayArguments,
}

impl SamplingFixture {
    fn new(device: &Device, mode: BenchMode, rows: u32, vocab: u32, top_k: u32) -> Self {
        let config = SamplerConfig {
            temperature: 0.8,
            top_k: top_k as usize,
            top_p: 0.9,
            seed: 12_345,
        };
        let bounds = TopKSamplingBounds {
            max_sampling_inputs: rows,
            vocab_size: vocab,
            top_k,
        };
        let sampler = TopKSampling::new(device, bounds);
        let output = TopKSamplingOutputBuffers::new(device, bounds);
        let logits = Buffer::from_slice(
            device,
            &sampling_logits(rows, vocab)
                .into_iter()
                .map(|value| bf16::from_f32(value).to_bits())
                .collect::<Vec<_>>(),
        );
        let configs = vec![config; rows as usize];
        let sample_positions = (0..rows).collect::<Vec<_>>();
        sampler.set_configs(&configs, &sample_positions, SamplingDomain::Target);
        let shape = sampler.active_shape(&configs);
        let stream = Stream::new(device);
        let runtime = MetalReplayRuntime::new(&stream);
        let mut recorder = runtime.create_recorder();
        match mode {
            BenchMode::Sample => {
                sampler.record_bf16(
                    &mut recorder,
                    shape,
                    TopKSamplingInputs {
                        logits: &logits,
                        logits_offset_bytes: 0,
                    },
                    output.as_output(),
                );
            },
            BenchMode::SampleAndSparseDistribution => {
                let distribution_token_ids =
                    Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<i32>());
                let distribution_probs = Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<f32>());
                let output_distribution_indices = Buffer::from_slice(device, &(0..rows).collect::<Vec<_>>());
                sampler.record_bf16_with_sparse_distribution(
                    &mut recorder,
                    shape,
                    TopKSamplingInputs {
                        logits: &logits,
                        logits_offset_bytes: 0,
                    },
                    output.as_output(),
                    TopKSamplingSparseDistributionOutput {
                        token_ids: &distribution_token_ids,
                        probs: &distribution_probs,
                        output_distribution_indices: &output_distribution_indices,
                        max_k: top_k,
                        num_output_distributions: rows,
                    },
                );
            },
        }
        let replay = recorder.build();
        let mut replay_arguments = ReplayArguments::new();
        sampler.add_replay_arguments(shape, &mut replay_arguments);
        let fixture = Self {
            stream,
            replay,
            replay_arguments,
        };
        fixture.run();
        fixture
    }

    fn run(&self) {
        MetalReplayRuntime::new(&self.stream)
            .submit_replay_with_arguments(&self.replay, &self.replay_arguments)
            .wait();
    }
}

fn sampling_logits(rows: u32, vocab: u32) -> Vec<f32> {
    (0..rows * vocab)
        .map(|index| ((index * 37 + index / vocab * 997) % 10_000) as f32 / 1000.0)
        .collect()
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run_once: impl FnMut()) -> Vec<Duration> {
    for _ in 0..warmup_iters {
        run_once();
    }
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iters {
                run_once();
            }
            start.elapsed()
        })
        .collect()
}

fn median_duration(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2
    } else {
        sorted[mid]
    }
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> String {
    iter.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_u32(value: &str, flag: &str) -> u32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a u32"))
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn parse_u32_list(value: &str) -> Vec<u32> {
    value
        .split(',')
        .map(|part| part.parse().expect("--rows requires comma-separated u32 values"))
        .collect()
}
