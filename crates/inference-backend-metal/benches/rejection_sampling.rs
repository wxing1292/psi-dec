//! Benchmarks top-k sampling and sparse rejection components.

use std::mem::size_of;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::components::DSparkMarkovTopKMapBuffers;
use inference_backend_metal::components::DSparkMarkovTopKMapConfig;
use inference_backend_metal::components::DSparkMarkovTopKMapKernel;
use inference_backend_metal::components::DSparkMarkovTopKMapShape;
use inference_backend_metal::components::SparseRejectionSampleBuffers;
use inference_backend_metal::components::SparseRejectionSampleKernel;
use inference_backend_metal::components::SparseRejectionSampleShape;
use inference_backend_metal::components::TopKMergeKernels;
use inference_backend_metal::components::TopKSampleAndWriteDistributionBuffers;
use inference_backend_metal::components::TopKSampleBuffers;
use inference_backend_metal::components::TopKSampleShape;
use inference_backend_metal::components::TopKSamplingOperation;
use inference_backend_metal::components::TopKTileBuffers;
use inference_backend_metal::components::TopKTileKernels;
use inference_backend_metal::components::TopKWriteDistributionBuffers;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::sampling::SamplingDomain;

fn sampling_runtime_params(device: &Device, rows: u32, top_k: u32) -> Buffer {
    let params = Buffer::new_zeroed(device, rows as usize * 6 * size_of::<u32>());
    for row in 0..rows as usize {
        let offset = row * 6;
        params.write_typed(offset, &[0.7f32, 0.8]);
        params.write_typed(offset + 2, &[12345u32, 0, top_k, SamplingDomain::Target as u32]);
    }
    params
}

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    match args.mode {
        BenchMode::TopKSample | BenchMode::TopKWriteDistribution | BenchMode::TopKSampleAndWriteDistribution => {
            for rows in args.rows {
                let fixture = TopKFixture::new(&device, args.mode, rows, args.top_k, args.vocab);
                let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run());
                print_topk_perf(args.mode, rows, args.top_k, args.vocab, args.iters, &samples);
            }
        },
        BenchMode::RejectionSparse => {
            for num_reqs in args.num_reqs {
                for spec_tokens in &args.spec_tokens {
                    let fixture =
                        RejectionFixture::new(&device, args.mode, num_reqs, *spec_tokens, args.top_k, args.vocab);
                    let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run());
                    print_rejection_perf(
                        args.mode,
                        num_reqs,
                        *spec_tokens,
                        args.top_k,
                        args.vocab,
                        args.iters,
                        &samples,
                    );
                }
            }
        },
        BenchMode::DSparkMarkovTopKMap => {
            let config = DSparkMarkovTopKMapConfig {
                vocab_size: args.vocab,
                rank: args.markov_rank,
                w1_group_size: args.markov_w1_group_size,
                w1_bits: args.markov_w1_bits,
                w2_group_size: args.markov_w2_group_size,
                w2_bits: args.markov_w2_bits,
            };
            for rows in args.rows {
                let fixture = DSparkMarkovFixture::new(&device, rows, args.top_k, config);
                let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run());
                print_dspark_markov_perf(rows, args.top_k, config, fixture.vocab_tile_size, args.iters, &samples);
            }
        },
    }
}

#[derive(Clone, Copy)]
enum BenchMode {
    TopKSample,
    TopKWriteDistribution,
    TopKSampleAndWriteDistribution,
    RejectionSparse,
    DSparkMarkovTopKMap,
}

impl BenchMode {
    fn parse(value: &str) -> Self {
        match value {
            "top-k-sample" => Self::TopKSample,
            "top-k-write-distribution" => Self::TopKWriteDistribution,
            "top-k-sample-and-write-distribution" => Self::TopKSampleAndWriteDistribution,
            "rejection-sparse" => Self::RejectionSparse,
            "dspark-markov-top-k-map" => Self::DSparkMarkovTopKMap,
            other => panic!("unknown --mode {other:?}; pass --help for usage"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::TopKSample => "top-k-sample",
            Self::TopKWriteDistribution => "top-k-write-distribution",
            Self::TopKSampleAndWriteDistribution => "top-k-sample-and-write-distribution",
            Self::RejectionSparse => "rejection-sparse",
            Self::DSparkMarkovTopKMap => "dspark-markov-top-k-map",
        }
    }
}

struct Args {
    mode: BenchMode,
    rows: Vec<u32>,
    num_reqs: Vec<u32>,
    spec_tokens: Vec<u32>,
    top_k: u32,
    vocab: u32,
    markov_rank: u32,
    markov_w1_group_size: u32,
    markov_w1_bits: u32,
    markov_w2_group_size: u32,
    markov_w2_bits: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            mode: BenchMode::RejectionSparse,
            rows: vec![1, 4],
            num_reqs: vec![1, 4],
            spec_tokens: vec![1, 4],
            top_k: 32,
            vocab: 32_768,
            markov_rank: 256,
            markov_w1_group_size: 64,
            markov_w1_bits: 4,
            markov_w2_group_size: 64,
            markov_w2_bits: 8,
            iters: 200,
            warmup_iters: 50,
            runs: 7,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--mode" => args.mode = BenchMode::parse(&next_arg(&mut iter, &arg)),
                "--rows" => args.rows = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--num-reqs" => args.num_reqs = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--spec-tokens" => args.spec_tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--top-k" => args.top_k = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--vocab" => args.vocab = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--markov-rank" => args.markov_rank = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--markov-w1-group-size" => args.markov_w1_group_size = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--markov-w1-bits" => args.markov_w1_bits = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--markov-w2-group-size" => args.markov_w2_group_size = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--markov-w2-bits" => args.markov_w2_bits = parse_u32_arg(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.rows.is_empty(), "--rows must include at least one value");
        assert!(!args.num_reqs.is_empty(), "--num-reqs must include at least one value");
        assert!(
            !args.spec_tokens.is_empty(),
            "--spec-tokens must include at least one value"
        );
        assert!(args.top_k > 0, "--top-k must be positive");
        assert!(args.vocab > 0, "--vocab must be positive");
        assert!(args.top_k <= args.vocab, "--top-k must be <= --vocab");
        assert!(args.markov_rank > 0, "--markov-rank must be positive");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

struct RejectionFixture(ReplayFixture);

impl RejectionFixture {
    fn new(device: &Device, mode: BenchMode, reqs: u32, spec_tokens: u32, top_k: u32, vocab: u32) -> Self {
        assert!(matches!(mode, BenchMode::RejectionSparse));
        Self(ReplayFixture::new_sparse_rejection(
            device,
            reqs,
            spec_tokens,
            top_k,
            vocab,
        ))
    }

    fn run(&self) {
        self.0.run();
    }
}

struct DSparkMarkovFixture {
    replay: ReplayFixture,
    vocab_tile_size: u32,
}

impl DSparkMarkovFixture {
    fn new(device: &Device, rows: u32, top_k: u32, config: DSparkMarkovTopKMapConfig) -> Self {
        assert!(rows > 0);
        config.validate();
        let shape = TopKSampleShape {
            num_total_sampling_inputs: rows,
            vocab_size: config.vocab_size,
            top_k,
        };
        let previous_token_ids = Buffer::new_zeroed_elements(device, rows as usize, Dtype::Int32);
        let base_logits =
            Buffer::new_zeroed_elements(device, rows as usize * config.vocab_size as usize, Dtype::Bfloat16);
        let w1_weight = Buffer::new_zeroed(
            device,
            affine_weight_bytes(config.vocab_size, config.rank, config.w1_bits),
        );
        let w1_scales = affine_param_buffer(device, config.vocab_size, config.rank, config.w1_group_size, 1.0);
        let w1_biases = affine_param_buffer(device, config.vocab_size, config.rank, config.w1_group_size, 0.0);
        let w2_weight = Buffer::new_zeroed(
            device,
            affine_weight_bytes(config.vocab_size, config.rank, config.w2_bits),
        );
        let w2_scales = affine_param_buffer(device, config.vocab_size, config.rank, config.w2_group_size, 1.0);
        let w2_biases = affine_param_buffer(device, config.vocab_size, config.rank, config.w2_group_size, 0.0);
        let kernel = DSparkMarkovTopKMapKernel::new(device, config);
        let candidate_count = kernel.candidate_count(shape);
        let tile_token_ids = Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32);
        let tile_logits = Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32);
        let stream = Stream::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            DSparkMarkovTopKMapShape {
                sampling: shape,
                base_logits_row_offset: 0,
            },
            DSparkMarkovTopKMapBuffers {
                previous_token_ids: &previous_token_ids,
                base_logits: &base_logits,
                w1_weight: &w1_weight,
                w1_scales: &w1_scales,
                w1_biases: &w1_biases,
                w2_weight: &w2_weight,
                w2_scales: &w2_scales,
                w2_biases: &w2_biases,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        let replay = builder.build();
        let mut arguments = ReplayArguments::new();
        kernel.add_replay_arguments(shape, rows, &mut arguments);
        let replay = ReplayFixture {
            stream,
            replay,
            arguments,
        };
        replay.run();
        Self {
            replay,
            vocab_tile_size: kernel.vocab_tile_size(),
        }
    }

    fn run(&self) {
        self.replay.run();
    }
}

enum TopKFixture {
    Sample(ReplayFixture),
    WriteDistribution(ReplayFixture),
    SampleAndWriteDistribution(ReplayFixture),
}

impl TopKFixture {
    fn new(device: &Device, mode: BenchMode, rows: u32, top_k: u32, vocab: u32) -> Self {
        match mode {
            BenchMode::TopKSample => Self::Sample(ReplayFixture::new_top_k_sample(device, rows, top_k, vocab)),
            BenchMode::TopKWriteDistribution => {
                Self::WriteDistribution(ReplayFixture::new_top_k_write_distribution(device, rows, top_k, vocab))
            },
            BenchMode::TopKSampleAndWriteDistribution => {
                Self::SampleAndWriteDistribution(ReplayFixture::new_top_k_sample_and_write_distribution(
                    device, rows, top_k, vocab,
                ))
            },
            BenchMode::RejectionSparse => {
                panic!("rejection bench mode cannot build top-k fixture")
            },
            BenchMode::DSparkMarkovTopKMap => {
                panic!("DSpark Markov bench mode cannot build generic top-k fixture")
            },
        }
    }

    fn run(&self) {
        match self {
            Self::Sample(fixture) | Self::WriteDistribution(fixture) | Self::SampleAndWriteDistribution(fixture) => {
                fixture.run()
            },
        }
    }
}

struct ReplayFixture {
    stream: Stream,
    replay: ReplayProgram,
    arguments: ReplayArguments,
}

impl ReplayFixture {
    fn new_top_k_sample(device: &Device, rows: u32, top_k: u32, vocab: u32) -> Self {
        let (stream, shape, logits, tile_token_ids, tile_logits, topk) = top_k_base(device, rows, top_k, vocab);
        let token_ids = Buffer::new_zeroed(device, rows as usize * size_of::<i32>());
        let token_probs = Buffer::new_zeroed(device, rows as usize * size_of::<f32>());
        let runtime_params = sampling_runtime_params(device, rows, top_k);
        let merge = TopKMergeKernels::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            Dtype::Float32,
            TopKSamplingOperation::Sample,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(merge.invoke_sample(
            shape,
            TopKSampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                token_probs: &token_probs,
                runtime_params: &runtime_params,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(&topk, &merge, shape, rows);
        let fixture = Self {
            stream,
            replay,
            arguments,
        };
        fixture.run();
        fixture
    }

    fn new_top_k_write_distribution(device: &Device, rows: u32, top_k: u32, vocab: u32) -> Self {
        let (stream, shape, logits, tile_token_ids, tile_logits, topk) = top_k_base(device, rows, top_k, vocab);
        let distribution_token_ids = Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<i32>());
        let distribution_probs = Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<f32>());
        let runtime_params = sampling_runtime_params(device, rows, top_k);
        let output_distribution_indices = Buffer::from_slice(device, &(0..rows).collect::<Vec<_>>());
        let merge = TopKMergeKernels::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            Dtype::Float32,
            TopKSamplingOperation::WriteDistribution,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(merge.invoke_write_distribution(
            shape,
            TopKWriteDistributionBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                runtime_params: &runtime_params,
                output_distribution_indices: &output_distribution_indices,
                max_k: top_k,
                num_output_distributions: rows,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(&topk, &merge, shape, rows);
        let fixture = Self {
            stream,
            replay,
            arguments,
        };
        fixture.run();
        fixture
    }

    fn new_top_k_sample_and_write_distribution(device: &Device, rows: u32, top_k: u32, vocab: u32) -> Self {
        let (stream, shape, logits, tile_token_ids, tile_logits, topk) = top_k_base(device, rows, top_k, vocab);
        let sampled_token_ids = Buffer::new_zeroed(device, rows as usize * size_of::<i32>());
        let sampled_token_probs = Buffer::new_zeroed(device, rows as usize * size_of::<f32>());
        let distribution_token_ids = Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<i32>());
        let distribution_probs = Buffer::new_zeroed(device, rows as usize * top_k as usize * size_of::<f32>());
        let runtime_params = sampling_runtime_params(device, rows, top_k);
        let output_distribution_indices = Buffer::from_slice(device, &(0..rows).collect::<Vec<_>>());
        let merge = TopKMergeKernels::new(device);
        let mut builder = stream.create_replay_program();
        builder.record(topk.invoke_replay(
            shape,
            Dtype::Float32,
            TopKSamplingOperation::SampleAndWriteDistribution,
            TopKTileBuffers {
                logits: &logits,
                logits_offset_bytes: 0,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(merge.invoke_sample_and_write_distribution(
            shape,
            TopKSampleAndWriteDistributionBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_token_probs,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                runtime_params: &runtime_params,
                output_distribution_indices: &output_distribution_indices,
                max_k: top_k,
                num_output_distributions: rows,
            },
        ));
        let replay = builder.build();
        let arguments = top_k_replay_arguments(&topk, &merge, shape, rows);
        let fixture = Self {
            stream,
            replay,
            arguments,
        };
        fixture.run();
        fixture
    }

    fn new_sparse_rejection(device: &Device, reqs: u32, spec_tokens: u32, top_k: u32, vocab: u32) -> Self {
        assert!(reqs > 0);
        assert!(top_k > 0);
        let rows_per_req = spec_tokens + 1;
        let target_rows = reqs * rows_per_req;
        let draft_rows = reqs * spec_tokens;
        let draft_tokens = draft_tokens(reqs, spec_tokens, vocab);
        let (target_token_ids, target_probs) = target_distribution(reqs, spec_tokens, top_k, vocab, &draft_tokens);
        let (mut draft_token_ids, mut draft_probs) = draft_distribution(reqs, spec_tokens, top_k, vocab, &draft_tokens);
        let mut draft_tokens = draft_tokens;
        if draft_tokens.is_empty() {
            draft_token_ids.resize(top_k as usize, -1);
            draft_probs.resize(top_k as usize, 0.0);
            draft_tokens.push(0);
        }
        let target_token_ids = Buffer::from_slice(device, &target_token_ids);
        let target_probs = Buffer::from_slice(device, &target_probs);
        let draft_distribution_token_ids = Buffer::from_slice(device, &draft_token_ids);
        let draft_probs = Buffer::from_slice(device, &draft_probs);
        let draft_tokens = Buffer::from_slice(device, &draft_tokens);
        let cu_target_distributions = Buffer::from_slice(
            device,
            &(0..=reqs).map(|req| (req * rows_per_req) as i32).collect::<Vec<_>>(),
        );
        let cu_draft_distributions = Buffer::from_slice(
            device,
            &(0..=reqs).map(|req| (req * spec_tokens) as i32).collect::<Vec<_>>(),
        );
        let accepted_token_ids = Buffer::new_zeroed(device, draft_rows.max(1) as usize * size_of::<i32>());
        let accepted_probs = Buffer::new_zeroed(device, draft_rows.max(1) as usize * size_of::<f32>());
        let num_accepted_tokens = Buffer::new_zeroed(device, reqs as usize * size_of::<i32>());
        let sampled_token_ids = Buffer::new_zeroed(device, reqs as usize * size_of::<i32>());
        let sampled_probs = Buffer::new_zeroed(device, reqs as usize * size_of::<f32>());
        let stream = Stream::new(device);
        let kernel = SparseRejectionSampleKernel::new(device);
        let runtime_params = Buffer::from_slice(
            device,
            &(0..reqs).flat_map(|req| [12345_u32, req, top_k, 0]).collect::<Vec<_>>(),
        );
        let shape = SparseRejectionSampleShape {
            num_total_reqs: reqs,
            num_total_target_distributions: target_rows,
            num_total_draft_distributions: draft_rows,
            top_k,
            max_target_k: top_k,
            max_draft_k: top_k,
        };
        let flat_draft_distribution_indices = Buffer::from_slice(device, &(0..draft_rows.max(1)).collect::<Vec<_>>());
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            shape,
            SparseRejectionSampleBuffers {
                target_distribution_token_ids: &target_token_ids,
                target_distribution_probs: &target_probs,
                draft_distribution_token_ids: &draft_distribution_token_ids,
                draft_distribution_probs: &draft_probs,
                flat_draft_token_ids: &draft_tokens,
                cu_target_distributions: &cu_target_distributions,
                cu_draft_distributions: &cu_draft_distributions,
                flat_draft_distribution_indices: &flat_draft_distribution_indices,
                flat_accepted_token_ids: &accepted_token_ids,
                flat_accepted_probs: &accepted_probs,
                num_accepted_tokens: &num_accepted_tokens,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_probs,
                runtime_params: &runtime_params,
            },
        ));
        let replay = builder.build();
        let arguments = rejection_replay_arguments(&kernel, shape, reqs, target_rows, draft_rows);
        let fixture = Self {
            stream,
            replay,
            arguments,
        };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream
            .submit_replay_with_arguments(&self.replay, &self.arguments)
            .wait();
    }
}

fn top_k_replay_arguments(
    topk: &TopKTileKernels,
    merge: &TopKMergeKernels,
    shape: TopKSampleShape,
    num_active_sampling_inputs: u32,
) -> ReplayArguments {
    let mut arguments = ReplayArguments::new();
    topk.add_replay_arguments(shape, num_active_sampling_inputs, &mut arguments);
    merge.add_replay_arguments(shape, num_active_sampling_inputs, &mut arguments);
    arguments
}

fn rejection_replay_arguments(
    kernel: &SparseRejectionSampleKernel,
    shape: SparseRejectionSampleShape,
    num_active_reqs: u32,
    num_active_target_distributions: u32,
    num_active_draft_distributions: u32,
) -> ReplayArguments {
    let expected_num_target_distributions = num_active_draft_distributions
        .checked_add(num_active_reqs)
        .expect("sparse rejection target row count must fit u32");
    assert_eq!(num_active_target_distributions, expected_num_target_distributions);
    let mut arguments = ReplayArguments::new();
    kernel.add_replay_arguments(
        shape,
        num_active_reqs,
        num_active_target_distributions,
        num_active_draft_distributions,
        &mut arguments,
    );
    arguments
}

fn top_k_base(
    device: &Device,
    rows: u32,
    top_k: u32,
    vocab: u32,
) -> (Stream, TopKSampleShape, Buffer, Buffer, Buffer, TopKTileKernels) {
    assert!(rows > 0);
    let shape = TopKSampleShape {
        num_total_sampling_inputs: rows,
        vocab_size: vocab,
        top_k,
    };
    let logits = Buffer::from_slice(device, &logits(rows, vocab));
    let topk = TopKTileKernels::new(device);
    let candidate_count = topk.candidate_count(shape);
    let tile_token_ids = Buffer::new_zeroed(device, candidate_count * size_of::<i32>());
    let tile_logits = Buffer::new_zeroed(device, candidate_count * size_of::<f32>());
    (Stream::new(device), shape, logits, tile_token_ids, tile_logits, topk)
}

fn affine_weight_bytes(vocab_size: u32, rank: u32, bits: u32) -> usize {
    vocab_size as usize * rank as usize * bits as usize / 8
}

fn affine_param_buffer(device: &Device, vocab_size: u32, rank: u32, group_size: u32, value: f32) -> Buffer {
    let len = vocab_size as usize * (rank / group_size) as usize;
    let bits = half::bf16::from_f32(value).to_bits();
    Buffer::from_slice(device, &vec![bits; len])
}

fn logits(rows: u32, vocab: u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows as usize * vocab as usize);
    for row in 0..rows {
        for token in 0..vocab {
            let bits = (row.wrapping_mul(1_103_515_245) ^ token.wrapping_mul(2_654_435_761)).rotate_left(13);
            values.push((bits % 4096) as f32 * (1.0 / 256.0));
        }
    }
    values
}

fn draft_tokens(reqs: u32, spec_tokens: u32, vocab: u32) -> Vec<i32> {
    let mut tokens = Vec::with_capacity((reqs * spec_tokens) as usize);
    for req in 0..reqs {
        for spec in 0..spec_tokens {
            tokens.push(((req * 97 + spec + 17) % vocab) as i32);
        }
    }
    tokens
}

fn target_distribution(
    reqs: u32,
    spec_tokens: u32,
    top_k: u32,
    vocab: u32,
    draft_tokens: &[i32],
) -> (Vec<i32>, Vec<f32>) {
    let rows_per_req = spec_tokens + 1;
    let mut token_ids = vec![-1_i32; (reqs * rows_per_req * top_k) as usize];
    let mut probs = vec![0.0_f32; (reqs * rows_per_req * top_k) as usize];
    for req in 0..reqs {
        for row in 0..rows_per_req {
            let target_row = req * rows_per_req + row;
            let token = if row < spec_tokens {
                draft_tokens[(req * spec_tokens + row) as usize]
            } else {
                ((20_000 + req) % vocab) as i32
            };
            let base = (target_row * top_k) as usize;
            token_ids[base] = token;
            probs[base] = 1.0;
        }
    }
    (token_ids, probs)
}

fn draft_distribution(
    reqs: u32,
    spec_tokens: u32,
    top_k: u32,
    vocab: u32,
    draft_tokens: &[i32],
) -> (Vec<i32>, Vec<f32>) {
    let _ = vocab;
    let mut token_ids = vec![-1_i32; (reqs * spec_tokens * top_k) as usize];
    let mut probs = vec![0.0_f32; (reqs * spec_tokens * top_k) as usize];
    for req in 0..reqs {
        for spec in 0..spec_tokens {
            let draft_row = req * spec_tokens + spec;
            let base = (draft_row * top_k) as usize;
            token_ids[base] = draft_tokens[(req * spec_tokens + spec) as usize];
            probs[base] = 1.0;
        }
    }
    (token_ids, probs)
}

fn measure_runs<F>(runs: usize, warmup_iters: usize, iters: usize, mut f: F) -> Vec<Duration>
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..warmup_iters {
        f();
    }
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        samples.push(start.elapsed());
    }
    samples
}

fn print_topk_perf(mode: BenchMode, rows: u32, top_k: u32, vocab: u32, iters: usize, samples: &[Duration]) {
    let per_iter_us = sorted_per_iter_us(iters, samples);
    println!(
        "perf component=sampling case={} rows={} top_k={} vocab={} iters={} runs={} median_us={:.3} min_us={:.3} \
         max_us={:.3}",
        mode.as_str(),
        rows,
        top_k,
        vocab,
        iters,
        samples.len(),
        median_of_sorted(&per_iter_us),
        per_iter_us[0],
        per_iter_us[per_iter_us.len() - 1]
    );
}

fn print_rejection_perf(
    mode: BenchMode,
    num_reqs: u32,
    spec_tokens: u32,
    top_k: u32,
    vocab: u32,
    iters: usize,
    samples: &[Duration],
) {
    let per_iter_us = sorted_per_iter_us(iters, samples);
    println!(
        "perf component=sampling case={} num_reqs={} spec_tokens={} top_k={} vocab={} iters={} runs={} \
         median_us={:.3} min_us={:.3} max_us={:.3}",
        mode.as_str(),
        num_reqs,
        spec_tokens,
        top_k,
        vocab,
        iters,
        samples.len(),
        median_of_sorted(&per_iter_us),
        per_iter_us[0],
        per_iter_us[per_iter_us.len() - 1]
    );
}

fn print_dspark_markov_perf(
    rows: u32,
    top_k: u32,
    config: DSparkMarkovTopKMapConfig,
    vocab_tile_size: u32,
    iters: usize,
    samples: &[Duration],
) {
    let per_iter_us = sorted_per_iter_us(iters, samples);
    println!(
        "perf component=sampling case=dspark-markov-top-k-map rows={} top_k={} vocab={} markov_rank={} \
         markov_w1_group_size={} markov_w1_bits={} markov_w2_group_size={} markov_w2_bits={} vocab_tile={} iters={} \
         runs={} median_us={:.3} min_us={:.3} max_us={:.3}",
        rows,
        top_k,
        config.vocab_size,
        config.rank,
        config.w1_group_size,
        config.w1_bits,
        config.w2_group_size,
        config.w2_bits,
        vocab_tile_size,
        iters,
        samples.len(),
        median_of_sorted(&per_iter_us),
        per_iter_us[0],
        per_iter_us[per_iter_us.len() - 1]
    );
}

fn sorted_per_iter_us(iters: usize, samples: &[Duration]) -> Vec<f64> {
    let mut per_iter_us = samples
        .iter()
        .map(|elapsed| elapsed.as_secs_f64() * 1_000_000.0 / iters as f64)
        .collect::<Vec<_>>();
    per_iter_us.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("timing sample must not be NaN"));
    per_iter_us
}

fn median_of_sorted(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| parse_u32_arg(part, name))
        .collect()
}

fn parse_u32_arg(value: &str, name: &str) -> u32 {
    value
        .parse::<u32>()
        .unwrap_or_else(|err| panic!("{name} expects a u32, got {value:?}: {err}"))
}

fn parse_usize_arg(value: &str, name: &str) -> usize {
    value
        .parse::<usize>()
        .unwrap_or_else(|err| panic!("{name} expects a usize, got {value:?}: {err}"))
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next().unwrap_or_else(|| panic!("{name} requires a value"))
}

fn print_help_and_exit() -> ! {
    println!(
        "sampling bench\n--mode \
         top-k-sample|top-k-write-distribution|top-k-sample-and-write-distribution|rejection-sparse|\
         dspark-markov-top-k-map\n--rows 1,4\n--num-reqs 1,4\n--spec-tokens 1,4\n--top-k 32\n--vocab \
         32768\n--markov-rank 256\n--markov-w1-group-size 64\n--markov-w1-bits 4\n--markov-w2-group-size \
         64\n--markov-w2-bits 8\n--iters 200 --warmup-iters 50 --runs 7"
    );
    std::process::exit(0);
}
