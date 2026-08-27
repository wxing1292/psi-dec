use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::model::qwen::v3_x::dspark::sampling::Qwen3xDSparkMarkov;
use inference_executor_metal::sampling::sampling_params::SamplingParamsStore;
use inference_executor_metal::sampling::spec_probs::SpecProbsStore;

fn main() {
    let args = Args::parse();
    let setup_start = Instant::now();
    let mut fixture = Fixture::new(
        &args.dspark_model_dir,
        args.num_requests,
        args.temperature,
        args.top_k,
        args.top_p,
        args.seed,
    );
    let setup = setup_start.elapsed();
    let cache_miss_start = Instant::now();
    fixture.run();
    let cache_miss = cache_miss_start.elapsed();
    for _ in 0..args.warmup_iters {
        fixture.run();
    }
    let samples = (0..args.runs)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..args.iters {
                fixture.run();
            }
            start.elapsed()
        })
        .collect::<Vec<_>>();
    let median = median_duration(samples);
    println!(
        "perf component=qwen3-dspark-sampling num_requests={} block_size={} vocab_size={} markov_rank={} \
         temperature={} top_k={} top_p={} seed={} setup_us={:.3} cache_miss_us={:.3} warmup_iters={} iters={} runs={} \
         median_us={:.3} per_iter_us={:.3}",
        args.num_requests,
        fixture.block_size,
        fixture.vocab_size,
        fixture.markov_rank,
        args.temperature,
        args.top_k,
        args.top_p,
        args.seed,
        setup.as_secs_f64() * 1.0e6,
        cache_miss.as_secs_f64() * 1.0e6,
        args.warmup_iters,
        args.iters,
        args.runs,
        median.as_secs_f64() * 1.0e6,
        median.as_secs_f64() * 1.0e6 / args.iters as f64,
    );
    fixture.print_output();
}

struct Args {
    dspark_model_dir: PathBuf,
    num_requests: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u32,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            dspark_model_dir: PathBuf::new(),
            num_requests: 1,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            seed: 42,
            warmup_iters: 20,
            iters: 100,
            runs: 7,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--dspark-model-dir" => args.dspark_model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--num-requests" => args.num_requests = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--temperature" => args.temperature = parse_f32(&next_arg(&mut values, &arg), &arg),
                "--top-k" => args.top_k = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--top-p" => args.top_p = parse_f32(&next_arg(&mut values, &arg), &arg),
                "--seed" => args.seed = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(
            !args.dspark_model_dir.as_os_str().is_empty(),
            "--dspark-model-dir is required"
        );
        assert!(args.num_requests > 0, "--num-requests must be positive");
        assert!(
            args.temperature.is_finite() && args.temperature >= 0.0,
            "--temperature must be finite and nonnegative"
        );
        assert!(args.top_k > 0, "--top-k must be positive");
        assert!(
            args.top_p.is_finite() && args.top_p > 0.0 && args.top_p <= 1.0,
            "--top-p must be finite and in (0, 1]"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

struct Fixture {
    runtime: MetalRuntime,
    replay: ReplayProgram,
    replay_arguments: ReplayArguments,
    markov: Qwen3xDSparkMarkov,
    distribution_store: SpecProbsStore,
    req_slots: Vec<u32>,
    _base_logits: Buffer,
    _hidden: Buffer,
    block_size: usize,
    vocab_size: usize,
    markov_rank: usize,
}

impl Fixture {
    fn new(
        dspark_model_dir: &Path,
        num_requests: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u32,
    ) -> Self {
        let config = init_qwen3x_dspark_config(dspark_model_dir).expect("unable to load Qwen3 DSpark benchmark config");
        assert!(top_k <= config.vocab_size, "--top-k exceeds the DSpark vocabulary");
        let runtime = MetalRuntime::system_default();
        let device = runtime.device();
        let mut store =
            SafeTensorStore::from_model_dir(dspark_model_dir).expect("unable to open Qwen3 DSpark benchmark weights");
        let bindings = resolve_qwen3x_dspark_weight_bindings(&config, store.index().tensor_names())
            .expect("unable to resolve Qwen3 DSpark benchmark weights");
        let bounds = TopKSamplingBounds {
            max_sampling_inputs: num_requests
                .try_into()
                .expect("DSpark sampling benchmark request count must fit u32"),
            vocab_size: config
                .vocab_size
                .try_into()
                .expect("DSpark sampling benchmark vocabulary must fit u32"),
            top_k: top_k.try_into().expect("DSpark sampling benchmark top_k must fit u32"),
        };
        let sampling_params = Rc::new(SamplingParamsStore::new(device, bounds, num_requests as u32));
        let mut markov = Qwen3xDSparkMarkov::new(
            device,
            &config,
            config.block_size,
            &bindings.markov,
            num_requests,
            Rc::clone(&sampling_params),
        )
        .expect("unable to construct Qwen3 DSpark Markov sampling");
        markov
            .load_weights(device, &mut store, &bindings.markov, &bindings.confidence)
            .expect("unable to load Qwen3 DSpark Markov sampling");
        let distribution_store = SpecProbsStore::new(
            device,
            config.block_size,
            num_requests,
            num_requests * (config.block_size + 1),
            top_k,
        );
        let req_slots = (0..num_requests)
            .map(|req_slot| {
                req_slot
                    .try_into()
                    .expect("DSpark sampling benchmark request slot must fit u32")
            })
            .collect::<Vec<_>>();
        let anchor_token_ids = vec![11; num_requests];
        let anchor_positions = vec![0; num_requests];
        let sampler_configs = vec![
            SamplerConfig {
                temperature,
                top_k,
                top_p,
                seed,
            };
            num_requests
        ];
        sampling_params.set(&req_slots, &sampler_configs);
        let shape = markov.prepare(
            &req_slots,
            &anchor_token_ids,
            &anchor_positions,
            &sampler_configs,
            &distribution_store,
        );
        let base_logits = Buffer::new_zeroed_elements(
            device,
            num_requests
                .checked_mul(config.block_size)
                .and_then(|rows| rows.checked_mul(config.vocab_size))
                .expect("DSpark sampling benchmark logit capacity must fit usize"),
            Dtype::Bfloat16,
        );
        let hidden = Buffer::new_zeroed_elements(
            device,
            num_requests
                .checked_mul(config.block_size)
                .and_then(|rows| rows.checked_mul(config.hidden_size))
                .expect("DSpark sampling benchmark hidden capacity must fit usize"),
            Dtype::Bfloat16,
        );
        let replay_runtime = MetalReplayRuntime::new(runtime.stream());
        let mut recorder = replay_runtime.create_recorder();
        markov.record(&mut recorder, shape, &base_logits, &hidden, &distribution_store);
        let replay = recorder.build();
        let mut replay_arguments = ReplayArguments::new();
        markov.add_replay_arguments(shape, &mut replay_arguments);
        Self {
            runtime,
            replay,
            replay_arguments,
            markov,
            distribution_store,
            req_slots,
            _base_logits: base_logits,
            _hidden: hidden,
            block_size: config.block_size,
            vocab_size: config.vocab_size,
            markov_rank: config.markov_rank,
        }
    }

    fn run(&self) {
        MetalReplayRuntime::new(self.runtime.stream())
            .submit_replay_with_arguments(&self.replay, &self.replay_arguments)
            .wait();
    }

    fn print_output(&mut self) {
        let proposal = self.markov.read_proposal(&self.req_slots, &mut self.distribution_store);
        let num_distribution_values = self
            .req_slots
            .len()
            .checked_mul(self.block_size)
            .and_then(|values| values.checked_mul(self.distribution_store.max_k()))
            .expect("DSpark sampling benchmark distribution size must fit usize");
        let distribution_token_ids = self
            .distribution_store
            .draft_token_ids()
            .read_typed::<i32>(0, num_distribution_values);
        let distribution_probs = self
            .distribution_store
            .draft_probs()
            .read_typed::<f32>(0, num_distribution_values);
        let proposal_prob_bits = proposal
            .token_probs
            .iter()
            .map(|probs| probs.iter().map(|prob| prob.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let confidence_bits = proposal
            .confidences
            .iter()
            .map(|values| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        println!(
            "output proposal_token_ids={:?} proposal_prob_bits={proposal_prob_bits:?} \
             confidence_bits={confidence_bits:?} write_distribution_fingerprint={:016x}",
            proposal.token_ids,
            distribution_fingerprint(&distribution_token_ids, &distribution_probs),
        );
    }
}

fn distribution_fingerprint(token_ids: &[i32], probs: &[f32]) -> u64 {
    assert_eq!(token_ids.len(), probs.len());
    let mut hash = 0xcbf29ce484222325u64;
    for byte in token_ids
        .iter()
        .flat_map(|token_id| token_id.to_le_bytes())
        .chain(probs.iter().flat_map(|prob| prob.to_bits().to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2
    } else {
        values[mid]
    }
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn parse_u32(value: &str, flag: &str) -> u32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a u32"))
}

fn parse_f32(value: &str, flag: &str) -> f32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires an f32"))
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen3_dspark_sampling bench\n--dspark-model-dir PATH\n--num-requests N\n--temperature F\n--top-k N\n--top-p \
         F\n--seed N\n--warmup-iters N\n--iters N\n--runs N"
    );
    std::process::exit(0);
}
