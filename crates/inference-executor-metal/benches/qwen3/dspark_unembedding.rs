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
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_weight_bindings;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembed;
use inference_executor_metal::model::qwen::v3_x::dspark::output::Qwen3xDSparkGatherUnembedArgs;
use inference_executor_metal::model::unembedding::Unembed;
use inference_executor_metal::model::unembedding::UnembedConfig;
use inference_executor_metal::replay::ReplayComponent;

fn main() {
    let args = Args::parse();
    let setup_start = Instant::now();
    let fixture = Fixture::new(&args.model_dir, &args.dspark_model_dir, args.num_requests);
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
        "perf component=qwen3-dspark-unembedding num_requests={} num_rows={} hidden_dim={} vocab_size={} \
         weight_source={} setup_us={:.3} cache_miss_us={:.3} warmup_iters={} iters={} runs={} median_us={:.3} \
         per_iter_us={:.3}",
        args.num_requests,
        fixture.num_rows,
        fixture.hidden_dim,
        fixture.vocab_size,
        fixture.weight_source,
        setup.as_secs_f64() * 1.0e6,
        cache_miss.as_secs_f64() * 1.0e6,
        args.warmup_iters,
        args.iters,
        args.runs,
        median.as_secs_f64() * 1.0e6,
        median.as_secs_f64() * 1.0e6 / args.iters as f64,
    );
}

struct Args {
    model_dir: PathBuf,
    dspark_model_dir: PathBuf,
    num_requests: usize,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            dspark_model_dir: PathBuf::new(),
            num_requests: 1,
            warmup_iters: 20,
            iters: 100,
            runs: 7,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--dspark-model-dir" => args.dspark_model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--num-requests" => args.num_requests = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(
            !args.dspark_model_dir.as_os_str().is_empty(),
            "--dspark-model-dir is required"
        );
        assert!(args.num_requests > 0, "--num-requests must be positive");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

struct Fixture {
    runtime: MetalRuntime,
    replay: ReplayProgram,
    replay_arguments: ReplayArguments,
    _gather_unembed: Qwen3xDSparkGatherUnembed,
    _hidden_input: Buffer,
    _hidden_output: Buffer,
    _logits: Buffer,
    num_rows: usize,
    hidden_dim: usize,
    vocab_size: usize,
    weight_source: &'static str,
}

impl Fixture {
    fn new(main_model_dir: &Path, dspark_model_dir: &Path, num_requests: usize) -> Self {
        let config = init_qwen3x_dspark_config(dspark_model_dir).expect("unable to load Qwen3 DSpark benchmark config");
        let num_rows = num_requests
            .checked_mul(config.block_size)
            .expect("DSpark GatherUnembed row count must fit usize");
        let runtime = MetalRuntime::system_default();
        let device = runtime.device();
        let mut store =
            SafeTensorStore::from_model_dir(dspark_model_dir).expect("unable to open Qwen3 DSpark benchmark weights");
        let bindings = resolve_qwen3x_dspark_weight_bindings(&config, store.index().tensor_names())
            .expect("unable to resolve Qwen3 DSpark benchmark weights");
        let (unembed, unembed_config, weight_source) = if let Some(unembed_bindings) = bindings.unembed {
            let quantization = config
                .quantization
                .as_ref()
                .expect("Qwen3 DSpark benchmark requires quantization")
                .resolve_for_tensor(&unembed_bindings.weight);
            let unembed_config = unembed_config(
                num_rows,
                config.vocab_size,
                config.hidden_size,
                quantization.group_size,
                quantization.bits,
            );
            let mut unembed = Unembed::new(device, unembed_config);
            unembed
                .load_weights(device, &mut store, unembed_bindings)
                .expect("unable to load Qwen3 DSpark unembed weights");
            (unembed, unembed_config, "dspark")
        } else {
            let main_config =
                init_qwen3_model_config(main_model_dir).expect("unable to load Qwen3 benchmark Main config");
            let mut main_store =
                SafeTensorStore::from_model_dir(main_model_dir).expect("unable to open Qwen3 benchmark Main weights");
            let main_bindings = resolve_qwen3_model_weight_bindings(&main_config, main_store.index().tensor_names())
                .expect("unable to resolve Qwen3 benchmark Main weights");
            let quantization = main_config
                .quantization
                .as_ref()
                .expect("Qwen3 Main benchmark requires quantization")
                .resolve_for_tensor(&main_bindings.unembed.weight);
            let unembed_config = unembed_config(
                num_rows,
                main_config.text_config.vocab_size,
                main_config.text_config.hidden_size,
                quantization.group_size,
                quantization.bits,
            );
            let mut unembed = Unembed::new(device, unembed_config);
            unembed
                .load_weights(device, &mut main_store, main_bindings.unembed)
                .expect("unable to load Qwen3 benchmark Main unembed weights");
            (unembed, unembed_config, "main")
        };
        let gather_unembed = Qwen3xDSparkGatherUnembed::new(
            device,
            config.block_size,
            num_requests,
            unembed_config.hidden_dim,
            Rc::new(unembed),
        );
        gather_unembed.prepare(num_requests);
        let hidden_elements = num_rows
            .checked_mul(config.hidden_size)
            .expect("DSpark GatherUnembed hidden capacity must fit usize");
        let hidden_input = Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16);
        let hidden_output = Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16);
        let logits = Buffer::new_zeroed(device, unembed_config.logits_bytes());
        let replay_runtime = MetalReplayRuntime::new(runtime.stream());
        let mut recorder = replay_runtime.create_recorder();
        gather_unembed.record(
            &mut recorder,
            &Qwen3xDSparkGatherUnembedArgs {
                num_requests: num_requests
                    .try_into()
                    .expect("DSpark GatherUnembed request count must fit u32"),
                hidden_input: &hidden_input,
                hidden_output: &hidden_output,
                logits: &logits,
            },
        );
        let replay = recorder.build();
        Self {
            runtime,
            replay,
            replay_arguments: ReplayArguments::new(),
            _gather_unembed: gather_unembed,
            _hidden_input: hidden_input,
            _hidden_output: hidden_output,
            _logits: logits,
            num_rows,
            hidden_dim: config.hidden_size,
            vocab_size: config.vocab_size,
            weight_source,
        }
    }

    fn run(&self) {
        MetalReplayRuntime::new(self.runtime.stream())
            .submit_replay_with_arguments(&self.replay, &self.replay_arguments)
            .wait();
    }
}

fn unembed_config(
    max_tokens: usize,
    vocab_size: usize,
    hidden_dim: usize,
    group_size: usize,
    bits: usize,
) -> UnembedConfig {
    UnembedConfig {
        max_tokens: max_tokens
            .try_into()
            .expect("DSpark GatherUnembed row count must fit u32"),
        vocab_size: vocab_size
            .try_into()
            .expect("DSpark GatherUnembed vocabulary must fit u32"),
        hidden_dim: hidden_dim
            .try_into()
            .expect("DSpark GatherUnembed hidden dimension must fit u32"),
        group_size: group_size
            .try_into()
            .expect("DSpark GatherUnembed group size must fit u32"),
        bits: bits.try_into().expect("DSpark GatherUnembed bits must fit u32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
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

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen3_dspark_unembedding bench\n--model-dir PATH\n--dspark-model-dir PATH\n--num-requests N\n--warmup-iters \
         N\n--iters N\n--runs N"
    );
    std::process::exit(0);
}
