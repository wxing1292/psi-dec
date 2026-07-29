use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;

#[path = "dspark/fixture.rs"]
mod fixture;

use fixture::ExecutionTiming;
use fixture::Fixture;
use fixture::Trajectory;

fn main() {
    let args = Args::parse();
    for &case in &args.cases {
        let setup_start = Instant::now();
        let mut fixture = Fixture::new(
            case,
            &args.model_dir,
            args.dspark_model_dir.as_deref(),
            args.num_requests,
            args.start_context,
            args.num_cache_pages,
        );
        let setup_elapsed = setup_start.elapsed();
        let cache_miss_start = Instant::now();
        let cache_miss = fixture.run();
        let cache_miss_wall = cache_miss_start.elapsed();
        for _ in 0..args.warmup_iters {
            let _ = fixture.run();
        }
        let samples = measure_runs(args.runs, args.iters, || fixture.run());
        print_result(
            case,
            fixture.block_size(),
            args.num_requests,
            args.start_context,
            setup_elapsed,
            cache_miss_wall,
            &cache_miss,
            args.iters,
            &samples,
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    Main,
    DSpark,
}

impl Case {
    fn key(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::DSpark => "dspark",
        }
    }
}

struct Args {
    model_dir: PathBuf,
    dspark_model_dir: Option<PathBuf>,
    cases: Vec<Case>,
    num_requests: usize,
    start_context: usize,
    num_cache_pages: usize,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            dspark_model_dir: None,
            cases: vec![Case::DSpark],
            num_requests: 1,
            start_context: 0,
            num_cache_pages: 32 * 1024,
            warmup_iters: 2,
            iters: 10,
            runs: 3,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--dspark-model-dir" => args.dspark_model_dir = Some(PathBuf::from(next_arg(&mut values, &arg))),
                "--cases" => args.cases = parse_cases(&next_arg(&mut values, &arg)),
                "--num-requests" => args.num_requests = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--start-context" => args.start_context = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--num-cache-pages" => args.num_cache_pages = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(!args.cases.is_empty(), "--cases must not be empty");
        assert!(args.num_requests > 0, "--num-requests must be positive");
        assert!(args.num_cache_pages > 0, "--num-cache-pages must be positive");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        let total_batches = 1usize
            .checked_add(args.warmup_iters)
            .and_then(|value| value.checked_add(args.iters.checked_mul(args.runs)?))
            .expect("Qwen3 DSpark benchmark batch count must fit usize");
        let max_token_advance = if args.cases.contains(&Case::DSpark) {
            let dspark_model_dir = args
                .dspark_model_dir
                .as_ref()
                .expect("the dspark case requires --dspark-model-dir");
            init_qwen3x_dspark_config(dspark_model_dir)
                .expect("unable to load Qwen3 DSpark benchmark config")
                .block_size
                .checked_add(1)
                .expect("Qwen3 DSpark benchmark token advance must fit usize")
        } else {
            1
        };
        let max_context = args
            .start_context
            .checked_add(
                total_batches
                    .checked_mul(max_token_advance)
                    .expect("Qwen3 DSpark benchmark context advance must fit usize"),
            )
            .expect("Qwen3 DSpark benchmark context must fit usize");
        assert!(
            max_context <= fixture::NUM_TOKENS_PER_BLOCK,
            "Qwen3 DSpark benchmark currently supports one cache block; requested run can reach context \
             {max_context}, block capacity is {}",
            fixture::NUM_TOKENS_PER_BLOCK
        );
        args
    }
}

#[derive(Default)]
struct RunSample {
    timing: ExecutionTiming,
    trajectory: Trajectory,
}

fn measure_runs(runs: usize, iters: usize, mut run: impl FnMut() -> (ExecutionTiming, Trajectory)) -> Vec<RunSample> {
    (0..runs)
        .map(|_| {
            let mut sample = RunSample::default();
            for _ in 0..iters {
                let (timing, trajectory) = run();
                sample.timing.add_assign(timing);
                sample.trajectory.add_assign(trajectory);
            }
            sample
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn print_result(
    case: Case,
    block_size: usize,
    num_requests: usize,
    start_context: usize,
    setup_elapsed: Duration,
    cache_miss_wall: Duration,
    cache_miss: &(ExecutionTiming, Trajectory),
    iters: usize,
    samples: &[RunSample],
) {
    let wall_us = sorted_per_iteration(samples, iters, |sample| sample.timing.wall);
    let prepare_us = sorted_per_iteration(samples, iters, |sample| sample.timing.prepare);
    let main_record_us = sorted_per_iteration(samples, iters, |sample| sample.timing.main_record);
    let main_submit_us = sorted_per_iteration(samples, iters, |sample| sample.timing.main_submit);
    let main_read_us = sorted_per_iteration(samples, iters, |sample| sample.timing.main_read);
    let spec_record_us = sorted_per_iteration(samples, iters, |sample| sample.timing.spec_record);
    let spec_submit_us = sorted_per_iteration(samples, iters, |sample| sample.timing.spec_submit);
    let spec_read_us = sorted_per_iteration(samples, iters, |sample| sample.timing.spec_read);
    let commit_us = sorted_per_iteration(samples, iters, |sample| sample.timing.commit);
    let trajectory = samples.iter().fold(Trajectory::default(), |mut total, sample| {
        total.add_assign(sample.trajectory);
        total
    });
    let acceptance = if trajectory.proposed_tokens == 0 {
        0.0
    } else {
        trajectory.accepted_tokens as f64 / trajectory.proposed_tokens as f64
    };
    println!(
        "perf component=qwen3-dspark-executor case={} num_requests={} block_size={} start_context={} setup_us={:.3} \
         cache_miss_wall_us={:.3} cache_miss_main_submit_us={:.3} cache_miss_spec_submit_us={:.3} iters={} runs={} \
         wall_median_us={:.3} prepare_median_us={:.3} main_record_median_us={:.3} main_submit_median_us={:.3} \
         main_read_median_us={:.3} spec_record_median_us={:.3} spec_submit_median_us={:.3} spec_read_median_us={:.3} \
         commit_median_us={:.3} proposed_tokens={} accepted_tokens={} generated_proposals={} sampled_tokens={} \
         acceptance={:.6}",
        case.key(),
        num_requests,
        block_size,
        start_context,
        setup_elapsed.as_secs_f64() * 1.0e6,
        cache_miss_wall.as_secs_f64() * 1.0e6,
        cache_miss.0.main_submit.as_secs_f64() * 1.0e6,
        cache_miss.0.spec_submit.as_secs_f64() * 1.0e6,
        iters,
        samples.len(),
        median_of_sorted(&wall_us),
        median_of_sorted(&prepare_us),
        median_of_sorted(&main_record_us),
        median_of_sorted(&main_submit_us),
        median_of_sorted(&main_read_us),
        median_of_sorted(&spec_record_us),
        median_of_sorted(&spec_submit_us),
        median_of_sorted(&spec_read_us),
        median_of_sorted(&commit_us),
        trajectory.proposed_tokens,
        trajectory.accepted_tokens,
        trajectory.generated_proposals,
        trajectory.sampled_tokens,
        acceptance,
    );
}

fn sorted_per_iteration(samples: &[RunSample], iters: usize, get: impl Fn(&RunSample) -> Duration) -> Vec<f64> {
    let mut values = samples
        .iter()
        .map(|sample| get(sample).as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values
}

fn median_of_sorted(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn parse_cases(value: &str) -> Vec<Case> {
    value
        .split(',')
        .map(|case| {
            match case.trim() {
                "main" => Case::Main,
                "dspark" => Case::DSpark,
                other => panic!("unknown Qwen3 DSpark benchmark case {other:?}"),
            }
        })
        .collect()
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen3_dspark bench\n--model-dir PATH\n--dspark-model-dir PATH\n--cases main,dspark\n--num-requests \
         N\n--start-context N\n--num-cache-pages N\n--warmup-iters N\n--iters N\n--runs N"
    );
    std::process::exit(0);
}
