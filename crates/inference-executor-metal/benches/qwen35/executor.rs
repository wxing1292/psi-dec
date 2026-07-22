use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_hf_mtp;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::ModelOutputTiming;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModelBatchExecutor;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::runtime::Token;

const NUM_TOKENS_PER_BLOCK: usize = 1024;
const NUM_CACHE_PAGES: usize = 32 * 1024;
const QWEN27_MAIN_GQA_PAGE_IDS_PER_BLOCK: usize = 2048;
const QWEN35_MAIN_GQA_PAGE_IDS_PER_BLOCK: usize = 640;

fn main() {
    let args = Args::parse();
    for &case in &args.cases {
        println!(
            "bench_start component=qwen35-executor case={} model_dir={} mtp_model_dir={}",
            case.key(),
            args.model_dir.display(),
            args.mtp_model_dir
                .as_ref()
                .map_or("none".into(), |path| path.display().to_string())
        );
        let setup_start = Instant::now();
        let mut fixture = ExecutorFixture::new(&args, case);
        let setup_elapsed = setup_start.elapsed();
        let cache_miss_start = Instant::now();
        let cache_miss = fixture.run();
        let cache_miss_wall = cache_miss_start.elapsed();
        fixture.warmup(args.warmup_iters);
        let samples = measure_runs(args.runs, args.iters, || fixture.run());
        print_result(case, setup_elapsed, cache_miss_wall, cache_miss, args.iters, &samples);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    E2EWithoutMTP,
    E2EWithMTP,
}

impl Case {
    fn key(self) -> &'static str {
        match self {
            Self::E2EWithoutMTP => "e2e_wo_mtp",
            Self::E2EWithMTP => "e2e_w_mtp",
        }
    }

    fn num_mtp_modules(self) -> usize {
        match self {
            Self::E2EWithoutMTP => 0,
            Self::E2EWithMTP => 1,
        }
    }

    fn max_spec_tokens(self) -> usize {
        match self {
            Self::E2EWithoutMTP => 0,
            Self::E2EWithMTP => self.num_mtp_modules(),
        }
    }
}

struct Args {
    model_dir: PathBuf,
    mtp_model_dir: Option<PathBuf>,
    cases: Vec<Case>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            mtp_model_dir: None,
            cases: vec![Case::E2EWithoutMTP],
            iters: 10,
            warmup_iters: 2,
            runs: 3,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--mtp-model-dir" => args.mtp_model_dir = Some(PathBuf::from(next_arg(&mut values, &arg))),
                "--cases" => args.cases = parse_cases(&next_arg(&mut values, &arg)),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--bench" => {},
                "--help" | "-h" => print_help_and_exit(),
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(!args.cases.is_empty(), "--cases must not be empty");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        if args.cases.contains(&Case::E2EWithMTP) {
            assert!(args.mtp_model_dir.is_some(), "e2e_w_mtp requires --mtp-model-dir");
        }
        args
    }
}

struct ExecutorFixture {
    model: Qwen35Executor,
    case: Case,
    main_gqa_page_ids_per_block: usize,
    mtp_gqa_page_ids_per_block: Vec<usize>,
}

impl ExecutorFixture {
    fn new(args: &Args, case: Case) -> Self {
        let config = Qwen35ExecutorConfig {
            max_requests: 1,
            max_tokens: 1 + case.max_spec_tokens(),
            max_tokens_per_request: NUM_TOKENS_PER_BLOCK,
            num_cache_pages: NUM_CACHE_PAGES,
            num_tokens_per_block: NUM_TOKENS_PER_BLOCK,
            num_mtp_modules: case.num_mtp_modules(),
        };
        let model = match case {
            Case::E2EWithoutMTP => init_qwen_3_5_model(&args.model_dir, config),
            Case::E2EWithMTP => {
                init_qwen_3_5_model_with_hf_mtp(
                    &args.model_dir,
                    args.mtp_model_dir
                        .as_ref()
                        .expect("e2e_w_mtp requires MTP model directory"),
                    config,
                )
            },
        }
        .unwrap_or_else(|err| panic!("unable to initialize {}: {err}", case.key()));
        let main_gqa_page_ids_per_block = match case {
            Case::E2EWithoutMTP => QWEN27_MAIN_GQA_PAGE_IDS_PER_BLOCK,
            Case::E2EWithMTP => QWEN35_MAIN_GQA_PAGE_IDS_PER_BLOCK,
        };
        let mtp_gqa_page_ids_per_block = model.num_mtp_gqa_page_ids_per_block();
        assert!(
            main_gqa_page_ids_per_block + mtp_gqa_page_ids_per_block.iter().sum::<usize>() <= NUM_CACHE_PAGES,
            "Qwen benchmark page IDs must fit the configured cache"
        );
        Self {
            model,
            case,
            main_gqa_page_ids_per_block,
            mtp_gqa_page_ids_per_block,
        }
    }

    fn warmup(&mut self, warmup_iters: usize) {
        for _ in 0..warmup_iters {
            let _ = self.run();
        }
    }

    fn run(&mut self) -> ExecutionTiming {
        match self.case {
            Case::E2EWithoutMTP => self.run_batch(self.batch_request(0, 0, Token::new(11), Vec::new())),
            Case::E2EWithMTP => self.run_mtp_cycle(),
        }
        .0
    }

    fn run_mtp_cycle(&mut self) -> (ExecutionTiming, BatchDeviceResponse) {
        let (proposal_timing, proposal_response) = self.run_batch(self.batch_request(0, 0, Token::new(11), Vec::new()));
        let (target_token, draft_tokens) = mtp_target_input(&proposal_response, self.case.max_spec_tokens());
        let (target_timing, target_response) = self.run_batch(self.batch_request(1, 1, target_token, draft_tokens));
        (proposal_timing.combine(target_timing), target_response)
    }

    fn run_batch(&mut self, core_batch_req: BatchDeviceRequest) -> (ExecutionTiming, BatchDeviceResponse) {
        let prepare_start = Instant::now();
        let model_batch_req = self.model.prepare_batch(&core_batch_req);
        let prepare = prepare_start.elapsed();
        let record_start = Instant::now();
        let mut recorder = self.model.begin_ops_recording(&model_batch_req);
        let hidden = self.model.embed(&mut recorder, &model_batch_req);
        let hidden = self.model.forward_main(&mut recorder, &model_batch_req, hidden);
        let output = self.model.unembed(&mut recorder, &model_batch_req, &hidden);
        let sampled = match self.case {
            Case::E2EWithoutMTP => self.model.sample(&mut recorder, &model_batch_req, &output),
            Case::E2EWithMTP if model_batch_req.microbatch().has_spec_tokens() => {
                self.model.rejection_sample(&mut recorder, &model_batch_req, &output)
            },
            Case::E2EWithMTP => self.model.sample(&mut recorder, &model_batch_req, &output),
        };
        let sampled = self
            .model
            .forward_mtp(&mut recorder, &model_batch_req, &hidden, sampled);
        let record = record_start.elapsed();
        let finish_start = Instant::now();
        let sampled = self.model.finish_ops_recording(recorder, sampled);
        let finish = finish_start.elapsed();
        let stage = self.model.sampled_output_timing(&sampled).unwrap_or_default();
        let feedback_start = Instant::now();
        let response = self.model.commit_batch(core_batch_req, sampled);
        (
            ExecutionTiming {
                stage,
                feedback: feedback_start.elapsed(),
                prepare,
                record,
                finish,
            },
            response,
        )
    }

    fn batch_request(
        &self,
        sequence: u64,
        token_index: usize,
        token: Token,
        spec_tokens: Vec<Token>,
    ) -> BatchDeviceRequest {
        let tokens = QueryTokens::Decode {
            epoch: 0,
            token_index,
            tokens: vec![token],
            spec_tokens,
        };
        BatchDeviceRequest::new(
            sequence,
            [DeviceRequest::new(
                0,
                0,
                tokens,
                DecoderSyncBlocks::new(0, self.kv_page_ids_by_lane(), vec![]),
                Default::default(),
            )],
        )
    }

    fn kv_page_ids_by_lane(&self) -> Vec<Vec<Vec<u32>>> {
        kv_page_ids_by_lane(self.main_gqa_page_ids_per_block, &self.mtp_gqa_page_ids_per_block)
    }
}

fn kv_page_ids_by_lane(main_page_ids_per_block: usize, mtp_page_ids_per_block: &[usize]) -> Vec<Vec<Vec<u32>>> {
    let mut next_page_id = 0_u32;
    let mut lanes = Vec::with_capacity(1 + mtp_page_ids_per_block.len());
    lanes.push(vec![next_page_ids(&mut next_page_id, main_page_ids_per_block)]);
    for &page_ids_per_block in mtp_page_ids_per_block {
        lanes.push(vec![next_page_ids(&mut next_page_id, page_ids_per_block)]);
    }
    lanes
}

fn next_page_ids(next_page_id: &mut u32, count: usize) -> Vec<u32> {
    let first = *next_page_id;
    *next_page_id = next_page_id
        .checked_add(count.try_into().expect("Qwen page ID count must fit u32"))
        .expect("Qwen page ID range must fit u32");
    (first..*next_page_id).collect()
}

fn mtp_target_input(response: &BatchDeviceResponse, expected_drafts: usize) -> (Token, Vec<Token>) {
    assert_eq!(
        response.dev_resps.len(),
        1,
        "MTP benchmark proposal must return one response"
    );
    let SampledTokens::Decode {
        sampled_token,
        spec_tokens,
        ..
    } = &response.dev_resps[0].sampled_tokens
    else {
        panic!("MTP benchmark proposal must return decode tokens");
    };
    assert_eq!(
        spec_tokens.len(),
        expected_drafts,
        "MTP benchmark proposal must produce one draft per configured module"
    );
    (*sampled_token, spec_tokens.clone())
}

struct RunSample {
    wall: Duration,
    stage: ModelOutputTiming,
    feedback: Duration,
    prepare: Duration,
    record: Duration,
    finish: Duration,
}

#[derive(Default)]
struct ExecutionTiming {
    stage: ModelOutputTiming,
    feedback: Duration,
    prepare: Duration,
    record: Duration,
    finish: Duration,
}

impl ExecutionTiming {
    fn combine(mut self, other: Self) -> Self {
        self.stage.add_assign(other.stage);
        self.feedback += other.feedback;
        self.prepare += other.prepare;
        self.record += other.record;
        self.finish += other.finish;
        self
    }

    fn cache_build_cpu_estimate(&self) -> Duration {
        (self.record + self.finish).saturating_sub(submit_wait_elapsed(&self.stage))
    }
}

fn submit_wait_elapsed(stage: &ModelOutputTiming) -> Duration {
    stage.main_replay_elapsed + stage.main_output_replay_elapsed + stage.mtp_replay_elapsed
}

fn measure_runs(runs: usize, iters: usize, mut run: impl FnMut() -> ExecutionTiming) -> Vec<RunSample> {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            let mut timing = ExecutionTiming::default();
            for _ in 0..iters {
                let iteration = run();
                timing.stage.add_assign(iteration.stage);
                timing.feedback += iteration.feedback;
                timing.prepare += iteration.prepare;
                timing.record += iteration.record;
                timing.finish += iteration.finish;
            }
            RunSample {
                wall: start.elapsed(),
                stage: timing.stage,
                feedback: timing.feedback,
                prepare: timing.prepare,
                record: timing.record,
                finish: timing.finish,
            }
        })
        .collect()
}

fn print_result(
    case: Case,
    setup_elapsed: Duration,
    cache_miss_wall: Duration,
    cache_miss: ExecutionTiming,
    iters: usize,
    samples: &[RunSample],
) {
    let mut wall_us = samples
        .iter()
        .map(|sample| sample.wall.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    wall_us.sort_by(f64::total_cmp);
    let mut main_us = samples
        .iter()
        .map(|sample| sample.stage.main_replay_elapsed.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    main_us.sort_by(f64::total_cmp);
    let mut output_us = samples
        .iter()
        .map(|sample| sample.stage.main_output_replay_elapsed.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    output_us.sort_by(f64::total_cmp);
    let mut mtp_us = samples
        .iter()
        .map(|sample| sample.stage.mtp_replay_elapsed.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    mtp_us.sort_by(f64::total_cmp);
    let mut feedback_us = samples
        .iter()
        .map(|sample| sample.feedback.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    feedback_us.sort_by(f64::total_cmp);
    let mut prepare_us = samples
        .iter()
        .map(|sample| sample.prepare.as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    prepare_us.sort_by(f64::total_cmp);
    let mut record_cpu_estimate_us = samples
        .iter()
        .map(|sample| {
            sample
                .record
                .saturating_sub(submit_wait_elapsed(&sample.stage))
                .as_secs_f64()
                * 1.0e6
                / iters as f64
        })
        .collect::<Vec<_>>();
    record_cpu_estimate_us.sort_by(f64::total_cmp);
    let mut finish_cpu_estimate_us = samples
        .iter()
        .map(|sample| {
            sample
                .finish
                .saturating_sub(submit_wait_elapsed(&sample.stage))
                .as_secs_f64()
                * 1.0e6
                / iters as f64
        })
        .collect::<Vec<_>>();
    finish_cpu_estimate_us.sort_by(f64::total_cmp);
    let median_wall_us = median_of_sorted(&wall_us);
    let cache_miss_wall_us = cache_miss_wall.as_secs_f64() * 1.0e6;
    let cache_build_estimate_us = cache_miss.cache_build_cpu_estimate().as_secs_f64() * 1.0e6;
    println!(
        "perf component=qwen35-executor case={} setup_us={:.3} cache_miss_wall_us={:.3} cache_build_estimate_us={:.3} \
         timing=executor-wall iters={} runs={} wall_median_us={:.3} main_median_us={:.3} output_median_us={:.3} \
         mtp_median_us={:.3} prepare_median_us={:.3} record_cpu_estimate_median_us={:.3} \
         finish_cpu_estimate_median_us={:.3} feedback_median_us={:.3}",
        case.key(),
        setup_elapsed.as_secs_f64() * 1.0e6,
        cache_miss_wall_us,
        cache_build_estimate_us,
        iters,
        samples.len(),
        median_wall_us,
        median_of_sorted(&main_us),
        median_of_sorted(&output_us),
        median_of_sorted(&mtp_us),
        median_of_sorted(&prepare_us),
        median_of_sorted(&record_cpu_estimate_us),
        median_of_sorted(&finish_cpu_estimate_us),
        median_of_sorted(&feedback_us),
    );
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
            match case {
                "e2e_wo_mtp" => Case::E2EWithoutMTP,
                "e2e_w_mtp" => Case::E2EWithMTP,
                _ => panic!("unknown qwen35 executor case {case:?}"),
            }
        })
        .collect()
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen35_executor bench\n--model-dir PATH\n--mtp-model-dir PATH\n--cases e2e_wo_mtp,e2e_w_mtp\n--iters \
         N\n--warmup-iters N\n--runs N"
    );
    std::process::exit(0);
}
