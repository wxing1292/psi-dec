use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::gather_flat_indices;
use inference_executor_core::model::qwen::v3_5::init_model_config;
use inference_executor_core::model::qwen::v3_5::num_target_hidden_states;
use inference_executor_core::model::qwen::v3_5::sample_sampler_configs;
use inference_executor_core::model::qwen::v3_5::sample_token_positions;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingLogitsDtype;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use inference_executor_metal::model::embed_unembed::Embed;
use inference_executor_metal::model::embed_unembed::EmbedConfig;
use inference_executor_metal::model::embed_unembed::EmbedInput;
use inference_executor_metal::model::embed_unembed::Unembed;
use inference_executor_metal::model::embed_unembed::UnembedConfig;
use inference_executor_metal::model::embed_unembed::UnembedInput;
use inference_executor_metal::model::gather::Gather;
use inference_executor_metal::model::qwen::v3_5::weight::load_qwen35_norm_weight;
use inference_executor_metal::model::rms_norm::RmsNorm;
use inference_executor_metal::sampling::top_k_sampling::TopKSampling;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingInput;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingInputs;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingOutputBuffers;

const DEFAULT_NUM_CACHE_PAGES: usize = 32 * 1024;
const DEFAULT_TOKENS_PER_BLOCK: usize = 8;

fn main() {
    run(
        vec![
            Case::FinalNormGather,
            Case::Unembed,
            Case::UnembedPath,
            Case::Sample,
            Case::SampleReadback,
        ],
        "qwen35_output",
    );
}

pub fn run(allowed_cases: Vec<Case>, bench_name: &str) {
    let args = Args::parse(allowed_cases, bench_name);
    let max_tokens = args
        .shapes
        .iter()
        .map(|shape| shape.num_tokens())
        .max()
        .expect("qwen35_output requires at least one shape");
    let fixture = HeadFixture::new(&args.model_dir, max_tokens);

    for shape in &args.shapes {
        let microbatch = bench_request(*shape, args.num_tokens_per_block, fixture.sampler_config);
        for case in &args.cases {
            let replay = fixture.build_replay(*case, &microbatch);
            fixture.prepare_case(*case, &microbatch);
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || {
                fixture.run_case(*case, &replay, &microbatch);
            });
            print_perf(&args.model_dir, *case, *shape, args.iters, &samples);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchShape {
    num_reqs: u32,
    tokens_per_req: u32,
    context: u32,
}

impl BenchShape {
    fn num_tokens(self) -> usize {
        self.num_reqs as usize * self.tokens_per_req as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Case {
    Embed,
    FinalNormGather,
    Unembed,
    UnembedPath,
    Sample,
    SampleReadback,
}

impl Case {
    fn key(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::FinalNormGather => "final_norm_gather",
            Self::Unembed => "unembed",
            Self::UnembedPath => "unembed_path",
            Self::Sample => "sample",
            Self::SampleReadback => "sample_readback",
        }
    }
}

struct Args {
    model_dir: PathBuf,
    cases: Vec<Case>,
    shapes: Vec<BenchShape>,
    num_cache_pages: usize,
    num_tokens_per_block: usize,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse(allowed_cases: Vec<Case>, bench_name: &str) -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            cases: allowed_cases.clone(),
            shapes: Vec::new(),
            num_cache_pages: DEFAULT_NUM_CACHE_PAGES,
            num_tokens_per_block: DEFAULT_TOKENS_PER_BLOCK,
            iters: 200,
            warmup_iters: 50,
            runs: 7,
        };
        let mut num_reqs = None;
        let mut tokens = None;
        let mut contexts = None;
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(bench_name, &allowed_cases),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--cases" => args.cases = parse_cases(&next_arg(&mut iter, &arg), &allowed_cases),
                "--num-reqs" => num_reqs = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--tokens-per-req" => tokens = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--contexts" => contexts = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--num-cache-pages" => args.num_cache_pages = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--num-tokens-per-block" => {
                    args.num_tokens_per_block = parse_usize_arg(&next_arg(&mut iter, &arg), &arg)
                },
                "--iters" => args.iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        let num_reqs = num_reqs.unwrap_or_else(|| vec![1, 2, 4]);
        let tokens = tokens.unwrap_or_else(|| vec![1]);
        let contexts = contexts.unwrap_or_else(|| vec![32]);
        assert!(!args.cases.is_empty(), "--cases must include at least one case");
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.num_cache_pages > 0, "--num-cache-pages must be positive");
        assert!(args.num_tokens_per_block > 0, "--num-tokens-per-block must be positive");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        for num_reqs in num_reqs {
            assert!(num_reqs > 0, "--num-reqs entries must be positive");
            for &tokens_per_req in &tokens {
                assert!(tokens_per_req > 0, "--tokens-per-req entries must be positive");
                for &context in &contexts {
                    args.shapes.push(BenchShape {
                        num_reqs,
                        tokens_per_req,
                        context,
                    });
                }
            }
        }
        args
    }
}

#[derive(Clone, Copy)]
struct HeadLayout {
    max_tokens: u32,
    vocab_size: u32,
    hidden_dim: u32,
    group_size: u32,
    bits: u32,
    affine_dtype: Dtype,
    hidden_dtype: Dtype,
    rms_norm_eps: f32,
}

impl HeadLayout {
    fn new(model_config: &inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig, max_tokens: usize) -> Self {
        let text = &model_config.text_config;
        let quant = model_config
            .quantization
            .as_ref()
            .expect("qwen35_output requires quantized Qwen3.5 weights");
        let layout = Self {
            max_tokens: max_tokens as u32,
            vocab_size: text.vocab_size as u32,
            hidden_dim: text.hidden_size as u32,
            group_size: quant.group_size as u32,
            bits: quant.bits as u32,
            affine_dtype: Dtype::Bfloat16,
            hidden_dtype: Dtype::Bfloat16,
            rms_norm_eps: text.rms_norm_eps,
        };
        layout.validate();
        layout
    }

    fn validate(self) {
        assert!(self.max_tokens > 0, "qwen35_output requires max_tokens > 0");
        assert!(self.vocab_size > 0, "qwen35_output requires vocab_size > 0");
        assert!(self.hidden_dim > 0, "qwen35_output requires hidden_dim > 0");
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert_eq!(self.hidden_dtype, Dtype::Bfloat16);
        assert!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0);
    }

    fn embedding_config(self) -> EmbedConfig {
        EmbedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            affine_dtype: self.affine_dtype,
            output_dtype: self.hidden_dtype,
        }
    }

    fn unembed_config(self) -> UnembedConfig {
        UnembedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.hidden_dtype,
            output_dtype: self.hidden_dtype,
            affine_dtype: self.affine_dtype,
        }
    }

    fn hidden_bytes(self) -> usize {
        self.max_tokens as usize * self.hidden_dim as usize * self.hidden_dtype.item_size()
    }

    fn token_id_bytes(self) -> usize {
        self.max_tokens as usize * std::mem::size_of::<i32>()
    }
}

struct HeadFixture {
    stream: Stream,
    sampler_config: SamplerConfig,
    token_ids: Buffer,
    input_hidden: Buffer,
    final_norm_hidden: Buffer,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    embedding: Embed,
    final_norm: RmsNorm,
    gather: Gather,
    unembedder: Unembed,
    sampler: TopKSampling,
    sampler_output: TopKSamplingOutputBuffers,
}

struct HeadReplay {
    program: ReplayProgram,
    arguments: ReplayArguments,
}

impl HeadFixture {
    fn new(model_dir: &std::path::Path, max_tokens: usize) -> Self {
        let device = Device::system_default();
        let model_config = init_model_config(model_dir)
            .unwrap_or_else(|err| panic!("unable to init Qwen3.5 config from {}: {err}", model_dir.display()));
        let sampler_config = SamplerConfig::load(model_dir)
            .unwrap_or_else(|err| panic!("unable to init sampler config from {}: {err}", model_dir.display()));
        let layout = HeadLayout::new(&model_config, max_tokens);
        let sampler_bounds = TopKSamplingBounds::from_config(&sampler_config, layout.max_tokens, layout.vocab_size)
            .unwrap_or_else(|err| panic!("unable to init qwen35_output sampling shape: {err}"));
        let unembed_config = layout.unembed_config();
        let mut store = SafeTensorStore::from_model_dir(model_dir)
            .unwrap_or_else(|err| panic!("unable to load safetensors store from {}: {err}", model_dir.display()));
        let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())
            .unwrap_or_else(|err| {
                panic!(
                    "unable to resolve Qwen3.5 weight layout from {}: {err}",
                    model_dir.display()
                )
            });
        let Qwen35ModelWeightBindings { embed, main, unembed } = weight_bindings;
        let embedding = Embed::load(&device, &mut store, layout.embedding_config(), embed)
            .unwrap_or_else(|err| panic!("unable to load qwen35 embedding: {err}"));
        let final_norm_weight = load_qwen35_norm_weight(
            &device,
            &mut store,
            &main.final_norm_weight,
            &[layout.hidden_dim as usize],
            model_config.quantization.is_some(),
        )
        .unwrap_or_else(|err| panic!("unable to load qwen35 final norm: {err}"));
        let unembedder = Unembed::load(&device, &mut store, unembed_config, unembed)
            .unwrap_or_else(|err| panic!("unable to load qwen35 unembed: {err}"));
        store.unload_all();
        Self {
            stream: Stream::new(&device),
            sampler_config,
            token_ids: Buffer::new_zeroed(&device, layout.token_id_bytes()),
            input_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
            final_norm_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
            gather_flat_indices: Buffer::new_zeroed(&device, max_tokens * std::mem::size_of::<u32>()),
            unembed_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
            unembed_logits: Buffer::new_zeroed(&device, unembed_config.logits_bytes()),
            embedding,
            final_norm: RmsNorm::new(
                layout.hidden_dim as usize,
                layout.rms_norm_eps,
                final_norm_weight,
                RmsNorm::kernel(&device),
            ),
            gather: Gather::new(&device),
            unembedder,
            sampler: TopKSampling::new(&device, sampler_bounds),
            sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        }
    }

    fn prepare_case(&self, case: Case, microbatch: &Qwen35Microbatch) {
        self.token_ids.write_typed(0, microbatch.flat_token_ids());
        if matches!(case, Case::FinalNormGather | Case::UnembedPath) {
            self.write_gather_flat_indices(microbatch);
        }
    }

    fn build_replay(&self, case: Case, microbatch: &Qwen35Microbatch) -> HeadReplay {
        let mut builder = MetalReplayRuntime::new(&self.stream).create_recorder();
        let mut arguments = ReplayArguments::new();
        match case {
            Case::Embed => {
                self.record_embed(&mut builder, microbatch.total_tokens() as u32);
            },
            Case::FinalNormGather => {
                self.record_final_norm_gather(&mut builder, microbatch);
            },
            Case::Unembed => {
                self.record_unembed(&mut builder, num_target_hidden_states(microbatch) as u32);
            },
            Case::UnembedPath => {
                self.record_final_norm_gather(&mut builder, microbatch);
                self.record_unembed(&mut builder, num_target_hidden_states(microbatch) as u32);
            },
            Case::Sample | Case::SampleReadback => {
                self.record_sampling(&mut builder, microbatch, &mut arguments);
            },
        }
        HeadReplay {
            program: builder.build(),
            arguments,
        }
    }

    fn run_case(&self, case: Case, replay: &HeadReplay, microbatch: &Qwen35Microbatch) {
        MetalReplayRuntime::new(&self.stream)
            .submit_replay_with_arguments(&replay.program, &replay.arguments)
            .wait();
        if case == Case::SampleReadback {
            self.read_sample_output(num_target_hidden_states(microbatch));
        }
    }

    fn record_embed<'a, R>(&'a self, recorder: &mut R, num_tokens: u32)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        <Embed as ReplayLayer>::record(
            &self.embedding,
            recorder,
            EmbedInput {
                num_tokens,
                token_ids: &self.token_ids,
                output_hidden: &self.input_hidden,
            },
        );
    }

    fn record_final_norm_gather<'a, R>(&'a self, recorder: &mut R, microbatch: &Qwen35Microbatch)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_rows = num_target_hidden_states(microbatch) as u32;
        self.final_norm.record(
            recorder,
            microbatch.total_tokens() as u32,
            &self.input_hidden,
            &self.final_norm_hidden,
        );
        self.gather.record(
            recorder,
            num_rows,
            self.final_norm_hidden.len_bytes() as u32
                / (microbatch.total_tokens() as u32 * Dtype::Bfloat16.item_size() as u32),
            &self.final_norm_hidden,
            &self.gather_flat_indices,
            &self.unembed_hidden,
        );
    }

    fn record_unembed<'a, R>(&'a self, recorder: &mut R, num_rows: u32)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        <Unembed as ReplayLayer>::record(
            &self.unembedder,
            recorder,
            UnembedInput {
                num_rows,
                hidden: &self.unembed_hidden,
                logits: &self.unembed_logits,
            },
        );
    }

    fn record_sampling<'a, R>(
        &'a self,
        recorder: &mut R,
        microbatch: &Qwen35Microbatch,
        arguments: &mut ReplayArguments,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let sampler_configs = sample_sampler_configs(microbatch);
        let sample_positions = sample_token_positions(microbatch);
        let shape = self.sampler.active_shape(&sampler_configs);
        self.sampler
            .set_configs(&sampler_configs, &sample_positions, SamplingDomain::Target);
        self.sampler.add_replay_arguments(shape, arguments);
        <TopKSampling as ReplayLayer>::record(
            &self.sampler,
            recorder,
            TopKSamplingInput {
                shape,
                logits_dtype: TopKSamplingLogitsDtype::Bfloat16,
                inputs: TopKSamplingInputs {
                    logits: &self.unembed_logits,
                    logits_offset_bytes: 0,
                },
                output: self.sampler_output.as_output(),
            },
        );
    }

    fn write_gather_flat_indices(&self, microbatch: &Qwen35Microbatch) {
        let flat_indices = gather_flat_indices(microbatch);
        assert!(!flat_indices.is_empty(), "qwen35_output requires sampled flat tokens");
        self.gather_flat_indices.write_typed(0, &flat_indices);
    }

    fn read_sample_output(&self, num_sampling_inputs: usize) {
        let _ = self.sampler_output.token_ids.read_typed::<i32>(0, num_sampling_inputs);
        let _ = self
            .sampler_output
            .token_probs
            .read_typed::<f32>(0, num_sampling_inputs);
    }
}

fn bench_request(shape: BenchShape, num_tokens_per_block: usize, sampler_config: SamplerConfig) -> Qwen35Microbatch {
    let num_tokens = shape.num_tokens();
    let mut req_slots = Vec::with_capacity(shape.num_reqs as usize);
    let mut block_indices = Vec::with_capacity(shape.num_reqs as usize);
    let mut token_indices = Vec::with_capacity(shape.num_reqs as usize);
    let mut flat_token_ids = Vec::with_capacity(num_tokens);
    let mut cu_tokens = Vec::with_capacity(shape.num_reqs as usize + 1);
    let mut gdn_state_txns = Vec::with_capacity(shape.num_reqs as usize);
    let mut cu_token = 0u32;
    cu_tokens.push(0);
    for req_index in 0..shape.num_reqs {
        let token_index = shape.context + req_index * (shape.context + shape.tokens_per_req + 17);
        req_slots.push(req_index);
        block_indices.push(token_index as usize / num_tokens_per_block);
        token_indices.push(token_index);
        for token_offset in 0..shape.tokens_per_req {
            flat_token_ids.push(1000 + (req_index * 31 + token_offset) as i32);
        }
        cu_token += shape.tokens_per_req;
        cu_tokens.push(cu_token);
        gdn_state_txns.push(GDNStateTxn::new(token_index, shape.tokens_per_req, 0));
    }
    Qwen35Microbatch::new(
        req_slots,
        block_indices,
        token_indices,
        flat_token_ids,
        cu_tokens,
        gdn_state_txns,
        vec![Vec::new(); shape.num_reqs as usize],
        vec![sampler_config; shape.num_reqs as usize],
        vec![true; num_tokens],
    )
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

fn print_perf(model_dir: &std::path::Path, case: Case, shape: BenchShape, iters: usize, samples: &[Duration]) {
    let mut per_iter_us = samples
        .iter()
        .map(|elapsed| elapsed.as_secs_f64() * 1_000_000.0 / iters as f64)
        .collect::<Vec<_>>();
    per_iter_us.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("timing sample must not be NaN"));
    let median = median_of_sorted(&per_iter_us);
    let min = per_iter_us[0];
    let max = per_iter_us[per_iter_us.len() - 1];
    println!(
        "perf component=qwen35-head model_dir={} case={} num_reqs={} tokens_per_req={} num_tokens={} context={} \
         iters={} runs={} median_us={:.3} min_us={:.3} max_us={:.3}",
        model_dir.display(),
        case.key(),
        shape.num_reqs,
        shape.tokens_per_req,
        shape.num_tokens(),
        shape.context,
        iters,
        samples.len(),
        median,
        min,
        max
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

fn parse_cases(value: &str, allowed_cases: &[Case]) -> Vec<Case> {
    let cases = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let case = match part {
                "embed" => Case::Embed,
                "final_norm_gather" => Case::FinalNormGather,
                "unembed" => Case::Unembed,
                "unembed_path" => Case::UnembedPath,
                "sample" => Case::Sample,
                "sample_readback" => Case::SampleReadback,
                other => {
                    panic!(
                        "invalid case {other:?}; expected embed, final_norm_gather, unembed, unembed_path, sample, or \
                         sample_readback"
                    )
                },
            };
            assert!(
                allowed_cases.contains(&case),
                "case {part:?} is not supported by this benchmark; expected {}",
                case_keys(allowed_cases)
            );
            case
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "--cases must include at least one case");
    cases
}

fn case_keys(cases: &[Case]) -> String {
    cases.iter().map(|case| case.key()).collect::<Vec<_>>().join(",")
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .unwrap_or_else(|err| panic!("{name} expects u32 values, got {part:?}: {err}"))
        })
        .collect()
}

fn parse_usize_arg(value: &str, name: &str) -> usize {
    value
        .parse::<usize>()
        .unwrap_or_else(|err| panic!("{name} expects a usize, got {value:?}: {err}"))
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next().unwrap_or_else(|| panic!("{name} requires a value"))
}

fn print_help_and_exit(bench_name: &str, allowed_cases: &[Case]) -> ! {
    println!(
        "{bench_name} bench\n--model-dir PATH\n--cases {}\n--num-reqs 1,2,4\n--tokens-per-req 1\n--contexts \
         0,32,128\n--num-cache-pages 32768\n--num-tokens-per-block 8\n--iters 200 --warmup-iters 50 --runs 7",
        case_keys(allowed_cases),
    );
    std::process::exit(0);
}
